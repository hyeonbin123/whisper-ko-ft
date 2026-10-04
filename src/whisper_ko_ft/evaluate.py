"""Transcribe a set with a Whisper model and write an error-rate report.

Usage:
    uv run python -m whisper_ko_ft.evaluate --model openai/whisper-small --set zeroth-val
    uv run python -m whisper_ko_ft.evaluate --model outputs/small-a/best --set fleurs-ko-val --name small-a

Decoding settings are fixed in docs/experiments.md ("디코딩"). Test sets need --allow-test, because they
are measured once per stage. The report keeps every reference and hypothesis so rates can be recomputed.
With --channel telephone the report is written as <set>@telephone.json. Reports are committed records: an
existing one is replaced only with --overwrite, and --adapter needs --name.

--fallback turns on the temperature fallback of transformers' Whisper generate (stage 8): a decoding whose
token bytes compress too well (a loop), or with a low mean log-probability ("ratio-logprob" only), is decoded
again at the next temperature. The report then records each utterance's final temperature.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_ko_ft.channel import CHANNELS
from whisper_ko_ft.metrics import error_rate, interval, score
from whisper_ko_ft.paths import REPORTS, ROOT
from whisper_ko_ft.store import SAMPLE_RATE, AudioStore, load_set

MAX_NEW_TOKENS = 256
WINDOW_SECONDS = 30
_DIGIT = re.compile(r"[0-9]")

# Stage 8 candidates (docs/experiments.md): D1a = "ratio", D1 = "ratio-logprob". The ratio is zlib on the
# token bytes, as transformers computes it (1.35 is its documented value); faster-whisper's 2.4 uses the text.
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
FALLBACKS: dict[str, dict[str, float]] = {
    "ratio": {"compression_ratio_threshold": 1.35},
    "ratio-logprob": {"compression_ratio_threshold": 1.35, "logprob_threshold": -1.0},
}


def decoding_settings(
    fallback: str | None,
    compression_ratio_threshold: float | None = None,
    logprob_threshold: float | None = None,
) -> dict:
    """Extra generate arguments: none for greedy decoding, the fallback candidate otherwise.

    The two thresholds replace the candidate's values in smoke runs (to force decoding again).
    """
    if fallback is None:
        if compression_ratio_threshold is not None or logprob_threshold is not None:
            raise ValueError("thresholds need --fallback")
        return {}
    settings: dict = {"temperature": TEMPERATURES, **FALLBACKS[fallback]}
    if compression_ratio_threshold is not None:
        settings["compression_ratio_threshold"] = compression_ratio_threshold
    if logprob_threshold is not None:
        if "logprob_threshold" not in settings:
            raise ValueError(f"the {fallback!r} fallback has no log-probability threshold")
        settings["logprob_threshold"] = logprob_threshold
    return settings


def final_temperatures(
    calls: list[tuple[float | None, bool]], batch_size: int, temperatures: Sequence[float]
) -> list[tuple[float, bool]]:
    """(final temperature, still failing at the last temperature) of each utterance of a batch.

    `calls` are the fallback decisions in the order transformers makes them: one per utterance at the first
    temperature, then one per utterance still failing, in batch order, at each higher temperature.
    """
    final = [(0.0, False)] * batch_size
    pending, position = list(range(batch_size)), 0
    while pending and position < len(calls):
        level = calls[position : position + len(pending)]
        if len(level) != len(pending) or len({t for t, _ in level}) != 1:
            raise RuntimeError(f"fallback calls do not match the batch: {calls}")
        still = []
        for utterance, (temperature, needs) in zip(pending, level, strict=True):
            final[utterance] = (float(temperature or 0.0), needs)
            if needs:
                still.append(utterance)
        position += len(pending)
        pending = still
    if position != len(calls) or (position == 0 and batch_size):
        raise RuntimeError(f"fallback calls do not match the batch: {calls}")
    for temperature, needs in final:
        if needs and temperature != temperatures[-1]:
            raise RuntimeError(f"decoded again at {temperature} but never at a higher temperature: {calls}")
    return final


class FallbackTrace:
    """Records which utterances transformers decoded again, and at what temperature.

    Whisper's generate decides per decoding in `_need_fallback` (transformers 5.17, generation_whisper.py) but
    does not return the decisions, so the method is wrapped on this model instance.
    """

    def __init__(self, model, settings: dict) -> None:
        self.temperatures = settings["temperature"]
        self.calls: list[tuple[float | None, bool]] = []
        decide = model._need_fallback

        def traced(*args, **kwargs):
            needs, skip = decide(*args, **kwargs)
            temperature = kwargs["temperature"] if "temperature" in kwargs else args[6]
            self.calls.append((temperature, bool(needs)))
            return needs, skip

        model._need_fallback = traced

    def take(self, batch_size: int) -> list[tuple[float, bool]]:
        calls, self.calls = self.calls, []
        return final_temperatures(calls, batch_size, self.temperatures)


def sequences(output) -> torch.Tensor:
    """Token ids from generate, which returns a tensor or, with return_dict_in_generate, an output object."""
    return output if isinstance(output, torch.Tensor) else output["sequences"]


def transcribe(
    model,
    processor,
    features,
    attention_mask,
    language: str,
    settings: dict,
    trace: FallbackTrace | None = None,
) -> tuple[list[str], list[tuple[float, bool]] | None]:
    """Hypotheses of one batch, and each utterance's final temperature when the fallback is traced."""
    with torch.inference_mode():
        output = model.generate(
            input_features=features,
            attention_mask=attention_mask,
            language=language,
            task="transcribe",
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            return_timestamps=False,
            **settings,
        )
    texts = processor.batch_decode(sequences(output), skip_special_tokens=True)
    return texts, trace.take(len(texts)) if trace else None


def score_rows(rows: list[dict], language: str) -> int:
    """Edits and reference length of each row (0 and 0 when the normalized reference is empty).

    Returns the number of rows skipped for an empty reference.
    """
    skipped = 0
    for row in rows:
        scored = score(row["reference"], row["hypothesis"], language)
        if scored is None:
            skipped += 1
            row["edits"] = row["length"] = 0
            continue
        row["edits"], row["length"] = scored.edits, scored.length
        row["has_digit"] = bool(_DIGIT.search(row["hypothesis"]))
    return skipped


def batch_seconds_summary(seconds: list[float]) -> dict:
    """Median and 90th percentile of the time per batch: the latency of one utterance at batch size 1."""
    if not seconds:
        return {"batch_seconds_median": None, "batch_seconds_p90": None}
    median, p90 = np.percentile(seconds, [50, 90])
    return {"batch_seconds_median": round(float(median), 3), "batch_seconds_p90": round(float(p90), 3)}


def decoding_counts(rows: list[dict], fallback: bool) -> dict:
    """Summary counts over the scored utterances: decoded again, loops (more edits than reference), empty."""
    kept = [row for row in rows if row["length"]]
    return {
        "redecoded": sum(row["temperature"] > 0 for row in kept) if fallback else None,
        "fallback_exhausted": sum(row["fallback_exhausted"] for row in kept) if fallback else None,
        "empty_hypotheses": sum(not row["hypothesis"].strip() for row in kept),
        "loops": sum(row["edits"] > row["length"] for row in kept),
    }


def git_commit(root: Path = ROOT) -> str:
    """Short HEAD hash; '+dirty' when code or dependency files differ from HEAD (not reports or docs)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"],
            cwd=root,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        return "unknown"
    if not head:
        return "uncommitted"
    return f"{head}+dirty" if dirty else head


def load_model(path: str, adapter: str | None) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    model = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float16)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    processor = WhisperProcessor.from_pretrained(path)
    return model.to("cuda").eval(), processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID or a local checkpoint folder")
    parser.add_argument("--adapter", help="LoRA adapter folder to merge into --model")
    parser.add_argument("--set", required=True, dest="set_name")
    parser.add_argument(
        "--name", help="report name; defaults to the last part of --model (required with --adapter)"
    )
    parser.add_argument(
        "--channel", choices=list(CHANNELS), help="pass the audio through a channel simulation"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="only the first N utterances (smoke runs, speed checks)")
    parser.add_argument("--allow-test", action="store_true", help="required for *-test sets")
    parser.add_argument("--no-report", action="store_true", help="print the summary only")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    parser.add_argument("--fallback", choices=list(FALLBACKS), help="temperature fallback (stage 8)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed of the fallback, per batch")
    parser.add_argument(
        "--compression-ratio-threshold", type=float, help="smoke runs only (--no-report): replace 1.35"
    )
    parser.add_argument("--logprob-threshold", type=float, help="smoke runs only (--no-report): replace -1.0")
    args = parser.parse_args()

    if args.set_name.endswith("-test") and not args.allow_test:
        parser.error("test sets are measured once per stage; pass --allow-test when the stage is done")
    if args.adapter and not args.name:
        parser.error("--adapter needs --name (the default is the base model's report folder)")
    try:
        settings = decoding_settings(args.fallback, args.compression_ratio_threshold, args.logprob_threshold)
    except ValueError as error:
        parser.error(str(error))
    overridden = args.compression_ratio_threshold is not None or args.logprob_threshold is not None
    if overridden and not args.no_report:
        parser.error("threshold overrides are not registered candidates; pass --no-report (smoke runs)")
    suffix = f"@{args.channel}" if args.channel else ""
    target = REPORTS / (args.name or Path(args.model).name) / f"{args.set_name}{suffix}.json"
    if not args.no_report and target.exists() and not args.overwrite:
        parser.error(f"{target.relative_to(ROOT)} exists; pass --overwrite to measure it again")
    commit = git_commit()  # the code that measures, before any of it can change

    language, utterances = load_set(args.set_name)
    if args.limit:
        utterances = utterances[: args.limit]
    stores: dict[str, AudioStore] = {}
    model, processor = load_model(args.model, args.adapter)
    trace = FallbackTrace(model, settings) if args.fallback else None

    rows: list[dict] = []
    batch_seconds: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for begin in range(0, len(utterances), args.batch_size):
        batch_started = time.perf_counter()
        batch = utterances[begin : begin + args.batch_size]
        audio = [stores.setdefault(u.store, AudioStore(u.store)).read(u) for u in batch]
        if args.channel:
            audio = [CHANNELS[args.channel](samples) for samples in audio]
        inputs = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True, device="cuda"
        )
        if args.fallback:
            torch.manual_seed(args.seed + begin)  # sampled decodings depend only on the batch, not on the run
        texts, decisions = transcribe(
            model,
            processor,
            inputs.input_features.to("cuda", dtype=torch.float16),
            inputs.attention_mask.to("cuda"),
            language,
            settings,
            trace,
        )
        batch_seconds.append(time.perf_counter() - batch_started)  # decoding the text waits for the GPU
        for index, (utterance, text) in enumerate(zip(batch, texts, strict=True)):
            row = {
                "id": utterance.id,
                "speaker": utterance.speaker,
                "seconds": round(utterance.seconds, 2),
                "reference": utterance.text,
                "hypothesis": text.strip(),
            }
            if decisions is not None:
                row["temperature"], row["fallback_exhausted"] = decisions[index]
            rows.append(row)
        if (begin // args.batch_size) % 20 == 0:
            print(f"{begin + len(batch):,}/{len(utterances):,}", flush=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

    skipped = score_rows(rows, language)
    kept = [r for r in rows if r["length"]]
    edits = np.array([r["edits"] for r in kept])
    lengths = np.array([r["length"] for r in kept])
    low, high = interval(edits, lengths)
    audio_seconds = sum(r["seconds"] for r in rows)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "set": args.set_name,
        "channel": args.channel,
        "language": language,
        "metric": "cer" if language == "ko" else "wer",
        "utterances": len(kept),
        "skipped_empty_reference": skipped,
        "error_rate": round(error_rate(edits, lengths), 5),
        "interval_95": [round(low, 5), round(high, 5)],
        "hypotheses_with_digit": sum(r.get("has_digit", False) for r in kept),
        "fallback": args.fallback,
        "decoding": {**settings, "seed": args.seed} if args.fallback else None,
        **decoding_counts(rows, fallback=bool(args.fallback)),
        "over_30_seconds": sum(r["seconds"] > WINDOW_SECONDS for r in rows),
        "audio_seconds": round(audio_seconds, 1),
        "wall_seconds": round(wall, 1),
        "audio_seconds_per_second": round(audio_seconds / wall, 1),
        **batch_seconds_summary(batch_seconds),
        "batch_size": args.batch_size,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "limit": args.limit,
        "commit": commit,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.no_report:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "utterances": rows}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
