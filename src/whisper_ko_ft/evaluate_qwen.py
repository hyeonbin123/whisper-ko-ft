"""Transcribe a set with Qwen3-ASR (stage 9) and write two reports from one decoding: raw and fixed.

Usage:
    uv run python -m whisper_ko_ft.evaluate_qwen --set zeroth-val
    uv run python -m whisper_ko_ft.evaluate_qwen --model Qwen/Qwen3-ASR-0.6B-hf --set zeroth-val
    uv run python -m whisper_ko_ft.evaluate_qwen --set zeroth-val500 --limit 100 --out <file>  (fp16 check)
    uv run python -m whisper_ko_ft.evaluate_qwen --adapter outputs/qwen-qn/checkpoint-1000 --name <name>
        --set zeroth-val [--channel telephone]  (stage 10)

Decoding is fixed in docs/experiments.md (9단계): the model's own Transformers classes (transformers >= 5.13)
at a pinned revision, fp16 (or fp32), SDPA attention, greedy, the language forced through the official prompt
(`apply_transcription_request`, which ends the prompt with "language Korean<asr_text>"), at most 256 new
tokens, batch 16, and the first 30 seconds of audio, as the Whisper harness hears them.

One decoding gives two outputs. "raw" is the generated text after the <asr_text> marker with nothing else
done to it; it goes to reports/<name>/. "fixed" is the official output, `processor.extract_transcription`,
which also runs the original implementation's repetition fix (`_detect_and_fix_repetitions`); it goes to
reports/<name>-fixed/. The Whisper harness has no such fix, so the verdict uses raw.

Rows also keep the number of generated tokens, whether the decoding ran into the token limit, and whether any
logits of that utterance were not finite (an fp16 overflow). A CPU run is never a measurement: it needs --out
(the fp32 half of the fp16 check) or --no-report.

Stage 10: --adapter merges a LoRA adapter (`train_qwen`) into the fp16 base before decoding, as `evaluate`
does for Whisper; it needs --name, and the reports record the adapter folder and the SHA-256 of its weights.
--out also takes the checkpoint runs on zeroth-val500. --channel telephone passes the audio through the
telephone channel of stage 4 before the first 30 seconds are kept, and the reports go to <set>@telephone.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import transformers

from whisper_ko_ft.channel import CHANNELS
from whisper_ko_ft.evaluate import (
    MAX_NEW_TOKENS,
    WINDOW_SECONDS,
    batch_seconds_summary,
    git_commit,
    score_rows,
)
from whisper_ko_ft.metrics import error_rate, interval
from whisper_ko_ft.paths import REPORTS
from whisper_ko_ft.store import SAMPLE_RATE, AudioStore, load_set

DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
# Model -> (pinned revision, report name). The revisions are the ones downloaded on 2026-10-04.
REGISTERED = {
    "Qwen/Qwen3-ASR-1.7B-hf": ("bcd2b5b7f32b480ab5790554cfa8347f246a14f3", "qwen3-asr-1.7b"),
    "Qwen/Qwen3-ASR-0.6B-hf": ("7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c", "qwen3-asr-0.6b"),
}
LANGUAGES = {"ko": "Korean", "en": "English"}
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
GENERATE = {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS}
MARKER = "<asr_text>"
OUTPUTS = ("raw", "fixed")
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE


def raw_transcription(text: str) -> str:
    """The official parse of a decoded output (`_parse_single_output`) without its repetition fix."""
    text = text.strip()
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[-1]
    if MARKER in text:
        text = text.split(MARKER, 1)[1]
    return text.strip()


def first_window(audio: np.ndarray) -> np.ndarray:
    """The first 30 seconds: what the Whisper harness hears of a longer utterance."""
    return audio[:WINDOW_SAMPLES] if audio.shape[0] > WINDOW_SAMPLES else audio


def prepare_audio(audio: np.ndarray, channel: str | None) -> np.ndarray:
    """The channel (if any) on the whole utterance, then the first 30 seconds, as the Whisper harness does."""
    return first_window(CHANNELS[channel](audio) if channel else audio)


def generated_lengths(generated: torch.Tensor, eos_token_id) -> list[tuple[int, bool]]:
    """(tokens before the first end token, ran into the token limit instead) for each row."""
    eos = torch.tensor(eos_token_id if isinstance(eos_token_id, list | tuple) else [eos_token_id])
    ended = torch.isin(generated.cpu(), eos)
    lengths = []
    for row in ended:
        hits = row.nonzero()
        lengths.append((int(hits[0]), False) if len(hits) else (int(row.shape[0]), True))
    return lengths


class NonFiniteWatch:
    """Flags the rows of a batch whose logits were ever not finite (an fp16 overflow), through a hook."""

    def __init__(self, head: torch.nn.Module) -> None:
        self.rows: torch.Tensor | None = None
        head.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        bad = ~torch.isfinite(output).flatten(1).all(dim=1)
        self.rows = bad if self.rows is None else self.rows | bad

    def take(self, batch_size: int) -> list[bool]:
        rows, self.rows = self.rows, None
        return [False] * batch_size if rows is None else [bool(flag) for flag in rows.tolist()]


def transcribe(
    model, processor, audio: list[np.ndarray], language: str, device: str, dtype, watch=None
) -> list[dict]:
    """Both outputs of one batch, with the token counts and the overflow flag of each utterance."""
    inputs = processor.apply_transcription_request(audio, language=LANGUAGES[language]).to(device, dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, **GENERATE)
    generated = output[:, inputs["input_ids"].shape[1] :]
    texts = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
    fixed = processor.extract_transcription(texts)
    lengths = generated_lengths(generated, model.generation_config.eos_token_id)
    nonfinite = watch.take(len(texts)) if watch else [False] * len(texts)
    return [
        {
            "raw": raw_transcription(text),
            "fixed": fix,
            "new_tokens": n,
            "hit_token_limit": hit,
            "nonfinite_logits": bad,
        }
        for text, fix, (n, hit), bad in zip(texts, fixed, lengths, nonfinite, strict=True)
    ]


def report_payloads(rows: list[dict], language: str, meta: dict) -> dict[str, dict]:
    """One report per output: the same utterances, each scored on its own hypothesis."""
    changed = sum(row["raw"] != row["fixed"] for row in rows)
    payloads = {}
    for output in OUTPUTS:
        scored = []
        for row in rows:
            out = {key: row[key] for key in ("id", "speaker", "seconds", "reference")}
            out["hypothesis"] = row[output]
            out.update({key: row[key] for key in ("new_tokens", "hit_token_limit", "nonfinite_logits")})
            scored.append(out)
        skipped = score_rows(scored, language)
        kept = [row for row in scored if row["length"]]
        edits = np.array([row["edits"] for row in kept])
        lengths = np.array([row["length"] for row in kept])
        low, high = interval(edits, lengths)
        summary = {
            **meta,
            "output": output,
            "metric": "cer" if language == "ko" else "wer",
            "utterances": len(kept),
            "skipped_empty_reference": skipped,
            "error_rate": round(error_rate(edits, lengths), 5),
            "interval_95": [round(low, 5), round(high, 5)],
            "hypotheses_with_digit": sum(row.get("has_digit", False) for row in kept),
            "empty_hypotheses": sum(not row["hypothesis"].strip() for row in kept),
            "loops": sum(row["edits"] > row["length"] for row in kept),
            "hit_token_limit": sum(row["hit_token_limit"] for row in kept),
            "nonfinite_utterances": sum(row["nonfinite_logits"] for row in kept),
            "non_speech_tags": sum("<non_speech>" in row["hypothesis"] for row in kept),
            "fixed_changed": changed,
        }
        payloads[output] = {"summary": summary, "utterances": scored}
    return payloads


def merge_adapter(model, adapter: str):
    """The base model with a LoRA adapter merged into its weights (in the base model's dtype)."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter).merge_and_unload()


def adapter_info(adapter: str | None) -> dict:
    """The adapter folder and the SHA-256 of its weights, for the report summary."""
    if adapter is None:
        return {"adapter": None, "adapter_sha256": None}
    weights = (Path(adapter) / "adapter_model.safetensors").read_bytes()
    return {"adapter": Path(adapter).as_posix(), "adapter_sha256": hashlib.sha256(weights).hexdigest()}


def load_model(model_id: str, revision: str, dtype, device: str, adapter: str | None = None):
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=dtype, attn_implementation="sdpa"
    )
    if adapter:
        model = merge_adapter(model, adapter)
    return model.to(device).eval(), processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID (a -hf checkpoint)")
    parser.add_argument("--revision", help="defaults to the pinned revision of a registered model")
    parser.add_argument("--set", required=True, dest="set_name")
    parser.add_argument("--name", help="report name of the raw output; the fixed one goes to <name>-fixed")
    parser.add_argument(
        "--adapter", help="LoRA adapter folder to merge into --model (stage 10; needs --name)"
    )
    parser.add_argument(
        "--channel", choices=list(CHANNELS), help="pass the audio through a channel simulation"
    )
    parser.add_argument("--precision", choices=list(DTYPES), default="fp16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="only the first N utterances (fp16 check, smoke, speed)")
    parser.add_argument("--allow-test", action="store_true", help="required for *-test sets")
    parser.add_argument("--no-report", action="store_true", help="print the summaries only")
    parser.add_argument("--out", help="write only the raw report, to this file (the fp16 check)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    args = parser.parse_args()

    if args.set_name.endswith("-test") and not args.allow_test:
        parser.error("test sets are measured once per stage; pass --allow-test when the stage is done")
    if args.adapter and not args.name:
        parser.error("--adapter needs --name (the default is the base model's report folder)")
    revision, name = REGISTERED.get(args.model, (args.revision, args.name))
    revision, name = args.revision or revision, args.name or name
    if not revision or not name:
        parser.error("a model that is not registered needs --revision and --name")
    if args.out and args.no_report:
        parser.error("--out writes a report; drop --no-report")
    if args.device == "cpu" and not (args.out or args.no_report):
        parser.error("a CPU run is not the measured condition; pass --out (the fp16 check) or --no-report")
    if args.out:
        targets = {"raw": Path(args.out)}
    elif args.no_report:
        targets = {}
    else:
        suffix = f"@{args.channel}" if args.channel else ""
        targets = {"raw": REPORTS / name / f"{args.set_name}{suffix}.json"}
        targets["fixed"] = REPORTS / f"{name}-fixed" / f"{args.set_name}{suffix}.json"
    for target in targets.values():
        if target.exists() and not args.overwrite:
            parser.error(f"{target} exists; pass --overwrite to measure it again")
    commit = git_commit()  # the code that measures, before any of it can change

    language, utterances = load_set(args.set_name)
    if args.limit:
        utterances = utterances[: args.limit]
    dtype, cuda = DTYPES[args.precision], args.device == "cuda"
    adapter = adapter_info(args.adapter)  # read before decoding, like the commit
    model, processor = load_model(args.model, revision, dtype, args.device, args.adapter)
    watch = NonFiniteWatch(model.lm_head)
    stores: dict[str, AudioStore] = {}

    rows: list[dict] = []
    batch_seconds: list[float] = []
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for begin in range(0, len(utterances), args.batch_size):
        batch_started = time.perf_counter()
        batch = utterances[begin : begin + args.batch_size]
        audio = [
            prepare_audio(stores.setdefault(u.store, AudioStore(u.store)).read(u), args.channel)
            for u in batch
        ]
        outputs = transcribe(model, processor, audio, language, args.device, dtype, watch)
        batch_seconds.append(time.perf_counter() - batch_started)  # decoding the text waits for the GPU
        for utterance, out in zip(batch, outputs, strict=True):
            rows.append(
                {
                    "id": utterance.id,
                    "speaker": utterance.speaker,
                    "seconds": round(utterance.seconds, 2),
                    "reference": utterance.text,
                    **out,
                }
            )
        if (begin // args.batch_size) % 20 == 0:
            print(f"{begin + len(batch):,}/{len(utterances):,}", flush=True)
    if cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - started

    audio_seconds = sum(row["seconds"] for row in rows)
    meta = {
        "model": args.model,
        "revision": revision,
        **adapter,
        "engine": "transformers qwen3_asr",
        "set": args.set_name,
        "channel": args.channel,
        "language": language,
        "precision": args.precision,
        "device": args.device,
        "attention": model.config._attn_implementation,
        "decoding": {**GENERATE, "language": LANGUAGES[language], "first_seconds": WINDOW_SECONDS},
        "over_30_seconds": sum(row["seconds"] > WINDOW_SECONDS for row in rows),
        "audio_seconds": round(audio_seconds, 1),
        "wall_seconds": round(wall, 1),
        "audio_seconds_per_second": round(audio_seconds / wall, 1),
        **batch_seconds_summary(batch_seconds),
        "batch_size": args.batch_size,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
        "limit": args.limit,
        "commit": commit,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }
    payloads = report_payloads(rows, language, meta)
    for output, payload in payloads.items():
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        if output in targets:
            target = targets[output]
            target.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(payload, ensure_ascii=False, indent=1)
            target.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {target}")


if __name__ == "__main__":
    main()
