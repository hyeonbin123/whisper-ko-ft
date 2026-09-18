"""Transcribe a set with a Whisper model and write an error-rate report.

Usage:
    uv run python -m whisper_ko_ft.evaluate --model openai/whisper-small --set zeroth-val
    uv run python -m whisper_ko_ft.evaluate --model outputs/small-a/best --set fleurs-ko-val --name small-a

Decoding settings are fixed in docs/experiments.md ("디코딩"). Test sets need --allow-test, because they
are measured once per stage. The report keeps every reference and hypothesis so rates can be recomputed.
With --channel telephone the report is written as <set>@telephone.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
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


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or "uncommitted"


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
    parser.add_argument("--name", help="report name; defaults to the last part of --model")
    parser.add_argument(
        "--channel", choices=list(CHANNELS), help="pass the audio through a channel simulation"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="only the first N utterances (smoke runs, speed checks)")
    parser.add_argument("--allow-test", action="store_true", help="required for *-test sets")
    parser.add_argument("--no-report", action="store_true", help="print the summary only")
    args = parser.parse_args()

    if args.set_name.endswith("-test") and not args.allow_test:
        parser.error("test sets are measured once per stage; pass --allow-test when the stage is done")

    language, utterances = load_set(args.set_name)
    if args.limit:
        utterances = utterances[: args.limit]
    stores: dict[str, AudioStore] = {}
    model, processor = load_model(args.model, args.adapter)

    rows: list[dict] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for begin in range(0, len(utterances), args.batch_size):
        batch = utterances[begin : begin + args.batch_size]
        audio = [stores.setdefault(u.store, AudioStore(u.store)).read(u) for u in batch]
        if args.channel:
            audio = [CHANNELS[args.channel](samples) for samples in audio]
        inputs = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True, device="cuda"
        )
        with torch.inference_mode():
            ids = model.generate(
                input_features=inputs.input_features.to("cuda", dtype=torch.float16),
                attention_mask=inputs.attention_mask.to("cuda"),
                language=language,
                task="transcribe",
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                return_timestamps=False,
            )
        for utterance, text in zip(batch, processor.batch_decode(ids, skip_special_tokens=True), strict=True):
            rows.append(
                {
                    "id": utterance.id,
                    "speaker": utterance.speaker,
                    "seconds": round(utterance.seconds, 2),
                    "reference": utterance.text,
                    "hypothesis": text.strip(),
                }
            )
        if (begin // args.batch_size) % 20 == 0:
            print(f"{begin + len(batch):,}/{len(utterances):,}", flush=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

    skipped = 0
    for row in rows:
        scored = score(row["reference"], row["hypothesis"], language)
        if scored is None:
            skipped += 1
            row["edits"] = row["length"] = 0
            continue
        row["edits"], row["length"] = scored.edits, scored.length
        row["has_digit"] = bool(_DIGIT.search(row["hypothesis"]))

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
        "over_30_seconds": sum(r["seconds"] > WINDOW_SECONDS for r in rows),
        "audio_seconds": round(audio_seconds, 1),
        "wall_seconds": round(wall, 1),
        "audio_seconds_per_second": round(audio_seconds / wall, 1),
        "batch_size": args.batch_size,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "limit": args.limit,
        "commit": git_commit(),
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.no_report:
        name = args.name or Path(args.model).name
        suffix = f"@{args.channel}" if args.channel else ""
        target = REPORTS / name / f"{args.set_name}{suffix}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "utterances": rows}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
