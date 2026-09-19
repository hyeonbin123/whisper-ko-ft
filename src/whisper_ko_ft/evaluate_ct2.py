"""Transcribe a set with faster-whisper (CTranslate2), with or without its decoding fallback.

Usage (needs `uv sync --group ct2` and a model converted with ct2-transformers-converter):
    uv run python -m whisper_ko_ft.evaluate_ct2 --model outputs/small-a/ct2 --name a-ct2 --set zeroth-val
    (add --fallback, with another --name, to turn on the decoding fallback)

One utterance at a time, greedy, fixed language, no timestamps, no VAD (docs/experiments.md, "5단계").
--fallback turns on faster-whisper's defaults: when the text compresses too well (a repeated phrase) or the
average log probability is low, it decodes again at a higher temperature.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime

import numpy as np
import torch

from whisper_ko_ft.evaluate import git_commit
from whisper_ko_ft.metrics import error_rate, interval, score
from whisper_ko_ft.paths import REPORTS, ROOT
from whisper_ko_ft.store import AudioStore, load_set

# CTranslate2 needs the CUDA 12 libraries; on Windows the ones that ship with torch are used.
_TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.name == "nt":
    os.add_dll_directory(_TORCH_LIB)
    os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel  # noqa: E402

NO_FALLBACK = {"temperature": 0.0, "compression_ratio_threshold": None, "log_prob_threshold": None}
FALLBACK = {
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="CTranslate2 model folder")
    parser.add_argument("--name", required=True, help="report name")
    parser.add_argument("--set", required=True, dest="set_name")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.set_name.endswith("-test") and not args.allow_test:
        parser.error("test sets are measured once per stage; pass --allow-test when the stage is done")

    language, utterances = load_set(args.set_name)
    if args.limit:
        utterances = utterances[: args.limit]
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    options = FALLBACK if args.fallback else NO_FALLBACK
    stores: dict[str, AudioStore] = {}

    rows: list[dict] = []
    started = time.perf_counter()
    for n, utterance in enumerate(utterances):
        audio = stores.setdefault(utterance.store, AudioStore(utterance.store)).read(utterance)
        segments, _ = model.transcribe(
            audio,
            language=language,
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            **options,
        )
        segments = list(segments)
        text = " ".join(segment.text.strip() for segment in segments)
        scored = score(utterance.text, text, language)
        rows.append(
            {
                "id": utterance.id,
                "seconds": round(utterance.seconds, 2),
                "reference": utterance.text,
                "hypothesis": text,
                "edits": scored.edits if scored else 0,
                "length": scored.length if scored else 0,
                "temperature": max((segment.temperature for segment in segments), default=0.0),
            }
        )
        if n % 400 == 0:
            print(f"{n:,}/{len(utterances):,}", flush=True)
    wall = time.perf_counter() - started

    kept = [r for r in rows if r["length"]]
    edits, lengths = np.array([r["edits"] for r in kept]), np.array([r["length"] for r in kept])
    low, high = interval(edits, lengths)
    audio_seconds = sum(r["seconds"] for r in rows)
    summary = {
        "model": args.model,
        "engine": "faster-whisper",
        "fallback": args.fallback,
        "set": args.set_name,
        "metric": "cer" if language == "ko" else "wer",
        "utterances": len(kept),
        "error_rate": round(error_rate(edits, lengths), 5),
        "interval_95": [round(low, 5), round(high, 5)],
        "more_edits_than_characters": sum(r["edits"] > r["length"] for r in kept),
        "decoded_again": sum(r["temperature"] > 0 for r in rows),
        "audio_seconds_per_second": round(audio_seconds / wall, 1),
        "limit": args.limit,
        "commit": git_commit(),
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    target = REPORTS / args.name / f"{args.set_name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "utterances": rows}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
