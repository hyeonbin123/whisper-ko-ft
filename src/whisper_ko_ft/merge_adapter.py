"""Merge a LoRA adapter into its base model and save it, so that ct2-transformers-converter can read it.

Usage:
    uv run python -m whisper_ko_ft.merge_adapter --model openai/whisper-large-v3-turbo \
        --adapter outputs/turbo-n/best --output outputs/turbo-n/merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    output = Path(args.output)
    merged.save_pretrained(output)
    WhisperProcessor.from_pretrained(args.model).save_pretrained(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
