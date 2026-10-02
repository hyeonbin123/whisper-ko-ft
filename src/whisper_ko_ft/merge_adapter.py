"""Merge a LoRA adapter into its base model and save it, so that ct2-transformers-converter can read it.

Usage:
    uv run python -m whisper_ko_ft.merge_adapter --model openai/whisper-large-v3-turbo \
        --adapter outputs/turbo-n/best --output outputs/turbo-n/merged
    uv run ct2-transformers-converter --model outputs/turbo-n/merged --output_dir outputs/turbo-n/ct2 \
        --quantization float16 --copy_files tokenizer.json preprocessor_config.json

Besides the processor, the folder gets the flat preprocessor_config.json that faster-whisper reads for the
number of mel bins (128 for large-v3 and turbo; see `ct2_features`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_ko_ft.ct2_features import write_preprocessor_config


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
    processor = WhisperProcessor.from_pretrained(args.model)
    processor.save_pretrained(output)
    write_preprocessor_config(processor.feature_extractor, merged.config.num_mel_bins, output)
    print(f"wrote {output} ({merged.config.num_mel_bins} mel bins)")


if __name__ == "__main__":
    main()
