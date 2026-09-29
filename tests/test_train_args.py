import argparse
import sys
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig

from whisper_ko_ft import train
from whisper_ko_ft.train import resolve_lora_args


def lora_args(**changes) -> argparse.Namespace:
    values = {
        "lora_r": 0,
        "lora_alpha": None,
        "lora_dropout": 0.05,
        "lora_targets": ["q_proj", "v_proj"],
        "init_adapter": None,
    }
    return argparse.Namespace(**{**values, **changes})


def save_adapter(folder) -> None:
    LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "fc1"], bias="none"
    ).save_pretrained(folder)


def test_init_adapter_implies_lora_with_the_saved_settings(tmp_path):
    save_adapter(tmp_path)
    args = lora_args(lora_dropout=0.1, init_adapter=str(tmp_path))

    resolve_lora_args(args)

    used = (args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_targets)
    assert used == (16, 32, 0.05, ["fc1", "q_proj"])  # not the --lora-* values


def test_default_alpha_is_twice_the_rank():
    args = lora_args(lora_r=8)
    resolve_lora_args(args)
    assert args.lora_alpha == 16


def test_full_fine_tuning_is_untouched():
    args = lora_args()
    resolve_lora_args(args)
    assert (args.lora_r, args.lora_alpha) == (0, None)


class Reached(Exception):
    pass


def test_init_adapter_without_lora_r_loads_the_base_in_fp16_and_the_adapter(tmp_path, monkeypatch):
    save_adapter(tmp_path)
    seen = {}

    def fake_model(*args, **kwargs):
        seen["dtype"] = kwargs["dtype"]
        return SimpleNamespace(
            generation_config=SimpleNamespace(), config=SimpleNamespace(decoder_start_token_id=1)
        )

    def fake_adapter(model, path, **kwargs):
        seen["adapter"] = path
        raise Reached

    def training_starts(**kwargs):
        raise Reached

    processor = SimpleNamespace(tokenizer=SimpleNamespace(set_prefix_tokens=lambda **kwargs: None))
    monkeypatch.setattr(train, "load_set", lambda name: ("ko", []))
    monkeypatch.setattr(train.WhisperProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(train.WhisperForConditionalGeneration, "from_pretrained", fake_model)
    monkeypatch.setattr("peft.PeftModel.from_pretrained", fake_adapter)
    monkeypatch.setattr(train, "Seq2SeqTrainingArguments", training_starts)  # stop if the adapter is skipped
    argv = ["train", "--run-name", "x", "--learning-rate", "5e-5", "--init-adapter", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(Reached):
        train.main()

    assert seen == {"dtype": torch.float16, "adapter": str(tmp_path)}
