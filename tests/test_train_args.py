# [테스트 묶음] train.py의 LoRA 인자 처리(resolve_lora_args)와, --init-adapter만 준 실행이 실제로 어댑터를
# 불러오는지 확인한다.
# [왜 중요한가] 예전 코드는 --lora-r만 보고 LoRA 여부를 정해서, --lora-r 없이 --init-adapter만 주면 어댑터를
# 버리고 기본 모델 전체를 fp32로 학습했다(커밋 8ad5a18에서 고침). 오류 없이 다른 실험이 되어 버리는 버그라
# 테스트로 막는다.
# [방법] 모델·처리기·Trainer를 monkeypatch로 가짜로 바꿔, 내려받기나 GPU 없이 main이 어디까지 가는지만 본다.

import argparse
import sys
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig

from whisper_ko_ft import train
from whisper_ko_ft.train import resolve_lora_args


# train.main이 parse_args로 만드는 것과 같은 LoRA 관련 인자 묶음. 기본값은 전체 파인튜닝(rank 0)이고 바꿀 값만
# 넘긴다.
def lora_args(**changes) -> argparse.Namespace:
    values = {
        "lora_r": 0,
        "lora_alpha": None,
        "lora_dropout": 0.05,
        "lora_targets": ["q_proj", "v_proj"],
        "init_adapter": None,
    }
    return argparse.Namespace(**{**values, **changes})


# rank 16, alpha 32, dropout 0.05, 대상 q_proj·fc1인 어댑터 설정을 폴더에 저장한다(저장된 어댑터 흉내).
def save_adapter(folder) -> None:
    LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "fc1"], bias="none"
    ).save_pretrained(folder)


# --init-adapter를 주면 명령줄의 --lora-* 값(dropout 0.1, q/v 대상)이 아니라 저장된 어댑터의 설정을 쓴다. 대상
# 층은 정렬된 목록으로 run.json에 남는다.
def test_init_adapter_implies_lora_with_the_saved_settings(tmp_path):
    save_adapter(tmp_path)
    args = lora_args(lora_dropout=0.1, init_adapter=str(tmp_path))

    resolve_lora_args(args)

    used = (args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_targets)
    assert used == (16, 32, 0.05, ["fc1", "q_proj"])  # not the --lora-* values


# 새 LoRA에서 alpha를 주지 않으면 2 × rank가 된다(3단계 rank 16 → alpha 32).
def test_default_alpha_is_twice_the_rank():
    args = lora_args(lora_r=8)
    resolve_lora_args(args)
    assert args.lora_alpha == 16


# LoRA를 쓰지 않는 전체 파인튜닝(2단계)은 인자가 바뀌지 않는다.
def test_full_fine_tuning_is_untouched():
    args = lora_args()
    resolve_lora_args(args)
    assert (args.lora_r, args.lora_alpha) == (0, None)


# main이 확인하려는 지점까지 왔음을 알리려고 일부러 던지는 예외.
class Reached(Exception):
    pass


# 명령줄에 --init-adapter만 주고 main을 돌리면 기본 모델을 fp16으로 올리고(LoRA 경로) 저장된 어댑터를 불러와야
# 한다. 어댑터를 건너뛰면 Trainer 설정 단계에서 멈추게 해 두어, 고치기 전 동작(fp32로 올리고 어댑터를
# 건너뜀)이면 seen 값이 달라져 실패한다.
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
    # 데이터, 처리기, 모델, PEFT 불러오기, Trainer 설정을 모두 가짜로 바꾼다. monkeypatch는 테스트가 끝나면
    # 원래대로 되돌린다.
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
