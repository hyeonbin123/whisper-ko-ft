"""Qwen3-ASR에 LoRA 어댑터를 학습한다(10단계): fp16 기본 가중치, fp32 어댑터, fp16 autocast와 GradScaler.

사용법:
    uv run --no-sync python -m whisper_ko_ft.train_qwen --run-name qwen-qn > outputs/qwen-qn/train.log 2>&1
    스모크(10단계의 관문): --run-name qwen-qn-smoke --max-steps 50 --longest-at 25 --save-steps 0
    학습 없이 처음 어댑터만(CPU로 충분): --run-name qwen-qn-init --max-steps 0 --device cpu
    멈춘 실행은 같은 명령에 --resume을 더하면 마지막으로 저장한 상태에서 이어 간다

설정은 docs/experiments.md(10단계)에 고정되어 있다. train.py의 Trainer가 아니라 직접 쓴 단순한 루프다.
옵티마이저 스텝마다 손실, 기울기 크기, 손실 배율, GradScaler가 그 스텝을 건너뛰었는지, GPU 메모리를 기록하고,
실행을 일시 정지할 수 있게 하려는 것이다: outputs/<run>/PAUSE가 있는 동안에는 새 스텝을 시작하지
않으며(Windows가 GPU 메모리를 공유 시스템 메모리로 옮기면 감시 스크립트가 이 파일을 만든다), 멈춘 시간은
어느 스텝 시간에도 들어가지 않는다. 6단계 모델 N에 Trainer가 준 설정은 그대로 지킨다: AdamW(GPU에서
fused, betas 0.9/0.999, eps 1e-8, weight decay 없음), 선형 warmup 뒤 0까지 선형 감소, 기울기 자르기 1.0,
GradScaler 기본값, 그리고 GradScaler가 건너뛴 스텝에서는 학습률 일정을 진행하지 않는 것.

손실은 정답 글의 토큰과 끝 토큰 <|im_end|>에 대해서만 계산한 모델의 인과 언어 모델 손실이다. 프롬프트는
추론과 같은 것(`apply_transcription_request`, "language Korean<asr_text>"로 끝남)이므로 템플릿, 오디오 자리
표시, 고정한 언어 표시는 패딩과 함께 가린다. 로짓은 그 위치에서만 계산한다(모든 위치에 어휘 크기의 로짓을
만들지 않으면서 모델 자체의 손실과 같은 값이 나온다). 한 스텝은 모든 마이크로 배치의 토큰에 대한 손실을
더해 그 토큰 수로 나눈다.

outputs/<run>/에 쓴다: run.json, steps.jsonl(스텝마다, 사건마다 한 줄. 이어 한 실행은 뒤에 덧붙인다),
checkpoint-<step>/(어댑터, --save-steps마다), resume/(어댑터와 옵티마이저·scaler·일정·난수 상태,
--resume-every 스텝마다와 끝에서).
"""

# [파일 역할] 10단계: Qwen3-ASR-1.7B에 LoRA를 학습해, 6단계 N(turbo + LoRA)을 대신할 범용 모델이 되는지 보려는
# 학습 스크립트다. 결과 어댑터는 evaluate_qwen --adapter로 잰다.
# [왜 직접 쓴 학습 루프인가] train.py는 Seq2SeqTrainer를 쓰지만, 여기서는 스텝마다 손실·자르기 전 기울기
# 크기·손실 배율·건너뜀·GPU 메모리를 남기고, 일시 정지와 이어 하기를 하려고 루프를 직접 썼다(위 docstring).
# Qwen3-ASR의 공식 학습은 bf16 전체 학습인데 이 GPU(RTX 2080 Ti)에는 bf16이 없어, fp16 기본 가중치 + fp32
# 어댑터 + fp16 autocast + GradScaler로 넘침 없이 도는지를 스모크 관문으로 먼저 확인했다(docs/experiments.md
# 10단계).
# [N과 맞춘 것] 같은 데이터(숫자를 바꾼 zeroth-train-itn), 같은 종류의 LoRA 위치(인코더·디코더의 어텐션과
# MLP), rank 16·alpha 32·dropout 0.05, 학습률 1e-4, warmup 500, 1,500스텝, 실효 배치 32, AdamW 설정, 기울기
# 자르기 1.0, 건너뛴 스텝에서 학습률 일정을 진행하지 않는 것.
# [N과 다른 것] 손실을 한 스텝의 정답 토큰 전체로 나눈 평균으로 한다. N의 Trainer는 누적한 묶음 평균을
# 더하기만 해서 자르기가 거의 매 스텝 걸렸다(docs/experiments.md 10단계 '재기 전에 아는 것').
# [흐름] 인자 확인 → 모델·처리기 불러오기(고정한 revision) → LoRA 붙이고 관문 (b) 확인 → run.json →
# 옵티마이저·일정·GradScaler → 고정된 순서의 데이터 → run_steps(스텝마다 steps.jsonl 한 줄, 500스텝마다
# 어댑터, 100스텝마다 이어 하기 상태) → run.json에 마무리 요약.

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# numpy·random: 이어 하기에 필요한 난수 상태 저장. torch: 학습, autocast, GradScaler. F: 교차 엔트로피.
# transformers: 선형 학습률 일정과 버전 기록(모델 클래스는 main에서 불러온다).
import numpy as np
import torch
import torch.nn.functional as F
import transformers

# evaluate.git_commit: 학습 코드의 커밋 기록. evaluate_qwen: 평가와 같은 모델·revision·언어 이름·앞 30초 창을
# 써서 학습 입력이 추론 입력과 같게 한다. store·train: 묶음 목록과 오디오를 읽는
# 데이터셋(train.UtteranceDataset)을 그대로 쓴다.
from whisper_ko_ft.evaluate import git_commit
from whisper_ko_ft.evaluate_qwen import DEFAULT_MODEL, LANGUAGES, REGISTERED, first_window
from whisper_ko_ft.paths import OUTPUTS
from whisper_ko_ft.store import SAMPLE_RATE, load_set
from whisper_ko_ft.train import UtteranceDataset

# LORA_TARGETS: LoRA를 붙일 층을 고르는 정규식. 오디오 인코더 24층의 q/k/v/out_proj와 fc1·fc2, 언어 모델
# 28층의 q/k/v/o_proj와 gate/up/down_proj. N의 어댑터가 붙은 층 종류(6단계: Whisper 인코더·디코더의 어텐션과
# MLP)에 맞춘 것이며, 프로젝터·합성곱·출력층에는 붙이지 않는다. 인코더의 어텐션에도 q_proj/k_proj/v_proj가
# 있어 이름 목록으로 주면 인코더와 언어 모델 양쪽이 함께 잡히므로 경로까지 적은 정규식을 쓴다.
# ENCODER_TARGETS_PER_LAYER(6), LANGUAGE_MODEL_TARGETS_PER_LAYER(7): 층마다 붙어야 할 LoRA 수. 6 × 24 + 7 × 28
# = 340개가 관문 (b)의 기대값이다.
LORA_TARGETS = (
    r"model\.audio_tower\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)"
    r"|model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)
ENCODER_TARGETS_PER_LAYER = 6
LANGUAGE_MODEL_TARGETS_PER_LAYER = 7
# END_TOKEN: 어시스턴트 답의 끝 토큰. 정답 뒤에 붙여 손실에 넣어, 모델이 받아 적은 뒤 멈추는 법을 함께 배운다.
# IGNORE: 손실에서 뺄 위치의 라벨 값(-100).
# MODEL_INPUTS: 모델 순전파에 넘길 키. DTYPES: --precision(얼린 기본 가중치의 dtype).
END_TOKEN = "<|im_end|>"
IGNORE = -100
MODEL_INPUTS = ("input_ids", "attention_mask", "input_features", "input_features_mask")
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


# --- 데이터 ---


# DataLoader의 batch_sampler. 스텝마다 쓸 마이크로 배치들의 발화 인덱스를 미리 정해진 순서로 내놓는다.
# 왜 필요한가: 학습이 멈췄다가 --resume으로 이어 할 때, 이어 한 실행이 멈추지 않은 실행과 똑같은 데이터를 같은
# 순서로 보게 하려는 것이다(docs/experiments.md 10단계 '학습 중'). 에폭 e의 순서는 seed + e로 섞은 순열이라,
# 스텝 번호만 알면 그 스텝의 데이터를 다시 만들 수 있다.
# longest_at(스모크 전용): 그 스텝 하나만 가장 긴 발화들로 채워 메모리가 가장 많이 드는 경우를 미리 돌려 본다.
# 그 스텝의 원래 마이크로 배치는 건너뛰므로 다른 스텝의 데이터는 바뀌지 않는다.
class StepBatches:
    """스텝마다 마이크로 배치들의 인덱스 목록을 내놓는다. 일정이 고정되어 있어 이어 한 실행도 같은 데이터를
    본다.

    에폭 e는 seed + e로 시드를 정한 순열을 마이크로 배치로 자른 것이다(나머지는 버린다). 스텝 s는 그 끝없는
    열의 (s-1)*accum ... s*accum-1번째 마이크로 배치를 쓴다. `longest_at`(스모크 실행)은 그 스텝 하나에만 대신
    `longest` 발화들을 주고, 그 스텝의 원래 마이크로 배치는 건너뛴다.
    """

    def __init__(
        self,
        count: int,
        batch_size: int,
        accum: int,
        seed: int = 0,
        first_step: int = 1,
        last_step: int = 1,
        longest: list[int] | None = None,
        longest_at: int | None = None,
    ) -> None:
        # 마이크로 배치 하나도 못 채우거나, 가장 긴 스텝에 넣을 발화 수가 장치당 배치 × 누적 횟수와 다르면
        # 거부한다.
        if count < batch_size:
            raise ValueError("fewer utterances than one micro-batch")
        if longest_at is not None and (longest is None or len(longest) != batch_size * accum):
            raise ValueError("the longest step needs batch_size * accum utterances")
        self.count, self.batch_size, self.accum, self.seed = count, batch_size, accum, seed
        self.first_step, self.last_step = first_step, last_step
        self.longest, self.longest_at = longest, longest_at
        # 에폭당 마이크로 배치 수. 나머지 발화는 버린다. _epoch는 지금 에폭의 순열을 한 번만 만들어 두는
        # 캐시다.
        self.per_epoch = count // batch_size
        self._epoch: tuple[int, list[int]] | None = None

    # 이 실행이 내놓을 마이크로 배치 수(남은 스텝 수 × 누적 횟수).
    def __len__(self) -> int:
        return max(0, self.last_step - self.first_step + 1) * self.accum

    # 에폭 번호로 시드를 정한 torch 생성기로 순열을 만든다. 전역 난수를 쓰지 않으므로 다른 난수 사용과 섞이지
    # 않는다.
    def _permutation(self, epoch: int) -> list[int]:
        if self._epoch is None or self._epoch[0] != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._epoch = (epoch, torch.randperm(self.count, generator=generator).tolist())
        return self._epoch[1]

    # 끝없이 이어지는 마이크로 배치 열의 index번째: 몇 번째 에폭의 몇 번째 묶음인지 계산해 그 순열에서 잘라
    # 낸다.
    def micro(self, index: int) -> list[int]:
        epoch, position = divmod(index, self.per_epoch)
        start = position * self.batch_size
        return self._permutation(epoch)[start : start + self.batch_size]

    # 스텝 s(1부터)의 마이크로 배치 목록: 열의 (s-1)×accum번째부터 accum개. 가장 긴 스텝이면 가장 긴 발화들을
    # accum개로 나눠 준다.
    def step(self, step: int) -> list[list[int]]:
        if step == self.longest_at:
            size = self.batch_size
            return [self.longest[m * size : (m + 1) * size] for m in range(self.accum)]
        return [self.micro((step - 1) * self.accum + m) for m in range(self.accum)]

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.first_step, self.last_step + 1):
            yield from self.step(step)


def longest_indices(utterances, count: int) -> list[int]:
    """가장 긴 발화 `count`개의 인덱스(오디오 길이, 다음으로 글자 수 기준). 긴 것부터 놓는다.

    스모크의 가장 긴 스텝(학습 데이터에서 가장 긴 32발화, 17.4~20.7초)에 쓴다.
    """
    order = sorted(range(len(utterances)), key=lambda i: (utterances[i].length, len(utterances[i].text)))
    return order[::-1][:count]


def answer_ids(tokenizer, text: str) -> list[int]:
    """모델이 프롬프트 뒤에 만들어야 할 토큰: 정답 글의 토큰(특수 토큰 없이), 그다음 끝 토큰 <|im_end|>."""
    return tokenizer(text, add_special_tokens=False)["input_ids"] + [
        tokenizer.convert_tokens_to_ids(END_TOKEN)
    ]


# 프롬프트와 정답을 이어 한 행씩 만들고 오른쪽으로 채운다. 돌려주는 값: input_ids, attention_mask, labels.
# 처리기는 프롬프트를 왼쪽으로 채워 주므로(추론의 공식 기본값) 마스크로 왼쪽 패딩을 떼어 낸 뒤 정답을 붙인다.
# labels는 정답 위치에만 토큰을 두고 나머지(템플릿, 오디오 자리, 'language Korean<asr_text>', 패딩)는 -100으로
# 가린다. 언어 표시는 추론 때 미리 채우는 부분이라 배울 필요가 없다(docs/experiments.md 10단계 '입력과 손실').
# labels는 입력과 같은 위치에 둔다. 한 칸 밀어 다음 토큰을 맞히게 하는 것은 answer_loss가 한다.
def assemble(
    prompt_ids: torch.Tensor, prompt_mask: torch.Tensor, answers: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """오른쪽으로 채운 입력 ID, attention mask, 라벨. 행마다 프롬프트(왼쪽 패딩을 뗀 것)와 정답을 잇고,
    정답에만 라벨을 둔다(라벨은 입력과 같은 위치에 두며, 한 칸 미는 것은 모델 쪽이 한다)."""
    prompts = [ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, prompt_mask, strict=True)]
    width = max(len(p) + len(a) for p, a in zip(prompts, answers, strict=True))
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(prompts), width), dtype=torch.long)
    labels = torch.full((len(prompts), width), IGNORE, dtype=torch.long)
    for row, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
        end = len(prompt) + len(answer)
        input_ids[row, :end] = torch.tensor(prompt + answer)
        attention[row, :end] = 1
        labels[row, len(prompt) : end] = torch.tensor(answer)
    return input_ids, attention, labels


# 학습용 콜레이터. 평가(evaluate_qwen)와 같은 앞 30초 창과 같은 공식 프롬프트(apply_transcription_request,
# 언어 고정, system 비움)로 입력을 만들어, 학습과 추론의 입력 모양을 맞춘다. 오디오 특징과 그 마스크, 그리고
# 로그용 발화 길이(초)를 함께 돌려준다.
# 이 입력이 공식 채팅 템플릿의 결과와 마지막 줄바꿈 하나만 빼고 같은지는 tests/test_train_qwen.py가 실제
# 처리기로 확인한다(모델 캐시가 있을 때만).
@dataclass
class QwenCollator:
    """오디오와 정답 글을 모델 입력으로 바꾼다. 프롬프트와 오디오 창은 `evaluate_qwen`과 같다."""

    processor: object
    language: str = "Korean"

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        audio = [first_window(row["audio"]) for row in rows]
        prompt = self.processor.apply_transcription_request(audio, language=self.language)
        tokenizer = self.processor.tokenizer
        answers = [answer_ids(tokenizer, row["text"]) for row in rows]
        input_ids, attention, labels = assemble(
            prompt["input_ids"], prompt["attention_mask"], answers, tokenizer.pad_token_id
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "input_features": prompt["input_features"],
            "input_features_mask": prompt["input_features_mask"],
            "seconds": torch.tensor([len(samples) / SAMPLE_RATE for samples in audio]),
        }


# --- 모델 ---


# 새 LoRA를 붙인다. 먼저 KV 캐시를 끄고 오디오 인코더·언어 모델 모두 gradient checkpointing을 켜 활성값
# 메모리를 줄인다. PEFT는 기본 가중치가 fp16이어도 어댑터를 fp32로 만든다(check_lora가 확인).
# 호출 위치: main(처음 학습), tests.
def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """두 탑(오디오 인코더, 언어 모델)에 gradient checkpointing을 켠 뒤 새 LoRA 어댑터를 붙인다(기본 가중치가
    fp16이어도 어댑터는 fp32)."""
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, bias="none")
    return get_peft_model(model, config)


def resume_model(model, folder: Path):
    """이어 하기: 저장된 어댑터를 학습 가능한 상태로 붙인 기본 모델.

    `add_lora`와 같은 설정(캐시 끔, gradient checkpointing)을 적용해, `add_lora`가 남겼을 상태와 같게 한다.
    """
    from peft import PeftModel

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return PeftModel.from_pretrained(model, str(folder), is_trainable=True)


# 설정에서 층 수를 읽어 LoRA가 붙어야 할 층 수를 계산한다(1.7B: 6 × 24 + 7 × 28 = 340).
def expected_lora_modules(config) -> int:
    audio, text = config.audio_config.encoder_layers, config.text_config.num_hidden_layers
    return ENCODER_TARGETS_PER_LAYER * audio + LANGUAGE_MODEL_TARGETS_PER_LAYER * text


# 관문 (b)의 기록: LoRA가 붙은 층 수(인코더·언어 모델 나눠서), 학습하는 파라미터 수(1.7B에서 24,510,464개),
# 학습하는 파라미터의 dtype, 어댑터가 아닌데 학습되는 가중치. run.json에 남는다.
def lora_summary(model) -> dict:
    """어댑터가 어디에 붙었고 무엇이 학습되는지(10단계 관문 (b))."""
    modules = {name.split(".lora_A.")[0] for name, _ in model.named_parameters() if ".lora_A." in name}
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "modules": len(modules),
        "encoder_modules": sum(".audio_tower." in name for name in modules),
        "language_model_modules": sum(".language_model." in name for name in modules),
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "trainable_dtypes": sorted({str(p.dtype) for _, p in trainable}),
        "other_trainable": [name for name, _ in trainable if ".lora_" not in name],
    }


# 관문 (b): LoRA 층 수가 기대값과 다르거나, 어댑터가 fp32가 아니거나, 어댑터 밖의 가중치가 학습되면 학습을
# 시작하지 않고 멈춘다(docs/experiments.md 10단계 '관문').
def check_lora(summary: dict, config) -> None:
    expected = expected_lora_modules(config)
    if summary["modules"] != expected:
        raise ValueError(f"LoRA is on {summary['modules']} modules, expected {expected}")
    if summary["trainable_dtypes"] != ["torch.float32"]:
        raise ValueError(f"the adapter must train in torch.float32, not {summary['trainable_dtypes']}")
    if summary["other_trainable"]:
        raise ValueError(f"other weights would train: {summary['other_trainable'][:3]}")


# 정답 토큰의 교차 엔트로피 합과 그 토큰 수. 모델의 손실(토큰 평균)에 토큰 수를 곱한 값과 같다(테스트로 확인).
# 왜 직접 계산하나: 모델의 손실은 모든 위치에서 어휘 크기의 로짓을 만든다. 여기서는 언어 모델의 은닉 상태만
# 받고, 다음 토큰이 정답인 위치에서만 lm_head를 계산해 메모리를 아낀다(위 docstring). 합으로 돌려주므로
# run_steps가 스텝 전체 토큰 수로 나눌 수 있다.
def answer_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """라벨이 있는 토큰의 교차 엔트로피 합과 그 토큰 수. 합은 모델의 손실 x 그 토큰 수와 같다."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = inner.model(**{key: batch[key] for key in MODEL_INPUTS}, use_cache=False).last_hidden_state
    # 위치 t의 은닉 상태가 t+1의 토큰을 맞히도록 라벨을 한 칸 민다. 가리지 않은 위치(keep)만 남긴다.
    targets = batch["labels"][:, 1:]
    keep = targets != IGNORE
    logits = inner.lm_head(hidden[:, :-1][keep])
    # autocast 안에서 로짓이 fp16으로 나오므로 fp32로 바꿔 교차 엔트로피를 계산한다(넘침·정밀도 손실 방지).
    return F.cross_entropy(logits.float(), targets[keep], reduction="sum"), int(keep.sum())


# --- 옵티마이저 한 스텝, 학습 루프, 저장 상태 ---


# warmup 동안 0에서 학습률까지 선형으로 올리고 그 뒤 0까지 선형으로 내리는 일정. N의 Trainer와 같은
# transformers 함수를 쓴다.
def linear_schedule(optimizer, warmup: int, total: int):
    return transformers.get_linear_schedule_with_warmup(optimizer, warmup, total)


# 옵티마이저 한 스텝. 배율을 먼저 풀어(unscale_) 실제 기울기 크기로 자르기(1.0)를 하고, GradScaler가 기울기에
# inf·nan이 있으면 그 스텝을 건너뛴다.
# 돌려주는 값: 자르기 전 기울기 크기(건너뛴 스텝은 nan일 수 있다), 이번에 쓴 배율, 건너뛰었는지. 건너뛰면
# update가 배율을 절반으로 줄이므로, 배율이 줄었는지로 건너뜀을 안다.
def optimizer_step(scaler, optimizer, params, max_grad_norm: float) -> tuple[float, float, bool]:
    """배율 풀기, 자르기, 스텝: (자르기 전 기울기 크기, 쓴 손실 배율, GradScaler가 건너뛰었는지)."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return float(norm), float(scale), scaler.get_scale() < scale


# PAUSE 파일이 있는 동안 5초마다 확인하며 기다리고, 기다린 시간을 돌려준다.
# 왜 필요한가: Windows는 GPU 메모리가 모자라면 공유(시스템) 메모리로 넘기는데 그러면 학습이 크게 느려진다.
# 저장소 밖의 메모리 감시 스크립트가 넘침과 다른 프로그램의 GPU 사용을 보고 이 파일을 만들거나
# 지운다(docs/experiments.md 10단계 '학습 중'). 스텝 사이에서만 멈추므로 가중치와 손실에는 영향이 없다.
def wait_while_paused(flag: Path, sleep: Callable[[float], None] = time.sleep, poll: float = 5.0) -> float:
    """`flag`가 사라지기를 기다린 초(처음부터 없었으면 0)."""
    if not flag.exists():
        return 0.0
    started = time.perf_counter()
    while flag.exists():
        sleep(poll)
    return time.perf_counter() - started


# run_steps 설정: 이번 실행의 첫·마지막 스텝, 누적 횟수, 장치, autocast 사용 여부, 자르기 기준, PAUSE 파일
# 위치, 가장 긴 스텝 번호.
@dataclass
class LoopSettings:
    first_step: int
    last_step: int
    accum: int
    device: str = "cuda"
    amp: bool = True
    max_grad_norm: float = 1.0
    pause_flag: Path | None = None
    longest_at: int | None = None


# 로그에 남길 현재 시각(초 단위).
def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# 학습 루프 본체. first_step부터 last_step까지 스텝마다 accum개의 마이크로 배치를 돌리고 steps.jsonl에 한 줄을
# 남긴다. 실행 요약(스텝 수, 건너뛴 스텝, 손실이 유한하지 않은 스텝, 최대 메모리, 시간)을 돌려준다.
# 호출 위치: main, tests/test_train_qwen.py(작은 무작위 Qwen3-ASR로 CPU에서).
def run_steps(
    model,
    batches: Iterator[dict],
    optimizer,
    scheduler,
    scaler,
    settings: LoopSettings,
    log: Callable[[dict], None],
    after_step: Callable[[int], None] = lambda step: None,
) -> dict:
    """옵티마이저 스텝 first_step..last_step을 돈다. 스텝마다 마이크로 배치 `accum`개, 로그 한 줄."""
    params = [p for p in model.parameters() if p.requires_grad]
    cuda = settings.device == "cuda"
    summary = {"steps": 0, "skipped_steps": [], "nonfinite_steps": [], "peak_allocated_mb": 0}
    summary.update({"seconds": 0.0, "paused_seconds": 0.0})
    for step in range(settings.first_step, settings.last_step + 1):
        # 다음 스텝을 시작하기 전에 PAUSE를 확인한다. 기다린 시간은 이벤트로 따로 남기고 스텝 시간에는 넣지
        # 않는다(시계는 그 뒤에 시작).
        if settings.pause_flag is not None:
            paused = wait_while_paused(settings.pause_flag)
            if paused:
                log({"event": "pause", "before_step": step, "seconds": round(paused, 1), "time": now()})
                summary["paused_seconds"] += paused
        # 스텝마다 피크 메모리를 새로 잰다.
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        # 이 스텝의 마이크로 배치를 모두 먼저 받는다(데이터를 기다린 시간도 따로 남긴다). 정답 토큰 수를 스텝
        # 전체로 세어 손실의 분모로 쓴다.
        micro = [next(batches) for _ in range(settings.accum)]
        data_seconds = time.perf_counter() - started
        tokens = sum(int((batch["labels"][:, 1:] != IGNORE).sum()) for batch in micro)
        loss_total = 0.0
        # 마이크로 배치마다 fp16 autocast로 순전파하고, '손실 합 ÷ 스텝 전체 토큰 수'에 배율을 곱해
        # 역전파한다. 기울기가 쌓이면 결국 스텝의 모든 정답 토큰에 대한 평균 손실의 기울기가 된다(N과의 차이,
        # 위 머리 주석).
        for batch in micro:
            inputs = {key: batch[key].to(settings.device) for key in (*MODEL_INPUTS, "labels")}
            with torch.autocast(device_type=settings.device, dtype=torch.float16, enabled=settings.amp):
                loss_sum, _ = answer_loss(model, inputs)
            scaler.scale(loss_sum / tokens).backward()
            loss_total += loss_sum.item()
        # 이번 스텝에 쓴 학습률을 기록하고 옵티마이저 스텝을 한다. GradScaler가 건너뛴 스텝에서는 학습률
        # 일정을 진행하지 않는다(N의 Trainer와 같다).
        lr = optimizer.param_groups[0]["lr"]
        norm, scale, skipped = optimizer_step(scaler, optimizer, params, settings.max_grad_norm)
        if not skipped:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        # GPU 작업이 끝난 뒤 스텝 시간을 잰다.
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        loss = loss_total / tokens
        peak = round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None
        seconds_per_utterance = torch.cat([batch["seconds"] for batch in micro])
        # steps.jsonl 한 줄: 손실(토큰 평균), 토큰·발화 수, 음성 길이 합과 최대, 자르기 전 기울기 크기, 배율,
        # 건너뜀, 학습률, 스텝·데이터 시간, 최대 할당·예약 메모리, 가장 긴 스텝 여부. docs/experiments.md
        # 10단계 결과의 손실·건너뜀·스텝 시간 통계가 이 기록에서 나왔다.
        log(
            {
                "step": step,
                "time": now(),
                "loss": loss,
                "tokens": tokens,
                "utterances": len(seconds_per_utterance),
                "audio_seconds": round(float(seconds_per_utterance.sum()), 2),
                "max_seconds": round(float(seconds_per_utterance.max()), 2),
                "grad_norm": norm,
                "scale": scale,
                "skipped": skipped,
                "lr": lr,
                "seconds": round(seconds, 3),
                "data_seconds": round(data_seconds, 3),
                "peak_allocated_mb": peak,
                "reserved_mb": round(torch.cuda.memory_reserved() / 2**20) if cuda else None,
                "longest": step == settings.longest_at,
            }
        )
        summary["steps"] += 1
        summary["seconds"] += seconds
        if skipped:
            summary["skipped_steps"].append(step)
        if not math.isfinite(loss):
            summary["nonfinite_steps"].append(step)
        if peak is not None:
            summary["peak_allocated_mb"] = max(summary["peak_allocated_mb"], peak)
        after_step(step)
    return summary


def _random_states() -> dict:
    """이어 하기용 난수 상태(python, numpy, torch CPU, CUDA).

    일반 자료형과 텐서만 담는다. 그래야 torch.load(weights_only=True)로 상태를 읽을 수 있다. weights_only는
    임의의 파이썬 객체를 풀지 않는 안전한 읽기 방식이다.
    """
    name, keys, position, has_gauss, gauss = np.random.get_state()
    states = {
        "python": random.getstate(),
        "numpy": [name, keys.tolist(), int(position), int(has_gauss), float(gauss)],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


# _random_states로 저장한 목록을 각 라이브러리가 받는 형태(튜플, uint32 배열)로 되돌려 난수 상태를 복원한다.
def _set_random_states(states: dict) -> None:
    version, internal, gauss = states["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, position, has_gauss, cached = states["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def save_state(folder: Path, model, optimizer, scaler, scheduler, step: int) -> None:
    """이어 하기 상태를 `folder` 옆에 다 쓴 뒤 제자리로 옮긴다.

    내용은 어댑터와 state.pt(마친 스텝, 옵티마이저, GradScaler, 학습률 일정, 난수 상태)다. 먼저
    <folder>.tmp에 다 쓴 뒤 원래 폴더를 지우고 이름을 바꾸므로, 저장 도중에 죽어도 이전 상태나 완성된 .tmp 중
    하나는 남는다.
    """
    partial = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(partial, ignore_errors=True)
    model.save_pretrained(str(partial))
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "random": _random_states(),
    }
    torch.save(state, partial / "state.pt")
    shutil.rmtree(folder, ignore_errors=True)
    partial.rename(folder)


def resume_folder(run_dir: Path) -> Path | None:
    """실행의 마지막 완성된 저장 상태 폴더. 없으면 None.

    resume을 먼저 보고, 없으면 제자리로 옮기기(이름 바꾸기) 직전에 죽어 남은 resume.tmp를 본다. 어댑터와
    state.pt가 모두 있어야 완성된 상태로 본다.
    """
    for name in ("resume", "resume.tmp"):
        folder = run_dir / name
        if (folder / "state.pt").exists() and (folder / "adapter_model.safetensors").exists():
            return folder
    return None


def saved_step(folder: Path) -> int:
    """저장 상태에서 마지막으로 끝낸 스텝 번호만 읽는다(데이터 순서를 그다음 스텝부터 잇는 데 쓴다)."""
    return int(torch.load(folder / "state.pt", weights_only=True)["step"])


def restore_state(folder: Path, optimizer, scaler, scheduler) -> int:
    """옵티마이저·GradScaler·학습률 일정·난수 상태를 복원하고, 마지막으로 끝낸 스텝을 돌려준다.

    어댑터 가중치는 resume_model이 따로 불러온다.
    """
    state = torch.load(folder / "state.pt", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    scheduler.load_state_dict(state["scheduler"])
    _set_random_states(state["random"])
    return int(state["step"])


# --- 실행 ---


# 명령줄 인자. 기본값이 10단계 규칙의 값이다: zeroth-train-itn, 학습률 1e-4, warmup 500, 1,500스텝,
# 500스텝마다 어댑터, 100스텝마다 이어 하기 상태, 장치당 2 × 누적 16(실효 32), 자르기 1.0, LoRA 16/32/0.05,
# seed 0.
# --max-steps 0은 학습 전 어댑터만 저장(관문 (a)용), --longest-at은 스모크, --limit-train과 --device cpu는 CPU
# 확인용이다.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="기본값은 등록한 모델의 고정 revision")
    parser.add_argument("--train-set", default="zeroth-train-itn")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1_500, help="0이면 처음 어댑터만 저장하고 멈춘다")
    parser.add_argument("--save-steps", type=int, default=500, help="어댑터 저장 간격. 0이면 저장 안 함")
    parser.add_argument("--resume-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--longest-at", type=int, help="스모크: 이 스텝에 가장 긴 발화들을 넣는다")
    parser.add_argument("--limit-train", type=int, help="CPU 확인 전용: 학습 발화를 앞의 N개로 줄인다")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=list(DTYPES), default="fp16", help="얼린 기본 가중치의 정밀도")
    parser.add_argument("--resume", action="store_true", help="outputs/<run>/resume에서 이어 간다")
    args = parser.parse_args(argv)
    # 등록한 모델이면 고정한 revision을 쓰고, 아니면 --revision을 요구한다.
    if args.revision is None:
        if args.model not in REGISTERED:
            parser.error("a model that is not registered needs --revision")
        args.revision = REGISTERED[args.model][0]
    # CPU 학습은 측정 실행이 아니며 fp32로만 한다. 학습 전 어댑터 저장(--max-steps 0)은 CPU fp16으로도 된다.
    if args.device == "cpu" and args.precision == "fp16" and args.max_steps > 0:
        parser.error("training on the CPU needs --precision fp32 (and is never a measured run)")
    # 실행 폴더 보호: --resume인데 이어 할 상태가 없으면 멈추고, 이미 run.json이 있는 폴더에 새 실행을
    # 덮어쓰지 않는다.
    run_dir = OUTPUTS / args.run_name
    if args.resume and resume_folder(run_dir) is None:
        parser.error(f"{run_dir} has no saved state to resume from")
    if not args.resume and (run_dir / "run.json").exists():
        parser.error(f"{run_dir} already has a run; pass --resume or pick another --run-name")
    if args.longest_at is not None and not 1 <= args.longest_at <= args.max_steps:
        parser.error("--longest-at must be one of the steps")
    return args


# 학습 실행. argv를 받을 수 있어 테스트가 명령줄 없이 부른다. peft와 Qwen3-ASR 모델 클래스는 여기서 불러온다.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from peft import __version__ as peft_version
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    run_dir = OUTPUTS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # autocast와 GradScaler는 GPU fp16에서만 켠다. 세 난수 생성기를 seed로 맞추고, 학습 코드의 커밋을
    # 기록한다.
    amp = args.device == "cuda" and args.precision == "fp16"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    commit = git_commit()

    language, utterances = load_set(args.train_set)
    if args.limit_train:
        utterances = utterances[: args.limit_train]
    # 평가와 같은 고정 revision으로 처리기와 모델을 불러온다. 기본 가중치는 fp16(또는 --precision), 어텐션은
    # SDPA.
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=DTYPES[args.precision], attn_implementation="sdpa"
    )
    # 이어 하기면 저장된 어댑터로, 아니면 새 LoRA를 붙인다. 모델이 아직 CPU에 있을 때 붙이므로 어댑터 초기값은
    # --device와 관계없이 같은 CPU 난수에서 나온다. 그 뒤 관문 (b)를 확인하고 장치로 옮긴다.
    state_folder = resume_folder(run_dir) if args.resume else None
    if state_folder is not None:
        model = resume_model(model, state_folder)
    else:
        model = add_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)  # CPU에서 초기화한다
    lora = lora_summary(model)
    check_lora(lora, model.config)
    model.to(args.device)

    # run.json: 실행을 시작할 때마다 starts에 시각·커밋·이어 한 상태를 덧붙이고, 인자·LoRA 정규식·관문 (b)
    # 요약·데이터 규모·라이브러리 버전·GPU를 적는다. 이어 하기면 기존 내용을 읽어 이어 쓴다.
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if args.resume else {}
    run.setdefault("starts", []).append({"time": now(), "commit": commit, "resume_from": str(state_folder)})
    run.update(
        {
            "args": vars(args),
            "commit": commit,
            "lora_targets": LORA_TARGETS,
            "lora": lora,
            "train_utterances": len(utterances),
            "train_hours": round(sum(u.seconds for u in utterances) / 3600, 2),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft_version,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        }
    )
    run_path.write_text(json.dumps(run, indent=1, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(lora), flush=True)
    # --max-steps 0: 학습 전 어댑터(B 행렬이 0이라 합쳐도 모델이 바뀌지 않는다)만 checkpoint-0에 저장하고
    # 끝낸다. 관문 (a)가 이 어댑터를 합쳐 9단계 디코딩과 가설이 같은지 확인한다(docs/experiments.md 10단계
    # '관문').
    if args.max_steps == 0:
        model.save_pretrained(str(run_dir / "checkpoint-0"))
        print(f"wrote {run_dir / 'checkpoint-0'} (the initial adapter)", flush=True)
        return

    # AdamW: N의 Trainer와 같은 설정(β 0.9/0.999, ε 1e-8, weight decay 0, GPU에서 fused 구현). 학습하는 어댑터
    # 파라미터만 넘긴다.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fused=args.device == "cuda",
    )
    # 학습률 일정과 GradScaler(기본값: 처음 배율 65,536, 넘치면 절반, 2,000스텝마다 두 배). 이어 하기면 마친
    # 스텝 다음부터 시작한다.
    scheduler = linear_schedule(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.amp.GradScaler(args.device, enabled=amp)
    done = saved_step(state_folder) if state_folder is not None else 0
    model.train()

    # 고정된 순서의 batch_sampler와 DataLoader. 데이터셋은 train.py의 것을 그대로 쓰고(전화 음질 변환 없음),
    # 콜레이터가 평가와 같은 프롬프트를 만든다.
    longest = longest_indices(utterances, args.batch_size * args.grad_accum) if args.longest_at else None
    sampler = StepBatches(
        len(utterances),
        args.batch_size,
        args.grad_accum,
        args.seed,
        done + 1,
        args.max_steps,
        longest,
        args.longest_at,
    )
    loader = torch.utils.data.DataLoader(
        UtteranceDataset(utterances),
        batch_sampler=sampler,
        collate_fn=QwenCollator(processor, LANGUAGES[language]),
        num_workers=args.workers,
        # train.py와 같은 이유로 워커를 실행 동안 살려 둔다(커밋 30e739b). Windows는 워커를 spawn으로
        # 띄우므로, 실행 전체에 이터레이터 하나를 써서 워커를 한 번만 띄운다.
        persistent_workers=args.workers > 0,
    )
    # 이터레이터를 만들 때 워커 기본 시드가 torch 난수에서 뽑히므로, 저장된 난수 상태는 그 뒤에 복원한다.
    # 그래야 복원한 상태가 학습 쪽 난수(dropout 등)로 그대로 이어진다.
    batches = iter(loader)  # 워커 기본 시드를 뽑으므로 저장된 난수 상태 복원보다 앞에 둔다
    if state_folder is not None:
        restore_state(state_folder, optimizer, scaler, scheduler)
    steps_path = run_dir / "steps.jsonl"

    # steps.jsonl에 한 줄씩 덧붙인다(LF). 화면에는 이벤트, 10스텝마다, 처음 세 스텝, 건너뛴 스텝만 찍는다.
    def log(entry: dict) -> None:
        with steps_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(entry) + "\n")
        if "event" in entry or entry["step"] % 10 == 0 or entry["step"] <= 3 or entry["skipped"]:
            print(json.dumps(entry), flush=True)

    # 스텝이 끝날 때마다: save_steps(500)의 배수면 어댑터 체크포인트(500/1,000/1,500을 학습 뒤 analyze
    # pick10으로 고른다), resume_every(100)의 배수이거나 마지막 스텝이면 이어 하기 상태를 저장한다.
    def after_step(step: int) -> None:
        if args.save_steps and step % args.save_steps == 0:
            model.save_pretrained(str(run_dir / f"checkpoint-{step}"))
            log({"event": "checkpoint", "step": step, "time": now()})
        if step % args.resume_every == 0 or step == args.max_steps:
            save_state(run_dir / "resume", model, optimizer, scaler, scheduler, step)

    log({"event": "start", "first_step": done + 1, "last_step": args.max_steps, "time": now()})
    settings = LoopSettings(
        first_step=done + 1,
        last_step=args.max_steps,
        accum=args.grad_accum,
        device=args.device,
        amp=amp,
        max_grad_norm=args.max_grad_norm,
        pause_flag=run_dir / "PAUSE",
        longest_at=args.longest_at,
    )
    summary = run_steps(model, batches, optimizer, scheduler, scaler, settings, log, after_step)
    log({"event": "end", **summary, "time": now()})

    # steps.jsonl 전체에서 마무리 요약을 만든다. 이어 한 실행은 일부 스텝을 두 번 기록하므로 스텝 번호로
    # 하나만 남긴다. 스텝 시간 중앙값은 가장 긴 스텝(스모크)을 빼고 낸다. 건너뛴 스텝, 손실이 유한하지 않은
    # 스텝, 최대 메모리, 일시 정지 시간을 run.json의 finished에 쓴다.
    entries = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()]
    steps = {e["step"]: e for e in entries if "event" not in e}  # 이어 한 실행은 일부 스텝을 두 번 남긴다
    times = [e["seconds"] for e in steps.values() if not e["longest"]]
    run["finished"] = {
        "time": now(),
        "steps": len(steps),
        "skipped_steps": sorted(s for s, e in steps.items() if e["skipped"]),
        "nonfinite_steps": sorted(s for s, e in steps.items() if not math.isfinite(e["loss"])),
        "median_step_seconds": round(statistics.median(times), 3) if times else None,
        "peak_allocated_mb": max((e["peak_allocated_mb"] or 0) for e in steps.values()),
        "paused_seconds": round(sum(e["seconds"] for e in entries if e.get("event") == "pause"), 1),
    }
    run_path.write_text(json.dumps(run, indent=1, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(run["finished"]), flush=True)


if __name__ == "__main__":
    main()
