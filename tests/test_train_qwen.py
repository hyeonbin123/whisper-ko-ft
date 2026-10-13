"""Stage 10: LoRA training on Qwen3-ASR (docs/experiments.md, 10단계). A tiny random model, no weights.

The tests that need the real processor or config read them from the Hugging Face cache and are skipped
where it is empty (CI).
"""

# [테스트 묶음] train_qwen.py(10단계 Qwen3-ASR LoRA 학습)를 작은 무작위 모델로 CPU에서 확인한다: 고정된 데이터
# 순서와 이어 하기, 입력·라벨 조립, 평가와 같은 프롬프트, LoRA 위치와 dtype(관문 (b)), 손실 계산, 옵티마이저
# 스텝과 GradScaler 건너뜀, 일시 정지, 저장·복원, 학습률 일정, 명령줄 보호. 실제 모델 파일이 필요한 두
# 테스트는 Hugging Face 캐시에 있을 때만 돈다(위 docstring).
# [왜 중요한가] 직접 쓴 학습 루프라 Trainer가 대신 해 주던 것(라벨 가리기, 배율 처리, 일정, 이어 하기)을 모두
# 스스로 지켜야 한다. 5시간 넘는 학습을 돌린 뒤에야 드러날 실수를 미리 잡으려는 것이다.

import json

import numpy as np
import pytest
import torch
from tiny_qwen import AUDIO, END, PAD, tiny_model
from transformers import Qwen3ASRForConditionalGeneration

from whisper_ko_ft import train_qwen
from whisper_ko_ft.evaluate_qwen import DEFAULT_MODEL, REGISTERED
from whisper_ko_ft.store import SAMPLE_RATE


# 작은 모델용 배치 하나를 만든다: 1초·2초 발화(오디오 토큰 13·26개), 처리기처럼 왼쪽을 채운 프롬프트, 정답
# 4·2토큰(끝 토큰 포함)을 assemble로 이은 것. seed로 특징 값을 바꿔 서로 다른 배치를 만든다.
def tiny_batch(seed: int = 0) -> dict[str, torch.Tensor]:
    """Utterances of 1 and 2 seconds (13 and 26 audio tokens), prompts left-padded as the processor does."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(2, 128, 200, generator=generator)
    features_mask = torch.zeros(2, 200, dtype=torch.long)
    features_mask[0, :100], features_mask[1, :] = 1, 1
    short, long = [1, 2, *[AUDIO] * 13, 3, 4], [1, 2, *[AUDIO] * 26, 3, 4]
    prompt = torch.tensor([[PAD] * 13 + short, long])
    mask = torch.tensor([[0] * 13 + [1] * len(short), [1] * len(long)])
    answers = [[20, 21, 22, END], [23, END]]
    input_ids, attention, labels = train_qwen.assemble(prompt, mask, answers, PAD)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "input_features": features,
        "input_features_mask": features_mask,
        "seconds": torch.tensor([1.0, 2.0]),
    }


# --- the data schedule ---


# StepBatches: 마이크로 배치 수와 크기가 맞고, 한 에폭 안에서 모든 발화가 한 번씩 나오며, 같은 시드면 같은
# 순서, 다른 시드면 다른 순서다.
def test_steps_take_fixed_micro_batches_from_seeded_epochs():
    batches = train_qwen.StepBatches(count=10, batch_size=2, accum=2, seed=0, first_step=1, last_step=5)
    seen = list(batches)
    assert len(seen) == len(batches) == 10 and all(len(b) == 2 for b in seen)
    first_epoch = sorted(i for b in seen[:5] for i in b)
    assert first_epoch == list(range(10))  # each utterance once per epoch
    assert seen == list(train_qwen.StepBatches(10, 2, 2, seed=0, first_step=1, last_step=5))
    assert seen != list(train_qwen.StepBatches(10, 2, 2, seed=1, first_step=1, last_step=5))


# 4번째 스텝부터 시작한 실행은 처음부터 돈 실행의 4번째 스텝 이후와 똑같은 데이터를 본다(스텝당 3개 × 3스텝 =
# 앞 9개를 건너뜀). 이어 하기가 데이터 순서를 바꾸지 않는다는 보장이다.
def test_a_resumed_run_sees_the_rest_of_the_same_schedule():
    full = list(train_qwen.StepBatches(count=11, batch_size=2, accum=3, seed=0, first_step=1, last_step=6))
    rest = list(train_qwen.StepBatches(count=11, batch_size=2, accum=3, seed=0, first_step=4, last_step=6))
    assert rest == full[9:]


# 스모크의 가장 긴 스텝은 그 스텝 하나만 가장 긴 발화로 바꾸고, 앞뒤 스텝의 데이터는 그대로다.
def test_the_longest_step_replaces_one_step_only():
    longest = [9, 8, 7, 6]
    plain = list(train_qwen.StepBatches(10, 2, 2, seed=0, first_step=1, last_step=3))
    smoke = list(
        train_qwen.StepBatches(10, 2, 2, seed=0, first_step=1, last_step=3, longest=longest, longest_at=2)
    )
    assert smoke[2:4] == [[9, 8], [7, 6]]
    assert smoke[:2] == plain[:2] and smoke[4:] == plain[4:]


# 가장 긴 발화 고르기: 오디오 길이가 먼저, 같으면 글자 수가 긴 쪽이 앞이다.
def test_longest_indices_order_by_audio_then_text():
    class U:
        def __init__(self, length, text):
            self.length, self.text = length, text

    utterances = [U(5, "a"), U(9, "a"), U(5, "abc"), U(7, "a")]
    assert train_qwen.longest_indices(utterances, 3) == [1, 3, 2]


# --- inputs and labels ---


# assemble: 프롬프트의 왼쪽 패딩을 떼고 정답을 이어 오른쪽으로 채우며, 라벨은 정답 위치에만 있고 프롬프트와
# 패딩은 -100이다.
def test_rows_are_prompt_then_answer_right_padded_and_only_the_answer_is_labelled():
    prompt = torch.tensor([[PAD, PAD, 5, 6, 7], [5, 6, 7, 8, 9]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    input_ids, attention, labels = train_qwen.assemble(prompt, mask, [[20, 21, END], [22, END]], PAD)
    assert input_ids.tolist() == [[5, 6, 7, 20, 21, END, PAD], [5, 6, 7, 8, 9, 22, END]]
    assert attention.tolist() == [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]
    ignore = train_qwen.IGNORE
    assert labels.tolist() == [
        [ignore, ignore, ignore, 20, 21, END, ignore],
        [ignore, ignore, ignore, ignore, ignore, 22, END],
    ]


# Qwen 토크나이저 대역: 글자마다 토큰 하나. 특수 토큰을 붙이지 않는지와 끝 토큰 이름을 확인한다.
class Tokenizer:
    """Stands in for the Qwen tokenizer: one id per character."""

    pad_token_id = PAD

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(c) % 90 for c in text]}

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return END


# 정답 토큰은 글의 토큰 뒤에 끝 토큰 <|im_end|>를 붙인 것이다.
def test_the_answer_is_the_transcript_and_the_end_token():
    assert train_qwen.answer_ids(Tokenizer(), "ab") == [ord("a") % 90, ord("b") % 90, END]


# 처리기 대역: apply_transcription_request 호출(발화별 표본 수, 언어)을 기록하고 고정된 프롬프트를 돌려준다.
class Processor:
    """Stands in for Qwen3ASRProcessor.apply_transcription_request: a fixed prompt per utterance."""

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.requests = []

    def apply_transcription_request(self, audio, language):
        self.requests.append(([len(a) for a in audio], language))
        return {
            "input_ids": torch.tensor([[PAD, 1, 2], [1, 2, 3]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "input_features": torch.zeros(2, 128, 100),
            "input_features_mask": torch.ones(2, 100, dtype=torch.int32),
        }


# QwenCollator: 31초 음성을 30초로 잘라 공식 입력 함수에 언어 Korean으로 한 번에 넘기고, 프롬프트 뒤에 정답을
# 이으며, 라벨에는 정답 토큰만 남고, 로그용 발화 길이와 모델 입력 키를 함께 돌려준다.
def test_the_collator_uses_the_inference_prompt_and_the_first_30_seconds():
    processor = Processor()
    collate = train_qwen.QwenCollator(processor, "Korean")
    rows = [
        {"audio": np.zeros(31 * SAMPLE_RATE, dtype=np.float32), "text": "가"},
        {"audio": np.zeros(SAMPLE_RATE, dtype=np.float32), "text": "나다"},
    ]
    batch = collate(rows)
    assert processor.requests == [([30 * SAMPLE_RATE, SAMPLE_RATE], "Korean")]
    assert batch["input_ids"].tolist()[0][:2] == [1, 2] and batch["input_ids"].tolist()[1][:3] == [1, 2, 3]
    kept = [[t for t in row if t != train_qwen.IGNORE] for row in batch["labels"].tolist()]
    assert kept == [
        train_qwen.answer_ids(processor.tokenizer, "가"),
        train_qwen.answer_ids(processor.tokenizer, "나다"),
    ]
    assert batch["seconds"].tolist() == [30.0, 1.0]
    assert {"input_features", "input_features_mask", "attention_mask"} <= set(batch)


# --- LoRA on the tiny model ---


# LoRA가 붙은 층 이름 목록(PEFT 접두사를 뗀 것).
def lora_names(model) -> set[str]:
    return {
        name.split(".lora_A.")[0].removeprefix("base_model.model.")
        for name, _ in model.named_parameters()
        if ".lora_A." in name
    }


# 관문 (b)의 작은 판: LoRA가 오디오 인코더의 어텐션·fc1·fc2와 언어 모델의 어텐션·MLP에만 붙고(투영층, 합성곱,
# lm_head에는 없음), 층 수가 기대값(2층씩이라 12 + 14 = 26)과 같으며, 어댑터는 fp32이고 그 밖의 가중치는
# fp16으로 얼어 있다.
def test_lora_goes_on_encoder_attention_and_mlp_and_language_model_attention_and_mlp_only():
    model = train_qwen.add_lora(tiny_model(torch.float16))
    names = lora_names(model)
    encoder = {
        f"model.audio_tower.layers.{i}.{m}"
        for i in range(2)
        for m in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        )
    }
    language = {
        f"model.language_model.layers.{i}.{m}"
        for i in range(2)
        for m in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    assert names == encoder | language  # not the projector, conv_out or lm_head
    summary = train_qwen.lora_summary(model)
    assert summary["modules"] == train_qwen.expected_lora_modules(model.config) == 26
    assert summary["encoder_modules"] == 12 and summary["language_model_modules"] == 14
    assert summary["trainable_dtypes"] == ["torch.float32"]  # fp32 adapter on an fp16 base
    assert summary["other_trainable"] == []
    assert {str(p.dtype) for n, p in model.named_parameters() if "lora_" not in n} == {"torch.float16"}
    train_qwen.check_lora(summary, model.config)  # does not raise


# check_lora는 층 수가 다르거나, 어댑터가 fp32가 아니거나, 어댑터 밖의 가중치가 학습되면 ValueError로 멈춘다.
def test_check_lora_refuses_a_wrong_adapter():
    model = train_qwen.add_lora(tiny_model(torch.float16))
    summary = train_qwen.lora_summary(model)
    with pytest.raises(ValueError, match="modules"):
        train_qwen.check_lora({**summary, "modules": 25}, model.config)
    with pytest.raises(ValueError, match="float32"):
        train_qwen.check_lora({**summary, "trainable_dtypes": ["torch.float16"]}, model.config)
    with pytest.raises(ValueError, match="other"):
        train_qwen.check_lora({**summary, "other_trainable": ["lm_head.weight"]}, model.config)


# 새 어댑터는 B가 0이라 붙여도 손실이 똑같다(관문 (a)의 전제). 또 오디오 인코더와 언어 모델 양쪽 층에 gradient
# checkpointing이 켜져 있다.
def test_a_fresh_adapter_changes_nothing_and_checkpointing_covers_both_towers():
    batch = tiny_batch()
    base = tiny_model()
    with torch.no_grad():
        before = base(**batch).loss
    model = train_qwen.add_lora(base)
    model.eval()
    with torch.no_grad():
        after = model(**batch).loss
    assert torch.equal(before, after)  # lora_B starts at zero
    layers = [
        m for n, m in model.named_modules() if n.endswith(("audio_tower.layers.0", "language_model.layers.1"))
    ]
    assert len(layers) == 2 and all(m.gradient_checkpointing for m in layers)


# answer_loss(정답 위치에서만 lm_head를 계산한 손실 합 ÷ 토큰 수)가 모델 자체 손실(라벨 위치 평균)과 같다.
# 토큰 수는 정답 4 + 2개(끝 토큰 포함)다.
def test_the_answer_loss_is_the_models_own_loss_on_the_answer_tokens():
    batch = tiny_batch()
    model = tiny_model()
    with torch.no_grad():
        total, count = train_qwen.answer_loss(model, batch)
        official = model(**batch).loss  # mean over the labelled positions
    assert count == 6  # 4 + 2 answer tokens, end token included
    assert torch.allclose(total / count, official, atol=1e-6)


# 역전파 한 번 뒤 기울기가 어댑터에만 생기고, 인코더 쪽과 언어 모델 쪽 B 행렬 모두에 0이 아닌 기울기가 있다(두
# tower 모두 학습된다). 얼린 가중치에는 기울기가 없다.
def test_one_step_trains_only_the_adapter_in_both_towers():
    model = train_qwen.add_lora(tiny_model())
    model.train()
    total, count = train_qwen.answer_loss(model, tiny_batch())
    (total / count).backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    assert all(".lora_" in n for n in grads)
    assert any("audio_tower" in n and "lora_B" in n and g.abs().sum() > 0 for n, g in grads.items())
    assert any("language_model" in n and "lora_B" in n and g.abs().sum() > 0 for n, g in grads.items())
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)


# --- the optimizer step and the loop ---


# optimizer_step: 기울기에 inf가 있으면 GradScaler가 그 스텝을 건너뛰고(가중치 그대로, 배율 8 → 4) 건너뜀으로
# 보고한다. 다음 스텝에서는 배율을 풀어 실제 기울기 크기(2.0)를 자르기 전 값으로 보고하고, 1.0으로 잘라
# 갱신한다.
def test_optimizer_step_reports_a_step_that_gradscaler_skipped():
    weight = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    scaler.scale(torch.tensor(0.0))  # GradScaler sets up its scale on the first scale()
    weight.grad = torch.tensor([1.0, float("inf"), 1.0])
    norm, scale, skipped = train_qwen.optimizer_step(scaler, optimizer, [weight], max_grad_norm=1.0)
    assert skipped and scale == 8.0 and not np.isfinite(norm)
    assert torch.equal(weight.detach(), torch.ones(3))  # not updated
    weight.grad = torch.tensor([8.0, 0.0, 0.0])  # the scale is now 4: the true gradient is (2, 0, 0)
    norm, scale, skipped = train_qwen.optimizer_step(scaler, optimizer, [weight], max_grad_norm=1.0)
    assert not skipped and scale == 4.0 and norm == pytest.approx(2.0)  # the norm before clipping
    assert weight.detach().tolist() == pytest.approx([0.9, 1.0, 1.0])  # clipped to (1, 0, 0)


# GradScaler를 끈 경우(CPU, fp32)는 배율 1.0이고 건너뜀이 없다.
def test_a_disabled_scaler_never_skips():
    weight = torch.nn.Parameter(torch.ones(1))
    weight.grad = torch.ones(1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    _, scale, skipped = train_qwen.optimizer_step(scaler, torch.optim.SGD([weight], lr=0.1), [weight], 1.0)
    assert not skipped and scale == 1.0


# PAUSE 파일이 없으면 바로 돌아오고, 있으면 지워질 때까지 poll 간격으로 기다린다(대역 sleep이 세 번째에 파일을
# 지운다).
def test_no_step_starts_while_the_pause_file_exists(tmp_path):
    flag = tmp_path / "PAUSE"
    assert train_qwen.wait_while_paused(flag, sleep=lambda s: None) == 0.0
    flag.write_text("", encoding="utf-8")
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 3:
            flag.unlink()

    assert train_qwen.wait_while_paused(flag, sleep=sleep, poll=2.0) >= 0.0
    assert calls == [2.0, 2.0, 2.0] and not flag.exists()


# 학습 루프 부품(AdamW, warmup 1·전체 4스텝 일정, 꺼진 GradScaler)을 만드는 도우미.
def make_loop(model, accum=2, lr=1e-2):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = train_qwen.linear_schedule(optimizer, warmup=1, total=4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return optimizer, scheduler, scaler


# 스텝 번호마다 정해진 시드의 배치 accum개. 같은 스텝이면 언제 만들어도 같은 데이터다.
def step_batches(step: int, accum: int = 2) -> list[dict]:
    return [tiny_batch(seed=10 * step + m) for m in range(accum)]


# first~last 스텝을 CPU에서 run_steps로 돌린다.
def run(model, optimizer, scheduler, scaler, first, last, log):
    batches = iter([b for s in range(first, last + 1) for b in step_batches(s)])
    settings = train_qwen.LoopSettings(first_step=first, last_step=last, accum=2, device="cpu", amp=False)
    return train_qwen.run_steps(model, batches, optimizer, scheduler, scaler, settings, log)


# 학습하는 파라미터(어댑터)의 복사본.
def adapter_weights(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


# 핵심 이어 하기 테스트. 4스텝을 끊지 않고 돈 결과와, 2스텝 뒤 저장한 상태에서 새 모델로 이어 3~4스텝을 돈
# 결과의 어댑터 가중치가 비트 단위로 같아야 한다. dropout(0.05)이 있어 난수 상태까지 복원해야 통과한다.
# 로그도 확인한다: 스텝마다 한 줄, 손실 유한, 토큰 12개(스텝당 정답 6 × 2), 발화 4개, 건너뜀 없음, 학습률이
# warmup 뒤 선형 감소(0 → 0.01 → 2/3 → 1/3), JSON으로 쓸 수 있음.
def test_steps_are_logged_and_a_resumed_run_ends_where_an_unbroken_one_does(tmp_path):
    torch.manual_seed(1)
    model = train_qwen.add_lora(tiny_model(), dropout=0.05)
    model.train()
    optimizer, scheduler, scaler = make_loop(model)
    log: list[dict] = []
    run(model, optimizer, scheduler, scaler, 1, 2, log.append)
    train_qwen.save_state(tmp_path / "resume", model, optimizer, scaler, scheduler, step=2)
    summary = run(model, optimizer, scheduler, scaler, 3, 4, log.append)
    unbroken = adapter_weights(model)

    steps = [e for e in log if "step" in e and "event" not in e]
    assert [e["step"] for e in steps] == [1, 2, 3, 4]
    assert all(np.isfinite(e["loss"]) and e["tokens"] == 12 and e["utterances"] == 4 for e in steps)
    assert all(not e["skipped"] and e["scale"] == 1.0 for e in steps)
    assert [round(e["lr"], 6) for e in steps] == [0.0, 0.01, round(0.01 * 2 / 3, 6), round(0.01 / 3, 6)]
    json.dumps(steps)
    assert summary["steps"] == 2 and summary["skipped_steps"] == [] and summary["nonfinite_steps"] == []

    resumed = train_qwen.resume_model(tiny_model(), tmp_path / "resume")
    resumed.train()
    optimizer, scheduler, scaler = make_loop(resumed)
    assert train_qwen.restore_state(tmp_path / "resume", optimizer, scaler, scheduler) == 2
    run(resumed, optimizer, scheduler, scaler, 3, 4, lambda entry: None)
    again = adapter_weights(resumed)
    assert unbroken.keys() == again.keys()
    assert all(torch.equal(unbroken[n], again[n]) for n in unbroken)


# state.pt가 없는(쓰다 만) 상태 폴더는 이어 하기에 쓰지 않는다.
def test_a_half_written_resume_state_is_not_used(tmp_path):
    model = train_qwen.add_lora(tiny_model())
    optimizer, scheduler, scaler = make_loop(model)
    train_qwen.save_state(tmp_path / "resume", model, optimizer, scaler, scheduler, step=7)
    assert train_qwen.resume_folder(tmp_path) == tmp_path / "resume"
    (tmp_path / "resume" / "state.pt").unlink()
    assert train_qwen.resume_folder(tmp_path) is None


# 학습률 일정: warmup 2스텝 동안 0 → 0.5 → 1.0으로 오르고 그 뒤 0까지 선형으로 내린다.
def test_the_schedule_warms_up_then_decays_to_zero():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    scheduler = train_qwen.linear_schedule(optimizer, warmup=2, total=4)
    rates = []
    for _ in range(4):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates == [0.0, 0.5, 1.0, 0.5] and optimizer.param_groups[0]["lr"] == 0.0


# --- the real model's files (Hugging Face cache; skipped when it is empty) ---


# 실제 모델 파일을 캐시에서만 읽고(local_files_only), 없으면 테스트를 건너뛴다. CI에는 캐시가 없어 이 두
# 테스트는 로컬에서만 돈다.
def cached(loader):
    try:
        return loader()
    except OSError:  # not in the cache
        pytest.skip("Qwen3-ASR-1.7B-hf is not in the Hugging Face cache")


# 실제 1.7B 설정으로 meta 장치에 모델 뼈대만 만들어(가중치 메모리 없이) LoRA를 붙이면 340층, 학습 파라미터
# 24,510,464개, fp32여야 한다(docs/experiments.md 10단계에 적은 수).
def test_on_the_real_config_lora_has_the_registered_size():
    from transformers import AutoConfig

    config = cached(
        lambda: AutoConfig.from_pretrained(
            DEFAULT_MODEL, revision=REGISTERED[DEFAULT_MODEL][0], local_files_only=True
        )
    )
    with torch.device("meta"):
        model = Qwen3ASRForConditionalGeneration._from_config(config, dtype=torch.float16)
    summary = train_qwen.lora_summary(train_qwen.add_lora(model))
    assert summary["modules"] == 24 * 6 + 28 * 7 == train_qwen.expected_lora_modules(config)
    assert summary["trainable_parameters"] == 24 * 294_912 + 28 * 622_592 == 24_510_464
    assert summary["trainable_dtypes"] == ["torch.float32"]


# 실제 처리기로 만든 학습 입력이 공식 채팅 템플릿(사용자 턴에 오디오, 어시스턴트 턴에 'language
# Korean<asr_text>정답')의 결과와 마지막 줄바꿈 하나만 빼고 같고, 라벨을 글로 되돌리면 정답 +
# <|im_end|>뿐이어야 한다. 학습 입력이 추론 프롬프트와 어긋나지 않는다는 확인이다.
def test_on_the_real_processor_inputs_are_the_chat_template_without_its_last_newline():
    from transformers import AutoProcessor

    processor = cached(
        lambda: AutoProcessor.from_pretrained(
            DEFAULT_MODEL, revision=REGISTERED[DEFAULT_MODEL][0], local_files_only=True
        )
    )
    text = "이천 십 팔 년 3월에 우리는 갔다"
    audio = [np.zeros(2 * SAMPLE_RATE, dtype=np.float32), np.zeros(5 * SAMPLE_RATE, dtype=np.float32)]
    batch = train_qwen.QwenCollator(processor, "Korean")([{"audio": a, "text": text} for a in audio])
    conversation = [
        [
            {"role": "user", "content": [{"type": "audio", "audio": audio[0]}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"language Korean<asr_text>{text}"}]},
        ]
    ]
    official = processor.apply_chat_template(conversation, tokenize=True, return_dict=True)["input_ids"][0]
    row = batch["input_ids"][0][batch["attention_mask"][0].bool()].tolist()
    assert row == official.tolist()[:-1]  # ends with <|im_end|>; the template adds "\n"
    kept = [t for t in batch["labels"][0].tolist() if t != train_qwen.IGNORE]
    assert processor.tokenizer.decode(kept) == f"{text}<|im_end|>"


# --- the command line ---


# parse_args 기본값이 10단계 규칙과 같고(revision, 2 × 16, 1e-4, warmup 500, 1,500스텝, 500스텝마다 저장, LoRA
# 16/32/0.05, 자르기 1.0, zeroth-train-itn), 저장 상태 없이 --resume하거나 이미 run.json이 있는 폴더에 새로
# 시작하면 멈춘다.
def test_a_run_is_never_overwritten_and_resume_needs_a_saved_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_qwen, "OUTPUTS", tmp_path)
    args = train_qwen.parse_args(["--run-name", "qn"])
    assert (args.revision, args.batch_size, args.grad_accum) == (REGISTERED[DEFAULT_MODEL][0], 2, 16)
    assert (args.learning_rate, args.warmup_steps, args.max_steps, args.save_steps) == (1e-4, 500, 1500, 500)
    assert (args.lora_r, args.lora_alpha, args.lora_dropout, args.max_grad_norm) == (16, 32, 0.05, 1.0)
    assert args.train_set == "zeroth-train-itn"
    with pytest.raises(SystemExit):
        train_qwen.parse_args(["--run-name", "qn", "--resume"])
    assert "no saved state" in capsys.readouterr().err
    (tmp_path / "qn").mkdir()
    (tmp_path / "qn" / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_qwen.parse_args(["--run-name", "qn"])
    assert "already has a run" in capsys.readouterr().err


# CPU 학습은 fp32를 요구하지만 학습 전 어댑터 저장(--max-steps 0)은 fp16으로도 된다. --longest-at은 학습할
# 스텝 범위 안이어야 한다.
def test_training_on_the_cpu_needs_fp32_but_the_initial_adapter_does_not(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_qwen, "OUTPUTS", tmp_path)
    with pytest.raises(SystemExit):
        train_qwen.parse_args(["--run-name", "a", "--device", "cpu"])
    assert "fp32" in capsys.readouterr().err
    init_only = train_qwen.parse_args(["--run-name", "a", "--device", "cpu", "--max-steps", "0"])
    assert init_only.precision == "fp16"
    with pytest.raises(SystemExit):
        train_qwen.parse_args(["--run-name", "a", "--max-steps", "50", "--longest-at", "51"])


# 일시 정지는 'pause' 이벤트로 따로 기록되고, 기다린 12.5초는 스텝 시간에 들어가지 않으며 요약의
# paused_seconds에 모인다.
def test_a_pause_is_logged_and_left_out_of_the_step_time(tmp_path, monkeypatch):
    monkeypatch.setattr(train_qwen, "wait_while_paused", lambda flag: 12.5 if flag.name == "PAUSE" else 0.0)
    model = train_qwen.add_lora(tiny_model())
    optimizer, scheduler, scaler = make_loop(model)
    batches = iter(step_batches(1))
    settings = train_qwen.LoopSettings(1, 1, accum=2, device="cpu", amp=False, pause_flag=tmp_path / "PAUSE")
    log: list[dict] = []
    summary = train_qwen.run_steps(model, batches, optimizer, scheduler, scaler, settings, log.append)
    assert log[0]["event"] == "pause" and log[0]["before_step"] == 1 and log[0]["seconds"] == 12.5
    assert log[1]["step"] == 1 and log[1]["seconds"] < 12.5
    assert summary["paused_seconds"] == 12.5
