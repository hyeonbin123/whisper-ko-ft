"""Stage 10: LoRA training on Qwen3-ASR (docs/experiments.md, 10단계). A tiny random model, no weights.

The tests that need the real processor or config read them from the Hugging Face cache and are skipped
where it is empty (CI).
"""

import json

import numpy as np
import pytest
import torch
from tiny_qwen import AUDIO, END, PAD, tiny_model
from transformers import Qwen3ASRForConditionalGeneration

from whisper_ko_ft import train_qwen
from whisper_ko_ft.evaluate_qwen import DEFAULT_MODEL, REGISTERED
from whisper_ko_ft.store import SAMPLE_RATE


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


def test_steps_take_fixed_micro_batches_from_seeded_epochs():
    batches = train_qwen.StepBatches(count=10, batch_size=2, accum=2, seed=0, first_step=1, last_step=5)
    seen = list(batches)
    assert len(seen) == len(batches) == 10 and all(len(b) == 2 for b in seen)
    first_epoch = sorted(i for b in seen[:5] for i in b)
    assert first_epoch == list(range(10))  # each utterance once per epoch
    assert seen == list(train_qwen.StepBatches(10, 2, 2, seed=0, first_step=1, last_step=5))
    assert seen != list(train_qwen.StepBatches(10, 2, 2, seed=1, first_step=1, last_step=5))


def test_a_resumed_run_sees_the_rest_of_the_same_schedule():
    full = list(train_qwen.StepBatches(count=11, batch_size=2, accum=3, seed=0, first_step=1, last_step=6))
    rest = list(train_qwen.StepBatches(count=11, batch_size=2, accum=3, seed=0, first_step=4, last_step=6))
    assert rest == full[9:]


def test_the_longest_step_replaces_one_step_only():
    longest = [9, 8, 7, 6]
    plain = list(train_qwen.StepBatches(10, 2, 2, seed=0, first_step=1, last_step=3))
    smoke = list(
        train_qwen.StepBatches(10, 2, 2, seed=0, first_step=1, last_step=3, longest=longest, longest_at=2)
    )
    assert smoke[2:4] == [[9, 8], [7, 6]]
    assert smoke[:2] == plain[:2] and smoke[4:] == plain[4:]


def test_longest_indices_order_by_audio_then_text():
    class U:
        def __init__(self, length, text):
            self.length, self.text = length, text

    utterances = [U(5, "a"), U(9, "a"), U(5, "abc"), U(7, "a")]
    assert train_qwen.longest_indices(utterances, 3) == [1, 3, 2]


# --- inputs and labels ---


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


class Tokenizer:
    """Stands in for the Qwen tokenizer: one id per character."""

    pad_token_id = PAD

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(c) % 90 for c in text]}

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return END


def test_the_answer_is_the_transcript_and_the_end_token():
    assert train_qwen.answer_ids(Tokenizer(), "ab") == [ord("a") % 90, ord("b") % 90, END]


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


def lora_names(model) -> set[str]:
    return {
        name.split(".lora_A.")[0].removeprefix("base_model.model.")
        for name, _ in model.named_parameters()
        if ".lora_A." in name
    }


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


def test_check_lora_refuses_a_wrong_adapter():
    model = train_qwen.add_lora(tiny_model(torch.float16))
    summary = train_qwen.lora_summary(model)
    with pytest.raises(ValueError, match="modules"):
        train_qwen.check_lora({**summary, "modules": 25}, model.config)
    with pytest.raises(ValueError, match="float32"):
        train_qwen.check_lora({**summary, "trainable_dtypes": ["torch.float16"]}, model.config)
    with pytest.raises(ValueError, match="other"):
        train_qwen.check_lora({**summary, "other_trainable": ["lm_head.weight"]}, model.config)


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


def test_the_answer_loss_is_the_models_own_loss_on_the_answer_tokens():
    batch = tiny_batch()
    model = tiny_model()
    with torch.no_grad():
        total, count = train_qwen.answer_loss(model, batch)
        official = model(**batch).loss  # mean over the labelled positions
    assert count == 6  # 4 + 2 answer tokens, end token included
    assert torch.allclose(total / count, official, atol=1e-6)


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


def test_a_disabled_scaler_never_skips():
    weight = torch.nn.Parameter(torch.ones(1))
    weight.grad = torch.ones(1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    _, scale, skipped = train_qwen.optimizer_step(scaler, torch.optim.SGD([weight], lr=0.1), [weight], 1.0)
    assert not skipped and scale == 1.0


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


def make_loop(model, accum=2, lr=1e-2):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = train_qwen.linear_schedule(optimizer, warmup=1, total=4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return optimizer, scheduler, scaler


def step_batches(step: int, accum: int = 2) -> list[dict]:
    return [tiny_batch(seed=10 * step + m) for m in range(accum)]


def run(model, optimizer, scheduler, scaler, first, last, log):
    batches = iter([b for s in range(first, last + 1) for b in step_batches(s)])
    settings = train_qwen.LoopSettings(first_step=first, last_step=last, accum=2, device="cpu", amp=False)
    return train_qwen.run_steps(model, batches, optimizer, scheduler, scaler, settings, log)


def adapter_weights(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


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


def test_a_half_written_resume_state_is_not_used(tmp_path):
    model = train_qwen.add_lora(tiny_model())
    optimizer, scheduler, scaler = make_loop(model)
    train_qwen.save_state(tmp_path / "resume", model, optimizer, scaler, scheduler, step=7)
    assert train_qwen.resume_folder(tmp_path) == tmp_path / "resume"
    (tmp_path / "resume" / "state.pt").unlink()
    assert train_qwen.resume_folder(tmp_path) is None


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


def cached(loader):
    try:
        return loader()
    except OSError:  # not in the cache
        pytest.skip("Qwen3-ASR-1.7B-hf is not in the Hugging Face cache")


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


def test_training_on_the_cpu_needs_fp32_but_the_initial_adapter_does_not(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_qwen, "OUTPUTS", tmp_path)
    with pytest.raises(SystemExit):
        train_qwen.parse_args(["--run-name", "a", "--device", "cpu"])
    assert "fp32" in capsys.readouterr().err
    init_only = train_qwen.parse_args(["--run-name", "a", "--device", "cpu", "--max-steps", "0"])
    assert init_only.precision == "fp16"
    with pytest.raises(SystemExit):
        train_qwen.parse_args(["--run-name", "a", "--max-steps", "50", "--longest-at", "51"])


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
