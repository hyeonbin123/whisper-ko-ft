"""Stage 9: the Qwen3-ASR harness (docs/experiments.md, 9단계). No weights: stubs and tiny modules."""

import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from tiny_qwen import tiny_model
from transformers.models.qwen3_asr.processing_qwen3_asr import _parse_single_output

from whisper_ko_ft import evaluate_qwen
from whisper_ko_ft.store import SAMPLE_RATE

EOS = (151643, 151645)
LOOP = "그리고 다시 " + "도와서는 그 " * 30


def official(text: str) -> str:
    """What processor.extract_transcription returns (the official parse, with its repetition fix)."""
    return _parse_single_output(text)["transcription"]


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("지난해 3월 김 전 장관의 동료인", "지난해 3월 김 전 장관의 동료인"),  # language forced in the prompt
        ("language Korean<asr_text>조계종이 처음이다.", "조계종이 처음이다."),  # the model wrote the marker
        ("  When you call someone.  ", "When you call someone."),
        ("", ""),
    ],
)
def test_raw_output_is_the_official_parse_without_the_repetition_fix(generated, expected):
    assert evaluate_qwen.raw_transcription(generated) == expected
    assert official(generated) == expected  # nothing repeats here, so the two outputs agree


def test_the_repetition_fix_only_changes_the_fixed_output():
    assert evaluate_qwen.raw_transcription(LOOP) == LOOP.strip()
    assert len(official(LOOP)) < len(LOOP.strip())


def test_audio_is_cut_to_the_first_30_seconds_like_the_whisper_harness():
    long = np.arange(31 * SAMPLE_RATE, dtype=np.float32)
    short = np.ones(5 * SAMPLE_RATE, dtype=np.float32)
    assert evaluate_qwen.first_window(long).shape == (30 * SAMPLE_RATE,)
    assert np.array_equal(evaluate_qwen.first_window(long), long[: 30 * SAMPLE_RATE])
    assert evaluate_qwen.first_window(short) is short


def test_generated_lengths_count_tokens_before_the_end_and_flag_the_limit():
    generated = torch.tensor(
        [
            [5, 6, EOS[1], EOS[1]],  # ended after two tokens, then padding
            [5, 6, 7, 8],  # never ended: ran into the token limit
            [EOS[0], EOS[1], EOS[1], EOS[1]],  # ended at once (empty output)
        ]
    )
    assert evaluate_qwen.generated_lengths(generated, list(EOS)) == [(2, False), (4, True), (0, False)]
    assert evaluate_qwen.generated_lengths(generated[:1], EOS[1]) == [(2, False)]


def test_nonfinite_watch_flags_the_rows_with_overflowing_logits():
    head = torch.nn.Linear(2, 3)
    watch = evaluate_qwen.NonFiniteWatch(head)
    with torch.inference_mode():
        head(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]))  # one decoding step, two rows
        head(torch.tensor([[[1.0, 0.0]], [[float("inf"), 1.0]]]))  # the second row overflows
    assert watch.take(2) == [False, True]
    assert watch.take(2) == [False, False]  # taken: the next batch starts clean


class Inputs(dict):
    """Stands in for the processor's BatchFeature."""

    def to(self, device, dtype):
        self.moved = (device, dtype)
        return self


class Processor:
    """Stands in for Qwen3ASRProcessor: token ids as text, end tokens dropped; the official parse."""

    def __init__(self) -> None:
        self.requests: list[tuple[int, str]] = []
        self.tokenizer = self

    def apply_transcription_request(self, audio, language):
        self.requests.append((len(audio), language))
        return Inputs(input_ids=torch.ones(len(audio), 3, dtype=torch.long), input_features="features")

    def batch_decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        texts = {11: "안녕하세요.", 12: LOOP}
        return ["".join(texts.get(t, "") for t in row.tolist() if t not in EOS) for row in ids]

    def extract_transcription(self, texts):
        return [official(text) for text in texts]


class Model:
    """Stands in for the model: keeps the arguments of generate; prompt (3 tokens) + generated ids."""

    class generation_config:  # noqa: N801
        eos_token_id = list(EOS)

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 1, 1, 11, EOS[1], EOS[1]], [1, 1, 1, 12, 12, 12]])


def test_one_decoding_gives_both_outputs_with_the_registered_arguments():
    model, processor = Model(), Processor()
    audio = [np.zeros(SAMPLE_RATE, dtype=np.float32)] * 2
    out = evaluate_qwen.transcribe(model, processor, audio, "ko", "cpu", torch.float16)
    assert processor.requests == [(2, "Korean")]  # the official prompt, language forced
    assert model.kwargs == {
        "input_ids": model.kwargs["input_ids"],
        "input_features": "features",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 256,
    }
    assert [o["raw"] for o in out] == ["안녕하세요.", (LOOP * 3).strip()]
    assert out[0]["fixed"] == "안녕하세요."
    assert len(out[1]["fixed"]) < len(out[1]["raw"])
    assert [(o["new_tokens"], o["hit_token_limit"], o["nonfinite_logits"]) for o in out] == [
        (1, False, False),
        (3, True, False),
    ]


def test_reports_score_each_output_on_its_own():
    rows = [
        {"id": "a", "speaker": "s", "seconds": 1.0, "reference": "안녕하세요", "raw": "안녕하세요.",
         "fixed": "안녕하세요.", "new_tokens": 3, "hit_token_limit": False, "nonfinite_logits": False},
        {"id": "b", "speaker": "s", "seconds": 1.0, "reference": "그리고 다시", "raw": LOOP.strip(),
         "fixed": official(LOOP), "new_tokens": 256, "hit_token_limit": True, "nonfinite_logits": False},
        {"id": "c", "speaker": "s", "seconds": 1.0, "reference": "...", "raw": "", "fixed": "",
         "new_tokens": 0, "hit_token_limit": False, "nonfinite_logits": False},
    ]  # fmt: skip
    payloads = evaluate_qwen.report_payloads(rows, "ko", {"model": "m"})
    raw, fixed = payloads["raw"], payloads["fixed"]
    assert [r["hypothesis"] for r in raw["utterances"]] == ["안녕하세요.", LOOP.strip(), ""]
    assert [r["hypothesis"] for r in fixed["utterances"]] == ["안녕하세요.", official(LOOP), ""]
    assert all("raw" not in r and "fixed" not in r for r in raw["utterances"] + fixed["utterances"])
    for payload, output in ((raw, "raw"), (fixed, "fixed")):
        summary = payload["summary"]
        assert summary["model"] == "m" and summary["output"] == output
        assert summary["utterances"] == 2 and summary["skipped_empty_reference"] == 1
        assert summary["hit_token_limit"] == 1 and summary["nonfinite_utterances"] == 0
        assert summary["fixed_changed"] == 1
    assert raw["summary"]["loops"] == 1  # more edits than reference characters
    assert raw["summary"]["error_rate"] > fixed["summary"]["error_rate"]
    json.dumps(payloads, ensure_ascii=False)  # what main writes


class Reached(Exception):
    """The argument checks passed and the set was about to be loaded."""


@pytest.fixture
def reports(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(evaluate_qwen, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_qwen, "load_set", stop)
    monkeypatch.setattr(evaluate_qwen, "load_model", stop)
    return tmp_path


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["evaluate_qwen", *args])
    evaluate_qwen.main()


def test_test_sets_need_allow_test(reports, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-test")
    assert "--allow-test" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, "--set", "zeroth-test", "--allow-test")


@pytest.mark.parametrize("existing", ["qwen3-asr-1.7b", "qwen3-asr-1.7b-fixed"])
def test_neither_report_is_replaced_without_overwrite(reports, monkeypatch, capsys, existing):
    (reports / existing).mkdir()
    (reports / existing / "zeroth-val.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val")
    assert "--overwrite" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, "--set", "zeroth-val", "--overwrite")


def test_a_cpu_run_is_not_a_measurement(reports, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val500", "--device", "cpu", "--precision", "fp32")
    assert "--out" in capsys.readouterr().err
    with pytest.raises(Reached):
        out = str(reports / "gate" / "fp32.json")
        run(monkeypatch, "--set", "zeroth-val500", "--device", "cpu", "--precision", "fp32", "--out", out)
    with pytest.raises(Reached):
        run(monkeypatch, "--set", "zeroth-val500", "--device", "cpu", "--no-report", "--limit", "4")


def test_out_is_not_replaced_without_overwrite_and_excludes_no_report(reports, monkeypatch, capsys):
    out = reports / "fp16.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val500", "--out", str(out))
    assert "--overwrite" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val500", "--out", str(out), "--overwrite", "--no-report")


def test_registered_models_have_a_pinned_revision_and_a_report_name(reports, monkeypatch, capsys):
    assert evaluate_qwen.REGISTERED["Qwen/Qwen3-ASR-1.7B-hf"] == (
        "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        "qwen3-asr-1.7b",
    )
    assert evaluate_qwen.REGISTERED["Qwen/Qwen3-ASR-0.6B-hf"][1] == "qwen3-asr-0.6b"
    with pytest.raises(SystemExit):
        run(monkeypatch, "--model", "someone/other-asr", "--set", "zeroth-val")
    assert "--revision" in capsys.readouterr().err
    with pytest.raises(Reached):
        other = ["--model", "someone/other-asr", "--revision", "abc", "--name", "x"]
        run(monkeypatch, *other, "--set", "zeroth-val")


# --- stage 10: a LoRA adapter merged into the base, and the telephone channel ---


def test_an_adapter_needs_a_name(reports, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val", "--adapter", "outputs/qwen-qn/checkpoint-500")
    assert "--name" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, "--set", "zeroth-val", "--adapter", "a", "--name", "qwen3-asr-1.7b-qn")


def test_channel_reports_are_written_as_set_at_channel(reports, monkeypatch, capsys):
    for name in ("qwen3-asr-1.7b", "qwen3-asr-1.7b-fixed"):
        (reports / name).mkdir()
        (reports / name / "zeroth-val@telephone.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Reached):  # the clean-audio reports are other files
        run(monkeypatch, "--set", "zeroth-val")
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val", "--channel", "telephone")
    assert "zeroth-val@telephone.json exists" in capsys.readouterr().err


def test_the_channel_is_applied_before_the_first_30_seconds_are_kept():
    rng = np.random.default_rng(0)
    long = (rng.standard_normal(31 * SAMPLE_RATE) * 0.1).astype(np.float32)
    clean = evaluate_qwen.prepare_audio(long, None)
    phone = evaluate_qwen.prepare_audio(long, "telephone")
    assert clean.shape == phone.shape == (30 * SAMPLE_RATE,)
    assert np.array_equal(clean, long[: 30 * SAMPLE_RATE]) and not np.allclose(phone, clean)
    assert np.array_equal(phone, evaluate_qwen.CHANNELS["telephone"](long)[: 30 * SAMPLE_RATE])


def save_adapter(folder, train_b: bool):
    from whisper_ko_ft.train_qwen import add_lora

    model = add_lora(tiny_model(torch.float16))
    if train_b:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.01)
    model.save_pretrained(str(folder))


def test_merging_a_fresh_adapter_leaves_every_weight_as_it_was(tmp_path):
    save_adapter(tmp_path / "fresh", train_b=False)
    base = {k: v.clone() for k, v in tiny_model(torch.float16).state_dict().items()}
    merged = evaluate_qwen.merge_adapter(tiny_model(torch.float16), str(tmp_path / "fresh"))
    after = merged.state_dict()
    assert type(merged).__name__ == "Qwen3ASRForConditionalGeneration"
    assert after.keys() == base.keys()
    assert all(torch.equal(after[k], base[k]) for k in base)


def test_merging_a_trained_adapter_changes_the_targeted_weights_only(tmp_path):
    save_adapter(tmp_path / "trained", train_b=True)
    base = {k: v.clone() for k, v in tiny_model(torch.float16).state_dict().items()}
    after = evaluate_qwen.merge_adapter(tiny_model(torch.float16), str(tmp_path / "trained")).state_dict()
    changed = {k for k in base if not torch.equal(after[k], base[k])}
    assert "model.language_model.layers.0.mlp.down_proj.weight" in changed
    assert "model.audio_tower.layers.1.fc1.weight" in changed
    assert not any(
        "multi_modal_projector" in k or "lm_head" in k or "conv" in k or "norm" in k for k in changed
    )
    assert all(after[k].dtype == torch.float16 for k in after)


def test_reports_name_the_adapter_and_its_weights(tmp_path):
    save_adapter(tmp_path / "checkpoint-500", train_b=False)
    info = evaluate_qwen.adapter_info(str(tmp_path / "checkpoint-500"))
    weights = (tmp_path / "checkpoint-500" / "adapter_model.safetensors").read_bytes()
    assert info["adapter"].endswith("checkpoint-500")
    assert info["adapter_sha256"] == hashlib.sha256(weights).hexdigest()
    assert evaluate_qwen.adapter_info(None) == {"adapter": None, "adapter_sha256": None}
