"""9단계: Qwen3-ASR 하네스(docs/experiments.md 9단계). 가중치 없이 대역과 작은 모듈로 확인한다."""

# [테스트 묶음] evaluate_qwen.py(9·10단계 Qwen3-ASR 하네스)를 가중치 없이 확인한다: raw 출력이 공식 파서에서
# 반복 제거만 뺀 것인지, 앞 30초 자르기, 생성 토큰 수와 한도 도달, fp16 넘침 감시, 한 디코딩에서 두 리포트,
# 인자 검사(test 묶음, 덮어쓰기, CPU 실행, 등록 모델, 어댑터 이름, 채널), 어댑터 병합과 SHA-256 기록.
# [왜 중요한가] 9단계 판정은 raw 출력으로 Whisper 하네스와 같은 조건에서 비교한다는 전제 위에 있다. raw에 반복
# 제거가 섞이거나, 30초 창·프롬프트·생성 인자가 달라지면 비교 조건이 무너진다. 10단계에서는 어댑터 병합이
# 겨냥한 층만 바꾸는지가 결과 해석의 전제다.
# [방법] 앞부분은 처리기·모델 대역으로 인자와 출력 모양을 보고, 뒷부분은 tiny_qwen의 작은 무작위 모델로 실제
# PEFT 병합을 한다. 공식 파서(_parse_single_output)는 transformers에서 직접 가져와 비교 기준으로 쓴다.

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

# EOS: 끝 토큰 번호 두 개(대역 모델에서 쓰는 값). LOOP: 같은 구절을 30번 되풀이한 글(공식 반복 제거가 줄이는
# 예).
EOS = (151643, 151645)
LOOP = "그리고 다시 " + "도와서는 그 " * 30


def official(text: str) -> str:
    """processor.extract_transcription이 돌려주는 것(반복 제거를 포함한 공식 파싱). fixed 출력과 같다."""
    return _parse_single_output(text)["transcription"]


# 반복이 없는 글에서는 raw와 공식 결과가 같아야 한다: 언어를 고정해 표시가 없는 경우, 모델이 'language
# Korean<asr_text>' 표시를 적은 경우, 앞뒤 공백, 빈 출력.
@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("지난해 3월 김 전 장관의 동료인", "지난해 3월 김 전 장관의 동료인"),  # 프롬프트에서 언어 고정
        ("language Korean<asr_text>조계종이 처음이다.", "조계종이 처음이다."),  # 모델이 표시를 적음
        ("  When you call someone.  ", "When you call someone."),
        ("", ""),
    ],
)
def test_raw_output_is_the_official_parse_without_the_repetition_fix(generated, expected):
    assert evaluate_qwen.raw_transcription(generated) == expected
    assert official(generated) == expected  # 되풀이가 없어 두 출력이 같다


# 되풀이가 있는 글은 raw에서 그대로 남고(Whisper 하네스와 같은 조건), 공식 결과에서만 짧아진다.
def test_the_repetition_fix_only_changes_the_fixed_output():
    assert evaluate_qwen.raw_transcription(LOOP) == LOOP.strip()
    assert len(official(LOOP)) < len(LOOP.strip())


# 31초 음성은 앞 30초만 남기고, 30초보다 짧으면 복사하지 않고 같은 배열을 돌려준다.
def test_audio_is_cut_to_the_first_30_seconds_like_the_whisper_harness():
    long = np.arange(31 * SAMPLE_RATE, dtype=np.float32)
    short = np.ones(5 * SAMPLE_RATE, dtype=np.float32)
    assert evaluate_qwen.first_window(long).shape == (30 * SAMPLE_RATE,)
    assert np.array_equal(evaluate_qwen.first_window(long), long[: 30 * SAMPLE_RATE])
    assert evaluate_qwen.first_window(short) is short


# 생성 토큰 수는 첫 끝 토큰 앞까지 센다. 끝 토큰이 없으면 한도까지 간 것으로 표시하고, 바로 끝나면 0이다. 끝
# 토큰이 하나(정수)로 와도 동작한다.
def test_generated_lengths_count_tokens_before_the_end_and_flag_the_limit():
    generated = torch.tensor(
        [
            [5, 6, EOS[1], EOS[1]],  # 두 토큰 뒤에 끝나고 그 뒤는 패딩
            [5, 6, 7, 8],  # 끝나지 않음: 토큰 한도에 닿았다
            [EOS[0], EOS[1], EOS[1], EOS[1]],  # 바로 끝남(빈 출력)
        ]
    )
    assert evaluate_qwen.generated_lengths(generated, list(EOS)) == [(2, False), (4, True), (0, False)]
    assert evaluate_qwen.generated_lengths(generated[:1], EOS[1]) == [(2, False)]


# 출력층 훅이 디코딩 단계마다 행별로 inf를 잡아 누적하는지, take 뒤에는 다음 배치를 위해 비워지는지.
def test_nonfinite_watch_flags_the_rows_with_overflowing_logits():
    head = torch.nn.Linear(2, 3)
    watch = evaluate_qwen.NonFiniteWatch(head)
    with torch.inference_mode():
        head(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]))  # 디코딩 한 단계, 두 행
        head(torch.tensor([[[1.0, 0.0]], [[float("inf"), 1.0]]]))  # 둘째 행이 넘친다
    assert watch.take(2) == [False, True]
    assert watch.take(2) == [False, False]  # 꺼낸 뒤에는 다음 배치가 깨끗하게 시작한다


class Inputs(dict):
    """처리기가 돌려주는 BatchFeature 대역. to(device, dtype)를 기록만 하고 자신을 돌려준다."""

    def to(self, device, dtype):
        self.moved = (device, dtype)
        return self


class Processor:
    """Qwen3ASRProcessor 대역: 토큰 번호를 글로 바꾸고(끝 토큰은 뺀다), 파싱은 공식 파서를 쓴다.

    공식 입력 함수 호출(발화 수, 언어)을 기록하고, 토큰 11은 '안녕하세요.', 12는 LOOP로 바꿔 글을 만든다.
    extract_transcription은 진짜 공식 파서를 쓴다.
    """

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
    """모델 대역: generate 인자를 저장하고, 프롬프트(3토큰) 뒤에 생성 토큰이 붙은 결과를 돌려준다.

    첫 행은 1토큰 뒤 끝, 둘째 행은 끝 없이 3토큰(한도 도달)이다.
    """

    class generation_config:  # noqa: N801
        eos_token_id = list(EOS)

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 1, 1, 11, EOS[1], EOS[1]], [1, 1, 1, 12, 12, 12]])


# transcribe 한 번에: 공식 입력 함수를 언어 Korean으로 한 번 부르고, generate에는 입력과 등록한 greedy
# 인자(표본 없음, 빔 1, 256토큰)만 넘기며, 프롬프트를 잘라 낸 생성 부분에서 raw·fixed·토큰 수·한도 도달·넘침을
# 모두 만든다.
def test_one_decoding_gives_both_outputs_with_the_registered_arguments():
    model, processor = Model(), Processor()
    audio = [np.zeros(SAMPLE_RATE, dtype=np.float32)] * 2
    out = evaluate_qwen.transcribe(model, processor, audio, "ko", "cpu", torch.float16)
    assert processor.requests == [(2, "Korean")]  # 공식 프롬프트, 언어 고정
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


# 같은 행에서 raw와 fixed 리포트가 각자의 가설로 채점되는지: 가설만 다르고 원본 키(raw, fixed)는 행에 남지
# 않으며, 요약의 발화 수·빈 정답 수·한도 도달·넘침·fixed_changed가 맞고, 되풀이 발화는 raw에서 세어지며 반복
# 제거 덕분에 fixed의 오류율이 낮다. JSON으로 쓸 수 있는지도 본다.
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
    assert raw["summary"]["loops"] == 1  # 편집 수가 정답 글자 수보다 많다
    assert raw["summary"]["error_rate"] > fixed["summary"]["error_rate"]
    json.dumps(payloads, ensure_ascii=False)  # main이 쓰는 것


class Reached(Exception):
    """인자 검사를 통과했고 묶음을 불러오려던 참이다(그 순간에 던지는 예외)."""


# 리포트 폴더를 임시 폴더로 바꾸고 묶음·모델 불러오기를 Reached로 바꾸는 fixture(실제 reports/와 GPU를 쓰지
# 않는다).
@pytest.fixture
def reports(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(evaluate_qwen, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_qwen, "load_set", stop)
    monkeypatch.setattr(evaluate_qwen, "load_model", stop)
    return tmp_path


# 명령줄 인자를 바꿔 evaluate_qwen.main을 한 번 부른다.
def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["evaluate_qwen", *args])
    evaluate_qwen.main()


# test 묶음은 --allow-test가 있어야 잰다(단계마다 한 번).
def test_test_sets_need_allow_test(reports, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-test")
    assert "--allow-test" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, "--set", "zeroth-test", "--allow-test")


# raw와 fixed 리포트 중 하나라도 이미 있으면 --overwrite 없이는 멈춘다.
@pytest.mark.parametrize("existing", ["qwen3-asr-1.7b", "qwen3-asr-1.7b-fixed"])
def test_neither_report_is_replaced_without_overwrite(reports, monkeypatch, capsys, existing):
    (reports / existing).mkdir()
    (reports / existing / "zeroth-val.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val")
    assert "--overwrite" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, "--set", "zeroth-val", "--overwrite")


# CPU 실행은 측정 조건이 아니므로 reports/에 쓰지 못하고, --out(fp16 점검의 fp32 쪽)이나 --no-report만 된다.
def test_a_cpu_run_is_not_a_measurement(reports, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val500", "--device", "cpu", "--precision", "fp32")
    assert "--out" in capsys.readouterr().err
    with pytest.raises(Reached):
        out = str(reports / "gate" / "fp32.json")
        run(monkeypatch, "--set", "zeroth-val500", "--device", "cpu", "--precision", "fp32", "--out", out)
    with pytest.raises(Reached):
        run(monkeypatch, "--set", "zeroth-val500", "--device", "cpu", "--no-report", "--limit", "4")


# --out 파일도 덮어쓰기 보호를 받고, 파일을 쓰는 --out과 쓰지 않는 --no-report는 함께 줄 수 없다.
def test_out_is_not_replaced_without_overwrite_and_excludes_no_report(reports, monkeypatch, capsys):
    out = reports / "fp16.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val500", "--out", str(out))
    assert "--overwrite" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val500", "--out", str(out), "--overwrite", "--no-report")


# 등록한 모델의 고정 revision과 리포트 이름이 docs/experiments.md 9단계에 적은 값과 같고, 등록하지 않은 모델은
# --revision과 --name을 직접 줘야 한다.
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


# --- 10단계: 기본 가중치에 합친 LoRA 어댑터, 그리고 전화 채널 ---


# 10단계: 어댑터를 합쳐 잴 때는 --name이 필수다(학습 전 모델의 리포트 폴더에 섞이지 않게).
def test_an_adapter_needs_a_name(reports, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val", "--adapter", "outputs/qwen-qn/checkpoint-500")
    assert "--name" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, "--set", "zeroth-val", "--adapter", "a", "--name", "qwen3-asr-1.7b-qn")


# 채널 리포트는 <묶음>@telephone.json이라 깨끗한 음성 리포트와 다른 파일이다. 전화 음질 리포트만 있으면 깨끗한
# 판은 잴 수 있고, 전화 음질 판은 덮어쓰기 보호에 걸린다.
def test_channel_reports_are_written_as_set_at_channel(reports, monkeypatch, capsys):
    for name in ("qwen3-asr-1.7b", "qwen3-asr-1.7b-fixed"):
        (reports / name).mkdir()
        (reports / name / "zeroth-val@telephone.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Reached):  # 깨끗한 음성 리포트는 다른 파일이다
        run(monkeypatch, "--set", "zeroth-val")
    with pytest.raises(SystemExit):
        run(monkeypatch, "--set", "zeroth-val", "--channel", "telephone")
    assert "zeroth-val@telephone.json exists" in capsys.readouterr().err


# 채널 변환을 발화 전체에 먼저 적용하고 그 뒤 앞 30초를 자르는지(Whisper 하네스와 같은 순서). 결과가 '변환 뒤
# 자르기'와 정확히 같아야 한다.
def test_the_channel_is_applied_before_the_first_30_seconds_are_kept():
    rng = np.random.default_rng(0)
    long = (rng.standard_normal(31 * SAMPLE_RATE) * 0.1).astype(np.float32)
    clean = evaluate_qwen.prepare_audio(long, None)
    phone = evaluate_qwen.prepare_audio(long, "telephone")
    assert clean.shape == phone.shape == (30 * SAMPLE_RATE,)
    assert np.array_equal(clean, long[: 30 * SAMPLE_RATE]) and not np.allclose(phone, clean)
    assert np.array_equal(phone, evaluate_qwen.CHANNELS["telephone"](long)[: 30 * SAMPLE_RATE])


# 작은 모델에 train_qwen.add_lora로 LoRA를 붙여 저장한다. train_b면 B 행렬을 0.01로 채워 '학습한' 어댑터를
# 흉내 내고, 아니면 B가 0인 학습 전 어댑터다.
def save_adapter(folder, train_b: bool):
    from whisper_ko_ft.train_qwen import add_lora

    model = add_lora(tiny_model(torch.float16))
    if train_b:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.01)
    model.save_pretrained(str(folder))


# 학습 전 어댑터(B=0)를 합치면 모든 가중치가 그대로이고 결과는 보통의 Qwen3-ASR 모델이다. 10단계 관문 (a)(학습
# 전 어댑터로 9단계 디코딩이 재현되는지)의 전제다.
def test_merging_a_fresh_adapter_leaves_every_weight_as_it_was(tmp_path):
    save_adapter(tmp_path / "fresh", train_b=False)
    base = {k: v.clone() for k, v in tiny_model(torch.float16).state_dict().items()}
    merged = evaluate_qwen.merge_adapter(tiny_model(torch.float16), str(tmp_path / "fresh"))
    after = merged.state_dict()
    assert type(merged).__name__ == "Qwen3ASRForConditionalGeneration"
    assert after.keys() == base.keys()
    assert all(torch.equal(after[k], base[k]) for k in base)


# B가 0이 아닌 어댑터를 합치면 겨냥한 층(언어 모델 MLP, 오디오 인코더 fc1 등)만 바뀌고,
# 투영층·출력층·합성곱·정규화는 그대로이며, 모든 가중치가 fp16으로 남는다.
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


# 리포트에 남길 어댑터 정보: 폴더 경로와 가중치 파일의 SHA-256. 어댑터가 없으면 둘 다 None.
def test_reports_name_the_adapter_and_its_weights(tmp_path):
    save_adapter(tmp_path / "checkpoint-500", train_b=False)
    info = evaluate_qwen.adapter_info(str(tmp_path / "checkpoint-500"))
    weights = (tmp_path / "checkpoint-500" / "adapter_model.safetensors").read_bytes()
    assert info["adapter"].endswith("checkpoint-500")
    assert info["adapter_sha256"] == hashlib.sha256(weights).hexdigest()
    assert evaluate_qwen.adapter_info(None) == {"adapter": None, "adapter_sha256": None}
