# [테스트 묶음] evaluate.py의 규칙 보호 장치(인자 검사)와 리포트 요약 도우미를 확인한다: --adapter에는 --name
# 필수, 있는 리포트는 --overwrite 없이 덮어쓰지 않음, 스모크용 기준값은 --no-report에서만, 행 채점과 배치 시간
# 요약, 등록한 Hub ID에만 고정 revision을 넘기는 모델 불러오기, 요약의 재현 지문(revision·라이브러리 판·어댑터
# SHA-256).
# [왜 중요한가] 리포트는 커밋된 측정 기록이고 모든 판정이 이것을 다시 읽는다. 어댑터 평가가 기준선 리포트를
# 덮어쓰거나, 일부만 잰 리포트가 전체 리포트를 대신하는 사고(커밋 a29a975가 막은 것)가 다시 생기지 않게 한다.
# 기본 모델의 revision과 어댑터 가중치의 지문이 없으면 Hub의 모델이 바뀌거나 outputs/의 어댑터를 다시 만든
# 뒤에 리포트만으로는 같은 가중치로 쟀는지 알 수 없다.
# [방법] 인자 검사를 통과하면 모델을 불러오기 직전에 Reached를 던지게 해, GPU 없이 '멈췄는지/통과했는지'만
# 본다. 요약의 지문은 모델·받아 적기·torch.cuda를 가짜로 바꿔 main을 끝까지 돌려 본다. 커밋 표시(git_commit)는
# provenance로 옮겨 tests/test_provenance.py가 확인한다.

import hashlib
import json
import sys
from types import SimpleNamespace

import peft
import pytest
import torch
import transformers

from whisper_ko_ft import evaluate
from whisper_ko_ft.provenance import WHISPER_REVISIONS
from whisper_ko_ft.store import Utterance


class Reached(Exception):
    """인자 검사를 통과했고 묶음을 불러오려던 참이다(아래 fixture가 load_set·load_model을 이것으로 바꾼다)."""


# 리포트 폴더와 저장소 루트를 임시 폴더로 바꾸고, 묶음·모델 불러오기를 Reached로 바꾸는 fixture. 실제
# reports/를 건드리지 않는다.
@pytest.fixture
def reports(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(evaluate, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "load_set", stop)
    monkeypatch.setattr(evaluate, "load_model", stop)
    return tmp_path


# 명령줄 인자를 바꿔 evaluate.main을 한 번 부른다.
def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["evaluate", *args])
    evaluate.main()


# --adapter인데 --name이 없으면 parser.error로 멈추고(오류 메시지에 --name), --name을 주면 측정 단계까지 간다.
def test_adapter_needs_a_report_name(reports, monkeypatch, capsys):
    adapter = ["--model", "openai/whisper-large-v3-turbo", "--adapter", "x", "--set", "zeroth-val"]
    # --name이 없으면 리포트가 기본 모델의 폴더에 들어간다.
    with pytest.raises(SystemExit):
        run(monkeypatch, *adapter)
    assert "--name" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, *adapter, "--name", "l2")


# 이미 있는 리포트(채널 판 포함)는 --overwrite 없이는 멈추고, --overwrite나 --no-report(리포트를 쓰지 않는
# 속도·스모크 실행)면 진행한다.
@pytest.mark.parametrize("channel", [[], ["--channel", "telephone"]])
def test_an_existing_report_is_not_replaced_without_overwrite(reports, monkeypatch, capsys, channel):
    suffix = "@telephone" if channel else ""
    (reports / "m").mkdir()
    (reports / "m" / f"zeroth-val{suffix}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val", *channel)
    assert "--overwrite" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val", *channel, "--overwrite")
    with pytest.raises(Reached):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val", *channel, "--no-report", "--limit", "5")


# 8단계 재시도 후보 D1a(ratio)로 재는 기본 인자.
SMOKE = ["--model", "m", "--set", "zeroth-val", "--fallback", "ratio"]


# 스모크용으로 기준값을 낮추면 --no-report가 있어야 한다. 등록하지 않은 기준으로 만든 리포트가 남지 않게
# 하려는 것이다.
def test_threshold_overrides_are_for_smoke_runs_only(reports, monkeypatch, capsys):
    # 낮춘 기준값은 등록한 후보가 아니므로 리포트를 남기면 안 된다.
    with pytest.raises(SystemExit):
        run(monkeypatch, *SMOKE, "--compression-ratio-threshold", "1.0")
    assert "--no-report" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, *SMOKE, "--compression-ratio-threshold", "1.0", "--no-report", "--limit", "16")


# 기준값은 맞는 후보가 있을 때만 받는다: 재시도 없이 압축비 기준만 주거나, 로그확률 기준이 없는 D1a에 로그확률
# 기준을 주면 멈춘다.
@pytest.mark.parametrize(
    "args",
    [
        ["--model", "m", "--set", "zeroth-val", "--compression-ratio-threshold", "1.0"],  # 재시도 없음
        [*SMOKE, "--logprob-threshold", "-0.5"],  # D1a에는 로그확률 기준이 없다
    ],
)
def test_threshold_overrides_need_the_matching_fallback(reports, monkeypatch, args):
    with pytest.raises(SystemExit):
        run(monkeypatch, *args, "--no-report")


# 등록한 두 후보(ratio, ratio-logprob)와 --seed는 검사를 통과해 측정 단계까지 간다.
def test_fallback_runs_reach_the_measurement(reports, monkeypatch):
    with pytest.raises(Reached):
        run(monkeypatch, *SMOKE)
    with pytest.raises(Reached):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val", "--fallback", "ratio-logprob", "--seed", "3")


# score_rows가 기록된 리포트와 같은 필드를 채우는지: 편집 수·길이·가설의 숫자 여부, 정규화하면 비는 정답은 0과
# 0으로 두고 그 수를 돌려준다. 9단계에서 Qwen3-ASR 하네스와 함께 쓰려고 main 밖으로 옮긴 함수라, 행이 기록된
# 리포트와 같은 모양으로 나오는지 고정한다.
def test_rows_are_scored_as_before():
    rows = [
        {"id": "a", "reference": "오는 오 월", "hypothesis": "오는 5월"},
        {"id": "b", "reference": "...", "hypothesis": "아무 말"},  # 정규화하면 비어 건너뛴다
        {"id": "c", "reference": "문장", "hypothesis": "문장"},
    ]
    assert evaluate.score_rows(rows, "ko") == 1
    assert (rows[0]["edits"], rows[0]["length"], rows[0]["has_digit"]) == (1, 4, True)
    assert (rows[1]["edits"], rows[1]["length"]) == (0, 0) and "has_digit" not in rows[1]
    assert (rows[2]["edits"], rows[2]["length"], rows[2]["has_digit"]) == (0, 2, False)


# 배치 시간의 중앙값과 90번째 백분위(numpy 선형 보간: 0.4와 1.0 사이 → 0.76), 빈 목록이면 None.
def test_batch_times_in_the_summary():
    # 배치 크기 1에서의 발화당 지연(9단계 속도 확인).
    assert evaluate.batch_seconds_summary([0.1, 0.2, 0.3, 0.4, 1.0]) == {
        "batch_seconds_median": 0.3,
        "batch_seconds_p90": 0.76,
    }
    assert evaluate.batch_seconds_summary([]) == {"batch_seconds_median": None, "batch_seconds_p90": None}


# 발화가 하나도 없는 묶음(빈 캐시로 만든 묶음)은 모델을 올리기 전에 멈춘다. 모델 불러오기(Reached)에 닿지
# 않아야 하고, 멈춘 이유에 묶음 이름이 나와야 한다.
def test_an_empty_set_is_refused_before_the_model_is_loaded(reports, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "load_set", lambda name: ("ko", []))
    with pytest.raises(SystemExit):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val")
    err = capsys.readouterr().err
    assert "zeroth-val" in err and "no utterances" in err


# 리포트 JSON은 NaN·inf를 거부한다(allow_nan=False). 정답이 모두 빈 묶음의 오류율은 0 ÷ 0 = NaN이 되는데, 표준
# JSON이 아닌 리포트가 '성공'으로 남으면 안 된다. 한국어는 그대로(ensure_ascii=False) 적는다.
def test_report_json_refuses_nan_and_keeps_korean():
    with pytest.raises(ValueError):
        evaluate.report_json({"summary": {"error_rate": float("nan")}})
    text = evaluate.report_json({"summary": {"error_rate": 0.5}, "utterances": [{"reference": "안녕"}]})
    assert "안녕" in text and json.loads(text)["summary"]["error_rate"] == 0.5


# --- 기본 모델 revision 고정과 요약의 재현 지문 ---


# load_model은 등록한 Hub ID(whisper-small, large-v3-turbo)에만 고정한 revision을 모델·처리기
# from_pretrained에 넘기고, outputs/ 아래 로컬 폴더에는 넘기지 않는다(로컬 폴더에는 revision이 없다). 가중치는
# 받지 않고 from_pretrained를 가짜로 바꿔 받은 인자만 본다.
def test_load_model_pins_the_registered_revision_for_hub_ids_only(monkeypatch):
    seen = {}

    class Loaded:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_model(path, **kwargs):
        seen["model"] = kwargs
        return Loaded()

    def fake_processor(path, **kwargs):
        seen["processor"] = kwargs
        return "processor"

    monkeypatch.setattr(evaluate.WhisperForConditionalGeneration, "from_pretrained", fake_model)
    monkeypatch.setattr(evaluate.WhisperProcessor, "from_pretrained", fake_processor)

    evaluate.load_model("openai/whisper-large-v3-turbo", None)
    turbo = WHISPER_REVISIONS["openai/whisper-large-v3-turbo"]
    assert seen["model"] == {"dtype": torch.float16, "revision": turbo}
    assert seen["processor"] == {"revision": turbo}
    evaluate.load_model("outputs/turbo-n/merged", None)
    assert seen["model"] == {"dtype": torch.float16} and seen["processor"] == {}


class Features:
    """특징 추출기 출력 대역. input_features·attention_mask가 자기 자신이고 to()는 자신을 돌려준다.

    배치의 '오디오'(아래 IdStore가 돌려주는 발화 ID)를 들고 있어 가짜 transcribe가 가설을 고른다.
    """

    def __init__(self, audio) -> None:
        self.audio = audio
        self.input_features = self.attention_mask = self

    def to(self, *args, **kwargs):
        return self


class IdStore:
    """오디오 저장소 대역: memmap 대신 발화 ID를 '오디오'로 돌려준다."""

    def __init__(self, name: str) -> None:
        pass

    def read(self, utterance):
        return utterance.id


# evaluate.main이 GPU·가중치 없이 끝까지 돌아 리포트를 쓰게 바꾼다: 묶음(정답), 모델 불러오기(아무 객체와 가짜
# 특징 추출기), 받아 적기(발화 ID → 가설), 오디오 저장소, torch.cuda의 메모리 통계·동기화·장치 이름.
def transcribing(monkeypatch, tmp_path, references: dict[str, str], hypotheses: dict[str, str]) -> None:
    utterances = [Utterance(i, "s", text, "x", 0, 16_000) for i, text in references.items()]
    processor = SimpleNamespace(feature_extractor=lambda audio, **kwargs: Features(audio))

    def transcribe(model, processor, features, attention_mask, language, settings, trace):
        return [hypotheses[a] for a in features.audio], None

    monkeypatch.setattr(evaluate, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "load_set", lambda name: ("ko", utterances))
    monkeypatch.setattr(evaluate, "load_model", lambda path, adapter: (object(), processor))
    monkeypatch.setattr(evaluate, "transcribe", transcribe)
    monkeypatch.setattr(evaluate, "AudioStore", IdStore)
    monkeypatch.setattr(evaluate.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(evaluate.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(evaluate.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(evaluate.torch.cuda, "get_device_name", lambda index: "Fake GPU")


# 요약에 재현 지문이 남는다: 기본 모델의 revision(등록한 Hub ID면 sha, 로컬 폴더면 null), transformers 판,
# 어댑터가 있으면 peft 판과 어댑터 가중치·설정의 SHA-256(기존 adapter 키는 준 경로 그대로). 어댑터가 없으면
# peft와 두 SHA는 null이다. 오류율 같은 기존 키의 계산은 그대로다.
def test_the_summary_records_the_revision_the_library_versions_and_the_adapter_hashes(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_bytes(b'{"r": 16}')
    transcribing(monkeypatch, tmp_path, {"a": "가나다", "b": "라마"}, {"a": "가나다", "b": "라마바"})
    turbo = [
        "--model",
        "openai/whisper-large-v3-turbo",
        "--adapter",
        str(adapter),
        "--name",
        "n",
        "--set",
        "x",
    ]

    run(monkeypatch, *turbo)

    summary = json.loads((tmp_path / "n" / "x.json").read_text(encoding="utf-8"))["summary"]
    assert summary["revision"] == WHISPER_REVISIONS["openai/whisper-large-v3-turbo"]
    assert summary["adapter"] == str(adapter)  # 기존 키: 준 경로 그대로
    assert summary["adapter_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert summary["adapter_config_sha256"] == hashlib.sha256(b'{"r": 16}').hexdigest()
    assert (summary["transformers"], summary["peft"]) == (transformers.__version__, peft.__version__)
    assert (summary["utterances"], summary["error_rate"]) == (2, 0.2)  # 편집 1 ÷ 글자 5

    run(monkeypatch, "--model", "outputs/small-a/best", "--set", "x")  # 로컬 폴더, 어댑터 없음

    summary = json.loads((tmp_path / "best" / "x.json").read_text(encoding="utf-8"))["summary"]
    assert (summary["revision"], summary["adapter"], summary["peft"]) == (None, None, None)
    assert (summary["adapter_sha256"], summary["adapter_config_sha256"]) == (None, None)
    assert summary["transformers"] == transformers.__version__


# 어댑터 지문은 모델을 올리기 전에 읽는다(커밋처럼 디코딩 전에). 그래서 요약의 SHA-256은 실제로 불러온 파일의
# 것이고, 파일이 없으면 모델을 올리고 묶음을 다 디코딩한 뒤가 아니라 시작할 때 멈춘다. 학습이 도는 실행의
# 체크포인트를 재는 동안 save_total_limit가 그 폴더를 지우는 경우, 끝에서 해시하면 다 잰 측정을 잃는다.
def test_the_adapter_is_fingerprinted_before_the_model_is_loaded(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")  # adapter_config.json이 없다
    transcribing(monkeypatch, tmp_path, {"a": "가나다"}, {"a": "가나다"})
    loaded: list[str] = []
    monkeypatch.setattr(evaluate, "load_model", lambda path, adapter: loaded.append(path))
    turbo = ["--model", "openai/whisper-large-v3-turbo", "--adapter", str(adapter), "--name", "n"]

    with pytest.raises(FileNotFoundError, match="adapter_config"):
        run(monkeypatch, *turbo, "--set", "x")
    assert loaded == []  # 모델을 올리기 전에 멈췄다
    assert not (tmp_path / "n").exists()
