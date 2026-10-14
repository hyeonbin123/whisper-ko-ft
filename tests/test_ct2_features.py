# [테스트 묶음] faster-whisper 경로의 멜 필터 수(ct2_features.py, evaluate_ct2.py)를 확인한다. 합친 모델
# 폴더에 128개 설정이 써지는지, 설정 파일이 없거나 틀린 폴더를 어떻게 다루는지, evaluate_ct2가 오디오를 읽기
# 전에 turbo에 128개를 쓰는지. 끝에 evaluate_ct2 요약의 빈 가설 수가 evaluate와 같은 기준인지도 본다.
# [왜 중요한가] 2026-09-20 재시도 측정이 turbo에 멜 80개로 계산된 특징 때문에 첫 발화에서 모두
# 멈췄다(docs/experiments.md '추가 측정', 커밋 61b4b2f에서 고침). 같은 실패가 다시 나지 않게 고정한다.
# evaluate_ct2 테스트는 고치기 전 코드에서 실패하는 것을 확인했다(커밋 61b4b2f 메시지).
# [방법] 가중치 없이 설정 객체만 만들고, evaluate_ct2 테스트는 faster_whisper 모델을 가짜로 바꾼다.
# faster-whisper는 선택 의존성(ct2 묶음)이라 없으면 evaluate_ct2 테스트는 건너뛴다(CI에는 설치하지 않는다).

import json
import sys

import pytest
from transformers import WhisperConfig, WhisperFeatureExtractor

from whisper_ko_ft.ct2_features import (
    COPY_FILES,
    PREPROCESSOR_CONFIG,
    feature_kwargs,
    write_preprocessor_config,
)

# whisper-large-v3와 large-v3-turbo는 멜 필터 128개, whisper-small은 80개를 받는다.
TURBO_MEL_BINS = 128
SMALL_MEL_BINS = 80


# merge_adapter가 쓰는 설정 파일: 이름이 변환기 복사 목록(COPY_FILES)에 있고, faster-whisper가 읽는 평평한
# 형식으로 feature_size 128이 들어 있으며, feature_kwargs가 그것을 그대로 읽는다.
def test_merged_folder_gets_the_128_bin_config_that_the_converter_copies(tmp_path):
    config = WhisperConfig(num_mel_bins=TURBO_MEL_BINS)  # 가중치는 불러오지 않는다
    extractor = WhisperFeatureExtractor(feature_size=TURBO_MEL_BINS)

    written = write_preprocessor_config(extractor, config.num_mel_bins, tmp_path)

    assert written.name == PREPROCESSOR_CONFIG and PREPROCESSOR_CONFIG in COPY_FILES
    saved = json.loads(written.read_text(encoding="utf-8"))
    assert saved["feature_size"] == TURBO_MEL_BINS  # faster-whisper가 읽는 평평한 형식
    assert feature_kwargs(tmp_path, TURBO_MEL_BINS)["feature_size"] == TURBO_MEL_BINS


# 기본 특징 추출기(80개)를 128개 모델에 쓰려고 하면 거부하고 파일을 남기지 않는다.
def test_a_feature_extractor_that_does_not_fit_the_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="128"):
        write_preprocessor_config(WhisperFeatureExtractor(), TURBO_MEL_BINS, tmp_path)
    assert not (tmp_path / PREPROCESSOR_CONFIG).exists()


# 설정 파일 없이 tokenizer.json만 있는 폴더(2026-09-20에 변환한 폴더와 같은 상태)는 모델의 멜 수만 넣은 인자를
# 받는다. turbo와 small 모두.
@pytest.mark.parametrize("n_mels", [TURBO_MEL_BINS, SMALL_MEL_BINS])
def test_a_folder_without_the_config_uses_the_model_mel_bins(tmp_path, n_mels):
    # 2026-09-20에 변환한 폴더에는 tokenizer.json만 있었고, faster-whisper는 turbo에 멜 80개를 썼다.
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert feature_kwargs(tmp_path, n_mels) == {"feature_size": n_mels}


# 설정 파일이 80개인데 모델이 128개면 조용히 틀리게 계산하지 않고 오류로 멈춘다.
def test_a_config_that_disagrees_with_the_model_is_refused(tmp_path):
    write_preprocessor_config(WhisperFeatureExtractor(), SMALL_MEL_BINS, tmp_path)
    with pytest.raises(ValueError, match="128 mel bins"):
        feature_kwargs(tmp_path, TURBO_MEL_BINS)


class Reached(Exception):
    """모델을 띄웠고 묶음을 읽으려던 참이다. 이 순서(모델 먼저, 오디오 나중)를 확인하는 데 쓰는 예외다."""


class FakeWhisperModel:
    """preprocessor_config.json이 없는 turbo 폴더를 연 faster_whisper.WhisperModel 흉내(가중치 없음).

    변환된 turbo 모델(n_mels 128)처럼 보이지만 faster-whisper 기본 특징 추출기(80개)를 가진 상태로 만든다.
    마지막으로 만든 객체를 last에 남겨 테스트가 꺼내 본다.
    """

    def __init__(self, path, device, compute_type):
        from faster_whisper.feature_extractor import FeatureExtractor

        self.model = type("Ct2Whisper", (), {"n_mels": TURBO_MEL_BINS})()
        self.feature_extractor = FeatureExtractor()  # faster-whisper가 만드는 것: 멜 80개
        FakeWhisperModel.last = self


# evaluate_ct2.main이 묶음을 읽기 전에 특징 추출기를 128개로 바꿔 두는지. 리포트 위치·커밋·모델·묶음 읽기는
# monkeypatch로 바꿔 실제 파일과 GPU를 쓰지 않는다.
def test_evaluate_ct2_uses_128_mel_bins_for_turbo_before_reading_audio(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper")  # ct2 의존성 묶음. CI에는 설치하지 않는다
    from whisper_ko_ft import evaluate_ct2

    def stop(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(evaluate_ct2, "load_set", stop)
    monkeypatch.setattr(sys, "argv", ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x"])

    with pytest.raises(Reached):
        evaluate_ct2.main()
    assert FakeWhisperModel.last.feature_extractor.mel_filters.shape[0] == TURBO_MEL_BINS


# evaluate_ct2도 있는 리포트를 --overwrite 없이 덮어쓰지 않고, --no-report(속도 측정)면 리포트를 건드리지
# 않으므로 그대로 진행한다.
def test_evaluate_ct2_no_report_skips_the_existing_report_check(tmp_path, monkeypatch, capsys):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2

    def stop(*args, **kwargs):
        raise Reached

    (tmp_path / "n").mkdir()
    (tmp_path / "n" / "x.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(evaluate_ct2, "load_set", stop)
    args = ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x"]

    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        evaluate_ct2.main()
    assert "--overwrite" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*args, "--no-report"])  # 속도 측정: 리포트를 바꿔 쓰지 않는다
    with pytest.raises(Reached):
        evaluate_ct2.main()


# faster-whisper 구간 대역: 리포트가 쓰는 text와 temperature만 가진다.
class Segment:
    def __init__(self, text: str, temperature: float = 0.0) -> None:
        self.text, self.temperature = text, temperature


# 받아 적기까지 하는 가짜 모델. 오디오 대신 받은 발화 ID로 미리 정한 구간 목록을 돌려준다.
class TranscribingModel(FakeWhisperModel):
    segments: dict[str, list[Segment]] = {}

    def transcribe(self, audio, **kwargs):
        return iter(self.segments[audio]), None


# 오디오 저장소 대역: memmap 대신 발화 ID를 '오디오'로 돌려준다.
class IdStore:
    def __init__(self, name: str) -> None:
        pass

    def read(self, utterance):
        return utterance.id


# 요약의 빈 가설 수는 evaluate.decoding_counts와 같은 기준이다: 채점한 발화(정답이 비지 않은 발화)만
# 세고, 공백만 남은 가설도 빈 가설로 센다. 고치기 전에는 정답이 빈 발화까지 세고 공백 가설은 세지 않아
# 3이었다.
def test_evaluate_ct2_counts_empty_hypotheses_over_scored_utterances_only(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2
    from whisper_ko_ft.store import Utterance

    texts = {"spoken": "가나다", "silent": "라마", "blank": "바사", "no-ref-1": "", "no-ref-2": ""}
    TranscribingModel.segments = {
        "spoken": [Segment("가나다")],
        "silent": [],  # 받아 적은 것이 없다
        "blank": [Segment(" "), Segment(" ")],  # 이으면 공백 하나가 된다
        "no-ref-1": [],
        "no-ref-2": [],
    }
    utterances = [Utterance(i, "s", text, "x", 0, 16_000) for i, text in texts.items()]
    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", TranscribingModel)
    monkeypatch.setattr(evaluate_ct2, "AudioStore", IdStore)
    monkeypatch.setattr(evaluate_ct2, "load_set", lambda name: ("ko", utterances))
    monkeypatch.setattr(sys, "argv", ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x"])

    evaluate_ct2.main()

    summary = json.loads((tmp_path / "n" / "x.json").read_text(encoding="utf-8"))["summary"]
    assert (summary["utterances"], summary["empty_hypotheses"]) == (3, 2)
