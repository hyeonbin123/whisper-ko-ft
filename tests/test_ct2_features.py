# [테스트 묶음] faster-whisper 경로의 멜 필터 수(ct2_features.py, evaluate_ct2.py)를 확인한다. 합친 모델
# 폴더에 128개 설정이 써지는지, 설정 파일이 없거나 틀린 폴더를 어떻게 다루는지, evaluate_ct2가 오디오를 읽기
# 전에 turbo에 128개를 쓰는지. 끝에 evaluate_ct2 요약의 빈 가설 수가 evaluate와 같은 기준인지, 요약에 변환
# 폴더의 지문(model.bin·config.json SHA-256)과 --seed가 남는지, merge_adapter가 등록한 기본 모델의 revision을
# 고정하는지도 본다.
# [왜 중요한가] 2026-09-20 재시도 측정이 turbo에 멜 80개로 계산된 특징 때문에 첫 발화에서 모두
# 멈췄다(docs/experiments.md '추가 측정', 커밋 61b4b2f에서 고침). 같은 실패가 다시 나지 않게 고정한다.
# evaluate_ct2 테스트는 고치기 전 코드에서 실패하는 것을 확인했다(커밋 61b4b2f 메시지). 변환 폴더는 같은
# 경로에 다시 만들어진 이력이 있어(outputs/turbo-n/ct2), 지문이 없으면 리포트만으로는 어느 변환으로 쟀는지
# 모른다.
# [방법] 가중치 없이 설정 객체만 만들고, evaluate_ct2 테스트는 faster_whisper 모델을 가짜로 바꾼다.
# faster-whisper는 선택 의존성(ct2 묶음)이라 없으면 evaluate_ct2 테스트는 건너뛴다. CI는 2026-10-15부터 ct2
# 묶음을 설치해 이 테스트들도 돌린다(.github/workflows/ci.yml). 모델은 내려받지 않는다.

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperConfig, WhisperFeatureExtractor

from whisper_ko_ft.ct2_features import (
    COPY_FILES,
    PREPROCESSOR_CONFIG,
    feature_kwargs,
    write_preprocessor_config,
)
from whisper_ko_ft.provenance import WHISPER_REVISIONS

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
    pytest.importorskip("faster_whisper")  # ct2 의존성 묶음. CI는 설치한다, 없는 로컬 환경에서는 건너뛴다
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


# 변환된 CTranslate2 폴더 흉내: 요약의 지문이 읽는 두 파일(model.bin, config.json)을 작은 바이트로 쓰고,
# 요약에 남아야 할 SHA-256을 돌려준다. 묶음을 끝까지 받아 적는 테스트는 이 폴더가 있어야 한다(실제 변환
# 폴더에는 두 파일이 늘 있다. model.bin이 없으면 WhisperModel이 불러오다 멈추고, config.json이 없으면 첫
# 디코딩에서 멈춘다).
def converted_folder(folder) -> dict[str, str]:
    files = {"model.bin": b"ct2 weights", "config.json": b'{"n_mels": 128}'}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return {
        "model_sha256": hashlib.sha256(files["model.bin"]).hexdigest(),
        "config_sha256": hashlib.sha256(files["config.json"]).hexdigest(),
    }


# 요약의 빈 가설 수는 evaluate.decoding_counts와 같은 기준이다: 채점한 발화(정답이 비지 않은 발화)만
# 세고, 공백만 남은 가설도 빈 가설로 센다. 고치기 전에는 정답이 빈 발화까지 세고 공백 가설은 세지 않아
# 3이었다.
def test_evaluate_ct2_counts_empty_hypotheses_over_scored_utterances_only(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2
    from whisper_ko_ft.store import Utterance

    converted_folder(tmp_path)
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


# 12단계: --compute-type이 WhisperModel에 그대로 가고(기본 float16은 기록과 같다), 요약에 정밀도·장치와 한
# 발화 지연(중앙값·90번째 백분위)이 남는다.
def test_evaluate_ct2_passes_the_compute_type_and_records_latency(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2
    from whisper_ko_ft.store import Utterance

    seen = {}

    class Recording(TranscribingModel):
        def __init__(self, path, device, compute_type):
            super().__init__(path, device, compute_type)
            seen.update(device=device, compute_type=compute_type)

    converted_folder(tmp_path)
    Recording.segments = {"a": [Segment("가나다")], "b": [Segment("라마")]}
    utterances = [Utterance(i, "s", text, "x", 0, 16_000) for i, text in (("a", "가나다"), ("b", "라마"))]
    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", Recording)
    monkeypatch.setattr(evaluate_ct2, "AudioStore", IdStore)
    monkeypatch.setattr(evaluate_ct2, "load_set", lambda name: ("ko", utterances))
    argv = ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x"]
    monkeypatch.setattr(sys, "argv", [*argv, "--compute-type", "int8_float16"])

    evaluate_ct2.main()

    summary = json.loads((tmp_path / "n" / "x.json").read_text(encoding="utf-8"))["summary"]
    assert seen == {"device": "cuda", "compute_type": "int8_float16"}
    assert (summary["compute_type"], summary["device"]) == ("int8_float16", "cuda")
    assert summary["batch_seconds_median"] is not None and summary["batch_seconds_p90"] is not None


# 오디오 저장소의 memmap은 저장소마다 한 번만 연다. 고치기 전에는 setdefault의 기본값 인자 AudioStore(...)가
# 키가 있어도 먼저 계산되어 발화마다 새 memmap을 열었고, 그 시간이 한 발화 지연(utterance_started부터) 안에
# 들어갔다. 두 저장소에 걸친 발화 셋을 재면 여는 것은 저장소 둘이고, 가설과 채점은 그대로다.
def test_evaluate_ct2_opens_each_audio_store_once(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2
    from whisper_ko_ft.store import Utterance

    opened: list[str] = []

    class CountingStore(IdStore):
        def __init__(self, name: str) -> None:
            opened.append(name)

    converted_folder(tmp_path)
    texts = {"a": "가나다", "b": "라마", "c": "바사"}
    TranscribingModel.segments = {i: [Segment(text)] for i, text in texts.items()}
    stores = zip(texts.items(), "xyx", strict=True)
    utterances = [Utterance(i, "s", text, name, 0, 16_000) for (i, text), name in stores]
    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", TranscribingModel)
    monkeypatch.setattr(evaluate_ct2, "AudioStore", CountingStore)
    monkeypatch.setattr(evaluate_ct2, "load_set", lambda name: ("ko", utterances))
    monkeypatch.setattr(sys, "argv", ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x"])

    evaluate_ct2.main()

    assert opened == ["x", "y"]
    summary = json.loads((tmp_path / "n" / "x.json").read_text(encoding="utf-8"))["summary"]
    assert (summary["utterances"], summary["error_rate"]) == (3, 0.0)


# CPU 실행은 측정 조건이 아니라 리포트를 남기지 못한다(--no-report만).
def test_a_cpu_ct2_run_needs_no_report(tmp_path, monkeypatch, capsys):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2

    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    argv = ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        evaluate_ct2.main()
    assert "--no-report" in capsys.readouterr().err


# 발화가 하나도 없는 묶음은 묶음을 읽은 자리에서 바로 멈춘다(오디오를 읽거나 리포트를 쓰지 않는다). 모델은
# 묶음보다 먼저 띄우는 순서라 멈추는 자리는 모델 뒤, 오디오 앞이다.
def test_evaluate_ct2_refuses_an_empty_set(tmp_path, monkeypatch, capsys):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2

    def no_audio(name):
        raise AssertionError("an empty set must not open an audio store")

    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(evaluate_ct2, "AudioStore", no_audio)
    monkeypatch.setattr(evaluate_ct2, "load_set", lambda name: ("ko", []))
    monkeypatch.setattr(sys, "argv", ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x"])

    with pytest.raises(SystemExit):
        evaluate_ct2.main()
    assert "no utterances" in capsys.readouterr().err
    assert not (tmp_path / "n").exists()


# 정답이 모두 빈 묶음은 오류율이 0 ÷ 0 = NaN이 되는데, 리포트 JSON은 NaN을 거부해(allow_nan=False) 파일을
# 남기지 않는다. 표준 JSON이 아닌 리포트가 '성공'으로 남는 것을 막는다.
def test_evaluate_ct2_does_not_write_a_report_with_nan(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2
    from whisper_ko_ft.store import Utterance

    converted_folder(tmp_path)
    TranscribingModel.segments = {"a": [Segment("가")], "b": []}
    utterances = [Utterance(i, "s", "", "x", 0, 16_000) for i in ("a", "b")]  # 정답이 모두 비었다
    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", TranscribingModel)
    monkeypatch.setattr(evaluate_ct2, "AudioStore", IdStore)
    monkeypatch.setattr(evaluate_ct2, "load_set", lambda name: ("ko", utterances))
    monkeypatch.setattr(sys, "argv", ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x"])

    with pytest.raises(ValueError):
        evaluate_ct2.main()
    assert not (tmp_path / "n" / "x.json").exists()


# 요약에 변환 폴더의 지문(model.bin·config.json의 SHA-256)이 남는다. --seed를 주면 모델을 띄우기 전에
# ctranslate2.set_random_seed(seed)를 한 번만 불러, 같은 명령의 재시도 temperature 표본이 실행마다 같게 한다.
# 띄운 뒤에(예: 발화마다) 부르면 소용이 없다: CTranslate2의 난수 생성기는 작업 스레드마다 처음 쓸 때 한 번만
# 시드를 받고, set_random_seed는 이미 만든 생성기를 다시 시드하지 않는다(CTranslate2 4.8.2
# src/random.cc). 그래서 순서(시드 → 모델)를 본다. --seed가 없으면(기본, 기록된 측정과 같은 경로) 시드를
# 걸지 않고 요약의 seed는 null이다.
def test_evaluate_ct2_records_the_folder_hashes_and_seeds_once_before_loading_only_when_asked(
    tmp_path, monkeypatch
):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2
    from whisper_ko_ft.store import Utterance

    expected = converted_folder(tmp_path)
    events: list[tuple] = []  # 시드 걸기와 모델 띄우기를 일어난 순서대로

    class Loading(TranscribingModel):
        def __init__(self, path, device, compute_type):
            super().__init__(path, device, compute_type)
            events.append(("load",))

    Loading.segments = {"a": [Segment("가나다", 0.2)], "b": [Segment("라마")]}
    utterances = [Utterance(i, "s", text, "x", 0, 16_000) for i, text in (("a", "가나다"), ("b", "라마"))]

    def seeding(seed):
        events.append(("seed", seed))

    monkeypatch.setattr(evaluate_ct2.ctranslate2, "set_random_seed", seeding)
    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", Loading)
    monkeypatch.setattr(evaluate_ct2, "AudioStore", IdStore)
    monkeypatch.setattr(evaluate_ct2, "load_set", lambda name: ("ko", utterances))
    argv = ["evaluate_ct2", "--model", str(tmp_path), "--set", "x", "--fallback"]

    monkeypatch.setattr(sys, "argv", [*argv, "--name", "n"])
    evaluate_ct2.main()

    summary = json.loads((tmp_path / "n" / "x.json").read_text(encoding="utf-8"))["summary"]
    assert events == [("load",)] and summary["seed"] is None  # 기본: 시드 없음(기록된 측정과 같다)
    assert (summary["model_sha256"], summary["config_sha256"]) == (
        expected["model_sha256"],
        expected["config_sha256"],
    )
    assert summary["decoded_again"] == 1

    events.clear()
    monkeypatch.setattr(sys, "argv", [*argv, "--name", "m", "--seed", "7"])
    evaluate_ct2.main()

    summary = json.loads((tmp_path / "m" / "x.json").read_text(encoding="utf-8"))["summary"]
    assert events == [("seed", 7), ("load",)]  # 모델을 띄우기 전에 한 번만(발화마다 다시 걸지 않는다)
    assert summary["seed"] == 7


# --seed는 CTranslate2가 받는 unsigned int 범위에서 '시드 없음' 표시(4294967295 = 2**32 - 1)를 뺀
# 0~4294967294만 받는다. 음수와 2**32 이상은 ctranslate2.set_random_seed가 TypeError를 내고, 4294967295는
# 받아도 시드가 걸리지 않는다(CTranslate2 4.8.2 src/random.cc의 default_seed. 요약에는 시드가 적히는데 표본은
# 실행마다 달라진다). 그래서 시드를 걸거나 모델을 띄우기 전에 거부하고, 경계값 0과 4294967294는 그대로 넘긴다.
@pytest.mark.parametrize(
    ("seed", "applied"),
    [("-1", None), ("0", 0), ("4294967294", 4294967294), ("4294967295", None), ("4294967296", None)],
)
def test_evaluate_ct2_refuses_a_seed_ctranslate2_would_not_apply(
    tmp_path, monkeypatch, capsys, seed, applied
):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2

    class Loaded(Exception):
        """인자 검사와 시드 걸기를 지나 모델을 띄우려 한 순간(여기서 멈춘다)."""

    def loading(*args, **kwargs):
        raise Loaded

    seeded: list[int] = []
    monkeypatch.setattr(evaluate_ct2.ctranslate2, "set_random_seed", seeded.append)
    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", loading)
    argv = ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x", "--seed", seed]
    monkeypatch.setattr(sys, "argv", argv)

    if applied is None:
        with pytest.raises(SystemExit):
            evaluate_ct2.main()
        assert seeded == []
        assert "--seed must be between 0 and 4294967294" in capsys.readouterr().err
    else:
        with pytest.raises(Loaded):
            evaluate_ct2.main()
        assert seeded == [applied]


# merge_adapter는 기본 모델이 등록한 Hub ID면 모델·처리기 from_pretrained에 고정한 revision을 넘기고(로컬
# 폴더면 넘기지 않음), 합친 폴더에 faster-whisper용 128개 설정을 쓴다. 가중치는 받지 않는다: 모델·PEFT·처리기
# 불러오기를 가짜로 바꾸고 설정 쓰기만 실제로 한다.
@pytest.mark.parametrize(
    ("model", "pinned"),
    [
        ("openai/whisper-large-v3-turbo", {"revision": WHISPER_REVISIONS["openai/whisper-large-v3-turbo"]}),
        ("outputs/turbo-n/base", {}),
    ],
)
def test_merge_adapter_pins_the_registered_revision_of_the_base_model(tmp_path, monkeypatch, model, pinned):
    from whisper_ko_ft import merge_adapter

    seen = {}
    output = tmp_path / "merged"
    merged = SimpleNamespace(
        config=SimpleNamespace(num_mel_bins=TURBO_MEL_BINS),
        save_pretrained=lambda path: Path(path).mkdir(parents=True),
    )
    processor = SimpleNamespace(
        feature_extractor=WhisperFeatureExtractor(feature_size=TURBO_MEL_BINS),
        save_pretrained=lambda path: None,
    )

    def base(path, **kwargs):
        seen["base"] = (path, kwargs)
        return "base"

    def adapter(base_model, path, **kwargs):
        seen["adapter"] = (base_model, path)
        return SimpleNamespace(merge_and_unload=lambda: merged)

    def processor_loader(path, **kwargs):
        seen["processor"] = (path, kwargs)
        return processor

    monkeypatch.setattr(merge_adapter.WhisperForConditionalGeneration, "from_pretrained", base)
    monkeypatch.setattr(merge_adapter.PeftModel, "from_pretrained", adapter)
    monkeypatch.setattr(merge_adapter.WhisperProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(
        sys, "argv", ["merge_adapter", "--model", model, "--adapter", "a", "--output", str(output)]
    )

    merge_adapter.main()

    assert seen["base"] == (model, {"dtype": torch.float16, **pinned})
    assert seen["adapter"] == ("base", "a")
    assert seen["processor"] == (model, pinned)
    saved = json.loads((output / PREPROCESSOR_CONFIG).read_text(encoding="utf-8"))
    assert saved["feature_size"] == TURBO_MEL_BINS
