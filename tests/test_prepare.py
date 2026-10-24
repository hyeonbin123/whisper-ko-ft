# [테스트 묶음] prepare.py의 입력 검사를 작은 parquet 픽스처로 확인한다: parquet이 없는 폴더 거부, 원본
# manifest(크기·SHA-256)와 다른 data/raw 거부, 색인 옆 원본 기록(<저장소>.sources.json)과 다른 parquet로 만든
# 색인 거부(--rebuild로만 다시 만듦), 기록이 없는 옛 색인의 이행, 기대 발화 수·validation 화자와 다르면
# 멈춤(--allow-different-data면 경고만), 중복 ID '#n' 규칙 보존.
# [왜 중요한가] 예전 prepare는 parquet이 없거나 일부만 있어도 빈·부분 저장소를 만들어 색인으로 영구 캐시했고,
# 그 색인에서 고른 validation 화자가 기록과 달라도 알리지 않았다. 내려받기가 끊긴 재현자가 다른 데이터로
# 측정하고도 오류를 보지 못하는 길이라, 기록된 묶음(docs/experiments.md 1단계 결과)과 다르면 멈추게 고정한다.
# [방법] soundfile로 만든 0.1초 WAV를 pyarrow parquet(id·audio·정답·화자 열)에 넣어 임시 data/raw를 만들고,
# prepare의 RAW·CACHE·STORES·MANIFEST·기대값을 임시 값으로 바꿔 main을 돌린다. 실제 데이터와 GPU는 쓰지
# 않는다.

import hashlib
import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from whisper_ko_ft import prepare
from whisper_ko_ft.splits import validation_speakers
from whisper_ko_ft.store import SAMPLE_RATE

# SAMPLES: 발화 하나의 길이 0.1초 = 1,600표본. 저장소 .int16 크기와 offset·length 확인에 쓴다.
SAMPLES = SAMPLE_RATE // 10
# TRAIN_SPEAKERS: 픽스처 Zeroth train의 화자 12명. validation_speakers가 그중 10명을 고르므로 학습 묶음에는
# 2명이 남는다(화자가 10명 이하면 학습 묶음이 비어 버린다).
TRAIN_SPEAKERS = [str(100 + n) for n in range(12)]
# LAYOUT: 저장소 이름 → (data/raw 아래 상대 폴더, 정답 열, 화자 열). prepare.STORES와 같은 꼴이고 폴더만
# 임시 폴더 아래로 바꾼다.
LAYOUT = {
    "zeroth_train": ("zeroth/default/train", "text", "speaker_id"),
    "zeroth_test": ("zeroth/default/test", "text", "speaker_id"),
    "fleurs_ko_validation": ("fleurs/ko_kr/validation", "transcription", None),
    "fleurs_ko_test": ("fleurs/ko_kr/test", "transcription", None),
    "fleurs_en_validation": ("fleurs/en_us/validation", "transcription", None),
    "fleurs_en_test": ("fleurs/en_us/test", "transcription", None),
}


def wav_bytes(seed: int) -> bytes:
    """0.1초 16 kHz 모노 WAV(PCM 16비트) 바이트. seed마다 진폭이 달라 발화마다 다른 소리가 된다."""
    samples = np.linspace(-0.5, 0.5, SAMPLES, dtype=np.float32) * ((seed % 7) + 1) / 8
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_parquet(path, rows, text_column, speaker_column=None):
    """Hugging Face parquet 변환본과 같은 열(id, audio{bytes, path}, 정답, 화자)의 parquet을 쓴다.

    rows는 (id, 정답, 화자) 튜플 목록이다. speaker_column이 None이면(FLEURS) 화자 열을 만들지 않는다.
    """
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    audio = [{"bytes": wav_bytes(n), "path": f"{n}.wav"} for n in range(len(rows))]
    columns = {
        "id": pa.array([row[0] for row in rows]),
        "audio": pa.array(audio, type=audio_type),
        text_column: pa.array([row[1] for row in rows]),
    }
    if speaker_column:
        columns[speaker_column] = pa.array([row[2] for row in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def write_manifest(path, raw):
    """raw 아래 모든 parquet의 경로·크기·SHA-256을 manifest로 쓴다(download --write-manifest가 쓰는 꼴)."""
    files = {
        p.relative_to(raw).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(raw.rglob("*.parquet"))
    }
    payload = {"created": "2026-10-24", "files": files, "upstream": {}}
    path.write_text(json.dumps(payload), encoding="utf-8")


# 임시 data/raw(저장소 6개분 parquet 7개)와 그에 맞는 manifest를 만들고, prepare가 보는 경로·기대값을 모두
# 임시 값으로 바꾼다. 기대값은 픽스처 그대로(학습 2, validation 10, val-500 10, test 2, FLEURS 2·1·1·1)다.
@pytest.fixture
def data(tmp_path, monkeypatch):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    train = [(f"{speaker}_001", f"문장 {speaker}", int(speaker)) for speaker in TRAIN_SPEAKERS]
    # train은 파일 두 개로 나눠 파일 번호순으로 이어 읽는지도 본다.
    write_parquet(raw / "zeroth/default/train/0.parquet", train[:6], "text", "speaker_id")
    write_parquet(raw / "zeroth/default/train/1.parquet", train[6:], "text", "speaker_id")
    write_parquet(
        raw / "zeroth/default/test/0.parquet", [("t1", "하나", 900), ("t2", "둘", 901)], "text", "speaker_id"
    )
    # FLEURS의 id는 문장 번호라 같은 문장을 읽은 두 녹음이 같은 id 7을 가진다.
    write_parquet(
        raw / "fleurs/ko_kr/validation/0.parquet",
        [(7, "같은 문장", None), (7, "같은 문장", None)],
        "transcription",
    )
    write_parquet(raw / "fleurs/ko_kr/test/0.parquet", [(8, "다른 문장", None)], "transcription")
    write_parquet(raw / "fleurs/en_us/validation/0.parquet", [(9, "hello there", None)], "transcription")
    write_parquet(raw / "fleurs/en_us/test/0.parquet", [(10, "good bye", None)], "transcription")
    manifest = tmp_path / "data_manifest.json"
    write_manifest(manifest, raw)

    held_out = sorted(validation_speakers(TRAIN_SPEAKERS))
    expected = {
        "zeroth-train": 2,
        "zeroth-val": 10,
        "zeroth-val500": 10,
        "zeroth-test": 2,
        "fleurs-ko-val": 2,
        "fleurs-ko-test": 1,
        "fleurs-en-val": 1,
        "fleurs-en-test": 1,
    }
    stores = {name: (raw / folder, text, speaker) for name, (folder, text, speaker) in LAYOUT.items()}
    monkeypatch.setattr(prepare, "RAW", raw)
    monkeypatch.setattr(prepare, "CACHE", cache)
    monkeypatch.setattr(prepare, "MANIFEST", manifest)
    monkeypatch.setattr(prepare, "STORES", stores)
    monkeypatch.setattr(prepare, "EXPECTED_UTTERANCES", expected)
    monkeypatch.setattr(prepare, "EXPECTED_VALIDATION_SPEAKERS", held_out)
    return SimpleNamespace(raw=raw, cache=cache, manifest=manifest, expected=expected, held_out=held_out)


# 명령줄 인자를 바꿔 prepare.main을 한 번 부른다.
def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prepare", *args])
    prepare.main()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# 저장소를 다시 만들면 안 되는 자리에서 decode 대신 꽂아, 디코딩이 한 번이라도 불리면 실패하게 한다.
def fail_decode(audio_bytes):
    raise AssertionError("decode must not run: the existing store should be kept")


# 처음 실행: 저장소 6개와 묶음 8개가 생기고, 묶음 수는 기대값 그대로이며, 색인 옆에 원본 기록(파일
# 이름·크기)이 남는다. 두 파일로 나뉜 train은 파일 번호순으로 이어진다.
def test_first_run_builds_stores_sets_and_source_records(data, monkeypatch):
    run(monkeypatch)

    for name, (folder, _, _) in prepare.STORES.items():
        assert (data.cache / f"{name}.int16").exists() and (data.cache / f"{name}.json").exists()
        files = sorted(folder.glob("*.parquet"), key=lambda p: int(p.stem))
        sources = read(data.cache / f"{name}.sources.json")["sources"]
        assert sources == [{"file": p.name, "bytes": p.stat().st_size} for p in files]
    for name, count in data.expected.items():
        assert len(read(data.cache / "sets" / f"{name}.json")["utterances"]) == count
    assert read(data.cache / "sets" / "fleurs-en-val.json")["language"] == "en"
    train = read(data.cache / "zeroth_train.json")
    assert [u["speaker"] for u in train] == TRAIN_SPEAKERS
    assert (data.cache / "zeroth_train.int16").stat().st_size == 12 * SAMPLES * 2
    validation = read(data.cache / "sets" / "zeroth-val.json")["utterances"]
    assert sorted({u["speaker"] for u in validation}) == data.held_out


# parquet이 하나도 없는 폴더(없는 폴더, 빈 폴더)는 빈 색인과 0바이트 저장소를 만드는 대신 멈춘다.
def test_a_store_without_parquet_files_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "CACHE", tmp_path / "cache")
    with pytest.raises(SystemExit, match="zeroth_test.*no parquet"):
        prepare.build_store("zeroth_test", tmp_path / "missing", "text", "speaker_id")
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no parquet"):
        prepare.build_store("zeroth_test", tmp_path / "empty", "text", "speaker_id")
    assert not (tmp_path / "cache").exists()


# 받지 않은 데이터셋의 폴더가 비어 있고 manifest도 그 파일을 모르면(manifest 검사는 지나감) 저장소를 만드는
# 자리에서 멈추고, 그 저장소의 색인을 남기지 않는다.
def test_prepare_stops_at_an_empty_store_folder(data, monkeypatch):
    for path in (data.raw / "fleurs/en_us/test").glob("*.parquet"):
        path.unlink()
    write_manifest(data.manifest, data.raw)

    with pytest.raises(SystemExit, match="fleurs_en_test.*no parquet"):
        run(monkeypatch)
    assert not (data.cache / "fleurs_en_test.json").exists()


# data/raw가 manifest와 다르면 멈춘다: 크기가 다른 파일, 없는 파일, manifest에 없는 파일. 크기가 같고 내용만
# 다른 파일은 기본 검사(크기)를 지나가고 --verify-hash(SHA-256)가 잡는다.
def test_raw_files_that_differ_from_the_manifest_stop_prepare(data, monkeypatch):
    run(monkeypatch)  # 색인이 생긴 뒤의 실행은 parquet을 읽지 않으므로 검사만 본다
    path = data.raw / "fleurs/en_us/test/0.parquet"
    original = path.read_bytes()

    path.write_bytes(original + b"\0")
    with pytest.raises(SystemExit) as info:
        run(monkeypatch)
    assert "fleurs/en_us/test/0.parquet" in str(info.value) and "--allow-different-data" in str(info.value)

    path.write_bytes(b"\0" + original[1:])
    run(monkeypatch)
    with pytest.raises(SystemExit, match="SHA-256"):
        run(monkeypatch, "--verify-hash")

    path.unlink()
    with pytest.raises(SystemExit, match="missing"):
        run(monkeypatch)

    path.write_bytes(original)
    (path.parent / "1.parquet").write_bytes(original)
    with pytest.raises(SystemExit, match="not in the manifest"):
        run(monkeypatch)


# 원본이 다시 변환되어(test parquet이 발화 3개로 바뀌고 manifest도 그에 맞게 다시 만듦) 색인 옆 원본 기록과
# 지금 파일이 다르면 옛 색인을 쓰지 않고 멈춘다. --rebuild면 다시 만들고 원본 기록도 새로 쓴다.
def test_an_index_built_from_other_files_is_refused_until_rebuild(data, monkeypatch):
    run(monkeypatch)
    rows = [("t1", "하나", 900), ("t2", "둘", 901), ("t3", "셋", 902)]
    write_parquet(data.raw / "zeroth/default/test/0.parquet", rows, "text", "speaker_id")
    write_manifest(data.manifest, data.raw)
    monkeypatch.setitem(prepare.EXPECTED_UTTERANCES, "zeroth-test", 3)

    with pytest.raises(SystemExit) as info:
        run(monkeypatch)
    assert "zeroth_test" in str(info.value) and "--rebuild" in str(info.value)
    assert [u["id"] for u in read(data.cache / "zeroth_test.json")] == ["t1", "t2"]  # 옛 색인은 그대로

    run(monkeypatch, "--rebuild")
    assert [u["id"] for u in read(data.cache / "zeroth_test.json")] == ["t1", "t2", "t3"]
    sources = read(data.cache / "zeroth_test.sources.json")["sources"]
    assert sources == [
        {"file": "0.parquet", "bytes": (data.raw / "zeroth/default/test/0.parquet").stat().st_size}
    ]
    assert len(read(data.cache / "sets" / "zeroth-test.json")["utterances"]) == 3


# --rebuild에 저장소 이름을 주면 그 저장소만 다시 만들고 나머지는 색인을 그대로 쓴다(디코딩은 그 저장소의 발화
# 수만큼만 불린다). 저장소 하나가 바뀌었을 때 6.5 GB짜리 zeroth_train까지 다시 만들지 않으려는 것이다.
def test_rebuild_with_a_name_rebuilds_that_store_only(data, monkeypatch):
    run(monkeypatch)
    before = {p.name: p.stat().st_mtime_ns for p in data.cache.glob("*.int16")}
    decoded = []
    real_decode = prepare.decode

    def counting_decode(audio_bytes):
        decoded.append(audio_bytes)
        return real_decode(audio_bytes)

    monkeypatch.setattr(prepare, "decode", counting_decode)
    run(monkeypatch, "--rebuild", "zeroth_test")

    assert len(decoded) == 2  # zeroth_test의 발화 2개만 다시 디코딩했다
    after = {p.name: p.stat().st_mtime_ns for p in data.cache.glob("*.int16")}
    assert all(after[name] == before[name] for name in before if name != "zeroth_test.int16")
    assert [u["id"] for u in read(data.cache / "zeroth_test.json")] == ["t1", "t2"]


# 원본 기록이 생기기 전에 만든 캐시(색인은 있고 .sources.json이 없음)는, 원본이 manifest와 같고 묶음 수가
# 기대값과 같으면 저장소를 다시 만들지 않고 원본 기록만 써서 이어 쓴다.
def test_an_old_index_without_a_source_record_is_adopted_when_the_data_matches(data, monkeypatch):
    run(monkeypatch)
    stores = {p.name: p.read_bytes() for p in data.cache.glob("*.int16")}
    for path in data.cache.glob("*.sources.json"):
        path.unlink()
    monkeypatch.setattr(prepare, "decode", fail_decode)

    run(monkeypatch)

    assert sorted(p.name for p in data.cache.glob("*.sources.json")) == sorted(
        f"{n}.sources.json" for n in LAYOUT
    )
    assert {p.name: p.read_bytes() for p in data.cache.glob("*.int16")} == stores


# 원본 기록이 없는 옛 색인이라도 data/raw가 manifest와 다르면 그 색인이 지금 파일에서 나왔다고 적지 않는다.
# --allow-different-data로 계속하더라도 원본 기록은 쓰지 않고 --rebuild를 안내한다.
def test_an_old_index_is_not_adopted_when_the_raw_files_differ(data, monkeypatch, capsys):
    run(monkeypatch)
    (data.cache / "zeroth_test.sources.json").unlink()
    path = data.raw / "zeroth/default/test/0.parquet"
    path.write_bytes(path.read_bytes() + b"\0")
    monkeypatch.setattr(prepare, "decode", fail_decode)

    with pytest.raises(SystemExit, match="--allow-different-data"):
        run(monkeypatch)
    run(monkeypatch, "--allow-different-data")

    assert not (data.cache / "zeroth_test.sources.json").exists()
    err = capsys.readouterr().err
    assert "zeroth_test" in err and "--rebuild" in err


# 묶음 발화 수나 validation 화자가 기록과 다르면 묶음을 쓰기 전에 멈추고 무엇이 다른지 적는다.
# --allow-different-data면 같은 내용을 경고로 적고 묶음을 쓴다.
def test_set_sizes_or_speakers_that_differ_from_the_record_stop_prepare(data, monkeypatch, capsys):
    monkeypatch.setitem(prepare.EXPECTED_UTTERANCES, "zeroth-test", 457)
    monkeypatch.setattr(prepare, "EXPECTED_VALIDATION_SPEAKERS", ["109", "125"])

    with pytest.raises(SystemExit) as info:
        run(monkeypatch)
    message = str(info.value)
    assert "zeroth-test: 2 utterances, recorded 457" in message and "validation speakers" in message
    assert "--allow-different-data" in message
    assert not (data.cache / "sets").exists()

    run(monkeypatch, "--allow-different-data")
    assert (data.cache / "sets" / "zeroth-test.json").exists()
    assert "zeroth-test: 2 utterances, recorded 457" in capsys.readouterr().err


# check_sets는 기록된 수·화자와 모두 같을 때만 비어 있고, 다르면 화자부터 적고 묶음 순서대로 적는다.
def test_check_sets_accepts_the_recorded_sizes_and_speakers_only():
    sets = {name: [None] * count for name, count in prepare.EXPECTED_UTTERANCES.items()}
    assert prepare.check_sets(sets, set(prepare.EXPECTED_VALIDATION_SPEAKERS)) == []

    problems = prepare.check_sets({**sets, "zeroth-val": sets["zeroth-val"][:-1]}, {"1", "2"})
    assert len(problems) == 2
    assert (
        "validation speakers" in problems[0] and "zeroth-val: 2,213 utterances, recorded 2,214" == problems[1]
    )


# 기대값은 docs/experiments.md 1단계 결과의 분리 수치 그대로다(실수로 바뀌지 않게 고정).
def test_the_recorded_sizes_and_speakers_are_the_defaults():
    assert prepare.EXPECTED_VALIDATION_SPEAKERS == [
        "109", "125", "135", "146", "152", "156", "178", "211", "216", "217",
    ]  # fmt: skip
    assert prepare.EXPECTED_UTTERANCES == {
        "zeroth-train": 20_049,
        "zeroth-val": 2_214,
        "zeroth-val500": 440,
        "zeroth-test": 457,
        "fleurs-ko-val": 226,
        "fleurs-ko-test": 382,
        "fleurs-en-val": 394,
        "fleurs-en-test": 647,
    }


# FLEURS처럼 같은 id가 되풀이되면 두 번째부터 '#2', '#3'을 붙이는 규칙은 그대로다(기존 리포트의 발화 ID 보존).
# 화자 열이 없는 저장소의 화자는 빈 문자열이고, offset·length는 표본 수 그대로다.
def test_repeated_source_ids_get_a_number_suffix(data, monkeypatch):
    run(monkeypatch)

    index = read(data.cache / "fleurs_ko_validation.json")
    assert [u["id"] for u in index] == ["7", "7#2"]
    assert [(u["offset"], u["length"]) for u in index] == [(0, SAMPLES), (SAMPLES, SAMPLES)]
    assert [u["speaker"] for u in index] == ["", ""]
    assert (data.cache / "fleurs_ko_validation.int16").stat().st_size == 2 * SAMPLES * 2
