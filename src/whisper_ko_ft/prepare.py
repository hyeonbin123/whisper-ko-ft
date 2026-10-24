"""내려받은 parquet 파일을 오디오 저장소와 평가·학습 묶음으로 바꾼다.

사용법:
    uv run python -m whisper_ko_ft.prepare
    uv run python -m whisper_ko_ft.prepare --verify-hash           # data/raw를 manifest와 SHA-256까지 견준다
    uv run python -m whisper_ko_ft.prepare --rebuild [저장소 ...]  # 색인이 있어도 저장소를 다시 만든다
    uv run python -m whisper_ko_ft.prepare --allow-different-data  # 기록과 달라도 경고만 찍고 계속한다

data/cache/<저장소>.int16 + <저장소>.json과 data/cache/sets/<묶음>.json을 쓴다:
    zeroth-train    validation 화자를 뺀 Zeroth-Korean train
    zeroth-val      validation 화자 10명의 발화 전부(docs/experiments.md "데이터 분리")
    zeroth-val500   validation 화자마다 50발화. 학습 중 체크포인트를 고를 때 쓴다
    zeroth-test, fleurs-ko-val, fleurs-ko-test, fleurs-en-val, fleurs-en-test

시작할 때 data/raw를 원본 manifest(src/whisper_ko_ft/data_manifest.json)와 견주고, 저장소 색인 옆의 원본
기록(<저장소>.sources.json)이 지금 parquet과 다르면 멈추며(--rebuild로만 다시 만든다), 만든 묶음의 발화 수와
validation 화자가 기록(docs/experiments.md 1단계 결과)과 다르면 멈춘다. 기록된 측정과 다른 데이터로 만든
묶음이 조용히 쓰이지 않게 하려는 것이다.
"""

# [파일 역할] 0단계(기반)의 두 번째 단계. download가 받은 parquet을 읽어 (1) 원천 split마다 16 kHz int16
# 오디오 저장소(data/cache/<저장소>.int16 + 색인 <저장소>.json)를 만들고 (2) 학습·validation·test 묶음
# 목록(data/cache/sets/<묶음>.json)을 쓴다.
# [설계 이유] 학습은 발화를 무작위 순서로 읽는데 parquet에서 한 행을 꺼낼 때마다 행 묶음 전체를 풀어야 한다.
# 그래서 한 번만 디코딩해 이어 붙인 int16 파일로 바꿔 두고, store.AudioStore가 memmap으로 잘라 읽게 한다.
# 오디오는 datasets 라이브러리(Windows에서 torchcodec 필요)가 아니라 soundfile로 직접 디코딩한다.
# [화자 기준 분리] Zeroth train에서 splits.validation_speakers가 고른 10명을 validation으로 떼고, 그 화자의
# 발화는 학습 묶음에서 뺀다. 학습 화자와 validation 화자가 겹치지 않게 하려는 것이다(docs/experiments.md
# '데이터 분리').
# [재현] 무작위 요소가 없어, 새 클론에서 다시 돌렸을 때 묶음 목록과 오디오 저장소가 바이트 단위로
# 같았다(docs/experiments.md '새 클론에서 재현').
# [입력 검사, 2026-10-24] 그 전에는 parquet이 없거나 일부만 있는 폴더로도 빈·부분 저장소를 만들어 색인으로
# 영구 캐시했고(색인이 있으면 원본을 다시 보지 않으므로 나중에 받아도 그대로였다), 일부 파일만 있으면 그 안의
# 화자 중에서 validation 10명을 다시 골라 기록과 다른 분리를 조용히 쓸 수 있었다. 이제 세 겹으로 막는다:
# (1) data/raw를 커밋된 manifest와 견준다(기본 크기, --verify-hash면 SHA-256). (2) 저장소마다 색인을 만든
# parquet 목록(이름·크기)을 <저장소>.sources.json에 남기고, 색인이 있어도 지금 parquet과 다르면 멈춘다. 기록이
# 없는 옛 색인은 원본이 manifest와 같고 묶음 수가 기록과 같을 때만 기록을 써서 이어 쓴다(약 6.5 GB를 다시
# 만들지 않는다). (3) 묶음의 발화 수와 validation 화자를 EXPECTED_*와 견준다. --allow-different-data는
# (1)·(3)을 경고로 바꾸고, (2)는 --rebuild로만 푼다.

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

# numpy: 표본 배열 계산과 int16 변환. pyarrow.parquet: parquet을 필요한 열만, 작은 배치 단위로 읽는다.
# soundfile: parquet 안에 파일 형식 그대로 든 오디오 바이트를 디코딩한다. resample_poly: 16 kHz가 아닌 음성이
# 있으면 정수비 다항 리샘플링으로 16 kHz에 맞춘다.
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

# download: manifest 읽기·견주기(check_raw)와 parquet 파일 순서(parquet_files)를 받는 쪽과 같은 함수로 한다.
# paths: 원본(RAW)·결과(CACHE)·manifest(MANIFEST) 위치. splits: 화자 기준 분리 규칙. store.SAMPLE_RATE: 저장할
# 표본화율 16 kHz.
from whisper_ko_ft.download import check_raw, load_manifest, parquet_files
from whisper_ko_ft.paths import CACHE, MANIFEST, RAW
from whisper_ko_ft.splits import val500_ids, validation_speakers
from whisper_ko_ft.store import SAMPLE_RATE

# STORES: 만들 오디오 저장소 6개와 각각의 원본 폴더, 정답 열 이름, 화자 열 이름. 두 데이터셋의 열 이름이
# 달라(Zeroth는 text·speaker_id, FLEURS는 transcription) 여기서 맞춘다. FLEURS는 화자 열을 쓰지 않아 None이다.
# 형식: 저장소 이름 -> (parquet 폴더, 정답 열, 화자 열 또는 None)
STORES = {
    "zeroth_train": (RAW / "zeroth" / "default" / "train", "text", "speaker_id"),
    "zeroth_test": (RAW / "zeroth" / "default" / "test", "text", "speaker_id"),
    "fleurs_ko_validation": (RAW / "fleurs" / "ko_kr" / "validation", "transcription", None),
    "fleurs_ko_test": (RAW / "fleurs" / "ko_kr" / "test", "transcription", None),
    "fleurs_en_validation": (RAW / "fleurs" / "en_us" / "validation", "transcription", None),
    "fleurs_en_test": (RAW / "fleurs" / "en_us" / "test", "transcription", None),
}
# EXPECTED_UTTERANCES: 묶음마다 기록된 발화 수. docs/experiments.md 1단계 결과의 분리 수치(학습 20,049발화,
# Zeroth validation 2,214, val-500 440, Zeroth test 457)와 FLEURS 묶음의 발화 수(결과 표의 발화 수 열)다.
# 2026-10-02 새 클론 재현에서도 같았다. 다른 수가 나오면 원본이나 분리 규칙이 바뀐 것이므로 main이 멈춘다.
EXPECTED_UTTERANCES = {
    "zeroth-train": 20_049,
    "zeroth-val": 2_214,
    "zeroth-val500": 440,
    "zeroth-test": 457,
    "fleurs-ko-val": 226,
    "fleurs-ko-test": 382,
    "fleurs-en-val": 394,
    "fleurs-en-test": 647,
}
# EXPECTED_VALIDATION_SPEAKERS: 기록된 validation 화자 10명(화자 ID 문자열, 정렬 순. docs/experiments.md 1단계
# 결과). 일부 parquet만으로 분리하면 SHA-256 순서가 바뀌어 다른 화자가 뽑히므로 수가 아니라 목록을 견준다.
EXPECTED_VALIDATION_SPEAKERS = ["109", "125", "135", "146", "152", "156", "178", "211", "216", "217"]
# SOURCES_SUFFIX: 색인 옆 원본 기록 파일의 꼬리. data/cache/<저장소>.sources.json에 그 색인을 만든 parquet
# 파일의 이름과 바이트 크기가 든다.
SOURCES_SUFFIX = ".sources.json"


def decode(audio_bytes: bytes) -> np.ndarray:
    """parquet에 든 오디오 바이트 하나를 16 kHz 모노 int16 표본으로 바꾼다. 호출 위치: build_store."""
    # always_2d=True로 받아 모노·스테레오를 같은 모양(표본 수 × 채널 수)으로 다루고, 채널 평균으로 모노를
    # 만든다.
    samples, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    # 16 kHz가 아니면 리샘플한다. Whisper 특징 추출기가 16 kHz 입력을 전제로 한다.
    if rate != SAMPLE_RATE:
        mono = resample_poly(mono, SAMPLE_RATE, rate).astype(np.float32)
    # -1~1 실수를 32768배해 int16 범위로 바꾼다. 반올림 뒤 범위를 넘는 값(예: +1.0 → 32768)은 잘라 넘침으로
    # 부호가 뒤집히지 않게 한다. store.AudioStore.read가 32768로 나눠 되돌린다.
    return np.clip(np.round(mono * 32768.0), -32768, 32767).astype(np.int16)


# 원본 기록 한 줄씩: parquet 파일 이름과 바이트 크기. SHA-256은 넣지 않는다(HDD에서 약 4 GB를 prepare마다 다시
# 해시하지 않으려고. 내용 확인은 manifest 쪽의 --verify-hash가 맡는다).
def source_record(files: Sequence[Path]) -> list[dict]:
    """parquet 파일 목록을 원본 기록 항목 [{file, bytes}, …]으로."""
    return [{"file": path.name, "bytes": path.stat().st_size} for path in files]


# 저장소의 원본 기록을 읽는다. 2026-10-24 전에 만든 캐시에는 이 파일이 없어 None이다(main의 이행 규칙 참고).
def read_sources(name: str) -> list[dict] | None:
    """<name>.sources.json의 sources 목록. 파일이 없으면 None."""
    path = CACHE / f"{name}{SOURCES_SUFFIX}"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["sources"]


# 저장소의 원본 기록을 쓴다. 호출 위치: build_store(만든 직후), main(기록 없는 옛 색인을 이어 쓸 때).
def write_sources(name: str, files: Sequence[Path]) -> None:
    """<name>.sources.json에 지금 parquet 파일 목록(이름·크기)을 적는다."""
    path = CACHE / f"{name}{SOURCES_SUFFIX}"
    path.write_text(json.dumps({"sources": source_record(files)}, indent=1), encoding="utf-8", newline="\n")


# 원본 기록과 지금 parquet 목록의 차이를 파일 이름순으로 적는다. 색인을 만든 뒤 원본이 바뀌었는지(다시 변환,
# 파일 추가·삭제) 가리는 데 쓴다. 호출 위치: main.
def source_differences(recorded: Sequence[dict], current: Sequence[dict]) -> list[str]:
    """차이 목록: 'N bytes, recorded M'(크기), 'missing'(기록에만 있음), 'not in the record'(지금만 있음)."""
    before = {entry["file"]: entry["bytes"] for entry in recorded}
    now = {entry["file"]: entry["bytes"] for entry in current}
    differences: list[str] = []
    for file in sorted(set(before) | set(now)):
        if file not in now:
            differences.append(f"{file}: missing")
        elif file not in before:
            differences.append(f"{file}: not in the record")
        elif before[file] != now[file]:
            differences.append(f"{file}: {now[file]:,} bytes, recorded {before[file]:,}")
    return differences


# 원본 split 하나를 <name>.int16(표본을 이어 붙인 파일)과 색인 항목 목록으로 바꾼다. 항목은 발화 ID, 화자,
# 정답, 저장소 이름, 시작 표본 위치(offset), 길이(length)를 담는다(store.Utterance와 같은 키).
# 색인 JSON이 이미 있으면(rebuild=False) 다시 디코딩하지 않고 그것을 읽어 돌려준다. 색인이 지금 parquet과
# 맞는지는 main이 원본 기록으로 확인한다. parquet이 하나도 없는 폴더(없는 폴더, 빈 폴더)는 빈 저장소를
# 만들지 않고 멈춘다. 2026-10-24 전에는 0바이트 저장소와 '[]' 색인을 써서 영구 캐시했다.
def build_store(
    name: str, folder: Path, text_column: str, speaker_column: str | None, rebuild: bool = False
) -> list[dict]:
    """<name>.int16과 원본 기록을 쓰고 색인 항목을 돌려준다. 색인이 있으면(rebuild=False) 그것을 읽는다."""
    index_path = CACHE / f"{name}.json"
    if index_path.exists() and not rebuild:
        return json.loads(index_path.read_text(encoding="utf-8"))

    files = parquet_files(folder)
    if not files:
        raise SystemExit(
            f"{name}: no parquet files in {folder}; run `python -m whisper_ko_ft.download` first"
        )
    CACHE.mkdir(parents=True, exist_ok=True)
    # 필요한 열(id, audio, 정답, 화자)만 읽는다.
    columns = ["id", "audio", text_column] + ([speaker_column] if speaker_column else [])
    entries: list[dict] = []
    # seen: 원본 ID가 몇 번 나왔는지 센다. 아래에서 중복 ID에 번호를 붙이는 데 쓴다.
    # offset: 지금까지 쓴 표본 수, 곧 다음 발화가 int16 파일에서 시작하는 위치다.
    seen: dict[str, int] = {}
    offset = 0
    # .part 파일에 쓰고 끝까지 쓴 뒤에만 .int16 이름으로 바꾼다. 색인 JSON은 그 뒤에 쓰므로, 색인이 있으면
    # 오디오 파일도 완성된 것이다(위 '이미 있으면 건너뛰기'가 안전한 이유).
    partial = CACHE / f"{name}.int16.part"
    with partial.open("wb") as out:
        # download가 붙인 파일 번호(0, 1, 2 …) 순서대로 읽는다(parquet_files). iter_batches로 64행씩 읽어 큰
        # parquet도 한꺼번에 메모리에 올리지 않는다.
        for parquet in files:
            reader = pq.ParquetFile(parquet)
            for batch in reader.iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    samples = decode(row["audio"]["bytes"])
                    out.write(samples.tobytes())
                    # 같은 ID가 다시 나오면 두 번째부터 '<ID>#2', '#3' …을 붙여 발화마다 고유한 ID를 만든다.
                    # 1단계에서 표를 만드는 코드가 ID로 발화를 찾다가 FLEURS 행 일부가 빠진 오류가 있어 이렇게
                    # 고쳤다(docs/experiments.md 1단계 결과). FLEURS의 id는 문장 ID라 여러 화자가 같은 문장을
                    # 읽으면 같은 id가 여러 번 나온다. 번호는 행 순서를 따르므로 기존 리포트의 발화 ID를
                    # 지키려면 규칙을 바꾸지 않는다(행 순서가 바뀌는 재변환은 manifest 검사가 잡는다).
                    source_id = str(row["id"])
                    seen[source_id] = seen.get(source_id, 0) + 1
                    entries.append(
                        {
                            "id": source_id if seen[source_id] == 1 else f"{source_id}#{seen[source_id]}",
                            "speaker": str(row[speaker_column]) if speaker_column else "",
                            "text": row[text_column].strip(),
                            "store": name,
                            "offset": offset,
                            "length": len(samples),
                        }
                    )
                    offset += len(samples)
            print(f"{name}: {parquet.name} done, {len(entries):,} utterances", flush=True)
    # 다 쓴 뒤 .part를 실제 이름으로 바꾸고 그다음 색인을, 마지막에 원본 기록을 쓴다. ensure_ascii=False로
    # 한국어 정답을 그대로 적고, newline='\n'으로 Windows에서도 줄바꿈을 LF로 고정한다.
    partial.replace(CACHE / f"{name}.int16")
    index_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8", newline="\n")
    write_sources(name, files)
    return entries


# 저장소 색인들로 묶음 8개를 만든다(쓰지는 않는다). 돌려주는 것: 묶음 이름 → (언어 코드, 발화 항목 목록)과
# validation 화자 집합. main이 기록과 견준 뒤 write_set으로 쓴다.
def make_sets(stores: Mapping[str, list[dict]]) -> tuple[dict[str, tuple[str, list[dict]]], set[str]]:
    """묶음 이름 → (언어, 발화 목록)과 validation 화자 집합."""
    # Zeroth train에서 validation 화자 10명을 고르고 그 화자의 발화를 모두 validation으로 뗀다. val-500은 그
    # validation에서 화자마다 앞의 50개다.
    train = stores["zeroth_train"]
    held_out = set(validation_speakers(u["speaker"] for u in train))
    validation = [u for u in train if u["speaker"] in held_out]
    picked = set(val500_ids(validation))
    # 학습 묶음은 validation 화자를 뺀 나머지다. zeroth-train, zeroth-val, zeroth-val500은 모두 같은
    # 저장소(zeroth_train)를 가리키는 서로 다른 목록이다. test와 FLEURS 묶음은 원본 split을 그대로 쓴다.
    # FLEURS 한국어는 다른 도메인 확인용, FLEURS 영어는 한국어만 학습했을 때 영어 인식이 얼마나 나빠지는지
    # 확인용이다(docs/plan.md 측정으로 답할 질문).
    sets = {
        "zeroth-train": ("ko", [u for u in train if u["speaker"] not in held_out]),
        "zeroth-val": ("ko", validation),
        "zeroth-val500": ("ko", [u for u in validation if u["id"] in picked]),
        "zeroth-test": ("ko", stores["zeroth_test"]),
        "fleurs-ko-val": ("ko", stores["fleurs_ko_validation"]),
        "fleurs-ko-test": ("ko", stores["fleurs_ko_test"]),
        "fleurs-en-val": ("en", stores["fleurs_en_validation"]),
        "fleurs-en-test": ("en", stores["fleurs_en_test"]),
    }
    return sets, held_out


# 묶음을 기록과 견준다. validation 화자 목록이 다르면 그것을 먼저 적고, 그다음 EXPECTED_UTTERANCES 순서로 발화
# 수가 다른 묶음을 적는다. 모두 같으면 빈 목록이다. sets는 묶음 이름 → 발화 목록(길이만 본다).
def check_sets(sets: Mapping[str, Sequence], held_out: Iterable[str]) -> list[str]:
    """기록(EXPECTED_VALIDATION_SPEAKERS, EXPECTED_UTTERANCES)과 다른 점의 목록. 모두 같으면 빈 목록."""
    problems: list[str] = []
    speakers = sorted(held_out)
    if speakers != EXPECTED_VALIDATION_SPEAKERS:
        problems.append(f"validation speakers {speakers}, recorded {EXPECTED_VALIDATION_SPEAKERS}")
    for name, expected in EXPECTED_UTTERANCES.items():
        count = len(sets[name])
        if count != expected:
            problems.append(f"{name}: {count:,} utterances, recorded {expected:,}")
    return problems


# 묶음 목록 하나를 data/cache/sets/<name>.json에 쓴다. 내용은 언어 코드와 발화 항목 목록이고 store.load_set이
# 읽는다. 오디오는 복사하지 않고 저장소 안의 위치만 가리키므로 묶음을 여러 개 만들어도 디스크가 늘지 않는다.
# 발화 수·시간·화자 수를 찍는다. docs/experiments.md 1단계 결과의 분리 수치(학습 20,049발화 46.38시간 95명
# 등)가 이 출력이다.
def write_set(name: str, language: str, utterances: list[dict]) -> None:
    target = CACHE / "sets" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"language": language, "utterances": utterances}
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    hours = sum(u["length"] for u in utterances) / SAMPLE_RATE / 3600
    speakers = len({u["speaker"] for u in utterances if u["speaker"]})
    print(f"set {name}: {len(utterances):,} utterances, {hours:.2f} h, {speakers} speakers")


# 명령줄 진입점. 흐름: manifest 검사 → 저장소 6개(색인이 있으면 원본 기록과 견주고, 없거나 --rebuild면 만든다)
# → 묶음 8개를 만들어 기록과 견준다 → 옛 색인의 원본 기록 쓰기 → 묶음 쓰기. 멈추는 자리마다 다음에 무엇을
# 줄지 오류 문구에 적는다.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="내려받은 parquet을 16 kHz 오디오 저장소와 학습·validation·test 묶음으로 바꾼다",
        epilog="data/raw는 원본 manifest와, 묶음은 docs/experiments.md 1단계의 기록과 다르면 멈춘다.",
    )
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="data/raw를 manifest와 크기뿐 아니라 SHA-256까지 견준다(약 4 GB를 읽는다)",
    )
    parser.add_argument(
        "--allow-different-data",
        action="store_true",
        help="data/raw나 묶음이 기록과 달라도 경고만 찍고 계속한다(그 묶음은 기록된 묶음이 아니다)",
    )
    parser.add_argument(
        "--rebuild",
        nargs="*",
        choices=list(STORES),
        metavar="STORE",
        help="색인이 있어도 저장소를 parquet에서 다시 만든다. 이름을 주면 그 저장소만, 없으면 6개 모두",
    )
    args = parser.parse_args()

    # (1) data/raw를 manifest와 견준다. 저장소별로 따로 보고 어느 저장소의 원본이 달랐는지를 (2)에서 쓴다.
    manifest = load_manifest(MANIFEST)
    folders = {name: spec[0].relative_to(RAW).as_posix() for name, spec in STORES.items()}
    raw_problems = {
        name: check_raw(manifest, RAW, [folder], hashes=args.verify_hash) for name, folder in folders.items()
    }
    found = [problem for problems in raw_problems.values() for problem in problems]
    if found:
        message = "data/raw differs from the recorded manifest:\n  " + "\n  ".join(found)
        if not args.allow_different_data:
            raise SystemExit(
                message
                + "\nre-download the listed files (`python -m whisper_ko_ft.download --verify` checks them),"
                " or pass --allow-different-data to continue with data that is not the recorded data"
            )
        print(f"warning: {message}", file=sys.stderr)

    # (2) 저장소 6개. 색인이 있으면 원본 기록과 지금 parquet을 견준다. 기록이 없는 옛 색인(2026-10-24 전
    # 캐시)은 그 폴더가 manifest와 같을 때 adopt 후보에 넣고, (3)의 묶음 검사까지 지나면 기록을 써서 이어
    # 쓴다. 폴더가 manifest와 다르면 그 색인이 지금 파일에서 나왔다고 적을 수 없으므로 기록 없이 두고
    # --rebuild를 안내한다.
    stores: dict[str, list[dict]] = {}
    adopt: list[str] = []
    for name, (folder, text_column, speaker_column) in STORES.items():
        rebuild = args.rebuild is not None and (not args.rebuild or name in args.rebuild)
        if not rebuild and (CACHE / f"{name}.json").exists():
            recorded = read_sources(name)
            if recorded is None:
                if raw_problems[name]:
                    print(
                        f"warning: {name}: the index has no source record and {folders[name]} differs from"
                        f" the manifest; pass --rebuild {name} to rebuild it from the current files",
                        file=sys.stderr,
                    )
                else:
                    adopt.append(name)
            else:
                differences = source_differences(recorded, source_record(parquet_files(folder)))
                if differences:
                    raise SystemExit(
                        f"{name}: data/cache/{name}.json was built from other parquet files than"
                        f" {folders[name]} holds now ({'; '.join(differences)}); pass --rebuild {name} to"
                        " rebuild the store from the current files (--rebuild alone rebuilds all six)"
                    )
            print(f"{name}: existing store kept", flush=True)
        stores[name] = build_store(name, folder, text_column, speaker_column, rebuild=rebuild)

    # (3) 묶음을 만들어 기록과 견준다. 쓰기 전에 견주므로 다르면 data/cache/sets에 아무것도 남지 않는다.
    sets, held_out = make_sets(stores)
    print(f"validation speakers: {sorted(held_out)}")
    problems = check_sets({name: utterances for name, (_, utterances) in sets.items()}, held_out)
    if problems:
        message = "the sets differ from the recorded ones:\n  " + "\n  ".join(problems)
        if not args.allow_different_data:
            raise SystemExit(
                message
                + "\nthe data or the split rule is not the recorded one; pass --allow-different-data to write"
                " these sets anyway (they are not the recorded sets)"
            )
        print(f"warning: {message}", file=sys.stderr)
        for name in adopt:
            print(
                f"warning: {name}: the index has no source record and the sets differ from the record, so the"
                f" record is not written; pass --rebuild {name} to rebuild the store",
                file=sys.stderr,
            )
    else:
        # 기록 없는 옛 색인이 지금 원본(manifest와 같음)에서 나왔다는 두 근거(원본 일치, 묶음 수·화자
        # 일치)가 모두 섰으므로 원본 기록을 써서 다음부터는 바로 견준다.
        for name in adopt:
            write_sources(name, parquet_files(STORES[name][0]))
            print(f"{name}: source record written for the existing index", flush=True)
    for name, (language, utterances) in sets.items():
        write_set(name, language, utterances)


if __name__ == "__main__":
    main()
