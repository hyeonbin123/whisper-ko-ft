"""Turn the downloaded parquet files into audio stores and the evaluation/training sets.

Usage:
    uv run python -m whisper_ko_ft.prepare

Writes data/cache/<store>.int16 + <store>.json, and data/cache/sets/<set>.json:
    zeroth-train    Zeroth-Korean train without the validation speakers
    zeroth-val      the 10 validation speakers (docs/experiments.md, "데이터 분리")
    zeroth-val500   50 utterances per validation speaker, for picking checkpoints during training
    zeroth-test, fleurs-ko-val, fleurs-ko-test, fleurs-en-val, fleurs-en-test
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

from __future__ import annotations

import io
import json
from pathlib import Path

# numpy: 표본 배열 계산과 int16 변환. pyarrow.parquet: parquet을 필요한 열만, 작은 배치 단위로 읽는다.
# soundfile: parquet 안에 파일 형식 그대로 든 오디오 바이트를 디코딩한다. resample_poly: 16 kHz가 아닌 음성이
# 있으면 정수비 다항 리샘플링으로 16 kHz에 맞춘다.
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

# paths: 원본(RAW)과 결과(CACHE) 위치. splits: 화자 기준 분리 규칙. store.SAMPLE_RATE: 저장할 표본화율 16 kHz.
from whisper_ko_ft.paths import CACHE, RAW
from whisper_ko_ft.splits import val500_ids, validation_speakers
from whisper_ko_ft.store import SAMPLE_RATE

# STORES: 만들 오디오 저장소 6개와 각각의 원본 폴더, 정답 열 이름, 화자 열 이름. 두 데이터셋의 열 이름이
# 달라(Zeroth는 text·speaker_id, FLEURS는 transcription) 여기서 맞춘다. FLEURS는 화자 열을 쓰지 않아 None이다.
# store -> (parquet folder, text column, speaker column or None)
STORES = {
    "zeroth_train": (RAW / "zeroth" / "default" / "train", "text", "speaker_id"),
    "zeroth_test": (RAW / "zeroth" / "default" / "test", "text", "speaker_id"),
    "fleurs_ko_validation": (RAW / "fleurs" / "ko_kr" / "validation", "transcription", None),
    "fleurs_ko_test": (RAW / "fleurs" / "ko_kr" / "test", "transcription", None),
    "fleurs_en_validation": (RAW / "fleurs" / "en_us" / "validation", "transcription", None),
    "fleurs_en_test": (RAW / "fleurs" / "en_us" / "test", "transcription", None),
}


# parquet에 든 오디오 바이트 하나를 16 kHz 모노 int16 표본으로 바꾼다. 호출 위치: build_store.
def decode(audio_bytes: bytes) -> np.ndarray:
    """16 kHz mono int16 samples."""
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


# 원본 split 하나를 <name>.int16(표본을 이어 붙인 파일)과 색인 항목 목록으로 바꾼다. 항목은 발화 ID, 화자,
# 정답, 저장소 이름, 시작 표본 위치(offset), 길이(length)를 담는다(store.Utterance와 같은 키).
# 색인 JSON이 이미 있으면 다시 디코딩하지 않고 그것을 읽어 돌려준다. 저장소를 다시 만들려면 색인 파일을
# 지운다.
def build_store(name: str, folder: Path, text_column: str, speaker_column: str | None) -> list[dict]:
    """Write <name>.int16 and return its index entries. Skips the work if the index already exists."""
    index_path = CACHE / f"{name}.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))

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
        # download가 붙인 파일 번호(0, 1, 2 …) 순서대로 읽는다. 이름을 문자열로 정렬하면 10이 2보다 앞에
        # 오므로 정수로 정렬한다. iter_batches로 64행씩 읽어 큰 parquet도 한꺼번에 메모리에 올리지 않는다.
        for parquet in sorted(folder.glob("*.parquet"), key=lambda p: int(p.stem)):
            reader = pq.ParquetFile(parquet)
            for batch in reader.iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    samples = decode(row["audio"]["bytes"])
                    out.write(samples.tobytes())
                    # 같은 ID가 다시 나오면 두 번째부터 '<ID>#2', '#3' …을 붙여 발화마다 고유한 ID를 만든다.
                    # 1단계에서 표를 만드는 코드가 ID로 발화를 찾다가 FLEURS 행 일부가 빠진 오류가 있어 이렇게
                    # 고쳤다(docs/experiments.md 1단계 결과).
                    # FLEURS ids are sentence ids: several speakers read the same sentence.
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
    # 다 쓴 뒤 .part를 실제 이름으로 바꾸고 그다음 색인을 쓴다. ensure_ascii=False로 한국어 정답을 그대로
    # 적고, newline='\n'으로 Windows에서도 줄바꿈을 LF로 고정한다.
    partial.replace(CACHE / f"{name}.int16")
    index_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8", newline="\n")
    return entries


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


def main() -> None:
    # 저장소 6개를 만들거나(처음 실행) 이미 만든 색인을 읽는다.
    stores = {name: build_store(name, *spec) for name, spec in STORES.items()}

    # Zeroth train에서 validation 화자 10명을 고르고 그 화자의 발화를 모두 validation으로 뗀다. val-500은 그
    # validation에서 화자마다 앞의 50개다.
    train = stores["zeroth_train"]
    held_out = set(validation_speakers(u["speaker"] for u in train))
    validation = [u for u in train if u["speaker"] in held_out]
    picked = set(val500_ids(validation))
    print(f"validation speakers: {sorted(held_out)}")

    # 학습 묶음은 validation 화자를 뺀 나머지다. zeroth-train, zeroth-val, zeroth-val500은 모두 같은
    # 저장소(zeroth_train)를 가리키는 서로 다른 목록이다.
    write_set("zeroth-train", "ko", [u for u in train if u["speaker"] not in held_out])
    write_set("zeroth-val", "ko", validation)
    write_set("zeroth-val500", "ko", [u for u in validation if u["id"] in picked])
    # test와 FLEURS 묶음은 원본 split을 그대로 쓴다. FLEURS 한국어는 다른 도메인 확인용, FLEURS 영어는
    # 한국어만 학습했을 때 영어 인식이 얼마나 나빠지는지 확인용이다(docs/plan.md 측정으로 답할 질문).
    write_set("zeroth-test", "ko", stores["zeroth_test"])
    write_set("fleurs-ko-val", "ko", stores["fleurs_ko_validation"])
    write_set("fleurs-ko-test", "ko", stores["fleurs_ko_test"])
    write_set("fleurs-en-val", "en", stores["fleurs_en_validation"])
    write_set("fleurs-en-test", "en", stores["fleurs_en_test"])


if __name__ == "__main__":
    main()
