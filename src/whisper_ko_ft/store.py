"""Audio stores: one int16 file per source split plus a JSON index, read through a memmap.

A store keeps 16 kHz mono samples back to back. Random access is a slice of the memmap, which is what
training needs; reading rows out of parquet at random would decode a whole row group each time.
"""

# [파일 역할] prepare.py가 만든 오디오 저장소(int16 원시 파일)와 묶음 목록(JSON)을 읽는 쪽이다. 학습(train.py,
# train_qwen.py는 train.UtteranceDataset을 거쳐)과 평가(evaluate.py, evaluate_ct2.py, evaluate_qwen.py)가
# 발화의 음성을 여기서 꺼낸다.
# [설계 이유] 데이터는 parquet로 받지만 학습은 발화를 무작위 순서로 읽는다. parquet에서 한 행을 꺼낼 때마다 행
# 묶음 전체를 풀어야 하므로(위 docstring), 미리 16 kHz 모노 int16으로 한 파일에 이어 붙여 두고 memmap으로
# 필요한 구간만 잘라 읽는다. int16은 float32의 절반 크기다.
# [연결] 발화 하나는 Utterance(어느 저장소의 몇 번째 표본부터 몇 개)로 가리킨다. 묶음 목록은
# data/cache/sets/<묶음>.json이고 load_set이 읽는다.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# numpy: int16 원시 파일을 memmap으로 열고, 잘라 낸 구간을 float32로 바꾼다.
import numpy as np

# CACHE: 오디오 저장소와 묶음 목록이 있는 data/cache. 테스트는 root 인자로 임시 폴더를 넘긴다.
from whisper_ko_ft.paths import CACHE

# SAMPLE_RATE: 모든 음성의 표본화율 16 kHz. Whisper 특징 추출기의 입력 규격이라 prepare가 모든 음성을 이
# 값으로 저장하고, channel·evaluate·evaluate_qwen·train·train_qwen이 이 상수를 가져다 쓴다.
SAMPLE_RATE = 16_000


# 발화 하나를 가리키는 레코드. 음성 자체는 담지 않고 '어느 저장소(store)의 몇 번째 표본(offset)부터 몇
# 개(length)'만 가진다. 묶음 JSON의 항목 키와 필드 이름이 같아 Utterance(**entry)로 바로 만든다. frozen=True라
# 학습·평가 중에 값이 바뀌지 않는다.
# id: 발화 ID(FLEURS는 같은 문장을 여러 사람이 읽어 '#2' 같은 번호가 붙을 수 있다). speaker: 화자 ID(FLEURS는
# 빈 문자열). text: 정답 문장.
@dataclass(frozen=True)
class Utterance:
    id: str
    speaker: str
    text: str
    store: str
    offset: int
    length: int

    # 음성 길이(초). 학습 데이터 시간 합계(run.json)와 리포트의 발화별 seconds에 쓴다.
    @property
    def seconds(self) -> float:
        return self.length / SAMPLE_RATE


# prepare가 만든 <이름>.int16 파일 하나를 읽기 전용 memmap으로 연다. 큰 파일 전체를 메모리에 올리지 않고, 읽는
# 구간만 운영체제가 디스크에서 가져온다.
# 호출 위치: train.UtteranceDataset(저장소 이름마다 하나씩 열어 둔다), evaluate·evaluate_ct2·evaluate_qwen의
# main.
class AudioStore:
    def __init__(self, name: str, root: Path = CACHE) -> None:
        self.name = name
        self._samples = np.memmap(root / f"{name}.int16", dtype=np.int16, mode="r")

    def read(self, utterance: Utterance) -> np.ndarray:
        """Float32 samples in [-1, 1]."""
        # offset부터 length개 표본을 잘라 float32로 바꾸고 32768로 나눠 -1~1 범위로 맞춘다(prepare.decode가
        # 곱한 값의 역). 슬라이스는 복사 없이 memmap을 가리키고, astype이 그 구간만 읽어 새 배열을 만든다.
        chunk = self._samples[utterance.offset : utterance.offset + utterance.length]
        return chunk.astype(np.float32) / 32768.0


# data/cache/sets/<name>.json을 읽어 (언어 코드, Utterance 목록)을 돌려준다. 언어('ko' 또는 'en')는 디코딩할
# 언어와 채점 방식(CER 또는 WER)을 정하는 데 쓰인다.
# 호출 위치: evaluate, evaluate_ct2, evaluate_qwen, train, train_qwen의 main.
def load_set(name: str, root: Path = CACHE) -> tuple[str, list[Utterance]]:
    """Language and utterances of a set written by `whisper_ko_ft.prepare`."""
    data = json.loads((root / "sets" / f"{name}.json").read_text(encoding="utf-8"))
    return data["language"], [Utterance(**entry) for entry in data["utterances"]]
