"""Zeroth-Korean train을 화자 기준으로 나눠 validation을 만든다(규칙은 docs/experiments.md "데이터 분리")."""

# [파일 역할] Zeroth-Korean train을 화자 기준으로 나누는 규칙이다. validation으로 뗄 화자 10명과, 학습 중
# 체크포인트를 고를 작은 묶음(val-500)의 발화를 정한다. prepare.py가 이 두 함수로 zeroth-val, zeroth-val500
# 목록을 만든다.
# [왜 필요한가] Zeroth-Korean에는 validation이 없다. 같은 화자가 학습과 validation에 함께 있으면 validation이
# 실제보다 쉬워지므로 발화가 아니라 화자 단위로 떼어 낸다(docs/plan.md 데이터).
# [설계 이유] 무작위 추출 대신 해시 정렬, ID 정렬 같은 결정적인 규칙만 쓴다. 시드를 관리하지 않아도 누가 다시
# 돌려도 같은 분리가 나온다. 실제로 새 클론에서 prepare 결과가 바이트 단위로 같았다(docs/experiments.md '새
# 클론에서 재현').

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

# VALIDATION_SPEAKERS: validation으로 뗄 화자 수(train 화자 105명 중 10명).
# VAL500_PER_SPEAKER: val-500에 화자마다 넣을 발화 수. 10명 × 50 = 500이 목표지만 발화가 50개보다 적은 화자가
# 있어 실제로는 440발화다(docs/experiments.md 1단계 결과).
VALIDATION_SPEAKERS = 10
VAL500_PER_SPEAKER = 50


# 화자 ID 문자열의 SHA-256 16진수 값을 정렬해 앞에서 count명을 고른다(docs/experiments.md '데이터 분리'에 적은
# 규칙 그대로).
# 왜 해시인가: ID 번호 순서와 무관하게 고르면서도 실행할 때마다 같은 화자가 나온다.
# 입력은 발화마다 화자 ID 하나씩이라 중복이 많다. 호출 위치: prepare.main.
def validation_speakers(speakers: Iterable[str], count: int = VALIDATION_SPEAKERS) -> list[str]:
    """화자 ID의 SHA-256 값으로 정렬했을 때 앞에 오는 `count`명의 화자."""
    # set으로 중복 화자를 없앤다. 그 뒤 해시 값으로 다시 정렬해 앞의 count명을 자른다.
    unique = sorted(set(speakers))
    return sorted(unique, key=lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())[:count]


# validation 발화에서 화자마다 발화 ID순으로 앞의 per_speaker개를 골라 ID 목록을 돌려준다. 입력 entries는
# 'speaker'와 'id' 키가 있는 dict 목록(prepare.py의 저장소 색인 항목).
# 왜 필요한가: 학습 중 체크포인트를 고를 때 쓰는 작은 묶음이다(train.py의 기본 --eval-set). 화자마다 같은 수를
# 뽑으므로 발화가 많은 화자 하나가 묶음을 차지하지 않는다.
# 주의: 작은 묶음이라 전체 validation과 순위가 다를 수 있다. 2단계에서 이 묶음으로는 학습률 3e-5가 나았지만
# 전체 2,214발화에서는 차이가 없었다(README 결과).
def val500_ids(entries: Sequence[dict], per_speaker: int = VAL500_PER_SPEAKER) -> list[str]:
    """화자마다 정렬 순서로 앞의 `per_speaker`개 발화 ID."""
    by_speaker: dict[str, list[str]] = {}
    for entry in entries:
        by_speaker.setdefault(entry["speaker"], []).append(entry["id"])
    picked: list[str] = []
    # 화자 이름순, 그 안에서 발화 ID순으로 정렬해 고르는 발화와 결과 순서까지 고정한다.
    for speaker in sorted(by_speaker):
        picked.extend(sorted(by_speaker[speaker])[:per_speaker])
    return picked
