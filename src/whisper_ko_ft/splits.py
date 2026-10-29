"""Zeroth-Korean train을 화자 기준으로 나눠 validation을 만든다(규칙은 docs/experiments.md "데이터 분리").

학습 묶음과 평가 묶음이 같은 문장을 얼마나 함께 갖는지도 센다(`sentence_overlap`, docs/experiments.md
"추가 확인: Zeroth 문장 겹침"). 화자 기준 분리는 화자만 떼므로 문장은 겹칠 수 있다. 13단계의 문장
hold-out은 문장 키를 해시 순서로 정렬해 일부 문장을 통째로 학습에서 뗀다(`heldout_sentences`,
docs/experiments.md 13단계).
"""

# [파일 역할] Zeroth-Korean train을 화자 기준으로 나누는 규칙이다. validation으로 뗄 화자 10명과, 학습 중
# 체크포인트를 고를 작은 묶음(val-500)의 발화를 정한다. prepare.py가 이 두 함수로 zeroth-val, zeroth-val500
# 목록을 만든다. 그 분리가 화자만 떼고 문장은 떼지 않는다는 점을 숫자로 보이는 문장 겹침 셈(sentence_key,
# sentence_overlap)도 여기 둔다. analyze overlap이 이 함수로 수를 찍고 reports/sentence_overlap/zeroth.json을
# 쓴다.
# [왜 필요한가] Zeroth-Korean에는 validation이 없다. 같은 화자가 학습과 validation에 함께 있으면 validation이
# 실제보다 쉬워지므로 발화가 아니라 화자 단위로 떼어 낸다(docs/plan.md 데이터). 그런데 Zeroth의
# validation·test 문장은 거의 모두 학습 묶음에 다른 화자의 녹음으로 들어 있다. 그래서 Zeroth 수치는 '처음
# 듣는 화자가 학습에서 본 문장을 읽는' 조건이고, 이것을 세어 두는 함수가 필요하다(docs/experiments.md '추가
# 확인: Zeroth 문장 겹침').
# [설계 이유] 무작위 추출 대신 해시 정렬, ID 정렬 같은 결정적인 규칙만 쓴다. 시드를 관리하지 않아도 누가 다시
# 돌려도 같은 분리가 나온다. 실제로 새 클론에서 prepare 결과가 바이트 단위로 같았다(docs/experiments.md '새
# 클론에서 재현'). 문장 겹침 셈도 파일을 읽지 않는 순수 함수라 가짜 묶음으로 테스트하고, 묶음을 읽는 일은
# 호출하는 쪽(analyze overlap)이 맡는다.
# [13단계 문장 hold-out] 위 겹침 때문에 기존 묶음으로는 '학습에서 본 적 없는 문장'을 잴 수 없다. 그래서 문장
# 키의 합집합에서 SHA-256 순서로 앞 15%를 H로 정해 학습에서 통째로 빼고 N과 같은 설정으로 다시 학습한다. H를
# 고르는 규칙(heldout_count, heldout_sentences)은 화자 고르기와 같은 해시 정렬이라 여기 둔다. 묶음 파일을 읽고
# 쓰는 일과 학습 전 관문은 prepare_heldout이 맡는다.

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

# text_norm.normalize: 채점(metrics.score)이 정답과 가설에 쓰는 정규화. 문장 키를 채점과 같은 규칙으로 만들어,
# 띄어쓰기·문장부호만 다른 문장을 같은 문장으로 센다.
from whisper_ko_ft.text_norm import normalize

# Utterance는 타입 표기에만 쓴다. 실행할 때 store(numpy를 부름)를 불러오지 않으려고 TYPE_CHECKING 안에 둔다.
if TYPE_CHECKING:
    from whisper_ko_ft.store import Utterance

# VALIDATION_SPEAKERS: validation으로 뗄 화자 수(train 화자 105명 중 10명).
# VAL500_PER_SPEAKER: val-500에 화자마다 넣을 발화 수. 10명 × 50 = 500이 목표지만 발화가 50개보다 적은 화자가
# 있어 실제로는 440발화다(docs/experiments.md 1단계 결과).
VALIDATION_SPEAKERS = 10
VAL500_PER_SPEAKER = 50
# HELDOUT_PERCENT: 13단계 문장 hold-out에서 학습에서 뗄 문장의 비율(%, docs/experiments.md 13단계). 15%면
# validation에서 그 문장을 읽은 판정 발화가 347개(화자 9명) 생기고 학습 발화는 84.9%가 남는다(같은 절 '묶음').
# HOLDOUT_DEFINITION: 13단계 정의 파일의 reports/ 아래 경로. prepare_heldout --write-definition이 쓰고
# 커밋하며 analyze verdict13이 읽는다. 두 모듈이 같은 위치를 쓰도록 한곳에 둔다.
HELDOUT_PERCENT = 15
HOLDOUT_DEFINITION = ("sentence_holdout", "zeroth.json")


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


# --- 문장 겹침 (docs/experiments.md '추가 확인: Zeroth 문장 겹침', 2026-10-24) ---

# OVERLAP_KEYS: 문장을 같다고 보는 두 기준. raw는 prepare가 저장한 정답 글 그대로(앞뒤 공백만 뗀 원문),
# normalized는 sentence_key(채점과 같은 정규화) 뒤의 글이다. 두 기준의 수가 다르면 띄어쓰기·문장부호만 다른
# 문장이 있다는 뜻이다.
OVERLAP_KEYS = ("raw", "normalized")


# 문장 키: 정답 글을 채점과 같은 규칙(text_norm.normalize)으로 정규화한 글. 한국어는
# 문장부호·띄어쓰기·대소문자 차이를 지운 글자열이 되므로 '학교에 간다'와 '학교에, 간다!'가 같은 키다. 영어는
# 단어 사이 공백 하나를 남긴다.
# 왜 채점 정규화인가: 채점이 같은 글로 보는 두 정답은 모델에게도 같은 문장이므로, 겹침도 같은 기준으로 센다.
# 문장 hold-out처럼 문장 단위로 묶음을 나누는 일도 이 키를 쓴다.
def sentence_key(text: str, language: str = "ko") -> str:
    """겹침을 셀 때 같은 문장으로 보는 키: `text_norm.normalize(text, language)`."""
    return normalize(text, language)


# --- 13단계 문장 hold-out (docs/experiments.md 13단계, 2026-10-29) ---


# 문장 키의 SHA-256(UTF-8) 16진수. H를 고르는 정렬 순서이고, 정의 파일에는 글 대신 이 값을 적는다(정답 글을
# 커밋하지 않고도 어떤 문장이 H인지 커밋된 리포트의 정답으로 다시 맞춰 볼 수 있다).
def sentence_hash(key: str) -> str:
    """문장 키의 SHA-256(UTF-8) 16진수."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# K개 문장 중 뗄 개수: percent% × K를 반올림한 것(.5는 올림). 0.15 × K를 부동소수점으로 계산해 round를 쓰면
# 4.5 같은 경계에서 은행가 반올림(짝수 쪽)과 부동소수점 오차가 끼므로, 정수 계산 (K × percent + 50) // 100으로
# 정한다. 실제 Zeroth의 K = 3,004는 451개다(0.15 × 3,004 = 450.6이라 어느 방식으로도 같다).
def heldout_count(total: int, percent: int = HELDOUT_PERCENT) -> int:
    """K개 중 뗄 문장 수: percent% × K를 반올림(.5는 올림)한 정수."""
    return (total * percent + 50) // 100


# 문장 키들의 합집합(중복은 하나로)을 키의 SHA-256 순으로 정렬해 앞의 heldout_count(K)개를 그 순서대로
# 돌려준다. 입력 순서와 중복에 흔들리지 않는다(validation_speakers와 같은 해시 정렬이라 문장 난도와 무관하게
# 고르면서 누가 돌려도 같다). 해시가 같은 두 키는 사실상 없지만 순서를 완전히 정하려고 키 자체를 둘째 정렬
# 기준으로 둔다. 호출 위치: prepare_heldout.holdout(입력은 zeroth-train·zeroth-val·zeroth-test의 원래 정답
# 키).
def heldout_sentences(keys: Iterable[str], percent: int = HELDOUT_PERCENT) -> list[str]:
    """문장 키 합집합에서 SHA-256 순으로 앞의 percent%(반올림)개."""
    unique = set(keys)
    ordered = sorted(unique, key=lambda key: (sentence_hash(key), key))
    return ordered[: heldout_count(len(unique), percent)]


# 한 기준(raw 또는 normalized)으로 발화마다 문장 키를 만든다.
def _keys(utterances: Sequence[Utterance], kind: str, language: str) -> list[str]:
    if kind == "raw":
        return [u.text for u in utterances]
    return [sentence_key(u.text, language) for u in utterances]


# 한 묶음 안에서의 문장 셈: 고유 문장 수, 두 번 이상 나온 문장 수와 그 문장들의 발화 수. counts는 문장 키 →
# 그 묶음에서 나온 횟수.
def _within(counts: Counter[str]) -> dict[str, int]:
    repeated = [n for n in counts.values() if n > 1]
    return {
        "sentences": len(counts),
        "repeated_sentences": len(repeated),
        "repeated_utterances": sum(repeated),
    }


# 학습 묶음과 평가 묶음들의 문장 겹침과 화자 겹침을 센다. 파일을 읽지 않는 순수 함수다(묶음은 호출하는 쪽이
# store.load_set으로 읽어 넘긴다).
# 입력: train은 학습 묶음의 발화 목록, sets는 묶음 이름 → 발화 목록, language는 문장 키를 만들 언어(학습과
# 평가 묶음이 같은 언어여야 뜻이 있다. analyze overlap이 다른 언어 묶음을 거부한다). 발화는
# store.Utterance처럼 id·speaker·text 속성이 있으면 된다.
# 출력(dict):
# - "train": 발화 수(utterances), 화자 수(speakers), 기준마다(raw, normalized) 고유 문장 수(sentences),
#   가장 많이 읽힌 문장의 낭독 수(max_reads), 두 번 이상 읽힌 문장 수와 그 발화 수(repeated_sentences,
#   repeated_utterances)
# - "sets"[이름]: 발화 수, 화자 수, 그중 학습 묶음에도 있는 화자 수(speakers_in_train), 기준마다 고유 문장
#   수(sentences), 그중 학습에도 있는 문장 수(sentences_in_train), 문장이 학습에 있는 발화
#   수(utterances_in_train), 묶음 안에서 두 번 이상 나온 문장 수와 그 발화 수(repeated_sentences,
#   repeated_utterances), 발화 ID → 그 문장이 학습 묶음에서 읽힌 횟수(train_reads, 묶음의 발화 순서), 문장이
#   학습에 없는 발화 ID(not_in_train, ID순)
# 주의: 화자 칸이 빈 발화(FLEURS)는 화자로 세지 않는다. 그래서 FLEURS의 화자 수와 화자 겹침 0은 겹침이 없다는
# 확인이 아니라 화자 정보가 없다는 뜻이다. 발화 ID가 겹치는 묶음은 발화별 횟수가 덮어써지므로 ValueError로
# 거부한다(1단계에서 FLEURS의 같은 ID 때문에 행이 빠졌던 오류와 같은 종류).
def sentence_overlap(
    train: Sequence[Utterance], sets: Mapping[str, Sequence[Utterance]], language: str = "ko"
) -> dict:
    """학습 묶음과 평가 묶음들의 문장·화자 겹침(원문 기준과 정규화 기준)."""
    train_speakers = {u.speaker for u in train if u.speaker}
    reads: dict[str, Counter[str]] = {}
    summary: dict = {"utterances": len(train), "speakers": len(train_speakers)}
    for kind in OVERLAP_KEYS:
        reads[kind] = Counter(_keys(train, kind, language))
        summary[kind] = {**_within(reads[kind]), "max_reads": max(reads[kind].values(), default=0)}

    result: dict = {"train": summary, "sets": {}}
    for name, utterances in sets.items():
        ids = [u.id for u in utterances]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: utterance ids are not unique")
        speakers = {u.speaker for u in utterances if u.speaker}
        entry: dict = {
            "utterances": len(utterances),
            "speakers": len(speakers),
            "speakers_in_train": len(speakers & train_speakers),
        }
        for kind in OVERLAP_KEYS:
            keys = _keys(utterances, kind, language)
            counts = Counter(keys)
            # 발화마다 그 문장이 학습에서 읽힌 횟수. 학습에 없으면 0이다.
            train_reads = {i: reads[kind][k] for i, k in zip(ids, keys, strict=True)}
            within = _within(counts)
            entry[kind] = {
                "sentences": within["sentences"],
                "sentences_in_train": sum(1 for k in counts if k in reads[kind]),
                "utterances_in_train": sum(1 for n in train_reads.values() if n),
                "repeated_sentences": within["repeated_sentences"],
                "repeated_utterances": within["repeated_utterances"],
                "train_reads": train_reads,
                "not_in_train": sorted(i for i, n in train_reads.items() if not n),
            }
        result["sets"][name] = entry
    return result
