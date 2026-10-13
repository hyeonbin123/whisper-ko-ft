"""Error rates with utterance-level bootstrap intervals (rules in docs/experiments.md, "지표")."""

# [파일 역할] 채점의 핵심. 발화 하나의 편집 수(score), 묶음 전체의 오류율(error_rate), 발화 단위 부트스트랩
# 신뢰구간(interval), 두 모델의 짝지은 차이 구간(difference_interval)을 계산한다.
# [설계 이유] 오류율은 발화별 오류율의 평균이 아니라 '편집 수 합 ÷ 정답 길이 합'(묶음 전체
# 기준)이다(docs/experiments.md 공통 규칙 '지표'). 그래서 짧은 발화 하나의 오류율이 긴 발화와 같은 무게를 갖지
# 않는다. 리포트에는 발화별 edits와 length를 남기므로, 숫자 발화 제외 같은 어떤 부분집합의 오류율과 구간도
# 나중에 다시 계산할 수 있다.
# [연결] evaluate·evaluate_ct2·evaluate_qwen이 리포트를 쓸 때, train이 학습 중 체크포인트를 고를 때, analyze가
# 표와 판정을 만들 때 쓴다.

from __future__ import annotations

from dataclasses import dataclass

# jiwer: 정렬 기반 편집 거리(맞음·치환·삭제·삽입 수)를 글자 단위·단어 단위로 계산한다. jiwer 버전이 바뀌어
# 수치가 움직이면 tests/test_recorded_reports.py가 잡는다.
# numpy: 부트스트랩 재표집을 배열 연산으로 한 번에 한다.
import jiwer
import numpy as np

# normalize: 채점 전에 정답과 가설을 같은 규칙으로 정리한다(text_norm.py).
from whisper_ko_ft.text_norm import normalize

# 부트스트랩 1,000번, seed 0(docs/experiments.md '지표'). 시드를 고정해, 같은 리포트에서 언제 다시 계산해도
# 같은 구간이 나온다.
BOOTSTRAP_SAMPLES = 1_000
BOOTSTRAP_SEED = 0


# score의 결과. 정규화한 정답·가설, 편집 수(치환 + 삭제 + 삽입), 정답 길이(한국어는 글자 수, 영어는 단어 수).
@dataclass(frozen=True)
class Scored:
    reference: str  # normalized
    hypothesis: str  # normalized
    edits: int
    length: int  # reference characters (ko) or words (en)


# 발화 하나를 채점한다. 정규화한 정답이 비면 None을 돌려준다(길이 0으로는 오류율을 낼 수 없다).
# evaluate.score_rows가 그런 발화를 길이 0으로 적고 그 수를 따로 센다.
# 호출 위치: evaluate.score_rows(evaluate·evaluate_qwen의 리포트), evaluate_ct2, train의 체크포인트 평가,
# analyze의 다시 채점, tests/test_recorded_reports.py.
def score(reference: str, hypothesis: str, language: str) -> Scored | None:
    """Edit counts of one utterance, or None when the normalized reference is empty."""
    ref, hyp = normalize(reference, language), normalize(hypothesis, language)
    if not ref:
        return None
    # 한국어는 글자 단위(CER), 영어는 단어 단위(WER)로 정렬한다. normalize가 한국어 공백을 모두 없앴으므로
    # 글자 비교에 띄어쓰기가 끼지 않는다.
    if language == "ko":
        out = jiwer.process_characters(ref, hyp)
    else:
        out = jiwer.process_words(ref, hyp)
    # 편집 수 = 치환 + 삭제 + 삽입. 정답 길이 = 맞음 + 치환 + 삭제(정답 쪽 토큰 수). 삽입은 길이에 들어가지
    # 않아 편집 수가 정답 길이를 넘을 수 있다. 이 '오류 수 > 정답 길이'가 되풀이 오류 발화를 가려내는
    # 기준이다(docs/experiments.md 7·8단계).
    edits = out.substitutions + out.deletions + out.insertions
    return Scored(ref, hyp, edits, out.hits + out.substitutions + out.deletions)


# 묶음 전체 오류율 = 편집 수 합 ÷ 정답 길이 합. 입력은 발화별 편집 수와 정답 길이 배열(같은 순서).
def error_rate(edits: np.ndarray, lengths: np.ndarray) -> float:
    """Corpus-level rate: total edits over total reference length."""
    return float(edits.sum() / lengths.sum())


# 부트스트랩 표본 1,000개의 발화 인덱스(복원 추출)를 한 번에 만든다. 모양은 (1,000, 발화 수). 부를 때마다 같은
# 시드로 새 생성기를 만들므로, 발화 수가 같으면 interval과 difference_interval이 언제나 같은 표본을 쓴다.
def _resample_indices(count: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, count, size=(BOOTSTRAP_SAMPLES, count))


# 묶음 오류율의 95% 구간. 발화를 복원 추출한 묶음 1,000개 각각에서 '편집 합 ÷ 길이 합'을 구하고 2.5·97.5
# 백분위를 낸다. 글자가 아니라 발화를 단위로 뽑는다(docs/experiments.md '지표').
# 호출 위치: evaluate·evaluate_ct2·evaluate_qwen의 리포트 요약, analyze의 표.
def interval(edits: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    """95% bootstrap interval of the corpus-level rate, resampling utterances."""
    picks = _resample_indices(len(edits))
    # picks로 인덱싱하면 (1,000, 발화 수) 배열이 되고, 행마다 합해 표본 묶음 1,000개의 오류율을 한 번에
    # 구한다.
    rates = edits[picks].sum(axis=1) / lengths[picks].sum(axis=1)
    low, high = np.percentile(rates, [2.5, 97.5])
    return float(low), float(high)


# 두 모델(a, b)을 같은 발화 묶음에서 비교할 때 rate(a) − rate(b)의 95% 구간. 같은 인덱스로 두 모델을 함께 뽑는
# 짝지은 재표집이라, 표본마다 발화 난이도의 차이가 양쪽에 똑같이 들어가 상쇄된다.
# 판정 규칙 대부분이 이 구간의 상한·하한으로 정해져 있다(예: 9단계는 상한 < 0이면 '더 나은 기준 모델'). 호출
# 위치: analyze.
# 주의: 세 배열은 같은 발화를 같은 순서로 담아야 한다.
def difference_interval(edits_a: np.ndarray, edits_b: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    """95% interval of rate(a) - rate(b) over the same utterances (paired resampling)."""
    picks = _resample_indices(len(lengths))
    # 정답 길이는 두 모델이 같으므로 한 번만 합해 양쪽 분모로 쓴다.
    totals = lengths[picks].sum(axis=1)
    deltas = edits_a[picks].sum(axis=1) / totals - edits_b[picks].sum(axis=1) / totals
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)
