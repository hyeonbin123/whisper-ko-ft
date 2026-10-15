"""오류율과 발화 단위 부트스트랩 구간(규칙은 docs/experiments.md "지표"), 민감도 분석용 블록 단위 구간."""

# [파일 역할] 채점의 핵심. 발화 하나의 편집 수(score), 묶음 전체의 오류율(error_rate), 발화 단위 부트스트랩
# 신뢰구간(interval), 두 모델의 짝지은 차이 구간(difference_interval)을 계산한다. 2026-10-15에 민감도
# 분석용으로 블록(화자·문장) 단위 짝지은 구간(block_difference_interval)과 블록 하나 빼기
# (leave_one_block_out)를 더했다. 기존 함수와 그 결과는 바뀌지 않는다.
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
    reference: str  # 정규화한 정답
    hypothesis: str  # 정규화한 가설
    edits: int
    length: int  # 정답 길이: 한국어(ko)는 글자 수, 영어(en)는 단어 수


# 발화 하나를 채점한다. 정규화한 정답이 비면 None을 돌려준다(길이 0으로는 오류율을 낼 수 없다).
# evaluate.score_rows가 그런 발화를 길이 0으로 적고 그 수를 따로 센다.
# 호출 위치: evaluate.score_rows(evaluate·evaluate_qwen의 리포트), evaluate_ct2, train의 체크포인트 평가,
# analyze의 다시 채점, tests/test_recorded_reports.py.
def score(reference: str, hypothesis: str, language: str) -> Scored | None:
    """발화 하나의 편집 수. 정규화한 정답이 비어 있으면 None."""
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


def error_rate(edits: np.ndarray, lengths: np.ndarray) -> float:
    """묶음 전체 기준 오류율 = 편집 수 합 ÷ 정답 길이 합.

    입력은 발화별 편집 수와 정답 길이 배열(같은 순서)이다.
    """
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
    """묶음 전체 기준 오류율의 95% 부트스트랩 구간. 발화를 단위로 재표집한다."""
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
    """같은 발화들에서 rate(a) - rate(b)의 95% 구간(짝지은 재표집)."""
    picks = _resample_indices(len(lengths))
    # 정답 길이는 두 모델이 같으므로 한 번만 합해 양쪽 분모로 쓴다.
    totals = lengths[picks].sum(axis=1)
    deltas = edits_a[picks].sum(axis=1) / totals - edits_b[picks].sum(axis=1) / totals
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


# 발화별 배열을 블록(화자나 문장)별 합으로 접는다. 블록은 이름 순으로 번호를 매긴다(docs/experiments.md
# '추가 분석: 화자·문장 단위 부트스트랩'의 규칙). 돌려주는 것: 블록 이름 목록과, 블록마다의 a 편집 합·b 편집
# 합·정답 길이 합 배열. 블록 단위 재표집에서는 블록 안 발화가 늘 함께 뽑히므로 합만 있으면 된다.
def _block_sums(
    edits_a: np.ndarray, edits_b: np.ndarray, lengths: np.ndarray, blocks: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    if not len(blocks) == len(lengths) == len(edits_a) == len(edits_b):
        raise ValueError("edits, lengths and blocks must describe the same utterances")
    names = sorted(set(blocks))
    position = {name: n for n, name in enumerate(names)}
    index = np.array([position[block] for block in blocks])
    sums = [
        np.bincount(index, weights=values, minlength=len(names)) for values in (edits_a, edits_b, lengths)
    ]
    return names, *sums


# 블록 단위 짝지은 부트스트랩: rate(a) − rate(b)의 95% 구간. 발화가 아니라 블록(Zeroth는 화자, FLEURS는
# 문장)을 복원 추출하고, 뽑힌 블록의 발화를 모두(같은 블록이 두 번 뽑히면 두 번) 넣어 '편집 합 ÷ 길이
# 합'의 차이를 구한다. 두 모델은 같은 블록을 함께 뽑는다.
# 왜 따로 두었나: 같은 화자·같은 문장의 발화는 서로 닮아, 발화를 독립으로 보고 뽑는 difference_interval의
# 구간이 실제보다 좁을 수 있다(arXiv 1912.09508, Liu & Peng). 기록된 판정을 이 구간으로 다시 보는 민감도
# 분석에 쓴다. 기존 구간과 판정은 바꾸지 않는다.
# 재현: 블록 K개를 이름 순으로 번호 매기고 default_rng(0).integers(0, K, size=(1,000, K))로 뽑는다. 블록
# 하나에 발화 하나씩이고 발화가 블록 이름 순서대로 놓여 있으면 difference_interval과 같은 값이
# 나온다(테스트로 고정).
# 주의: 블록이 10개뿐이면(Zeroth 화자) 백분위 구간이 거칠고 불안정하다. leave_one_block_out을 함께 본다.
def block_difference_interval(
    edits_a: np.ndarray, edits_b: np.ndarray, lengths: np.ndarray, blocks: list[str]
) -> tuple[float, float]:
    """같은 발화들에서 rate(a) - rate(b)의 95% 구간(블록 단위 짝지은 재표집)."""
    names, sum_a, sum_b, sum_length = _block_sums(edits_a, edits_b, lengths, blocks)
    picks = _resample_indices(len(names))
    totals = sum_length[picks].sum(axis=1)
    deltas = sum_a[picks].sum(axis=1) / totals - sum_b[picks].sum(axis=1) / totals
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


# 블록 하나씩 빼고 계산한 rate(a) − rate(b)(점 추정). 블록 이름 → 그 블록을 뺀 차이. 블록이 적은 Zeroth(화자
# 10명)에서 블록 부트스트랩 구간이 거친 것을 보완하려고, 어느 화자 하나가 차이를 만드는지 보여 준다.
def leave_one_block_out(
    edits_a: np.ndarray, edits_b: np.ndarray, lengths: np.ndarray, blocks: list[str]
) -> dict[str, float]:
    """블록마다, 그 블록의 발화를 모두 뺀 나머지에서 잰 rate(a) - rate(b)."""
    names, sum_a, sum_b, sum_length = _block_sums(edits_a, edits_b, lengths, blocks)
    rest_a, rest_b, rest_length = sum_a.sum() - sum_a, sum_b.sum() - sum_b, sum_length.sum() - sum_length
    return {name: float((rest_a[n] - rest_b[n]) / rest_length[n]) for n, name in enumerate(names)}
