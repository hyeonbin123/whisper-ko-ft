# [테스트 묶음] metrics.py(채점, 묶음 오류율, 부트스트랩 구간)의 정의를 작은 예로 고정한다(docs/experiments.md
# '지표').
# [왜 중요한가] 모든 판정이 이 함수들의 결과로 정해진다. 오류율을 발화 평균으로 바꾸거나 구간이 실행마다
# 달라지면 기록된 수치와 판정을 다시 낼 수 없다.

import numpy as np

from whisper_ko_ft.metrics import difference_interval, error_rate, interval, score


# 한국어 CER: 띄어쓰기를 무시하고 글자 단위로 센다. '할수있다' 대 '할수없다'는 치환 1개, 정답 4글자다.
def test_korean_cer_counts_characters_without_spaces():
    scored = score("할 수 있다", "할수 없다", "ko")
    assert (scored.edits, scored.length) == (1, 4)


# 영어 WER: 단어 단위. 단어 하나를 더 붙이면 삽입 1개, 정답 3단어다.
def test_english_wer_counts_words():
    scored = score("the cat sat", "the cat sat down", "en")
    assert (scored.edits, scored.length) == (1, 3)


# 정규화하면 비는 정답(문장부호뿐)은 채점하지 않고 None을 돌려준다.
def test_empty_reference_is_skipped():
    assert score("...", "무언가", "ko") is None


# 오류율은 발화 평균이 아니라 '편집 수 합 ÷ 정답 길이 합'이다(옆 주석의 예).
def test_error_rate_is_corpus_level():
    # 1 error in 2 characters and 0 errors in 8: 1/10, not the mean of 0.5 and 0.
    assert error_rate(np.array([1, 0]), np.array([2, 8])) == 0.1


# 부트스트랩 구간이 점 추정을 포함하고, 시드가 고정되어 두 번 계산해도 같은 값이 나오는지.
def test_interval_contains_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    lengths = rng.integers(5, 40, size=200)
    edits = rng.binomial(lengths, 0.1)
    low, high = interval(edits, lengths)
    assert low < error_rate(edits, lengths) < high
    assert (low, high) == interval(edits, lengths)


# 모든 발화에서 a가 b보다 정확히 1/10 더 틀리면 차이 구간은 0.1 한 점이어야 한다. 차이의 방향(a − b)과 같은
# 분모(정답 길이)를 쓰는지 확인한다.
def test_difference_interval_is_paired():
    lengths = np.full(100, 10)
    worse = np.full(100, 2)
    better = np.full(100, 1)
    low, high = difference_interval(worse, better, lengths)
    assert low == high == 0.1
