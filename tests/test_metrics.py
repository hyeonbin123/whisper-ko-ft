# [테스트 묶음] metrics.py(채점, 묶음 오류율, 부트스트랩 구간)의 정의를 작은 예로 고정한다(docs/experiments.md
# '지표').
# [왜 중요한가] 모든 판정이 이 함수들의 결과로 정해진다. 오류율을 발화 평균으로 바꾸거나 구간이 실행마다
# 달라지면 기록된 수치와 판정을 다시 낼 수 없다.

import numpy as np
import pytest

from whisper_ko_ft.metrics import (
    block_difference_interval,
    difference_interval,
    error_rate,
    interval,
    leave_one_block_out,
    score,
)


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


# 오류율은 발화 평균이 아니라 '편집 수 합 ÷ 정답 길이 합'이다.
def test_error_rate_is_corpus_level():
    # 2글자 중 오류 1개, 8글자 중 오류 0개: 0.5와 0의 평균이 아니라 1/10이다.
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


# --- 블록(화자·문장) 단위 구간 (민감도 분석, 2026-10-15) ---


# 블록 하나에 발화 하나씩이고 발화가 블록 이름 순서대로 놓여 있으면, 블록 단위 재표집은 발화 단위 재표집과
# 같은 인덱스를 뽑으므로 difference_interval과 똑같은 구간이 나와야 한다. 같은 시드·같은 뽑기 방식인지
# 고정한다.
def test_block_interval_with_one_utterance_per_block_equals_the_utterance_interval():
    rng = np.random.default_rng(2)
    lengths = rng.integers(5, 40, size=150)
    edits_a, edits_b = rng.binomial(lengths, 0.12), rng.binomial(lengths, 0.10)
    blocks = [f"b{n:03d}" for n in range(150)]  # 이름 순서 = 발화 순서
    assert block_difference_interval(edits_a, edits_b, lengths, blocks) == difference_interval(
        edits_a, edits_b, lengths
    )


# 같은 블록의 발화는 함께 뽑힌다: 블록 안 발화를 하나로 합친 배열과 구간이 같아야 한다.
def test_utterances_of_a_block_are_drawn_together():
    lengths = np.array([10, 10, 20, 5, 5, 5, 30])
    edits_a = np.array([1, 3, 2, 0, 4, 1, 6])
    edits_b = np.array([0, 1, 2, 1, 1, 1, 2])
    blocks = ["s1", "s1", "s2", "s3", "s3", "s3", "s4"]
    merged = block_difference_interval(
        np.array([4, 2, 5, 6]), np.array([1, 2, 3, 2]), np.array([20, 20, 15, 30]), ["s1", "s2", "s3", "s4"]
    )
    assert block_difference_interval(edits_a, edits_b, lengths, blocks) == merged


# 차이가 한 블록에 몰려 있으면 블록 단위 구간이 발화 단위보다 넓다(같은 화자의 발화가 서로 닮은 경우의 그림).
def test_a_difference_concentrated_in_one_block_widens_the_interval():
    lengths = np.full(200, 10)
    edits_b = np.ones(200, dtype=int)
    edits_a = edits_b.copy()
    edits_a[:20] += 3  # 첫 블록의 발화만 더 틀린다
    blocks = [f"s{n // 20}" for n in range(200)]
    low, high = block_difference_interval(edits_a, edits_b, lengths, blocks)
    u_low, u_high = difference_interval(edits_a, edits_b, lengths)
    assert high - low > u_high - u_low


# 블록 하나 빼기: 블록마다 그 블록의 발화를 모두 뺀 나머지에서 잰 차이(점 추정)다.
def test_leave_one_block_out_drops_every_utterance_of_the_block():
    lengths = np.array([10, 10, 20, 40])
    edits_a = np.array([2, 2, 1, 4])
    edits_b = np.array([1, 1, 1, 4])
    out = leave_one_block_out(edits_a, edits_b, lengths, ["x", "x", "y", "z"])
    assert out == {"x": 0 / 60, "y": 2 / 60, "z": 2 / 40}


# 배열 길이가 맞지 않으면 조용히 틀리게 계산하지 않고 멈춘다.
def test_block_arrays_must_describe_the_same_utterances():
    with pytest.raises(ValueError, match="same utterances"):
        block_difference_interval(np.array([1]), np.array([1]), np.array([5]), ["a", "b"])
