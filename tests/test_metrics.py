import numpy as np

from whisper_ko_ft.metrics import difference_interval, error_rate, interval, score


def test_korean_cer_counts_characters_without_spaces():
    scored = score("할 수 있다", "할수 없다", "ko")
    assert (scored.edits, scored.length) == (1, 4)


def test_english_wer_counts_words():
    scored = score("the cat sat", "the cat sat down", "en")
    assert (scored.edits, scored.length) == (1, 3)


def test_empty_reference_is_skipped():
    assert score("...", "무언가", "ko") is None


def test_error_rate_is_corpus_level():
    # 1 error in 2 characters and 0 errors in 8: 1/10, not the mean of 0.5 and 0.
    assert error_rate(np.array([1, 0]), np.array([2, 8])) == 0.1


def test_interval_contains_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    lengths = rng.integers(5, 40, size=200)
    edits = rng.binomial(lengths, 0.1)
    low, high = interval(edits, lengths)
    assert low < error_rate(edits, lengths) < high
    assert (low, high) == interval(edits, lengths)


def test_difference_interval_is_paired():
    lengths = np.full(100, 10)
    worse = np.full(100, 2)
    better = np.full(100, 1)
    low, high = difference_interval(worse, better, lengths)
    assert low == high == 0.1
