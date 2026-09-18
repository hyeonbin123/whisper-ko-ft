"""Error rates with utterance-level bootstrap intervals (rules in docs/experiments.md, "지표")."""

from __future__ import annotations

from dataclasses import dataclass

import jiwer
import numpy as np

from whisper_ko_ft.text_norm import normalize

BOOTSTRAP_SAMPLES = 1_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Scored:
    reference: str  # normalized
    hypothesis: str  # normalized
    edits: int
    length: int  # reference characters (ko) or words (en)


def score(reference: str, hypothesis: str, language: str) -> Scored | None:
    """Edit counts of one utterance, or None when the normalized reference is empty."""
    ref, hyp = normalize(reference, language), normalize(hypothesis, language)
    if not ref:
        return None
    if language == "ko":
        out = jiwer.process_characters(ref, hyp)
    else:
        out = jiwer.process_words(ref, hyp)
    edits = out.substitutions + out.deletions + out.insertions
    return Scored(ref, hyp, edits, out.hits + out.substitutions + out.deletions)


def error_rate(edits: np.ndarray, lengths: np.ndarray) -> float:
    """Corpus-level rate: total edits over total reference length."""
    return float(edits.sum() / lengths.sum())


def _resample_indices(count: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, count, size=(BOOTSTRAP_SAMPLES, count))


def interval(edits: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    """95% bootstrap interval of the corpus-level rate, resampling utterances."""
    picks = _resample_indices(len(edits))
    rates = edits[picks].sum(axis=1) / lengths[picks].sum(axis=1)
    low, high = np.percentile(rates, [2.5, 97.5])
    return float(low), float(high)


def difference_interval(edits_a: np.ndarray, edits_b: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    """95% interval of rate(a) - rate(b) over the same utterances (paired resampling)."""
    picks = _resample_indices(len(lengths))
    totals = lengths[picks].sum(axis=1)
    deltas = edits_a[picks].sum(axis=1) / totals - edits_b[picks].sum(axis=1) / totals
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)
