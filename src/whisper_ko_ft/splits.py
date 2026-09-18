"""Speaker-based validation split of Zeroth-Korean train (rules in docs/experiments.md, "데이터 분리")."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

VALIDATION_SPEAKERS = 10
VAL500_PER_SPEAKER = 50


def validation_speakers(speakers: Iterable[str], count: int = VALIDATION_SPEAKERS) -> list[str]:
    """The `count` speakers whose SHA-256 of the speaker ID sorts first."""
    unique = sorted(set(speakers))
    return sorted(unique, key=lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())[:count]


def val500_ids(entries: Sequence[dict], per_speaker: int = VAL500_PER_SPEAKER) -> list[str]:
    """For each speaker, the first `per_speaker` utterance IDs in sorted order."""
    by_speaker: dict[str, list[str]] = {}
    for entry in entries:
        by_speaker.setdefault(entry["speaker"], []).append(entry["id"])
    picked: list[str] = []
    for speaker in sorted(by_speaker):
        picked.extend(sorted(by_speaker[speaker])[:per_speaker])
    return picked
