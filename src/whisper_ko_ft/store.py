"""Audio stores: one int16 file per source split plus a JSON index, read through a memmap.

A store keeps 16 kHz mono samples back to back. Random access is a slice of the memmap, which is what
training needs; reading rows out of parquet at random would decode a whole row group each time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from whisper_ko_ft.paths import CACHE

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Utterance:
    id: str
    speaker: str
    text: str
    store: str
    offset: int
    length: int

    @property
    def seconds(self) -> float:
        return self.length / SAMPLE_RATE


class AudioStore:
    def __init__(self, name: str, root: Path = CACHE) -> None:
        self.name = name
        self._samples = np.memmap(root / f"{name}.int16", dtype=np.int16, mode="r")

    def read(self, utterance: Utterance) -> np.ndarray:
        """Float32 samples in [-1, 1]."""
        chunk = self._samples[utterance.offset : utterance.offset + utterance.length]
        return chunk.astype(np.float32) / 32768.0


def load_set(name: str, root: Path = CACHE) -> tuple[str, list[Utterance]]:
    """Language and utterances of a set written by `whisper_ko_ft.prepare`."""
    data = json.loads((root / "sets" / f"{name}.json").read_text(encoding="utf-8"))
    return data["language"], [Utterance(**entry) for entry in data["utterances"]]
