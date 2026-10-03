"""Rewrite Latin letters spelled with their Korean names as the letters ("에이아이" -> "AI").

The training references (Zeroth-Korean) have no Latin letters, so the fine-tuned models write an initialism as
the names of its letters ("유에스오씨"). A run of tokens is rewritten only when every token is made of letter
names (a particle may follow the last one), the run has at least two letters, and at least one name in it is
not also an everyday syllable or word: "비디오", "오디오", "이유", "티비" are left alone. A rule, not a list
of known abbreviations; it misses initialisms made only of such names. docs/experiments.md, stage 8.
"""

from __future__ import annotations

import re

LETTER_NAMES = {
    "에이": "A", "비": "B", "씨": "C", "디": "D", "이": "E", "에프": "F", "지": "G", "에이치": "H",
    "아이": "I", "제이": "J", "케이": "K", "엘": "L", "엠": "M", "엔": "N", "오": "O", "피": "P",
    "큐": "Q", "알": "R", "에스": "S", "티": "T", "유": "U", "브이": "V", "더블유": "W", "엑스": "X",
    "와이": "Y", "제트": "Z",
}  # fmt: skip
# Names that are also everyday syllables or words; a run of only these is not rewritten.
COMMON_NAMES = {"비", "씨", "디", "이", "지", "아이", "오", "피", "티", "유", "알", "엔"}
# Particles that may follow the last token of a run, longest first.
PARTICLES = (
    "이라는", "에서", "으로", "라는", "의", "는", "은", "가", "를",
    "을", "와", "과", "에", "로", "도", "만", "이",
)  # fmt: skip
_LONGEST = max(len(name) for name in LETTER_NAMES)
_TRAILING = re.compile(r"^(.*?)(\W*)$")


def _spell(word: str) -> list[str] | None:
    """The fewest letter names that make up the whole word, or None."""
    best: list[list[str] | None] = [[]] + [None] * len(word)
    for end in range(1, len(word) + 1):
        for start in range(max(0, end - _LONGEST), end):
            before = best[start]
            if before is None or word[start:end] not in LETTER_NAMES:
                continue
            if best[end] is None or len(before) + 1 < len(best[end]):
                best[end] = [*before, word[start:end]]
    return best[len(word)] or None


def _read(token: str) -> tuple[list[str], str, bool] | None:
    """(letter names, what follows them, whether a particle ended the run) of a spelled token, or None."""
    word, tail = _TRAILING.match(token).groups()
    names = _spell(word)
    if names:
        return names, tail, bool(tail)
    for particle in PARTICLES:
        if word.endswith(particle) and len(word) > len(particle):
            names = _spell(word[: -len(particle)])
            if names:
                return names, particle + tail, True
    return None


def letters_to_latin(text: str) -> str:
    tokens = text.split()
    out: list[str] = []
    index = 0
    while index < len(tokens):
        names: list[str] = []
        tail, end = "", index
        while end < len(tokens):
            read = _read(tokens[end])
            if read is None:
                break
            names += read[0]
            tail = read[1]
            end += 1
            if read[2]:  # a particle or punctuation closes the run
                break
        if len(names) >= 2 and any(name not in COMMON_NAMES for name in names):
            out.append("".join(LETTER_NAMES[name] for name in names) + tail)
            index = end
        else:
            out.append(tokens[index])
            index += 1
    return " ".join(out)
