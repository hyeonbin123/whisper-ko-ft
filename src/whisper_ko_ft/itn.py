"""Rewrite spelled-out Korean numbers as Arabic digits ("이천 십 팔 년" -> "2018년").

Zeroth-Korean spells numbers out token by token; Whisper and FLEURS write digits. This is a rule-based
inverse text normalization for that one convention, not a general one. A run of number tokens is rewritten
only when the next token is a counting unit, because most single number words are also ordinary words
("이" this, "일" work, "사", "구"). Rules and known gaps: docs/experiments.md, "6단계".
"""

from __future__ import annotations

DIGITS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
SMALL_UNITS = {"십": 10, "백": 100, "천": 1000}
LARGE_UNITS = ("만", "억", "조")
NATIVE_ONES = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
NATIVE_TENS = {
    "열": 10,
    "스물": 20,
    "서른": 30,
    "마흔": 40,
    "쉰": 50,
    "예순": 60,
    "일흔": 70,
    "여든": 80,
    "아흔": 90,
}
# "한", "두", "세", "네" alone are left as they are: "한 관계자", "세 부담", "네" (yes).
NATIVE_SAFE_ALONE = {"다섯", "여섯", "일곱", "여덟", "아홉"} | set(NATIVE_TENS)

UNITS = (
    "년대", "년", "개월", "월", "일", "시간", "시", "분", "초", "주일", "주", "세기",
    "퍼센트포인트", "퍼센트", "프로", "원", "달러", "유로", "엔", "위안", "명", "개", "위", "호", "번",
    "세", "살", "층", "도", "킬로미터", "킬로그램", "센티미터", "밀리미터", "미터", "그램", "톤", "리터",
    "배", "회", "차", "학년", "대", "건", "곳", "점", "승", "패", "골", "가지", "마리", "차례", "편", "장",
    "권", "척", "평", "인", "석", "표", "조", "억", "만", "천",
)  # fmt: skip
# What may follow a unit inside the same token: particles and a few suffixes.
AFTER_UNIT = {
    "", "에", "의", "은", "는", "이", "가", "을", "를", "로", "으로", "부터", "까지", "도", "만", "과", "와",
    "에서", "에는", "에도", "이다", "입니다", "째", "간", "쯤", "씩", "여", "경", "께", "보다", "이나", "나",
    "이라고", "이라는", "인", "이던", "였던", "에서는", "까지는", "부터는", "마다", "이며", "이고",
    "당", "짜리",
}  # fmt: skip
# A run that is only one of these is an ordinary word far more often than a number.
AMBIGUOUS_ALONE = {"이", "일", "사", "구", "오", "천", "백", "십", "만"}
STRICT_UNITS = {
    "년",
    "개월",
    "월",
    "퍼센트",
    "퍼센트포인트",
    "프로",
    "학년",
    "차",
    "위",
    "달러",
    "배",
    "세",
    "층",
    "회",
}
MONTH_FORMS = {"시": 10, "유": 6}  # "시 월" (10월), "유 월" (6월)


def _small_value(token: str) -> tuple[int, int] | None:
    """(value, lowest place filled) of a token like "삼", "십", "이천", or None."""
    if token in DIGITS:
        return DIGITS[token], 1
    if token in SMALL_UNITS:
        return SMALL_UNITS[token], SMALL_UNITS[token]
    if len(token) == 2 and token[0] in DIGITS and token[1] in SMALL_UNITS:
        return DIGITS[token[0]] * SMALL_UNITS[token[1]], SMALL_UNITS[token[1]]
    return None


def _unit_of(token: str, strict: bool) -> str | None:
    for unit in UNITS:
        if token.startswith(unit) and token[len(unit) :] in AFTER_UNIT:
            if strict and unit not in STRICT_UNITS:
                return None
            return unit
    return None


def _read_sino(tokens: list[str], start: int) -> tuple[str, int] | None:
    """Digits for the longest valid number starting at `start`, and the index after it.

    Groups below 10,000 become digits; 만/억/조 stay as words ("일 만 칠 천" -> "1만 7000").
    """
    parts: list[str] = []
    index = start
    while index < len(tokens):
        value, lowest, begin = 0, 10_000, index
        while index < len(tokens):
            small = _small_value(tokens[index])
            if small is None or small[1] >= lowest:  # places must go down: "팔 일" is 8 then the unit 일
                break
            value, lowest = value + small[0], small[1]
            index += 1
        if index == begin:
            break
        if index < len(tokens) and tokens[index] in LARGE_UNITS:
            parts.append(f"{value}{tokens[index]}")
            index += 1
            continue
        parts.append(str(value))
        break
    if not parts:
        return None
    return " ".join(parts), index


def _read_native(tokens: list[str], start: int) -> tuple[str, int] | None:
    token = tokens[start]
    if token in NATIVE_TENS:
        following = tokens[start + 1] if start + 1 < len(tokens) else ""
        if following in NATIVE_ONES:
            return str(NATIVE_TENS[token] + NATIVE_ONES[following]), start + 2
        return str(NATIVE_TENS[token]), start + 1
    if token in NATIVE_SAFE_ALONE:
        return str(NATIVE_ONES[token]), start + 1
    return None


def to_digits(text: str) -> str:
    tokens = text.split()
    out: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token in MONTH_FORMS and following.startswith("월") and following[1:] in AFTER_UNIT:
            out.append(f"{MONTH_FORMS[token]}{following}")
            index += 2
            continue
        read = _read_sino(tokens, index) or _read_native(tokens, index)
        if read is None:
            out.append(token)
            index += 1
            continue
        digits, after = read
        alone = after - index == 1 and token in AMBIGUOUS_ALONE
        unit_token = tokens[after] if after < len(tokens) else ""
        if _unit_of(unit_token, strict=alone) is None:
            out.append(token)  # not followed by a unit: leave the words as they are
            index += 1
            continue
        out.append(f"{digits}{unit_token}")
        index = after + 1
    return " ".join(out)


def harmonize(text: str) -> str:
    """One notation for references and hypotheses alike: "%" as a word, then numbers as digits."""
    return to_digits(text.replace("%", "퍼센트"))
