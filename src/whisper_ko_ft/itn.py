"""한글로 풀어 쓴 한국어 숫자를 아라비아 숫자로 바꾼다("이천 십 팔 년" -> "2018년").

Zeroth-Korean은 숫자를 토큰마다 한글로 풀어 쓰고, Whisper와 FLEURS는 아라비아 숫자로 적는다. 이 모듈은 그
한 가지 표기 관례만 다루는 규칙 기반 역정규화(inverse text normalization)이며 범용 변환기가 아니다. 숫자 토큰
묶음은 바로 다음 토큰이 세는 단위일 때만 바꾼다. 숫자 낱말 하나하나는 대부분 보통 낱말이기도 하기
때문이다("이"(지시어), "일"(업무), "사", "구"). 규칙과 알려진 빈틈은 docs/experiments.md "6단계".
"""

# [파일 역할] Zeroth-Korean식으로 띄어 쓴 한국어 숫자 낱말을 아라비아 숫자로 바꾸는 규칙 기반 역정규화(ITN)다.
# 6단계부터 쓴다.
# [쓰이는 곳] (1) prepare_itn: 학습 정답의 숫자를 바꿔 6·7·10단계 모델(N, N2, QN)을 학습한다. (2) analyze의
# '맞춘 CER'(harmonize): 정답과 가설 양쪽에 같은 변환을 적용해 어느 표기로 적든 같은 점수가 나오게 한다. (3)
# analyze --scoring itn(to_digits): 8단계의 출력 숫자 변환 후처리로, 가설에만 적용한다. (4) analyze --scoring
# harmonized2(harmonize2, 2026-10-29 판정 없는 보조 채점): 붙여 쓴 한글 숫자('이천십육년')를 Zeroth식으로
# 띄어 쓴 뒤(split_attached) harmonize를 적용한다. 규칙은 docs/experiments.md '보조 지표 (판정 없음): 붙여 쓴
# 한글 숫자까지 맞춘 채점' 절. 기존 to_digits·harmonize는 바꾸지 않고 함수만 더했다.
# [설계 이유] 숫자 낱말 대부분이 보통 낱말과 겹치므로(위 docstring의 '이', '일' 등) 숫자 묶음 바로 뒤에
# 단위(년, 월, 명 …)가 올 때만 바꾼다. 흔한 낱말과 겹치는 한 글자 숫자는 년·월·퍼센트 같은 좁은 단위 앞에서만
# 바꾼다. 만·억·조는 글자로 둔다.
# [한계] 규칙 기반이라 빠뜨리는 것이 있다. 단위 없는 숫자('갤럭시노트 칠'), 붙여 쓴 숫자는 그대로 남아 학습
# 정답에 두 표기가 섞인다(docs/experiments.md 6단계). 붙여 쓴 숫자는 보조 채점 harmonize2만 다룬다.
# [주의] 이 규칙을 바꾸면 6단계 이후의 맞춘 CER이 모두 달라진다. tests/test_recorded_reports.py가 맞춘 CER
# 총합을 고정해 이런 변경을 잡는다. harmonize2는 학습 정답(prepare_itn)과 등록 채점에 쓰지 않는다.

from __future__ import annotations

import re

# DIGITS: 한자어 숫자 한 글자(일~구)의 값.
# SMALL_UNITS: 만 아래의 자리 단위(십·백·천). '이천'처럼 숫자와 붙거나 '십'처럼 혼자 쓰인다.
# LARGE_UNITS: 만·억·조. 숫자로 바꾸지 않고 글자로 남긴다('일 만 칠 천' → '1만 7000').
DIGITS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
SMALL_UNITS = {"십": 10, "백": 100, "천": 1000}
LARGE_UNITS = ("만", "억", "조")
# NATIVE_ONES, NATIVE_TENS: 고유어 수사(한·두·세 …, 열·스물 …)의 값. '다섯 번' → '5번', '스물 다섯 살' →
# '25살'처럼 쓴다.
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
# NATIVE_SAFE_ALONE: 혼자 나와도 숫자로 바꿔도 되는 고유어 수사(다섯~아홉, 열·스물 …). 한·두·세·네는 관형사나
# 대답('네')과 겹쳐 혼자서는 바꾸지 않는다. 예: '한 관계자', '세 부담', '네'(대답)는 그대로 남는다.
NATIVE_SAFE_ALONE = {"다섯", "여섯", "일곱", "여덟", "아홉"} | set(NATIVE_TENS)

# UNITS: 숫자 묶음 바로 뒤에 와야 하는 단위. 토큰이 단위로 시작하고 나머지가 AFTER_UNIT(조사·접미사)이면
# 단위로 본다. 앞에서부터 처음 맞는 것을 쓴다.
# 끝의 조·억·만·천도 단위로 둔다. '칠 천여'처럼 자리 단위에 접미사가 붙은 토큰에서 숫자 읽기가 멈추면, 그
# 토큰을 단위로 받아 '7천여'가 되게 하려는 것이다(tests/test_itn.py의 '일 만 칠 천여 년' → '1만 7천여 년').
UNITS = (
    "년대", "년", "개월", "월", "일", "시간", "시", "분", "초", "주일", "주", "세기",
    "퍼센트포인트", "퍼센트", "프로", "원", "달러", "유로", "엔", "위안", "명", "개", "위", "호", "번",
    "세", "살", "층", "도", "킬로미터", "킬로그램", "센티미터", "밀리미터", "미터", "그램", "톤", "리터",
    "배", "회", "차", "학년", "대", "건", "곳", "점", "승", "패", "골", "가지", "마리", "차례", "편", "장",
    "권", "척", "평", "인", "석", "표", "조", "억", "만", "천",
)  # fmt: skip
# AFTER_UNIT: 같은 토큰 안에서 단위 뒤에 붙어도 되는 조사·접미사. 빈 문자열은 단위만 있는 토큰('년')이다.
# '년에는', '번째' 같은 형태를 받는다.
AFTER_UNIT = {
    "", "에", "의", "은", "는", "이", "가", "을", "를", "로", "으로", "부터", "까지", "도", "만", "과", "와",
    "에서", "에는", "에도", "이다", "입니다", "째", "간", "쯤", "씩", "여", "경", "께", "보다", "이나", "나",
    "이라고", "이라는", "인", "이던", "였던", "에서는", "까지는", "부터는", "마다", "이며", "이고",
    "당", "짜리",
}  # fmt: skip
# AMBIGUOUS_ALONE: 한 토큰으로만 나오면 숫자보다 보통 낱말일 때가 훨씬 많은 것. 이것 하나로 된 숫자는
# STRICT_UNITS 앞에서만 바꾼다.
# STRICT_UNITS: 그런 애매한 숫자 뒤에서도 숫자로 읽어도 될 만큼 좁은 단위(년, 월, 퍼센트 …). 예: '이 년'은
# '2년'이 되지만 '이 분'은 '분'이 좁은 단위가 아니라 그대로 남는다.
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
# MONTH_FORMS: 소리가 바뀐 달 이름. '시 월'은 10월, '유 월'은 6월이다. '시', '유'는 숫자 글자가 아니라서 숫자
# 읽기와 따로 처리한다.
MONTH_FORMS = {"시": 10, "유": 6}  # "시 월"은 10월, "유 월"은 6월


# 만 아래 토큰 하나의 (값, 채운 가장 낮은 자리)를 돌려준다. 예: '삼' → (3, 1), '십' → (10, 10), '이천' →
# (2000, 1000). 숫자가 아니면 None. '가장 낮은 자리'는 _read_sino가 자리가 내려가는지 확인하는 데 쓴다.
def _small_value(token: str) -> tuple[int, int] | None:
    """토큰("삼", "십", "이천" 같은 것)의 (값, 채운 가장 낮은 자리). 숫자가 아니면 None."""
    if token in DIGITS:
        return DIGITS[token], 1
    if token in SMALL_UNITS:
        return SMALL_UNITS[token], SMALL_UNITS[token]
    if len(token) == 2 and token[0] in DIGITS and token[1] in SMALL_UNITS:
        return DIGITS[token[0]] * SMALL_UNITS[token[1]], SMALL_UNITS[token[1]]
    return None


# token이 단위(+ 허용된 조사·접미사)면 그 단위를, 아니면 None을 돌려준다. strict=True면 STRICT_UNITS에 든
# 단위만 인정한다.
def _unit_of(token: str, strict: bool) -> str | None:
    for unit in UNITS:
        if token.startswith(unit) and token[len(unit) :] in AFTER_UNIT:
            if strict and unit not in STRICT_UNITS:
                return None
            return unit
    return None


# start에서 시작하는 가장 긴 한자어 숫자를 읽어 (숫자 문자열, 다음 토큰 위치)를 돌려준다. 예: ['이천', '십',
# '팔', '년'] → ('2018', 3).
# 자리가 내려가는 동안만 한 묶음으로 더한다(이천 → 십 → 팔). 만·억·조가 오면 그 묶음을 '<값>만'으로 적고 다음
# 묶음을 읽는다.
def _read_sino(tokens: list[str], start: int) -> tuple[str, int] | None:
    """`start`에서 시작하는 가장 긴 올바른 숫자를 아라비아 숫자로, 그리고 그 다음 위치를 돌려준다.

    10,000 아래의 묶음은 아라비아 숫자가 되고, 만/억/조는 글자로 남는다("일 만 칠 천" -> "1만 7000").
    """
    parts: list[str] = []
    index = start
    while index < len(tokens):
        # 만 아래 한 묶음을 읽는다. lowest는 지금까지 채운 가장 낮은 자리이고, 다음 토큰의 자리가 그보다
        # 낮아야 같은 묶음으로 이어진다.
        value, lowest, begin = 0, 10_000, index
        while index < len(tokens):
            small = _small_value(tokens[index])
            if small is None or small[1] >= lowest:  # 자리는 내려가야 한다: "팔 일"은 8 다음에 단위 '일'
                break
            value, lowest = value + small[0], small[1]
            index += 1
        # 숫자를 하나도 읽지 못했으면 끝낸다(첫 토큰이 숫자가 아니거나, 만·억·조 뒤에 숫자가 없다).
        if index == begin:
            break
        # 묶음 뒤에 만·억·조가 오면 '<값>만'처럼 글자를 붙여 두고 다음 묶음을 읽는다. 아니면 이 묶음이
        # 마지막이다.
        if index < len(tokens) and tokens[index] in LARGE_UNITS:
            parts.append(f"{value}{tokens[index]}")
            index += 1
            continue
        parts.append(str(value))
        break
    if not parts:
        return None
    return " ".join(parts), index


# start의 고유어 수사를 읽어 (숫자 문자열, 다음 토큰 위치)를 돌려준다. 열·스물 … 뒤에 한·두 …가 오면
# 더한다('스물 다섯' → 25). 한·두·세·네는 혼자일 때 읽지 않는다(NATIVE_SAFE_ALONE에 없다). 숫자가 아니면 None.
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


# 문장의 숫자 낱말 묶음 중 바로 뒤에 단위가 오는 것만 아라비아 숫자로 바꾸고 단위와 붙여 쓴다('이천 십 팔 년'
# → '2018년', '다섯 번' → '5번'). 다른 토큰은 그대로 둔다.
# 호출 위치: prepare_itn(학습 정답), harmonize, analyze의 itn 계열 채점(출력 후처리).
def to_digits(text: str) -> str:
    tokens = text.split()
    out: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        # '시 월'·'유 월'은 숫자 읽기 규칙으로 읽히지 않으므로 먼저 따로 바꾼다.
        if token in MONTH_FORMS and following.startswith("월") and following[1:] in AFTER_UNIT:
            out.append(f"{MONTH_FORMS[token]}{following}")
            index += 2
            continue
        # 한자어 숫자를 먼저 읽어 보고, 아니면 고유어 수사를 읽는다.
        read = _read_sino(tokens, index) or _read_native(tokens, index)
        if read is None:
            out.append(token)
            index += 1
            continue
        digits, after = read
        # 숫자가 한 토큰뿐이고 그 토큰이 흔한 낱말과 겹치면(AMBIGUOUS_ALONE) 좁은 단위만 인정한다.
        alone = after - index == 1 and token in AMBIGUOUS_ALONE
        unit_token = tokens[after] if after < len(tokens) else ""
        # 뒤에 단위가 없으면 첫 토큰만 그대로 내보내고 한 칸만 나아간다. 다음 토큰부터 다시 숫자 묶음을
        # 찾는다.
        if _unit_of(unit_token, strict=alone) is None:
            out.append(token)  # 뒤에 단위가 없으면 낱말을 그대로 둔다
            index += 1
            continue
        # 숫자와 단위 토큰(조사 포함)을 붙여 한 토큰으로 내보내고 단위 다음으로 건너뛴다.
        out.append(f"{digits}{unit_token}")
        index = after + 1
    return " ".join(out)


# 맞춘 CER용 표기 통일. '%'를 '퍼센트'로 바꾼 뒤 to_digits를 적용해, 정답의 '오십 퍼센트'와 가설의 '50%'가
# 모두 '50퍼센트'가 된다. 정답과 가설 양쪽에 똑같이 적용한다.
# 8단계 출력 후처리에 이 함수가 아니라 to_digits를 쓰는 이유: FLEURS 한국어 정답은 '100%'처럼 '%'를 쓰므로,
# 가설에만 '%' → '퍼센트'를 적용하면 정답 표기와 반대 방향이 된다(docs/experiments.md 8단계).
# 호출 위치: analyze의 harmonized·itn-harmonized·agnostic 채점.
def harmonize(text: str) -> str:
    """정답과 가설을 같은 표기로 맞춘다: "%"를 낱말로 바꾼 뒤 숫자를 아라비아 숫자로 바꾼다."""
    return to_digits(text.replace("%", "퍼센트"))


# --- 보조 채점 harmonized2: 붙여 쓴 한글 숫자 (규칙 2026-10-29, docs/experiments.md '보조 지표' 절) ---
# to_digits는 공백 토큰마다 읽으므로 Zeroth 정답처럼 띄어 쓴 '이천 십 육 년'만 숫자로 바꾸고, 한글 맞춤법대로
# 붙여 쓴 '이천십육년'·'이천십육 년'(제44항: 수는 만 단위로 띄어 쓴다)·'스물다섯 살'은 그대로 둔다. 9단계
# Qwen처럼 한글 숫자를 붙여 쓰는 모델은 그래서 맞춘 CER에서 숫자 표기 차이를 오류로 받는다. 아래 함수는 그런
# 토큰을 Zeroth식 조각으로 띄어 쓴 뒤 기존 harmonize를 그대로 적용한다. 숫자로 바꿀지는 여전히 to_digits가
# 띄어 쓴 정답에서와 똑같이 정하므로(뒤에 단위가 와야 한다 등), 더 맞춰지는 것은 숫자를 붙여 쓴
# 표기(띄어쓰기와 숫자 토큰에 붙은 문장부호)의 몫뿐이다. 규칙은 한국어 수사 문법과 이 모듈의 기존 목록만으로
# 정했고 데이터에서 예외를 고르지 않았다. 알려진 오탐('사회는' → '4회는', '육회' → '6회')은
# tests/test_itn.py에 고정했다.

# PUNCTUATION: 등록 정규화(text_norm.normalize의 한국어 경로)가 공백으로 바꾸는 글자와 같다(text_norm의
# 문장부호 글자에 작은따옴표를 더한 것. 한국어 경로는 작은따옴표도 지운다). harmonize2가 숫자를 읽기 전에
# 공백으로 바꿔 '오월,'처럼 문장부호가 붙은 토큰에서도 변환이 막히지 않게 한다. 어차피 채점 전에 지워지는
# 글자라 숫자 변환 밖의 결과는 바뀌지 않는다. analyze scoring-diff도 바뀐 낱말을 보여 줄 때 같은 글자로 토큰을
# 나눈다.
PUNCTUATION = re.compile(r"[^\w\s]|_")
# _PLACES: 수의 자리 글자(십·백·천, 만·억·조). 붙여 쓴 토큰을 나눌 때는 단위로 보지 않는다. UNITS 끝의
# 조·억·만·천은 띄어 쓴 '칠 천여'에서 숫자 읽기가 멈춘 토큰을 받으려고 넣은 것이라, 붙여 쓴 '이천에'를 '이
# 천에'로 나누면 한 수가 둘로 깨진다.
_PLACES = set(SMALL_UNITS) | set(LARGE_UNITS)
# _ATTACHED_UNITS: 붙여 쓴 토큰에서 수사 뒤에 올 수 있는 단위(UNITS에서 자리 글자를 뺀 것, 순서는 그대로라
# _unit_of처럼 앞에서부터 처음 맞는 것을 쓴다).
_ATTACHED_UNITS = tuple(unit for unit in UNITS if unit not in _PLACES)


# 글자열이 한자어 수사 하나면 Zeroth식 조각 목록을, 아니면 None을 돌려준다. 예: '이천십육' → ['이천', '십',
# '육'], '일만칠천' → ['일', '만', '칠천']. 숫자 글자 바로 뒤에 십·백·천이 오면 두 글자를 한 조각으로
# 묶고('이천'), 나머지 숫자·자리 글자는 한 글자씩 조각으로 둔다. 그 조각들을 _read_sino가 처음부터 끝까지 한
# 수로 읽어야 수사로 본다(자리가 내려가야 하고, 만·억·조는 앞에 수가 있어야 한다). 그래서 '삼삼', '일일',
# '만'은 수사가 아니다.
def _sino_pieces(text: str) -> list[str] | None:
    pieces: list[str] = []
    index = 0
    while index < len(text):
        pair = text[index : index + 2]
        if len(pair) == 2 and pair[0] in DIGITS and pair[1] in SMALL_UNITS:
            pieces.append(pair)
            index += 2
        elif text[index] in DIGITS or text[index] in SMALL_UNITS or text[index] in LARGE_UNITS:
            pieces.append(text[index])
            index += 1
        else:
            return None
    read = _read_sino(pieces, 0)
    return pieces if read is not None and read[1] == len(pieces) else None


# token이 고유어 수사로 시작하면 (수사 조각 목록, 나머지)들을 긴 수사부터 돌려준다. 고유어 수사는 열·스물 …
# 아흔(NATIVE_TENS) 하나, 그 뒤에 한·두 … 아홉(NATIVE_ONES)을 붙인 것('스물다섯' → ['스물', '다섯']),
# 다섯~아홉 하나다. 한·두·세·네만 쓴 것은 관형사·대답과 겹치고 to_digits도 혼자서는 숫자로 읽지
# 않으므로(NATIVE_SAFE_ALONE에 없다) 수사로 보지 않는다.
def _native_prefixes(token: str) -> list[tuple[list[str], str]]:
    found: list[tuple[list[str], str]] = []
    for tens in NATIVE_TENS:
        if token.startswith(tens):
            rest = token[len(tens) :]
            found += [([tens, ones], rest[len(ones) :]) for ones in NATIVE_ONES if rest.startswith(ones)]
            found.append(([tens], rest))
    found += [
        ([ones], token[len(ones) :])
        for ones in NATIVE_ONES
        if ones in NATIVE_SAFE_ALONE and token.startswith(ones)
    ]
    return sorted(found, key=lambda item: -len("".join(item[0])))


# 수사 뒤의 나머지(rest)를 본다. 나머지가 단위 + 조사·접미사(AFTER_UNIT)면 [수사 조각들, 나머지]를 돌려준다.
# 나머지가 없으면(수사만 있는 토큰) 조각이 둘 이상일 때만 나눈다(이미 한 조각인 '이천', '오', '스물'은
# 그대로). strict면 단위가 좁은 단위(STRICT_UNITS)여야 한다. 단위는 _unit_of처럼 앞에서부터 처음 맞는 것
# 하나로 정한다. 나눌 수 없으면 None.
def _with_unit(numeral: list[str], rest: str, strict: bool) -> list[str] | None:
    if not rest:
        return numeral if len(numeral) > 1 else None
    for unit in _ATTACHED_UNITS:
        if rest.startswith(unit) and rest[len(unit) :] in AFTER_UNIT:
            return [*numeral, rest] if not strict or unit in STRICT_UNITS else None
    return None


# 공백 토큰 하나를 Zeroth식 조각으로 나눈 목록을 돌려준다. 남김없이 나뉘지 않으면 None(그대로 둔다).
# 순서: (1) 소리가 바뀐 달 이름 '시월'·'유월'(+조사)은 '시 월'·'유 월'로(to_digits의 MONTH_FORMS). (2) 한자어
# 수사 + (단위 + 조사): 수사 앞부분을 가장 길게 잡고, 나머지가 맞지 않으면 짧게 줄여 본다. 수사가 한 조각이고
# 흔한 낱말과 겹치면(AMBIGUOUS_ALONE: '이', '일', '사' …) 좁은 단위 앞에서만 나눈다. to_digits가 띄어 쓴 글에
# 쓰는 규칙과 같아 '이년'은 나누고 '이분'·'이번'·'사건'은 그대로 둔다. (3) 고유어 수사 + (단위 + 조사).
def _pieces(token: str) -> list[str] | None:
    for form in MONTH_FORMS:
        rest = token[len(form) :]
        if token.startswith(form) and rest.startswith("월") and rest[1:] in AFTER_UNIT:
            return [form, rest]
    for cut in range(len(token), 0, -1):
        numeral = _sino_pieces(token[:cut])
        if numeral is not None:
            strict = len(numeral) == 1 and numeral[0] in AMBIGUOUS_ALONE
            found = _with_unit(numeral, token[cut:], strict)
            if found is not None:
                return found
    for numeral, rest in _native_prefixes(token):
        found = _with_unit(numeral, rest, strict=False)
        if found is not None:
            return found
    return None


# 붙여 쓴 한글 수사와 그 뒤에 붙은 단위(+조사)를 Zeroth 정답처럼 자리마다 띄어 쓴다. 띄어쓰기만 바꾸고 숫자로
# 바꾸지는 않는다(그것은 harmonize가 한다). 예: '이천십육년 오월' → '이천 십 육 년 오 월', '스물다섯살이' →
# '스물 다섯 살이'. 나뉘지 않는 토큰과 이미 Zeroth식으로 띄어 쓴 글은 그대로다(공백은 한 칸씩으로 맞춘다).
# 호출 위치: harmonize2.
def split_attached(text: str) -> str:
    """붙여 쓴 한글 수사·단위를 Zeroth 정답처럼 자리마다 띄어 쓴다."""
    out: list[str] = []
    for token in text.split():
        out.extend(_pieces(token) or [token])
    return " ".join(out)


# 보조 채점 harmonized2의 표기 통일: '%'를 '퍼센트'로 바꾸고 문장부호를 공백으로 바꾼 뒤 split_attached로 붙여
# 쓴 한글 숫자를 띄우고, 기존 harmonize를 그대로 적용한다('%' 치환은 split_attached가 '삼십%'를 '삼십
# 퍼센트'로 나눌 수 있게 먼저 한다. harmonize 안의 같은 치환은 그때 할 일이 없다). 정답과 가설 양쪽에 똑같이
# 적용한다. 등록 채점(harmonize)과 학습 정답(prepare_itn)은 이 함수를 쓰지 않는다. 호출 위치: analyze의
# harmonized2 채점(판정 없는 보조 지표)과 scoring-diff.
def harmonize2(text: str) -> str:
    """붙여 쓴 한글 숫자까지 맞춘다: 문장부호를 지우고 수사를 Zeroth식으로 띄운 뒤 harmonize를 적용한다."""
    return harmonize(split_attached(PUNCTUATION.sub(" ", text.replace("%", "퍼센트"))))
