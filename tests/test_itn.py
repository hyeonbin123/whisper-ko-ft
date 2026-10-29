# [테스트 묶음] itn.to_digits(6단계 숫자 변환)의 규칙을 실제 Zeroth 정답과 같은 꼴의 예문으로 고정한다. 바꿔야
# 하는 경우(단위 앞의 숫자)와 그대로 둬야 하는 경우(보통 낱말과 겹치는 숫자 낱말)를 나눠 확인한다.
# [왜 중요한가] 이 변환은 학습 정답(N, N2, QN)과 맞춘 CER 채점에 함께 쓰인다. 규칙이 너무 넓으면 '이 일은'처럼
# 보통 문장을 망가뜨리고, 너무 좁으면 표기 차이가 남는다. 규칙을 고칠 때 이전 동작이 깨지지 않게 막는다.
# 끝 절은 보조 채점 harmonized2(2026-10-29, itn.split_attached·harmonize2)의 규칙 예시다. 규칙은 결과를
# 계산하기 전에 docs/experiments.md '보조 지표 (판정 없음): 붙여 쓴 한글 숫자까지 맞춘 채점' 절에 적었고,
# 여기의 예시는 그 규칙을 그대로 옮긴 것이다(데이터에서 고른 예외가 아니다).

import pytest

from whisper_ko_ft.itn import harmonize, harmonize2, split_attached, to_digits

# 단위 앞의 숫자가 아라비아 숫자로 바뀌는 경우들: 연도, 달(시 월 → 10월), 날짜, 자리가 내려가는지로 숫자와
# 단위 '일'을 가르는 경우('삼십 일 일' → '31일', '십 이 일 일부' → '12일 일부'), 퍼센트, 도, 만 단위('1만
# 7천여'), 학년, 고유어 수사('다섯 번', '스물 한 명'). '달러대'처럼 알려진 조사·접미사가 아닌 것이 붙으면
# 그대로 둔다. 아래 harmonize2 불변 테스트도 같은 목록을 쓴다.
SPELLED = [
    ("이천 십 팔 년 완공을 목표로", "2018년 완공을 목표로"),
    ("천 구백 이십 이 년에는", "1922년에는"),
    ("지난해 시 월과 올해 일 월 잇달아", "지난해 10월과 올해 1월 잇달아"),
    ("이십 팔 일 러시아 타스 통신은", "28일 러시아 타스 통신은"),
    ("일 월 삼십 일 일 서울", "1월 31일 서울"),
    ("지난 십 이 일 일부 언론에", "지난 12일 일부 언론에"),
    ("생산량을 오 퍼센트씩 줄이자고", "생산량을 5퍼센트씩 줄이자고"),
    ("대구가 삼십 오 도 울산이 삼십 삼 도", "대구가 35도 울산이 33도"),
    ("육십 달러대를 회복하는", "육십 달러대를 회복하는"),  # "달러대": 알려진 어미가 아니라 그대로 둔다
    ("일 만 칠 천여 년 전", "1만 7천여 년 전"),
    ("초등학교 육 학년 부장교사", "초등학교 6학년 부장교사"),
    ("다섯 번 환자처럼", "5번 환자처럼"),
    ("스물 한 명이 모였다", "21명이 모였다"),
]
# 숫자로 바꾸면 안 되는 보통 문장들: '이 일은'(이것), '한 관계자', '사 측', '구 대표', '세 부담', 단위처럼
# 시작만 하는 낱말('오 일본'), 단위가 빠진 숫자. 흔한 낱말과 겹치는 한 글자 숫자는 좁은 단위 앞에서만 바꾼다는
# 규칙을 확인한다.
ORDINARY = [
    "이 일은 쉽지 않다",  # '2일'이 아니라 '이(지시어) 일(업무)'
    "이 사람은 한 관계자에게 말했다",
    "사 측은 구 대표와 만났다",
    "세 부담이 늘었다",
    "오 일본 대표",  # 숫자 낱말 뒤의 낱말이 단위처럼 시작하기만 한다
    "금요일 대구의 낮 최고기온은 이십 육 로 예보됐습니다",  # 데이터에 단위가 빠져 있어 그대로 둔다
]


# 각 예문이 기대한 문장과 글자까지 같아야 한다.
@pytest.mark.parametrize(("spelled", "digits"), SPELLED)
def test_numbers_before_a_unit_become_digits(spelled, digits):
    assert to_digits(spelled) == digits


# 입력과 출력이 같아야 한다.
@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_words_are_left_alone(text):
    assert to_digits(text) == text


# 이미 아라비아 숫자인 글은 그대로다. 출력 후처리(8단계)를 숫자로 적는 모델의 가설에 적용해도 망가지지 않는다.
def test_text_with_digits_already_is_unchanged():
    assert to_digits("2018년 5월 3일") == "2018년 5월 3일"


# --- 보조 채점 harmonized2: 붙여 쓴 한글 숫자 (규칙 2026-10-29, 계산 전) ---


# 붙여 쓴 수사와 단위를 Zeroth 정답처럼 자리마다 띄어 쓴 뒤 harmonize를 적용하면 띄어 쓴 정답과 같은 숫자가
# 된다. 수사만 붙여 쓰고 단위를 띄어 쓴 꼴('이천십육 년', '스물다섯 살')은 한글 맞춤법(제44항 수는 만 단위로
# 띄어 쓴다, 제43항 단위 명사는 띄어 쓴다)의 표준형이라 함께 다룬다. 한 글자 수사는 좁은 단위 앞에서만
# 바뀐다('이년' → '2년'). '%'와 숫자 토큰에 붙은 문장부호는 변환을 막지 않는다.
@pytest.mark.parametrize(
    ("attached", "harmonized"),
    [
        ("이천십육년 오월", "2016년 5월"),
        ("삼십퍼센트", "30퍼센트"),
        ("스물다섯살", "25살"),
        ("천구백칠십팔년도", "1978년도"),
        ("이천십육 년", "2016년"),
        ("스물다섯 살", "25살"),
        ("이년", "2년"),
        ("시월에", "10월에"),
        ("유월", "6월"),
        ("삼십%", "30퍼센트"),
        ("이천십육년, 오월.", "2016년 5월"),
        ("십일월 이십일일", "11월 21일"),
        ("열한시", "11시"),
        ("일만칠천명", "1만 7000명"),
        ("다섯번째", "5번째"),
    ],
)
def test_harmonize2_reads_attached_numbers(attached, harmonized):
    assert harmonize2(attached) == harmonized


# 수사 + 단위 + 조사로 남김없이 나뉘지 않는 낱말('사과를', '구월동에서', '천사의', '일부'), 한 글자 수사 뒤의
# 넓은 단위('이분', '이번', '오일'), 혼자서는 숫자로 읽지 않는 고유어 수사('한명'), 앞에 수가 없는
# '만'('만세'), 이미 한 조각인 수사('이천')는 바꾸지 않는다.
@pytest.mark.parametrize(
    "text",
    [
        "사과를 먹었다",
        "구월동에서 만났다",
        "천사의 날개",
        "일부 언론에",
        "이분이 오셨다",
        "이번에는 오일 가격이",
        "한국 사람 한명",
        "만세를 불렀다",
        "이천에 갔다",
    ],
)
def test_harmonize2_leaves_words_that_do_not_split(text):
    assert split_attached(text) == text
    assert harmonize2(text) == harmonize(text) == text


# 알려진 오탐(규칙에 적음): 붙여 쓴 보통 낱말이 한 글자 수사 + 좁은 단위로 남김없이 나뉘면 숫자로 바뀐다.
# 데이터에서 예외를 고르지 않는다는 규칙대로 그대로 두고, 정답·가설에 똑같이 적용되므로 같은 낱말끼리는 편집
# 수가 바뀌지 않는다. 오탐 점검(analyze scoring-diff)이 이런 발화를 나열한다.
@pytest.mark.parametrize(("word", "converted"), [("육회", "6회"), ("사회는", "4회는")])
def test_harmonize2_known_false_positives(word, converted):
    assert harmonize2(word) == converted


# 기존 harmonize의 결과는 그대로다: Zeroth식으로 이미 띄어 쓴 예문(위 6단계 목록)에서는 harmonize2와
# harmonize가 글자까지 같다. 아라비아 숫자로 적은 글도 같다.
@pytest.mark.parametrize("text", [s for s, _ in SPELLED] + ORDINARY + ["2018년 5월 3일"])
def test_harmonize2_equals_harmonize_on_zeroth_spacing(text):
    assert split_attached(text) == text
    assert harmonize2(text) == harmonize(text)


# split_attached는 띄어쓰기만 바꾼다: 수사를 자리마다 띄우고 단위(+조사)를 따로 둔다. 숫자로 바꾸는 것은
# harmonize다. 수사만 있는 토큰도 두 조각 이상이면 띄운다('이십일' → '이십 일').
@pytest.mark.parametrize(
    ("attached", "spaced"),
    [
        ("이천십육년 오월", "이천 십 육 년 오 월"),
        ("스물다섯살이", "스물 다섯 살이"),
        ("천구백칠십팔년도", "천 구백 칠십 팔 년도"),
        ("삼천만원", "삼천 만 원"),
        ("시월에", "시 월에"),
        ("이십일", "이십 일"),
        ("열세", "열 세"),
    ],
)
def test_split_attached_only_respaces(attached, spaced):
    assert split_attached(attached) == spaced
