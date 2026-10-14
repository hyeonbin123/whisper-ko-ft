# [테스트 묶음] latin.letters_to_latin(8단계 보조 후보, 한글 글자 이름 → 라틴 글자)의 규칙을 예문으로
# 고정한다.
# [왜 중요한가] 사전 없이 규칙만으로 바꾸므로 보통 낱말을 약어로 잘못 바꾸기 쉽다. 바꿔야 하는 경우와 그대로
# 둬야 하는 경우를 함께 고정해 규칙이 넓어지거나 좁아지는 것을 막는다.

import pytest

from whisper_ko_ft.latin import letters_to_latin


# 바꾸는 경우: 붙여 쓴 약어('에이아이'), 조사가 붙은 약어('유에스오씨는'), 띄어 쓴 글자 이름 묶음과 마지막
# 토큰의 조사('에프 에이에이의'), 세 토큰 묶음('에이치 아이 브이'), 문장부호가 붙은 경우('에이아이.' → 'AI.').
@pytest.mark.parametrize(
    ("spelled", "latin"),
    [
        ("에이아이 기술이", "AI 기술이"),
        ("유에스오씨는 성명에서", "USOC는 성명에서"),
        ("에프 에이에이의 발표", "FAA의 발표"),  # 글자 이름 토큰 묶음, 마지막 토큰에 조사
        ("엘지 전자", "LG 전자"),
        ("에이치 아이 브이", "HIV"),
        ("더블유 에이치 오가", "WHO가"),
        ("에이아이.", "AI."),  # 문장부호는 제자리에 남는다
    ],
)
# 각 예문이 기대한 문장과 같아야 한다.
def test_spelled_letters_become_latin(spelled, latin):
    assert letters_to_latin(spelled) == latin


# 그대로 두는 경우: 흔한 음절과 겹치는 이름으로만 된 낱말('비디오', '오디오', '이유', '오이', '티비'), 글자
# 하나뿐인 경우('아이가', '에스 자', '에이 그건'), 글자 이름으로만 이루어지지 않은 낱말('케이팝').
@pytest.mark.parametrize(
    "text",
    [
        "비디오를 보았다",  # 안의 이름이 모두 흔한 음절이기도 하다
        "오디오 장비",
        "이유가 있다",
        "오이 한 개",
        "티비를 켰다",
        "아이가 웃었다",  # 글자 하나뿐
        "에스 자 모양",
        "에이 그건 아니다",
        "이 사람은",
        "케이팝 공연",  # 글자 이름으로만 이루어지지 않았다
    ],
)
# 입력과 출력이 같아야 한다.
def test_ordinary_words_are_left_alone(text):
    assert letters_to_latin(text) == text


# 글자 이름이 없는 글과 빈 문자열은 그대로다.
def test_text_without_letter_names_is_unchanged():
    assert letters_to_latin("2018년 5월 3일 서울") == "2018년 5월 3일 서울"
    assert letters_to_latin("") == ""
