import pytest

from whisper_ko_ft.itn import to_digits


@pytest.mark.parametrize(
    ("spelled", "digits"),
    [
        ("이천 십 팔 년 완공을 목표로", "2018년 완공을 목표로"),
        ("천 구백 이십 이 년에는", "1922년에는"),
        ("지난해 시 월과 올해 일 월 잇달아", "지난해 10월과 올해 1월 잇달아"),
        ("이십 팔 일 러시아 타스 통신은", "28일 러시아 타스 통신은"),
        ("일 월 삼십 일 일 서울", "1월 31일 서울"),
        ("지난 십 이 일 일부 언론에", "지난 12일 일부 언론에"),
        ("생산량을 오 퍼센트씩 줄이자고", "생산량을 5퍼센트씩 줄이자고"),
        ("대구가 삼십 오 도 울산이 삼십 삼 도", "대구가 35도 울산이 33도"),
        ("육십 달러대를 회복하는", "육십 달러대를 회복하는"),  # "달러대": not a known ending, left alone
        ("일 만 칠 천여 년 전", "1만 7천여 년 전"),
        ("초등학교 육 학년 부장교사", "초등학교 6학년 부장교사"),
        ("다섯 번 환자처럼", "5번 환자처럼"),
        ("스물 한 명이 모였다", "21명이 모였다"),
    ],
)
def test_numbers_before_a_unit_become_digits(spelled, digits):
    assert to_digits(spelled) == digits


@pytest.mark.parametrize(
    "text",
    [
        "이 일은 쉽지 않다",  # "this work", not "2일"
        "이 사람은 한 관계자에게 말했다",
        "사 측은 구 대표와 만났다",
        "세 부담이 늘었다",
        "오 일본 대표",  # a number word followed by a word that only starts like a unit
        "금요일 대구의 낮 최고기온은 이십 육 로 예보됐습니다",  # no unit in the data: left as it is
    ],
)
def test_ordinary_words_are_left_alone(text):
    assert to_digits(text) == text


def test_text_with_digits_already_is_unchanged():
    assert to_digits("2018년 5월 3일") == "2018년 5월 3일"
