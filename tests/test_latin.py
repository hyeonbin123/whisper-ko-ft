import pytest

from whisper_ko_ft.latin import letters_to_latin


@pytest.mark.parametrize(
    ("spelled", "latin"),
    [
        ("에이아이 기술이", "AI 기술이"),
        ("유에스오씨는 성명에서", "USOC는 성명에서"),
        ("에프 에이에이의 발표", "FAA의 발표"),  # a run of spelled tokens, a particle on the last one
        ("엘지 전자", "LG 전자"),
        ("에이치 아이 브이", "HIV"),
        ("더블유 에이치 오가", "WHO가"),
        ("에이아이.", "AI."),  # punctuation stays where it was
    ],
)
def test_spelled_letters_become_latin(spelled, latin):
    assert letters_to_latin(spelled) == latin


@pytest.mark.parametrize(
    "text",
    [
        "비디오를 보았다",  # every name in it is also an ordinary syllable
        "오디오 장비",
        "이유가 있다",
        "오이 한 개",
        "티비를 켰다",
        "아이가 웃었다",  # one letter only
        "에스 자 모양",
        "에이 그건 아니다",
        "이 사람은",
        "케이팝 공연",  # not made of letter names only
    ],
)
def test_ordinary_words_are_left_alone(text):
    assert letters_to_latin(text) == text


def test_text_without_letter_names_is_unchanged():
    assert letters_to_latin("2018년 5월 3일 서울") == "2018년 5월 3일 서울"
    assert letters_to_latin("") == ""
