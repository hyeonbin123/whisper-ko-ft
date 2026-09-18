from whisper_ko_ft.text_norm import normalize


def test_korean_drops_spaces_and_punctuation():
    assert normalize("할 수 있다, '희망'을!", "ko") == "할수있다희망을"


def test_korean_spacing_difference_is_not_an_error():
    assert normalize("할 수", "ko") == normalize("할수", "ko")


def test_english_keeps_contractions_and_single_spaces():
    assert normalize("It's  a   Test.", "en") == "it's a test"


def test_digits_are_kept():
    assert normalize("2020년 3월", "ko") == "2020년3월"
