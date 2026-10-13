# [테스트 묶음] text_norm.normalize(채점 전 정규화, docs/experiments.md '지표')의 규칙을 예시로 고정한다.
# [왜 중요한가] 정규화가 바뀌면 커밋된 모든 리포트의 수치가 달라진다. 띄어쓰기·문장부호·대소문자를 문서와 같게
# 다루는지 지킨다.

from whisper_ko_ft.text_norm import normalize


# 한국어: 공백, 쉼표, 작은따옴표, 느낌표가 모두 사라진다.
def test_korean_drops_spaces_and_punctuation():
    assert normalize("할 수 있다, '희망'을!", "ko") == "할수있다희망을"


# 띄어쓰기만 다른 두 문장은 같은 글자열이 되어 CER에서 오류가 0이다.
def test_korean_spacing_difference_is_not_an_error():
    assert normalize("할 수", "ko") == normalize("할수", "ko")


# 영어: 소문자로 바꾸고, 축약형의 작은따옴표는 남기며, 연속 공백은 하나로 줄인다.
def test_english_keeps_contractions_and_single_spaces():
    assert normalize("It's  a   Test.", "en") == "it's a test"


# 아라비아 숫자는 지우지 않는다. 숫자 표기 차이는 정규화가 아니라 맞춘 CER(itn.harmonize)에서 따로 다룬다.
def test_digits_are_kept():
    assert normalize("2020년 3월", "ko") == "2020년3월"
