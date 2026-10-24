"""영어 보조 지표의 표준 정규화(en_norm.normalize_english)와 함께 둔 사전 사본을 고정한다."""

# [테스트 묶음] en_norm.normalize_english(Whisper 표준 영어 정규화, docs/experiments.md '보조 지표' 절)가
# 숫자 표기·기호·영미 철자 차이를 같은 글로 맞추는지와, 함께 둔 사전 사본(english.json)이 규칙에 적은 그
# 파일인지 확인한다.
# [왜 중요한가] 보조 지표 en-whisper의 값은 이 정규화와 사전으로 정해진다. transformers 버전이 바뀌어 정규화
# 결과가 달라지거나 사전 파일이 바뀌면 여기서 드러나야 한다. 커밋된 리포트의 보조 지표 총합은
# tests/test_analyze.py가 따로 고정한다.

import hashlib

import pytest

from whisper_ko_ft import en_norm
from whisper_ko_ft.en_norm import ENGLISH_JSON, ENGLISH_JSON_SHA256, normalize_english
from whisper_ko_ft.text_norm import normalize

# 등록 정규화(text_norm.normalize)에서는 다른 글로 남고 표준 정규화에서는 같은 글이 되는 짝. 한국어로 학습한
# 모델이 영어 숫자를 풀어 쓰거나('seven wickets') 기호를 낱말로 적는('100 percent') 표기 변화의 예다.
PAIRS = [
    ("7 wickets", "seven wickets"),
    ("10,000 years", "ten thousand years"),
    ("100%", "100 percent"),
    ("behaviour", "behavior"),
]


# 네 짝 모두 표준 정규화로는 같은 글, 등록 정규화로는 다른 글이다(등록 지표는 그대로 표기 차이를 센다).
@pytest.mark.parametrize(("a", "b"), PAIRS)
def test_standard_normalization_matches_numbers_symbols_and_spellings(a, b):
    assert normalize_english(a) == normalize_english(b)
    assert normalize(a, "en") != normalize(b, "en")


# 정규화한 글의 모양을 몇 가지 고정한다: 소문자, 문장부호 제거, 숫자는 아라비아 숫자로, 'percent'는 '%'로,
# 영국 철자는 미국 철자로, 축약형은 펴고(it's → it is), 군말(hmm)만 있으면 빈 글이 된다.
def test_examples_of_the_normalized_form():
    assert (
        normalize_english("He took seven wickets in ten thousand years, one hundred percent behaviour!")
        == "he took 7 wickets in 10000 years 100% behavior"
    )
    assert normalize_english("It's a Test.") == "it is a test"
    assert normalize_english("Hmm.") == ""


# 사전 사본이 규칙에 적은 파일(SHA-256)과 같고, 영미 철자 1,740쌍을 담고 있다.
def test_dictionary_file_is_the_registered_copy():
    digest = hashlib.sha256(ENGLISH_JSON.read_bytes()).hexdigest()
    assert digest == ENGLISH_JSON_SHA256 == en_norm.english_json_sha256()
    mapping = en_norm.load_spelling_mapping()
    assert len(mapping) == 1740
    assert mapping["behaviour"] == "behavior"


# 다른 내용의 사전은 거부한다. 사전을 바꾸면 보조 지표가 조용히 달라지므로, 바꿀 때는 기록된 해시와 규칙을
# 함께 고치게 한다.
def test_a_changed_dictionary_is_refused(tmp_path):
    other = tmp_path / "english.json"
    other.write_text('{"colour": "color"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256"):
        en_norm.load_spelling_mapping(other)
