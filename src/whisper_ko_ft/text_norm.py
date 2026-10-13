"""Text normalization for error rates (rules in docs/experiments.md, "지표")."""

# [파일 역할] 오류율(CER·WER)을 재기 전에 정답과 가설을 같은 형태로 맞추는 정규화 함수다. metrics.score가 모든
# 채점에서 정답·가설 양쪽에 이 함수를 적용한다.
# [왜 필요한가] 대소문자, 문장부호, 띄어쓰기 차이는 인식 오류가 아니라 표기 차이다. 이런 차이까지 오류로 세면
# 모델 비교가 흐려지므로, docs/experiments.md 공통 규칙 '지표'에 미리 적은 정규화를 한곳에서 적용한다.
# [주의] 이 규칙을 바꾸면 커밋된 리포트의 모든 수치가 달라진다. tests/test_recorded_reports.py가 커밋된
# 리포트를 지금 코드로 다시 채점해 이런 변경을 잡는다. 일부러 바꿀 때는 그 테스트의 고정값을 고치고 이유를
# docs/experiments.md에 남긴다(그 테스트의 docstring).

import re

# _PUNCTUATION: 공백으로 바꿀 문자. [^\w\s']는 글자·숫자·공백·작은따옴표가 아닌 모든 것이고, 뒤의 |_는 \w에
# 포함되는 밑줄(_)도 문장부호로 다루려고 따로 더한 것이다. 모듈을 불러올 때 한 번만 컴파일한다.
# Anything that isn't a letter, digit, whitespace or apostrophe. `\w` covers Hangul.
_PUNCTUATION = re.compile(r"[^\w\s']|_")


# 한 문장을 채점용 형태로 바꾼다. 입력: 원문과 언어 코드('ko' 또는 'en'). 출력: 정규화된 문자열.
# 호출 위치: metrics.score. 같은 함수를 정답과 가설 양쪽에 적용해야 표기 차이가 상쇄된다.
def normalize(text: str, language: str) -> str:
    """Lowercase and drop punctuation; Korean also drops every space and apostrophe.

    Korean CER is measured on characters without spaces, so a spacing difference
    ("할 수" vs "할수") doesn't count as an error. Apostrophes are kept only for English
    contractions ("it's"); in Korean they are quotation marks and count as punctuation.
    English keeps single spaces for WER.
    """
    # 1) 소문자로 바꾸고 문장부호를 공백으로 바꾼다. 지우지 않고 공백으로 바꾸므로 문장부호로만 나뉜 영어
    #    단어가 하나로 붙지 않는다.
    text = _PUNCTUATION.sub(" ", text.lower())
    # 2) 한국어: 작은따옴표도 공백으로 바꾼 뒤 공백을 모두 없애 글자열 하나로 만든다. CER은 이 글자열을 한
    #    글자씩 비교하므로 띄어쓰기 차이는 오류로 세지 않는다.
    if language == "ko":
        return "".join(text.replace("'", " ").split())
    # 3) 영어: 연속된 공백을 하나로 줄여 WER의 단어 경계를 일정하게 한다. 작은따옴표는 남겨 축약형(it's)이 한
    #    단어로 유지된다.
    return " ".join(text.split())
