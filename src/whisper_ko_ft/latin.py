"""Rewrite Latin letters spelled with their Korean names as the letters ("에이아이" -> "AI").

The training references (Zeroth-Korean) have no Latin letters, so the fine-tuned models write an initialism as
the names of its letters ("유에스오씨"). A run of tokens is rewritten only when every token is made of letter
names (a particle may follow the last one), the run has at least two letters, and at least one name in it is
not also an everyday syllable or word: "비디오", "오디오", "이유", "티비" are left alone. A rule, not a list
of known abbreviations; it misses initialisms made only of such names. docs/experiments.md, stage 8.
"""

# [파일 역할] 8단계의 보조 후보(판정 없음)인 '영문 글자 이름 변환'. 한글로 적힌 영문 약어('에이아이', '에프
# 에이에이의')를 라틴 글자('AI', 'FAA의')로 바꾼다.
# [왜 필요한가] 학습 정답(Zeroth-Korean)에 라틴 글자가 없어서, 파인튜닝한 모델은 약어를 글자 이름으로
# 적는다(위 docstring). FLEURS 정답은 영문으로 적으므로 이 차이가 오류로 잡힌다. 이 표기 몫이 얼마인지 보려는
# 보조 지표다.
# [설계 이유] 알려진 약어 사전을 쓰지 않고 규칙으로 했다. 예시 약어(USOC, FAA, AI)가 FLEURS 한국어 test를
# 분석해서 나온 것이라, 그것으로 사전을 만들면 test에 맞춘 지표가 되기 때문이다. 규칙은 알파벳의 한글 이름표와
# 흔한 낱말 목록만으로 정했다(docs/experiments.md 8단계 보조 후보).
# [쓰이는 곳] analyze의 --scoring itn-latin(가설에만, 숫자 변환 뒤)과 agnostic(정답·가설 양쪽, 표기 무관 CER).

from __future__ import annotations

import re

# LETTER_NAMES: 영문 알파벳 26자의 한글 이름 → 라틴 대문자. 닫는 괄호 옆의 포맷 생략 지시는 한 줄에 여러
# 항목을 두려고 붙인 것이다.
LETTER_NAMES = {
    "에이": "A", "비": "B", "씨": "C", "디": "D", "이": "E", "에프": "F", "지": "G", "에이치": "H",
    "아이": "I", "제이": "J", "케이": "K", "엘": "L", "엠": "M", "엔": "N", "오": "O", "피": "P",
    "큐": "Q", "알": "R", "에스": "S", "티": "T", "유": "U", "브이": "V", "더블유": "W", "엑스": "X",
    "와이": "Y", "제트": "Z",
}  # fmt: skip
# COMMON_NAMES: 흔한 음절·낱말과 겹치는 이름. 이것들만으로 된 묶음은 바꾸지 않아 '비디오'(비·디·오),
# '이유'(이·유), '티비'(티·비)가 그대로 남는다. 대신 이런 이름만으로 된 약어는 놓친다.
# Names that are also everyday syllables or words; a run of only these is not rewritten.
COMMON_NAMES = {"비", "씨", "디", "이", "지", "아이", "오", "피", "티", "유", "알", "엔"}
# PARTICLES: 묶음의 마지막 토큰 뒤에 붙을 수 있는 조사. 앞에서부터 처음 맞는 것을 쓰므로 긴 것('이라는')을
# 짧은 것('이')보다 먼저 둔다.
# Particles that may follow the last token of a run, longest first.
PARTICLES = (
    "이라는", "에서", "으로", "라는", "의", "는", "은", "가", "를",
    "을", "와", "과", "에", "로", "도", "만", "이",
)  # fmt: skip
# _LONGEST: 가장 긴 글자 이름의 글자 수(3, '에이치'·'더블유'). _spell이 이보다 긴 조각은 보지 않는다.
# _TRAILING: 토큰을 본문과 끝에 붙은 문장부호(\W*)로 나눈다.
_LONGEST = max(len(name) for name in LETTER_NAMES)
_TRAILING = re.compile(r"^(.*?)(\W*)$")


# word 전체를 글자 이름의 연결로 읽을 수 있으면 그 이름 목록을, 아니면 None을 돌려준다. 여러 방법으로 나뉘면
# 이름 수가 가장 적은 쪽을 고른다.
# 방법: 동적 계획법. best[i]는 word의 앞 i글자를 이루는 가장 짧은 이름 목록(못 만들면 None)이다. 끝 위치마다
# 길이 _LONGEST 이하의 마지막 조각을 모두 시도한다.
def _spell(word: str) -> list[str] | None:
    """The fewest letter names that make up the whole word, or None."""
    best: list[list[str] | None] = [[]] + [None] * len(word)
    for end in range(1, len(word) + 1):
        for start in range(max(0, end - _LONGEST), end):
            before = best[start]
            if before is None or word[start:end] not in LETTER_NAMES:
                continue
            if best[end] is None or len(before) + 1 < len(best[end]):
                best[end] = [*before, word[start:end]]
    # 빈 단어는 빈 목록 대신 None이 된다.
    return best[len(word)] or None


# 토큰 하나를 글자 이름으로 읽는다. 돌려주는 값: (이름 목록, 이름 뒤에 남는 꼬리(조사·문장부호), 이 토큰에서
# 묶음이 끝나는지). 읽을 수 없으면 None.
# 먼저 문장부호를 뺀 본문 전체를 읽어 보고, 안 되면 끝의 조사를 떼고 다시 읽는다. 조사나 문장부호가 붙은
# 토큰은 묶음의 마지막이 된다.
def _read(token: str) -> tuple[list[str], str, bool] | None:
    """(letter names, what follows them, whether a particle ended the run) of a spelled token, or None."""
    word, tail = _TRAILING.match(token).groups()
    names = _spell(word)
    if names:
        return names, tail, bool(tail)
    for particle in PARTICLES:
        if word.endswith(particle) and len(word) > len(particle):
            names = _spell(word[: -len(particle)])
            if names:
                return names, particle + tail, True
    return None


# 문장에서 글자 이름으로만 된 연속 토큰 묶음을 찾아 라틴 글자로 바꾼다('에프 에이에이의' → 'FAA의'). 글자가 둘
# 이상이고 COMMON_NAMES가 아닌 이름이 하나 이상 있을 때만 바꾼다.
# 호출 위치: analyze의 itn-latin·agnostic 채점.
def letters_to_latin(text: str) -> str:
    tokens = text.split()
    out: list[str] = []
    index = 0
    while index < len(tokens):
        # index에서 시작해 글자 이름으로 읽히는 토큰을 이어 붙인다. 읽히지 않는 토큰이 나오거나
        # 조사·문장부호로 묶음이 닫히면 멈춘다.
        names: list[str] = []
        tail, end = "", index
        while end < len(tokens):
            read = _read(tokens[end])
            if read is None:
                break
            names += read[0]
            tail = read[1]
            end += 1
            if read[2]:  # a particle or punctuation closes the run
                break
        # 바꿀 조건을 만족하면 묶음 전체를 '라틴 글자 + 꼬리' 한 토큰으로 내보내고 묶음 뒤로 건너뛴다.
        if len(names) >= 2 and any(name not in COMMON_NAMES for name in names):
            out.append("".join(LETTER_NAMES[name] for name in names) + tail)
            index = end
        else:
            # 아니면 첫 토큰만 그대로 내보내고 한 칸 나아가, 다음 토큰에서 시작하는 묶음을 다시 찾는다.
            out.append(tokens[index])
            index += 1
    return " ".join(out)
