"""영어 보조 지표(판정 없음)용 표준 정규화. 규칙은 docs/experiments.md '보조 지표' 절."""

# [파일 역할] FLEURS 영어 WER의 보조 지표 en-whisper(`analyze --scoring en-whisper`)에 쓰는 정규화다.
# transformers의 EnglishTextNormalizer(openai/whisper의 영어 정규화 normalizers/english.py를 옮긴 것)에, 그
# 정규화가 요구하는 영미 철자 사전 사본 english.json(같은 폴더)을 넘겨 만든다. 등록 지표의
# 정규화(text_norm.normalize)와 커밋된 리포트의 수치는 건드리지 않는다.
# [왜 필요한가] 등록 정규화는 소문자·문장부호·공백만 맞추고 숫자 표기('7'과 'seven'), '%'와 'percent', 영미
# 철자('behaviour'와 'behavior')는 맞추지 않는다. 한국어로 학습한 모델이 영어 숫자를 풀어 쓰게 된 표기 변화가
# 등록 지표의 영어 WER 차이에 들어가므로, 그 몫을 걷어 낸 값을 보조 지표로 함께 본다. 판정에는 쓰지 않는다.
# [사전 사본의 출처] english.json은 Hugging Face `openai/whisper-small` 저장소가 `normalizer.json`으로
# 배포하는 영미 철자 사전(영국 철자 → 미국 철자 1,740쌍)을 2026-10-24에 그대로 복사한 것이다. transformers의
# WhisperTokenizer가 바로 이 파일을 읽어 EnglishTextNormalizer에 넘기고(tokenization_whisper.py의
# `english_spelling_normalizer`), 같은 파일이 `openai/whisper-large-v3-turbo` 저장소에도 바이트까지 같게 들어
# 있다. 아래 ENGLISH_JSON_SHA256이 복사한 파일의 SHA-256이다.
# GitHub openai/whisper와의 대조(2026-10-29): 내용은 그 저장소 첫 커밋 6e3be77(2022-09-21)의
# `whisper/normalizers/english.json` 1,740쌍과 같다. 바이트는 키를 정렬하고 2칸으로 들여 다시 쓴 꼴이라
# 다르다(그 파일을 json.dumps(..., indent=2, sort_keys=True)로 쓰고 줄바꿈 하나를 붙이면 이 사본과 바이트까지
# 같다). 지금 openai/whisper main은 커밋 ea1c266(2023-01-18, '20mm'의 mm이 hmm으로 바뀌던 버그를 고침)에서
# "mm" → "hmm" 한 쌍을 뺀 1,739쌍이다. 군말로 쓴 'mm'은 EnglishTextNormalizer가 사전을 적용하기 전에 지우므로
# 이 쌍은 '35mm'처럼 숫자에서 떨어져 나온 'mm'에만 걸리고, 정답과 가설에 같게 걸린다. 커밋된 FLEURS 영어
# 리포트에서는 두 사전의 발화별 편집 수·단어 수가 모두 같다(docs/experiments.md '보조 지표' 절의 2026-10-29
# 덧붙임).
# [라이선스] openai/whisper의 정규화 코드와 사전은 MIT 라이선스다. 사본에 붙이는 고지:
#   MIT License. Copyright (c) 2022 OpenAI
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#   associated documentation files (the "Software"), to deal in the Software without restriction, including
#   without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions: The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
#   KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#   PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
#   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# [주의] transformers 버전이 바뀌면 EnglishTextNormalizer의 동작이 바뀔 수 있다(규칙은 5.17.0으로 적었다).
# tests/test_en_norm.py의 예시와 tests/test_analyze.py의 커밋된 리포트 총합 고정이 그 변화를 잡는다. 사전
# 파일이 기록된 해시와 다르면 불러오기를 거부해, 보조 지표가 조용히 다른 값이 되지 않게 한다.

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

# EnglishTextNormalizer: 소문자로 바꾸고, 괄호 안과 군말(hmm, uh 등)을 지우고, 축약형을 펴고(it's → it is,
# won't → will not), 숫자 낱말을 아라비아 숫자로(seven → 7, ten thousand → 10000), 'percent'를 '%'로,
# 통화 낱말을 기호로 바꾸고, 기호·문장부호를 공백으로 바꾼 뒤 영미 철자 사전으로 철자를 맞춘다. 정답과 가설
# 양쪽에 같이 적용해야 표기 차이가 상쇄된다.
# 이 클래스는 normalizer()가 처음 불릴 때 불러온다. 이 모듈 머리에서 불러오면 transformers 전체를 불러오게
# 되어(이 PC에서 약 2초), 이 모듈을 가져다 쓰는 analyze의 모든 하위 명령이 en-whisper 채점을 쓰지 않을 때도
# 그만큼 늦게 시작했다. 여기서는 타입 표기에 쓰는 이름만 가져온다(실행 중에는 불러오지 않는다).
if TYPE_CHECKING:
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# ENGLISH_JSON: 영미 철자 사전 사본(위 '사전 사본의 출처'). ENGLISH_JSON_SHA256: 그 파일의 SHA-256. 규칙
# (docs/experiments.md '보조 지표')에 같은 값을 적었다.
ENGLISH_JSON = Path(__file__).with_name("english.json")
ENGLISH_JSON_SHA256 = "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"


def english_json_sha256(path: Path = ENGLISH_JSON) -> str:
    """사전 파일의 SHA-256(16진수 64자). 규칙에 적은 값과 같은지 확인하는 데 쓴다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 사전을 읽는다. 기록된 해시와 다른 파일은 거부한다: 사전이 바뀌면 보조 지표가 달라지므로, 바꿀 때는 이 상수와
# 규칙(docs/experiments.md)을 함께 고치게 한다. 호출 위치: normalizer, tests/test_en_norm.py.
def load_spelling_mapping(path: Path = ENGLISH_JSON) -> dict[str, str]:
    """영미 철자 사전(영국 철자 → 미국 철자)을 읽는다. 기록된 SHA-256과 다른 파일은 거부한다."""
    digest = english_json_sha256(path)
    if digest != ENGLISH_JSON_SHA256:
        raise SystemExit(
            f"{path}: SHA-256 {digest} is not the registered {ENGLISH_JSON_SHA256}; "
            "the en-whisper scoring is defined with the registered dictionary"
        )
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
    ):
        raise SystemExit(f"{path}: expected a JSON object mapping words to words")
    return mapping


# 정규화기는 사전을 읽어 한 번만 만든다(발화마다 52KB JSON을 다시 읽지 않게). transformers도 이때 처음
# 불러온다(위 import 주석).
@lru_cache(maxsize=1)
def normalizer() -> EnglishTextNormalizer:
    """사전 사본으로 만든 EnglishTextNormalizer 하나. 처음 부를 때 만들고 그 뒤로는 재사용한다."""
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer(load_spelling_mapping())


# 호출 위치: analyze의 en-whisper 채점(정답과 가설 양쪽), tests/test_en_norm.py.
def normalize_english(text: str) -> str:
    """영어 문장을 Whisper 표준 영어 정규화로 바꾼다. 군말만 있는 문장은 빈 문자열이 된다."""
    return normalizer()(text)
