"""리포트에서 표를 만든다: 숫자 발화, 구간을 붙인 오류율, 짝지은 비교.

사용법:
    uv run python -m whisper_ko_ft.analyze digits --set zeroth-val
    uv run python -m whisper_ko_ft.analyze table --set zeroth-val --reports whisper-small small-a \
        --baseline whisper-small

    uv run python -m whisper_ko_ft.analyze verdict --baseline whisper-small --candidate small-a

`digits`는 묶음의 "숫자 발화" 목록을 확정한다(docs/experiments.md "숫자 표기"): 기본 모델 둘 중 하나라도
아라비아 숫자를 쓴 발화다. 한 번 쓰고 --force 없이는 덮어쓰지 않는다.
`verdict`는 2·3단계 판정 규칙을 validation 리포트에 적용한다(3단계의 구간 조건은 --no-interval-rule이
없으면 켜진다).
`verdict6`은 6단계 규칙을 적용한다. `--harmonized`는 저장된 정답과 가설에 `itn.harmonize`를 적용한 뒤 다시
채점해, 숫자를 어느 표기로 적든 같은 점수를 받게 한다(docs/experiments.md 6단계).
`loops`는 리포트마다 편집 수가 정답 글자 수보다 많은 발화(되풀이 오류)를 보여 준다.

8단계: `--scoring`은 저장된 글을 다른 방식으로 다시 채점한다(SCORINGS. 리포트는 절대 바꾸지 않는다).
`changes`는 재시도 디코딩이 기록된 greedy 리포트에 비해 무엇을 바꿨는지 보여 주고, `verdict8`은 디코딩 규칙을,
`itn8`은 출력 후처리의 규칙을 적용한다.

9단계(다른 기준 모델 Qwen3-ASR): `gate9`는 zeroth-val500 앞 100발화에서 잰 두 실행으로 fp16과 fp32 중
무엇을 쓸지 정하고, `verdict9`는 다른 도메인 규칙을 적용해 그와 함께 보고할 것을 찍는다.

10단계(Qwen3-ASR에 LoRA): `gate10`은 새 어댑터(B=0)가 9단계 fp16 점검을 재현하는지 확인하고, `smoke10`은
스모크 실행의 스텝 기록과 GPU 메모리 표본을 관문에 비춰 읽고, `pick10`은 zeroth-val500의 맞춘 CER로
체크포인트를 고르고, `verdict10`은 N에 대한 규칙을 적용한다.

11단계(QN 어댑터를 줄여 합치기): `gate11`은 스케일 1.0과 0으로 합친 실행이 기록된 파일과 발화 단위로 같은지
보고, 판정은 10단계와 같은 `verdict10`이다. `blocks`는 비교 하나의 블록 단위 구간을 찍는다(보고만).
12단계(N을 CTranslate2 int8_float16으로): `verdict12`는 정확도 조건과 속도 파일(scripts/speed.py의
speed.json)의 메모리·지연 조건을 적용한다.

민감도 분석(2026-10-15): `robust`는 2~10단계의 판정·선택에 쓴 validation 구간 25개를 화자(Zeroth)·문장(FLEURS)
단위 부트스트랩으로 다시 계산해, 블록 단위에서도 조건이 기록과 같은지 찍는다. 판정은 바꾸지 않는다.

문장 겹침(2026-10-24): `overlap`은 학습 묶음(zeroth-train)과 평가 묶음들이 같은 문장을 얼마나 함께 갖는지
(원문 기준과 채점 정규화 기준), 화자가 겹치는지 센다. `--write`는 발화 ID → 학습 낭독 횟수와 요약 수를
`reports/sentence_overlap/zeroth.json`에 쓴다(정답 글은 넣지 않는다). 다른 하위 명령과 달리 리포트가 아니라
`prepare`가 만든 묶음 목록(data/cache/sets)을 읽는다. 학습 문장은 리포트에 없기 때문이다.

    uv run python -m whisper_ko_ft.analyze overlap --train zeroth-train \
        --sets zeroth-val zeroth-val500 zeroth-test fleurs-ko-val fleurs-ko-test --write

영어 보조 지표(2026-10-24, 판정 없음): `--scoring en-whisper`는 FLEURS 영어 묶음의 WER을 Whisper 표준 영어
정규화(en_norm.normalize_english)로 정답·가설을 바꾼 뒤 다시 센다. 숫자 표기·'%'·영미 철자 차이를 걷어 낸
값이고, 등록 지표와 그 판정은 그대로다. 채점마다 쓸 수 있는 묶음의 언어가 정해져 있어(한국어 채점은
Zeroth·FLEURS 한국어, en-whisper는 FLEURS 영어) 다른 언어의 묶음에 주면 거부한다(docs/experiments.md
'보조 지표').

    uv run python -m whisper_ko_ft.analyze table --set fleurs-en-val \
        --reports whisper-large-v3-turbo turbo-n --baseline whisper-large-v3-turbo \
        --no-digit-split --scoring en-whisper

표의 부트스트랩 표본 수(2026-10-29): `table --bootstrap-samples N`은 구간을 N번(seed 0) 재표집해 계산하고,
`--decimals`는 %·%p 값의 소수 자릿수를 바꾼다. 기록된 구간과 판정은 모두 기본값(1,000번, 두 자리)이다. 판정
문턱에 가까운 구간 끝이 표본 수에 따라 얼마나 움직이는지 사후에 볼 때 쓴다(docs/experiments.md 12단계 결과의
2026-10-29 덧붙임).

    uv run python -m whisper_ko_ft.analyze table --set zeroth-val --reports turbo-n-ct2 turbo-n-ct2-int8 \
        --baseline turbo-n-ct2 --harmonized --bootstrap-samples 100000 --decimals 3
"""

# [파일 역할] 커밋된 리포트(reports/)만 읽어 표와 판정을 내는 분석 도구. 모델을 다시 돌리지 않는다. 예외는
# overlap 하나다: 학습 묶음의 문장은 리포트에 없으므로 overlap만 prepare가 만든 묶음 목록(data/cache/sets)을
# 읽고, 그 결과를 파생 파일(reports/sentence_overlap/zeroth.json)로 남겨 데이터가 없는 곳(CI)에서도 수를
# 확인할 수 있게 한다.
# docs/experiments.md의 수치와 판정 문장은 대부분 이 파일의 하위 명령 출력이다.
# [왜 필요한가] 이 프로젝트는 판정 규칙을 재기 전에 적고 커밋한다. 규칙을 코드(상수와 verdict 함수)로 옮겨
# 두면, 결과를 본 뒤 기준을 슬쩍 바꾸거나 손으로 계산하다 틀리는 일을 막고, 누구든 같은 리포트에서 같은 판정을
# 다시 낼 수 있다.
# [주요 구성]
# - 채점 방식: 원래 채점, 맞춘 CER(harmonized, 6단계), 출력 숫자 변환(itn, 8단계), 글자 이름 변환(itn-latin),
#   표기 무관(agnostic), 영어 표준 정규화(en-whisper, 2026-10-24 보조 지표, FLEURS 영어 묶음만). 리포트의 글은
#   바꾸지 않고 채점할 때만 적용한다(SCORINGS. 채점마다 쓸 수 있는 묶음의 언어는 SCORING_LANGUAGE).
# - 공통 도구: load_rows(리포트 읽기와 검사), compare(두 리포트의 짝지은 차이와 구간), table(표), loops(되풀이
#   발화), changes(재시도가 바꾼 것).
# - 단계별 판정: verdict(2·3단계), verdict6, verdict8·itn8, gate9·verdict9, gate10·smoke10·pick10·verdict10.
# - 민감도 분석: robust(기록된 판정 구간을 화자·문장 단위 부트스트랩으로 다시 계산, 2026-10-15).
# - 문장 겹침: overlap(학습 묶음과 평가 묶음의 문장·화자 겹침, 2026-10-24. 셈은 splits.sentence_overlap).
# [연결] metrics(채점·구간), itn·latin(표기 변환), en_norm(영어 표준 정규화), paths.REPORTS를 쓴다. overlap만
# splits·store.load_set과 paths.CACHE를 더 쓴다. tests/test_recorded_reports.py가 load_rows·arrays로 커밋된
# 리포트의 맞춘 CER 총합을 고정하고, tests/test_analyze.py가 문장 겹침 파생 파일의 수와 영어 보조 지표의
# 총합을 고정한다.

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics

# jiwer: en-whisper 채점의 단어 단위 편집 거리(표준 정규화의 출력을 더 손대지 않고 세기 위해 metrics.score를
# 거치지 않는다. word_edits). numpy: 발화별 편집 수·길이 배열.
import jiwer
import numpy as np

# itn·latin: 다시 채점할 때 쓰는 표기 변환(한국어). en_norm: 영어 보조 지표의 표준 정규화. metrics: 채점,
# 오류율, 부트스트랩 구간과 짝지은 차이 구간. paths.REPORTS: 리포트 폴더. paths.CACHE·store.load_set·splits:
# overlap만 쓴다(묶음 목록 읽기와 문장 겹침 셈). CACHE는 load_set에 매번 넘겨, 테스트가 analyze.CACHE를 임시
# 폴더로 바꿀 수 있게 한다.
from whisper_ko_ft.en_norm import normalize_english
from whisper_ko_ft.itn import harmonize, to_digits
from whisper_ko_ft.latin import letters_to_latin
from whisper_ko_ft.metrics import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    block_difference_interval,
    difference_interval,
    error_rate,
    interval,
    leave_one_block_out,
    score,
)
from whisper_ko_ft.paths import CACHE, REPORTS
from whisper_ko_ft.splits import OVERLAP_KEYS, sentence_overlap
from whisper_ko_ft.store import load_set

# BASE_MODELS: '숫자 발화' 목록을 정할 때 보는 기본 모델 두 개(1단계 기준선).
BASE_MODELS = ("whisper-small", "whisper-large-v3-turbo")
# 2·3단계 판정 기준(docs/experiments.md 2단계 '판정 규칙'): 같은 도메인은 Zeroth validation CER이 기준선보다
# 상대 20% 이상 낮고(-0.20), 숫자 발화를 빼고도 상대 10% 이상 낮아야 한다(-0.10). 다른 도메인은 FLEURS 한국어
# validation CER이 절대 1.0%p(0.010)를 넘게 나빠지지 않아야 한다. 이 1.0%p는 6·7단계와 10단계 망각 줄에도
# 그대로 쓰인다.
IN_DOMAIN_RELATIVE = -0.20
IN_DOMAIN_NO_DIGITS_RELATIVE = -0.10
OUT_OF_DOMAIN_MAX_INCREASE = 0.010
# 6단계: N의 맞춘 CER이 3단계 L2보다 0.3%p를 넘게 높으면 같은 도메인 이득을 잃은 것으로 본다.
STAGE6_MAX_ABOVE_L2 = 0.003
# 8단계: 재시도 후보는 어느 오류율도 짝지은 95% 구간 상한 기준 +0.1%p를 넘게 올리면 안 되고, 출력 숫자
# 변환은 Zeroth 맞춘 CER을 +0.05%p(STAGE8_ITN_MAX_INCREASE)를 넘게 올리면 안 된다(docs/experiments.md 8단계
# 판정 규칙).
STAGE8_MAX_INCREASE = 0.001
STAGE8_ITN_MAX_INCREASE = 0.0005
# 8단계 디코딩 판정에 쓰는 validation 세 묶음과, 각 묶음의 되풀이와 오류율을 셀 채점: Zeroth는 맞춘 채점,
# FLEURS 한국어·영어는 원래 채점. 9·10단계의 되풀이 보고도 같은 묶음을 쓴다.
STAGE8_SETS = {"zeroth-val": "harmonized", "fleurs-ko-val": None, "fleurs-en-val": None}
# 9단계 fp16 점검: zeroth-val500 앞 100발화에서 fp16의 원래 CER이 fp32 실행과 어느 방향으로든 ±0.20%p
# 안이고, 빈 가설과 로짓이 유한하지 않은(넘친) 발화가 없어야 fp16으로 잰다(docs/experiments.md 9단계 'fp16
# 점검').
STAGE9_FP16_MAX_GAP = 0.002
# 10단계 스모크 관문(docs/experiments.md 10단계 '관문' (c)): 앞 20스텝(GradScaler가 배율을 맞추는 구간)은
# 건너뜀을 세지 않고 그 뒤 5개까지 허용, 학습 프로세스의 공유 GPU 메모리 512MiB 이하, nvidia-smi 카드 전체
# 10,752MiB 미만, torch 최대 할당 7,168MiB 이하. 스텝 시간 중앙값 × 1,500이 8시간을 넘으면 1,000스텝으로
# 줄인다.
# 공유 메모리는 Windows "GPU Process Memory" 카운터의 학습 프로세스 공유 사용량이고, 카드 전체는 nvidia-smi의
# memory.used다. 공유 메모리 기준이 0이 아니라 512MiB인 이유: 9단계 디코딩 프로세스는 3.4GB로 넘치기 전까지도
# 이 카운터가 약 230MB에 머물렀다.
STAGE10_SKIP_GRACE = 20
STAGE10_MAX_SKIPS = 5
STAGE10_MAX_SHARED_MB = 512
STAGE10_MAX_GPU_MB = 10_752  # 10.5 GiB(nvidia-smi가 세는 MiB 단위)
STAGE10_MAX_ALLOCATED_MB = 7_168  # 7 GiB
STAGE10_HOURS = 8.0
STAGE10_STEPS, STAGE10_FEWER_STEPS = 1_500, 1_000
# 12단계 판정(docs/experiments.md 12단계): 정확도는 Zeroth 맞춘 CER에서 int8 − fp16 상한 ≤ +0.10%p(8단계의
# '정확도를 잃지 않는다'와 같은 값), 서빙 이득은 nvidia-smi 증가가 20% 넘게 작거나 한 발화 지연 중앙값이 두 번
# 모두 5% 넘게 짧은 것.
STAGE12_MAX_INCREASE = 0.001
STAGE12_MEMORY_CUT = 0.20
STAGE12_LATENCY_CUT = 0.05
# 10단계 판정: 조건 1은 Zeroth 맞춘 CER에서 QN − N의 짝지은 95% 구간 상한 < +0.10%p. 망각 보고 줄은 FLEURS
# 한국어에서 QN − 학습 전 Qwen 상한 ≤ +1.0%p(판정은 바꾸지 않는 보고용).
# _DIGIT: 정답에 아라비아 숫자가 있는지(FLEURS를 숫자 유무로 나눌 때).
STAGE10_ZEROTH_MAX_ABOVE_N = 0.001
STAGE10_FORGETTING_MAX = 0.010
_DIGIT = re.compile(r"[0-9]")


# 채점 방식 여섯 가지(한국어 다섯, 영어 하나). 모두 (정답, 가설)을 받아 바꾼 (정답, 가설)을 돌려주고, 저장된
# 리포트는 바꾸지 않는다.
def _harmonized(reference: str, hypothesis: str) -> tuple[str, str]:
    """harmonized(6단계 맞춘 CER): 정답과 가설 양쪽에 같은 숫자 표기 통일을 적용한다.

    그래서 '오 월'과 '5월'이 같은 점수를 받는다.
    """
    return harmonize(reference), harmonize(hypothesis)


def _itn(reference: str, hypothesis: str) -> tuple[str, str]:
    """itn(8단계 출력 후처리): 가설의 풀어 쓴 숫자만 아라비아 숫자로 바꾸고 정답은 그대로 둔다.

    서비스에서 출력에 후처리를 붙였을 때의 원래 CER이다.
    """
    return reference, to_digits(hypothesis)


def _itn_harmonized(reference: str, hypothesis: str) -> tuple[str, str]:
    """itn-harmonized: 후처리한 가설을 6단계 채점(맞춘 CER)으로 다시 잰다.

    후처리가 맞춘 채점에서 다른 것을 바꾸지 않는지 보는 데 쓴다(itn8의 조건 2).
    """
    return _harmonized(*_itn(reference, hypothesis))


def _itn_latin(reference: str, hypothesis: str) -> tuple[str, str]:
    """itn-latin(8단계, 판정 없음): 숫자 후처리에 더해 가설의 한글 글자 이름을 라틴 글자로 바꾼다."""
    return reference, letters_to_latin(to_digits(hypothesis))


def _agnostic(reference: str, hypothesis: str) -> tuple[str, str]:
    """agnostic(8단계 판정 없는 보조 지표, 표기 무관 CER): 양쪽의 숫자와 글자 이름을 같은 표기로 맞춘다."""
    return letters_to_latin(harmonize(reference)), letters_to_latin(harmonize(hypothesis))


def _en_whisper(reference: str, hypothesis: str) -> tuple[str, str]:
    """en-whisper(2026-10-24 판정 없는 보조 지표, 영어 WER): 양쪽에 Whisper 표준 영어 정규화를 적용한다.

    숫자 낱말과 아라비아 숫자, 'percent'와 '%', 영미 철자가 같은 글이 되어 그 표기 차이는 오류로 세지
    않는다(docs/experiments.md '보조 지표'). FLEURS 영어 묶음에만 쓴다.
    """
    return normalize_english(reference), normalize_english(hypothesis)


# SCORINGS: --scoring 이름 → 변환 함수. 오류율을 다시 계산하기 전에 저장된 정답과 가설을 어떻게 바꿀지 정한다.
# SCORING_LANGUAGE: 채점마다 쓸 수 있는 묶음의 언어. 적지 않은 채점은 한국어 CER용이다. load_rows가 묶음의
# 언어(set_language)와 다르면 거부한다.
# FINAL_SCORINGS: 변환 결과가 이미 채점용 글이라 등록 정규화(text_norm.normalize)를 덧씌우지 않고 그대로 세는
# 채점. en-whisper가 그렇다: Whisper의 WER 절차는 표준 정규화의 출력을 바로 단어로 나눠 세고, 등록 정규화를 더
# 적용하면 숫자에 붙은 '%'·'$'가 지워지고 소수점이 단어 경계가 되어 표준과 다른 값이 된다. 한국어 채점들은
# 변환 뒤 등록 정규화를 거친다(기존 동작 그대로).
SCORINGS = {
    "harmonized": _harmonized,
    "itn": _itn,
    "itn-harmonized": _itn_harmonized,
    "itn-latin": _itn_latin,
    "agnostic": _agnostic,
    "en-whisper": _en_whisper,
}
SCORING_LANGUAGE = {"en-whisper": "en"}
FINAL_SCORINGS = {"en-whisper"}


def scoring_language(scoring: str) -> str:
    """채점 방식이 전제하는 언어 코드: en-whisper는 'en', 나머지는 'ko'."""
    return SCORING_LANGUAGE.get(scoring, "ko")


def set_language(set_name: str) -> str:
    """묶음 이름의 언어 코드: fleurs-en-*('@telephone' 판 포함)은 'en', Zeroth·FLEURS 한국어는 'ko'."""
    return "en" if set_name.startswith("fleurs-en") else "ko"


# 리포트 하나를 읽어 발화 ID → 행 dict로 돌려준다. 정답이 빈 행(길이 0)은 뺀다. scoring을 주면 행마다 다시
# 채점한 편집 수·길이로 바꾼다(harmonized=True는 scoring='harmonized'와 같다). 다시 채점해 정답이 비는
# 행(en-whisper가 군말만 있는 정답을 지운 경우)도 같은 이유로 뺀다.
# 거부하는 경우(규칙을 지키게 하는 검사): 채점의 언어와 묶음의 언어가 다르면(한국어 채점을 영어 묶음에,
# en-whisper를 한국어 묶음에), --limit으로 일부만 잰 리포트(커밋 a29a975), 발화 ID가 겹치는 리포트(1단계에서
# FLEURS 행이 빠진 오류를 되풀이하지 않게).
# 호출 위치: table, compare, loops, changes, write_digits, tests/test_recorded_reports.py.
def load_rows(
    report: str, set_name: str, harmonized: bool = False, scoring: str | None = None
) -> dict[str, dict]:
    """리포트의 채점된 발화를 ID별로 돌려준다. `harmonized`는 `scoring="harmonized"`와 같다."""
    scoring = "harmonized" if harmonized else scoring
    language = set_language(set_name)
    if scoring and scoring_language(scoring) != language:
        wanted = "Korean" if scoring_language(scoring) == "ko" else "English"
        raise SystemExit(f"{set_name}: the {scoring!r} scoring is for {wanted} sets")
    data = json.loads((REPORTS / report / f"{set_name}.json").read_text(encoding="utf-8"))
    if data["summary"].get("limit"):
        raise SystemExit(f"{report}/{set_name}: measured with --limit, not the whole set; measure it again")
    rows = {row["id"]: row for row in data["utterances"] if row["length"]}
    if len(rows) != sum(1 for row in data["utterances"] if row["length"]):
        raise SystemExit(f"{report}/{set_name}: utterance ids are not unique; measure it again")
    if scoring:
        rows = {i: rescored(row, scoring, language) for i, row in rows.items()}
        rows = {i: row for i, row in rows.items() if row["length"]}
    return rows


# 이미 채점용 형태인 영어 글의 단어 단위 편집 수와 정답 단어 수. metrics.score와 같은 셈(치환 + 삭제 + 삽입,
# 정답 길이 = 맞음 + 치환 + 삭제)이지만 정규화를 더 하지 않는다. 정답에 단어가 없으면 None(score와 같다).
def word_edits(reference: str, hypothesis: str) -> tuple[int, int] | None:
    """정규화를 더 하지 않고 센 (편집 수, 정답 단어 수). 정답이 비면 None."""
    if not reference.split():
        return None
    out = jiwer.process_words(reference, hypothesis)
    return out.substitutions + out.deletions + out.insertions, out.hits + out.substitutions + out.deletions


def rescored(row: dict, scoring: str = "harmonized", language: str = "ko") -> dict:
    """행 하나를 SCORINGS[scoring]으로 바꾼 글로 그 언어의 오류율(ko는 CER, en은 WER)을 다시 계산한 새 행.

    edits·length만 바뀌고, 저장된 정답·가설 글은 그대로 남는다. FINAL_SCORINGS의 채점은 바꾼 글을 그대로
    단어로 세고, 나머지는 metrics.score(등록 정규화 포함)로 센다. 바꾼 정답이 비면 편집 수와 길이를 0으로 적어
    load_rows가 그 행을 뺄 수 있게 한다.
    """
    reference, hypothesis = SCORINGS[scoring](row["reference"], row["hypothesis"])
    if scoring in FINAL_SCORINGS:
        counted = word_edits(reference, hypothesis)
    else:
        scored = score(reference, hypothesis, language)
        counted = (scored.edits, scored.length) if scored else None
    edits, length = counted or (0, 0)
    return {**row, "edits": edits, "length": length}


def digit_ids(set_name: str) -> set[str]:
    """묶음의 확정된 '숫자 발화' ID 집합(reports/digit_utterances/<묶음>.json).

    목록은 깨끗한 음성에서 정했으므로, "zeroth-val@telephone"처럼 채널이 붙은 묶음도 '@' 앞
    이름("zeroth-val")의 목록을 쓴다.
    """
    path = REPORTS / "digit_utterances" / f"{set_name.split('@')[0]}.json"
    return set(json.loads(path.read_text(encoding="utf-8"))["ids"])


# digits 명령: 기본 모델 둘(small, turbo) 중 하나라도 가설에 아라비아 숫자가 있는 발화를 '숫자 발화'로 확정해
# 쓴다(docs/experiments.md '숫자 표기'). 1단계에서 한 번 정하고 바꾸지 않기로 했으므로 이미 있으면 --force
# 없이 멈춘다.
# 왜 필요한가: Zeroth 정답은 숫자를 한글로 풀어 써서, 인식은 맞아도 표기 때문에 오류로 잡히는 발화가 많다. 이
# 발화를 뺀 수치를 함께 보고해 표기 차이를 걷어 낸다. 모델마다 다른 목록을 쓰면 비교가 흔들리므로 하나로
# 고정한다.
def write_digits(set_name: str, force: bool) -> None:
    target = REPORTS / "digit_utterances" / f"{set_name}.json"
    if target.exists() and not force:
        raise SystemExit(f"{target} exists; the list is fixed at stage 1 (--force overwrites it)")
    per_model = {
        m: {i for i, row in load_rows(m, set_name).items() if row.get("has_digit")} for m in BASE_MODELS
    }
    ids = sorted(set().union(*per_model.values()))
    total = len(load_rows(BASE_MODELS[0], set_name))
    payload = {
        "set": set_name,
        "rule": "either base model's hypothesis contains [0-9]",
        "per_model": {m: len(v) for m, v in per_model.items()},
        "count": len(ids),
        "of": total,
        "ids": ids,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"{set_name}: {len(ids)} of {total} utterances, per model {payload['per_model']}")


# 주어진 ID 순서대로 편집 수와 정답 길이 배열을 만든다. 두 리포트를 같은 순서로 맞추는 데 쓴다.
def arrays(rows: dict[str, dict], ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([rows[i]["edits"] for i in ids]), np.array([rows[i]["length"] for i in ids])


# table 명령: 여러 리포트의 오류율과 95% 구간을 한 표로 찍는다. --baseline을 주면 각 모델과 기준선의 짝지은
# 차이(%p, 구간)와 상대 변화도 찍는다. Zeroth 묶음은 '전체'와 '숫자 발화 제외'를 함께 보인다(FLEURS는 목록이
# 없어 --no-digit-split).
# 모든 리포트에 공통으로 있는 발화만 써서 같은 발화끼리 비교한다. 발화는 ID 순으로 정렬해 쓰므로, 하네스가
# 묶음 순서로 계산해 리포트 요약에 적은 interval_95와 구간 끝이 조금 다를 수 있다(문서의 표와 판정은 이 명령의
# 값이다. docs/experiments.md 공통 규칙 '지표'의 2026-10-29 덧붙임).
# samples(2026-10-29, --bootstrap-samples): 오류율 구간과 차이 구간의 부트스트랩 표본 수. 기본 1,000(seed 0)이
# 기록된 모든 값이고, 다른 값은 판정 문턱에 가까운 구간 끝이 표본 수에 따라 얼마나 움직이는지 사후에 볼 때만
# 쓴다. 기본값이 아니면 묶음 머리 줄에 표본 수를 덧붙인다. decimals(--decimals): %·%p 값의 소수 자릿수(기본 2,
# 상대 변화는 늘 한 자리). 둘 다 기본값이면 출력은 예전과 같다.
def table(
    set_name: str,
    reports: list[str],
    baseline: str | None,
    split_digits: bool,
    harmonized: bool = False,
    scoring: str | None = None,
    samples: int = BOOTSTRAP_SAMPLES,
    decimals: int = 2,
) -> None:
    loaded = {name: load_rows(name, set_name, harmonized, scoring) for name in reports}
    shared = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    subsets = {"all": shared}
    if split_digits:
        digits = digit_ids(set_name)
        subsets["no digits"] = [i for i in shared if i not in digits]
    # 오류율 칸의 너비는 자릿수 + 4(기본 2자리면 예전과 같은 6칸).
    places, width = decimals, decimals + 4
    note = "" if samples == BOOTSTRAP_SAMPLES else f", bootstrap {samples:,} samples (seed {BOOTSTRAP_SEED})"
    for label, ids in subsets.items():
        print(f"\n{set_name} / {label}: {len(ids)} utterances{note}")
        for name, rows in loaded.items():
            edits, lengths = arrays(rows, ids)
            low, high = interval(edits, lengths, samples=samples)
            line = (
                f"  {name:28s} {error_rate(edits, lengths) * 100:{width}.{places}f}%"
                f"  [{low * 100:.{places}f}, {high * 100:.{places}f}]"
            )
            if baseline and name != baseline:
                base_edits, _ = arrays(loaded[baseline], ids)
                base_rate = error_rate(base_edits, lengths)
                d_low, d_high = difference_interval(edits, base_edits, lengths, samples=samples)
                delta = error_rate(edits, lengths) - base_rate
                # 기준선 오류율이 0이면 상대 변화를 낼 수 없어 compare처럼 nan으로 찍는다(차이와 구간은
                # 그대로 찍는다).
                relative = delta / base_rate if base_rate else float("nan")
                line += (
                    f"  vs {baseline}: {delta * 100:+.{places}f}%p"
                    f" [{d_low * 100:+.{places}f}, {d_high * 100:+.{places}f}]"
                    f", relative {relative * 100:+.1f}%"
                )
            print(line)


# 민감도 분석(robust)의 블록 이름: Zeroth는 화자(발화 ID의 첫 '_' 앞, 리포트의 speaker 칸과 같다.
# CTranslate2 리포트에는 speaker 칸이 없어 ID로 정한다), FLEURS는 문장(ID의 '#' 앞. 같은 문장의 두 번째·
# 세 번째 녹음은 '1563#2'처럼 적혀 있다). 규칙은 docs/experiments.md '추가 분석: 화자·문장 단위 부트스트랩'.
def block_of(set_name: str, utterance_id: str) -> str:
    """발화 ID가 속한 블록: Zeroth는 화자, FLEURS는 문장."""
    if set_name.startswith("zeroth"):
        return utterance_id.split("_", 1)[0]
    if set_name.startswith("fleurs"):
        return utterance_id.split("#", 1)[0]
    raise ValueError(f"{set_name}: no block rule for this set")


# 판정 함수들이 공통으로 쓰는 비교. 두 리포트의 공통 발화에서 기준선·후보 오류율, 차이, 상대 변화, 짝지은
# 부트스트랩 95% 구간을 돌려준다.
# without_digits: Zeroth의 숫자 발화 제외. reference_digits: 정답에 숫자가 있는(True)/없는(False)
# 발화만(FLEURS를 나눠 보는 사후 분석이 3단계부터 정식 보고가 됐다). baseline_scoring: 기준선을 다른 방식으로
# 채점(8단계에서 같은 리포트를 후처리 전후로 비교할 때). 기본 'same'은 후보와 같은 방식이다.
def compare(
    set_name: str,
    candidate: str,
    baseline: str,
    without_digits: bool = False,
    reference_digits: bool | None = None,
    harmonized: bool = False,
    scoring: str | None = None,
    baseline_scoring: str | None = "same",
    with_blocks: bool = False,
) -> dict:
    """두 리포트의 공통 발화에서 잰 오류율과, 그 차이의 짝지은 구간.

    `without_digits`는 Zeroth 묶음의 확정된 숫자 발화를 뺀다. `reference_digits`는 정답에 아라비아 숫자가
    있는(True) 또는 없는(False) 발화만 남긴다. FLEURS처럼 정답이 숫자를 아라비아 숫자로 적는 묶음용이다.
    `baseline_scoring`은 기준선을 후보와 다른 방식으로 채점한다(8단계: 한 리포트를 후처리한 것과 원래 것의
    비교). 두 채점 모두 정답 길이를 같게 남겨야 한다. `with_blocks`는 같은 발화에서 블록(화자·문장) 단위
    구간과 블록 하나 빼기 값을 더한다(민감도 분석, `robust`).
    """
    scoring = "harmonized" if harmonized else scoring
    cand_rows = load_rows(candidate, set_name, scoring=scoring)
    base_rows = load_rows(
        baseline, set_name, scoring=scoring if baseline_scoring == "same" else baseline_scoring
    )
    # 공통 발화만 남기고 조건에 따라 부분집합을 고른다. 숫자 유무는 기준선 리포트의 정답 글로 판단한다.
    ids = sorted(set(cand_rows) & set(base_rows))
    if without_digits:
        digits = digit_ids(set_name)
        ids = [i for i in ids if i not in digits]
    if reference_digits is not None:
        ids = [i for i in ids if bool(_DIGIT.search(base_rows[i]["reference"])) == reference_digits]
    if not ids:
        return {"utterances": 0}
    cand_edits, lengths = arrays(cand_rows, ids)
    base_edits, base_lengths = arrays(base_rows, ids)
    # 짝지은 비교는 분모(정답 길이)가 같아야 뜻이 있다. 두 채점이 정답 길이를 다르게 만들면(예: 한쪽만 정답을
    # 바꾸는 채점) 비교를 거부한다.
    if not np.array_equal(lengths, base_lengths):
        raise SystemExit(
            f"{set_name}: the two scorings give different reference lengths; compare rates instead"
        )
    base, cand = error_rate(base_edits, lengths), error_rate(cand_edits, lengths)
    result = {
        "utterances": len(ids),
        "baseline": base,
        "candidate": cand,
        "delta": cand - base,
        "relative": (cand - base) / base if base else float("nan"),
        "delta_interval": difference_interval(cand_edits, base_edits, lengths),
    }
    # 민감도 분석(robust)일 때만: 같은 발화·같은 채점에서 블록 단위 구간과 블록 하나 빼기. 기본값에서는 결과가
    # 예전과 똑같다.
    if with_blocks:
        blocks = [block_of(set_name, i) for i in ids]
        result["blocks"] = len(set(blocks))
        result["block_interval"] = block_difference_interval(cand_edits, base_edits, lengths, blocks)
        result["leave_one_out"] = leave_one_block_out(cand_edits, base_edits, lengths, blocks)
    return result


# verdict 명령(2·3단계 판정): 후보를 기준선과 Zeroth validation(전체, 숫자 발화 제외)과 FLEURS 한국어
# validation에서 비교해 '범용 개선 / 도메인 전용 / 효과 없음'을 낸다. 판정 표는 docs/experiments.md 2단계와
# 같다.
# require_interval: 3단계에서 더한 조건(숫자 발화 제외 차이의 구간 상한 < 0). 기준선 turbo의 숫자 발화 제외
# CER이 2.18%라 상대 10%가 0.22%p뿐이어서, 작은 차이를 '개선'이라 부르지 않으려고 넣었다(docs/experiments.md
# 3단계). --no-interval-rule이면 2단계 규칙 그대로다(커밋 037ad21에서 기본값으로 바꿨고
# small-a·turbo-l1·turbo-l2의 판정은 같았다).
def verdict(baseline: str, candidate: str, require_interval: bool = True) -> str:
    overall = compare("zeroth-val", candidate, baseline)
    no_digits = compare("zeroth-val", candidate, baseline, without_digits=True)
    other = compare("fleurs-ko-val", candidate, baseline)
    shown = [
        ("zeroth-val all", overall),
        ("zeroth-val no digits", no_digits),
        ("fleurs-ko-val", other),
        # 3단계부터 보고하지만 판정에는 들어가지 않는다(docs/experiments.md 2단계 사후 분석).
        ("  reference has digit", compare("fleurs-ko-val", candidate, baseline, reference_digits=True)),
        ("  reference has none", compare("fleurs-ko-val", candidate, baseline, reference_digits=False)),
    ]
    for label, result in shown:
        if not result["utterances"]:
            continue
        low, high = result["delta_interval"]
        print(
            f"{label:22s} {result['baseline'] * 100:5.2f}% -> {result['candidate'] * 100:5.2f}%"
            f"  {result['delta'] * 100:+.2f}%p [{low * 100:+.2f}, {high * 100:+.2f}]"
            f"  relative {result['relative'] * 100:+.1f}%  ({result['utterances']} utterances)"
        )
    # 조건 1(같은 도메인 개선)과 조건 2(다른 도메인 유지)를 계산하고 판정 표대로 이름을 붙인다. 출력의 한국어
    # 판정 문장은 docs/experiments.md에 그대로 옮겨 적는다.
    in_domain = (
        overall["relative"] <= IN_DOMAIN_RELATIVE
        and no_digits["relative"] <= IN_DOMAIN_NO_DIGITS_RELATIVE
        # 3단계: 숫자 발화 제외 차이의 짝지은 구간이 0을 포함하면 개선이 아니다.
        and (not require_interval or no_digits["delta_interval"][1] < 0)
    )
    kept = other["delta"] <= OUT_OF_DOMAIN_MAX_INCREASE
    if not in_domain:
        label = "효과 없음"
    else:
        label = "범용 개선" if kept else "도메인 전용"
    print(f"1. 같은 도메인 개선: {'만족' if in_domain else '불만족'}")
    print(f"2. 다른 도메인 유지: {'만족' if kept else '불만족'}")
    print(f"판정: {label}")
    return label


# loops 명령: 리포트마다 '편집 수 > 정답 길이'인 발화(한 구절을 끝까지 되풀이한 발화)를 찾아 찍는다. 되풀이
# 발화 하나가 묶음 오류의 17~37%를 차지한 적이 있어 따로 센다(docs/experiments.md 8단계). Zeroth는 맞춘
# 채점으로 셀 수 있다(--harmonized).
def loops(
    set_name: str, reports: list[str], harmonized: bool = False, scoring: str | None = None
) -> dict[str, list[str]]:
    """리포트마다 편집 수가 정답 글자 수보다 많은 발화(한 구절을 끝까지 되풀이한 발화)."""
    found: dict[str, list[str]] = {}
    for name in reports:
        rows = load_rows(name, set_name, harmonized, scoring)
        ids = sorted(i for i, row in rows.items() if row["edits"] > row["length"])
        found[name] = ids
        detail = ", ".join(f"{i} ({rows[i]['edits']} edits, {rows[i]['length']} characters)" for i in ids)
        print(f"{name:28s} {len(ids)} of {len(rows)}" + (f": {detail}" if ids else ""))
    return found


# compare 결과 한 줄 출력: 기준선 → 후보 오류율, 차이(%p)와 짝지은 구간, 상대 변화, 발화 수.
def show(label: str, result: dict) -> None:
    low, high = result["delta_interval"]
    print(
        f"{label:26s} {result['baseline'] * 100:5.2f}% -> {result['candidate'] * 100:5.2f}%"
        f"  {result['delta'] * 100:+.2f}%p [{low * 100:+.2f}, {high * 100:+.2f}]"
        f"  relative {result['relative'] * 100:+.1f}%  ({result['utterances']} utterances)"
    )


# verdict6 명령(6단계 판정): N을 기준선 turbo와 L2(원래 표기로 학습한 3단계 모델)에 비교한다. 조건 1: Zeroth
# 맞춘 CER이 기준선보다 상대 20% 이상 낮고 L2보다 0.3%p를 넘게 높지 않다. 조건 2: FLEURS 한국어가 기준선보다
# 1.0%p를 넘게 나빠지지 않는다. FLEURS는 정답의 숫자 유무로 나눈 수치도 함께 찍는다.
def verdict6(baseline: str, previous: str, candidate: str) -> str:
    """6단계: Zeroth validation의 맞춘 CER을 기준 모델, 3단계 모델과 비교한다."""
    overall = compare("zeroth-val", candidate, baseline, harmonized=True)
    against_previous = compare("zeroth-val", candidate, previous, harmonized=True)
    other = compare("fleurs-ko-val", candidate, baseline)
    show("zeroth-val harmonized", overall)
    show(f"  vs {previous}", against_previous)
    show("fleurs-ko-val", other)
    for label, flag in (("  reference has digit", True), ("  reference has none", False)):
        part = compare("fleurs-ko-val", candidate, baseline, reference_digits=flag)
        if part["utterances"]:
            show(label, part)
    in_domain = overall["relative"] <= IN_DOMAIN_RELATIVE and against_previous["delta"] <= STAGE6_MAX_ABOVE_L2
    kept = other["delta"] <= OUT_OF_DOMAIN_MAX_INCREASE
    if not in_domain:
        label = "표기를 바꾸면 같은 도메인의 이득을 잃는다"
    elif kept:
        label = "표기를 바꿔 학습하면 범용으로 쓸 수 있다"
    else:
        label = "표기만으로는 해결되지 않는다"
    print(f"1. 같은 도메인 개선 유지: {'만족' if in_domain else '불만족'}")
    print(f"2. 다른 도메인 유지: {'만족' if kept else '불만족'}")
    print(f"판정: {label}")
    return label


# 되풀이 발화(편집 수 > 정답 길이)의 ID 집합.
def _loop_ids(rows: dict[str, dict]) -> set[str]:
    return {i for i, row in rows.items() if row["edits"] > row["length"]}


# changes 명령(8단계): 재시도 디코딩 리포트가 기록된 greedy 리포트(base)에 비해 무엇을 바꿨는지 센다.
# - redecoded: 마지막 temperature > 0인 발화(다시 디코딩됨). exhausted: 마지막 temperature까지 기준을 넘은
#   발화.
# - changed_not_redecoded: 다시 디코딩하지 않았는데 가설이 바뀐 발화. 첫 디코딩(greedy)이 기록과 같지 않았다는
#   뜻이라 0을 기대한다(docs/experiments.md 8단계).
# - changed_outside_loops: 되풀이가 아니던 발화 중 바뀐 것. loops_before·loops_after: 전후의 되풀이 발화.
#   empty: 빈 가설.
def changes(set_name: str, base: str, reports: list[str], scoring: str | None = None) -> dict[str, dict]:
    """리포트마다 재시도 디코딩이 기록된 greedy 리포트에 비해 바꾼 것(8단계).

    다시 디코딩됨: 리포트의 마지막 temperature가 0보다 크다. 다시 디코딩되지 않았는데 가설이 바뀌었다면 첫
    (greedy) 디코딩이 기록된 것을 그대로 재현하지 못했다는 뜻이다.
    """
    before = load_rows(base, set_name, scoring=scoring)
    found: dict[str, dict] = {}
    for name in reports:
        after = load_rows(name, set_name, scoring=scoring)
        ids = set(before) & set(after)
        redecoded = {i for i in ids if after[i].get("temperature")}
        changed = {i for i in ids if after[i]["hypothesis"] != before[i]["hypothesis"]}
        loops_before = _loop_ids(before) & ids
        result = {
            "utterances": len(ids),
            "redecoded": sorted(redecoded),
            "exhausted": sorted(i for i in ids if after[i].get("fallback_exhausted")),
            "changed": sorted(changed),
            "changed_not_redecoded": sorted(changed - redecoded),
            "changed_outside_loops": sorted(changed - loops_before),
            "loops_before": sorted(loops_before),
            "loops_after": sorted(_loop_ids(after) & ids),
            "empty": sorted(i for i in ids if not after[i]["hypothesis"].strip()),
        }
        counts = ", ".join(f"{key} {len(value)}" for key, value in result.items() if key != "utterances")
        print(f"{set_name} {name} against {base} ({result['utterances']} utterances): {counts}")
        for key in ("redecoded", "loops_before", "loops_after", "exhausted"):
            if result[key] and len(result[key]) <= 20:
                print(f"  {key}: {', '.join(result[key])}")
        found[name] = result
    return found


# 재시도 후보 하나(예: d1a)에 대한 8단계 조건 계산. 리포트 이름은 '<모델>-<후보>'이고 '<모델>'(기록된
# greedy)과 비교한다.
# 조건 1(되풀이를 막는다): 기록된 되풀이가 하나 이상 있고, 모두 다시 디코딩됐고, 후보에서 세 묶음 모두
# 되풀이가 0개. 조건 2(정확도를 잃지 않는다): 모델마다 Zeroth 맞춘 CER(전체, 숫자 발화 제외)과 FLEURS 한국어
# CER의 짝지은 차이 구간 상한이 모두 +0.1%p 이하. FLEURS 영어는 보고만 한다.
# tested: 기록된 되풀이가 실제로 다시 디코딩되어 재시도를 시험할 수 있었는지.
def stage8_arm(models: list[str], arm: str) -> dict:
    """재시도 후보 하나의 8단계 조건: 리포트 `<model>-<arm>`을 greedy `<model>`과 비교한다."""
    recorded_loops, untested, loops_left, upper_ends = 0, [], 0, []
    for model in models:
        for set_name, scoring in STAGE8_SETS.items():
            found = changes(set_name, model, [f"{model}-{arm}"], scoring)[f"{model}-{arm}"]
            recorded_loops += len(found["loops_before"])
            untested += [
                f"{model} {set_name} {i}" for i in found["loops_before"] if i not in found["redecoded"]
            ]
            loops_left += len(found["loops_after"])
        parts = [
            ("zeroth-val harmonized", "zeroth-val", "harmonized", False),
            ("zeroth-val harmonized, no digits", "zeroth-val", "harmonized", True),
            ("fleurs-ko-val", "fleurs-ko-val", None, False),
        ]
        for label, set_name, scoring, without_digits in parts:
            result = compare(
                set_name, f"{model}-{arm}", model, without_digits=without_digits, scoring=scoring
            )
            show(f"{model}-{arm} {label}", result)
            upper_ends.append(result["delta_interval"][1])
        show(
            f"{model}-{arm} fleurs-en-val (WER, reported)", compare("fleurs-en-val", f"{model}-{arm}", model)
        )
    stops = recorded_loops > 0 and not untested and loops_left == 0
    harmless = max(upper_ends) <= STAGE8_MAX_INCREASE
    print(
        f"{arm}: recorded loops {recorded_loops}, loops left {loops_left}, not decoded again {len(untested)}"
    )
    for item in untested:
        print(f"  not decoded again: {item}")
    print(f"{arm}: 1. 되풀이를 막는다: {'만족' if stops else '불만족'}")
    print(
        f"{arm}: 2. 오류율 상한 +{STAGE8_MAX_INCREASE * 100:.1f}%p 이하: {'만족' if harmless else '불만족'}"
    )
    return {"tested": recorded_loops > 0 and not untested, "stops": stops, "harmless": harmless}


# verdict8 명령(8단계 디코딩 판정): 후보를 단순한 것부터(--arms 순서) 보고, 두 조건을 모두 만족한 첫 후보를
# 고른다. 없으면 재시도를 쓰지 않는다.
# 기록된 되풀이가 첫 디코딩에서 되풀이하지 않아 다시 디코딩되지 않았으면 재시도를 시험하지 못한 것이므로
# 판정하지 않고 멈춘다(2026-10-02 CTranslate2 측정과 같은 처리, docs/experiments.md 8단계).
def verdict8(models: list[str], arms: list[str]) -> str | None:
    """8단계: 정확도를 잃지 않고 기록된 되풀이를 막는 첫 후보(단순한 것부터)."""
    results = {arm: stage8_arm(models, arm) for arm in arms}
    if not all(result["tested"] for result in results.values()):
        print("판정: 하지 않는다 (기록된 되풀이가 첫 디코딩에서 나오지 않아 재시도를 시험하지 못했다)")
        raise SystemExit("no verdict: a recorded loop was not decoded again")
    picked = next((arm for arm in arms if results[arm]["stops"] and results[arm]["harmless"]), None)
    print(f"판정: {picked}를 고른다" if picked else "판정: 재시도를 쓰지 않는다 (greedy 그대로)")
    return picked


# 출력 숫자 변환의 판정 이름. Zeroth 맞춘 CER이 +0.05%p를 넘게 오르면 '해친다'가 먼저이고, 아니면 FLEURS
# 한국어 구간 상한 < 0일 때 '쓴다'.
def itn8_label(fleurs_upper: float, zeroth_delta: float) -> str:
    if zeroth_delta > STAGE8_ITN_MAX_INCREASE:
        return "출력 숫자 변환은 같은 도메인을 해친다"
    if fleurs_upper < 0:
        return "출력 숫자 변환을 쓴다"
    return "출력 숫자 변환은 이득이 없다"


# itn8 명령(8단계 출력 숫자 변환 판정): 같은 리포트를 후처리 전후로 비교한다. 조건 1: FLEURS 한국어 원래
# CER에서 후처리 − 원래의 구간 상한 < 0. 조건 2: Zeroth 맞춘 CER 변화가 +0.05%p 이하(맞춘 채점은 양쪽에 같은
# 변환을 하므로 0에 가까워야 정상이고, 다른 것을 바꾸지 않는지만 본다).
# 판정 없이 함께 보고: 정답의 숫자 유무로 나눈 FLEURS, 같은 후처리를 한 기준선, 후보와 기준선의 비교, 글자
# 이름 변환, 표기 무관 CER, Zeroth 원래 CER의 변화(Zeroth 정답은 숫자를 풀어 써서 나빠지는 것이 정상이다).
# 주의: 이 효과는 규칙을 적기 전에 사후 계산으로 이미 알려져 있었다. 그래서 이 판정은 눈을 가린 측정이 아니라
# 미리 정한 규칙으로 다시 계산한 확인이다(README 8단계).
def itn8(model: str, baseline: str, fleurs: str = "fleurs-ko-val", zeroth: str = "zeroth-val") -> str:
    """8단계 출력 후처리: 후처리한 `model`을 원래 `model`과 비교한다. 나머지는 보고만 한다."""
    other = compare(fleurs, model, model, scoring="itn", baseline_scoring=None)
    same = compare(zeroth, model, model, scoring="itn-harmonized", baseline_scoring="harmonized")
    show(f"{fleurs} {model}, post-processed", other)
    show(f"{zeroth} {model} harmonized, post-processed", same)
    for label, flag in (("  reference has digit", True), ("  reference has none", False)):
        part = compare(fleurs, model, model, reference_digits=flag, scoring="itn", baseline_scoring=None)
        if part["utterances"]:
            show(label, part)
    print("reported, no verdict:")
    show(
        f"{fleurs} {baseline}, post-processed",
        compare(fleurs, baseline, baseline, scoring="itn", baseline_scoring=None),
    )
    show(
        f"{fleurs} {model} vs {baseline}, both post-processed",
        compare(fleurs, model, baseline, scoring="itn"),
    )
    show(
        f"{fleurs} {model} post-processed vs {baseline} as is",
        compare(fleurs, model, baseline, scoring="itn", baseline_scoring=None),
    )
    show(
        f"{fleurs} {model}, plus letter names",
        compare(fleurs, model, model, scoring="itn-latin", baseline_scoring=None),
    )
    show(
        f"{fleurs} notation-agnostic, {model} vs {baseline}",
        compare(fleurs, model, baseline, scoring="agnostic"),
    )
    show(
        f"{zeroth} {model} original CER, post-processed",
        compare(zeroth, model, model, scoring="itn", baseline_scoring=None),
    )
    label = itn8_label(other["delta_interval"][1], same["delta"])
    helps, harmless = other["delta_interval"][1] < 0, same["delta"] <= STAGE8_ITN_MAX_INCREASE
    print(f"1. {fleurs} 짝지은 차이의 구간 상한 < 0: {'만족' if helps else '불만족'}")
    limit = f"+{STAGE8_ITN_MAX_INCREASE * 100:.2f}%p"
    print(f"2. {zeroth} 맞춘 CER 변화 {limit} 이하: {'만족' if harmless else '불만족'}")
    print(f"판정: {label}")
    return label


# evaluate_qwen --out으로 쓴 파일(fp16 점검, 체크포인트 고르기)의 요약과 ID별 행. 이 파일들은 limit이 있어
# load_rows를 거치지 않고 직접 읽는다.
def _check_rows(path: str) -> tuple[dict, dict[str, dict]]:
    """9단계 fp16 점검 파일(--limit 실행의 raw 리포트)의 요약과 ID별 행."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return data["summary"], {row["id"]: row for row in data["utterances"]}


# gate9 명령(9단계 fp16 점검): 앞 100발화를 fp16(GPU)과 fp32(CPU)로 디코딩한 두 파일을 비교한다. 두 파일이
# 정밀도 표시와 발화 순서까지 맞는지 먼저 확인하고, 원래 CER 차이가 ±0.20%p 안이며 fp16 쪽에 빈 가설과 넘침
# 발화가 없으면 'fp16을 쓴다'. 아니면 fp32로 잰다. 가설이 다른 발화 수도 보고한다.
# CPU와 GPU의 계산 차이도 fp16 쪽 차이로 세게 되므로 점검이 그만큼 엄격해질 뿐이다(docs/experiments.md 9단계).
def gate9(fp16: str, fp32: str) -> bool:
    """9단계: fp16을 써도 되면 True(docs/experiments.md 9단계 "fp16 점검")."""
    (half_summary, half), (full_summary, full) = _check_rows(fp16), _check_rows(fp32)
    if (half_summary.get("precision"), full_summary.get("precision")) != ("fp16", "fp32"):
        raise SystemExit("gate9 needs an fp16 file and an fp32 file (summary precision)")
    if list(half) != list(full):
        raise SystemExit("the two runs did not decode the same utterances in the same order")
    ids = [i for i, row in full.items() if row["length"]]
    half_edits, lengths = arrays(half, ids)
    full_edits, _ = arrays(full, ids)
    half_rate, full_rate = error_rate(half_edits, lengths), error_rate(full_edits, lengths)
    gap = half_rate - full_rate
    empty = sum(not half[i]["hypothesis"].strip() for i in ids)
    nonfinite = sum(bool(half[i].get("nonfinite_logits")) for i in ids)
    differ = sum(half[i]["hypothesis"] != full[i]["hypothesis"] for i in ids)
    print(
        f"{len(ids)} utterances: fp16 ({half_summary.get('device')}) {half_rate * 100:.2f}%"
        f", fp32 ({full_summary.get('device')}) {full_rate * 100:.2f}%"
        f", difference {gap * 100:+.2f}%p; hypotheses that differ {differ}"
    )
    print(f"fp16: empty hypotheses {empty}, utterances with non-finite logits {nonfinite}")
    close = abs(gap) <= STAGE9_FP16_MAX_GAP
    clean = empty == 0 and nonfinite == 0
    print(f"1. CER 차이 {STAGE9_FP16_MAX_GAP * 100:.1f}%p 이하: {'만족' if close else '불만족'}")
    print(f"2. fp16 빈 가설 0, 넘침 0: {'만족' if clean else '불만족'}")
    print(f"판정: {'fp16을 쓴다' if close and clean else 'fp32로 잰다'}")
    return close and clean


# 9단계 판정 이름: FLEURS 한국어에서 (후보 − 기준선 turbo) 짝지은 구간의 상한 < 0이면 '더 나은 기준 모델',
# 하한 > 0이면 '나쁘다', 0을 포함하면 '가리지 못했다'.
def verdict9_label(low: float, high: float) -> str:
    """FLEURS 한국어 validation에서 (후보 - 기준선 turbo)의 짝지은 95% 구간에 적용하는 9단계 규칙."""
    if high < 0:
        return "다른 도메인에서 더 나은 기준 모델이다"
    if low > 0:
        return "다른 도메인에서 기준선보다 나쁘다"
    return "다른 도메인에서 차이를 가리지 못했다"


# verdict9 명령(9단계 판정): 후보의 raw 출력을 기준선 turbo와 FLEURS 한국어 validation 원래 CER로 비교해
# 판정한다. Zeroth는 학습 데이터에 들어 있을 수 있어 판정에 넣지 않고 보고만 한다(docs/experiments.md 9단계
# 판정 규칙).
# 함께 보고: 숫자 유무 분리, 맞춘 CER·표기 무관 CER, 파인튜닝한 모델들(--others)과의 비교, Zeroth 맞춘
# CER(숫자 발화 제외 포함), 영어 WER, 공식 반복 제거(fixed)가 바꾼 것, 세 묶음의 되풀이 발화.
def verdict9(candidate: str, baseline: str, others: list[str]) -> str:
    """9단계: FLEURS 한국어 validation(원래 CER)에서 후보의 raw 출력을 기준선 turbo와 비교한다.

    나머지는 모두 보고만 한다: 기준선 turbo와 파인튜닝한 모델들에 견준 Zeroth 수치, 다른 채점 방식, 영어,
    되풀이, 그리고 공식 반복 제거(`<candidate>-fixed`)가 바꾸는 것.
    """
    fixed = f"{candidate}-fixed"
    rule = compare("fleurs-ko-val", candidate, baseline)
    show(f"fleurs-ko-val {candidate} vs {baseline}", rule)
    print("reported, no verdict:")
    for label, flag in (("  reference has digit", True), ("  reference has none", False)):
        part = compare("fleurs-ko-val", candidate, baseline, reference_digits=flag)
        if part["utterances"]:
            show(label, part)
    for scoring in ("harmonized", "agnostic"):
        show(f"fleurs-ko-val {scoring}", compare("fleurs-ko-val", candidate, baseline, scoring=scoring))
    for other in others:
        show(f"fleurs-ko-val vs {other}", compare("fleurs-ko-val", candidate, other))
    for reference in (baseline, *others):
        overall = compare("zeroth-val", candidate, reference, harmonized=True)
        show(f"zeroth-val harmonized vs {reference}", overall)
        no_digits = compare("zeroth-val", candidate, reference, without_digits=True, harmonized=True)
        show("  no digit utterances", no_digits)
    show(f"zeroth-val original CER vs {baseline}", compare("zeroth-val", candidate, baseline))
    show(f"fleurs-en-val (WER) vs {baseline}", compare("fleurs-en-val", candidate, baseline))
    print(f"official repetition fix ({fixed} vs {candidate}):")
    for label, set_name, scoring in (
        ("zeroth-val harmonized", "zeroth-val", "harmonized"),
        ("fleurs-ko-val", "fleurs-ko-val", None),
        ("fleurs-en-val (WER)", "fleurs-en-val", None),
    ):
        show(f"  {label}", compare(set_name, fixed, candidate, scoring=scoring))
    print("loops (more edits than reference length):")
    for set_name, scoring in STAGE8_SETS.items():
        loops(set_name, [baseline, candidate, fixed], scoring=scoring)
    label = verdict9_label(*rule["delta_interval"])
    print("규칙: fleurs-ko-val CER 차이(후보 raw - 기준선)의 짝지은 95% 구간 상한 < 0")
    print(f"판정: {label}")
    return label


# gate10 명령(10단계 관문 (a)): 학습 전 어댑터(B=0)를 합쳐 디코딩한 결과가 9단계 fp16 점검 파일과 발화 단위로
# 같은지 본다. 어댑터 병합 경로와 하네스가 9단계와 같은 조건임을 확인하려는 것이다. 한쪽만 어댑터가 있어야
# 하고, 정밀도·묶음·limit과 발화 순서가 같아야 비교한다. 가설이 하나라도 다르면 실패다.
def gate10(path: str, recorded: str) -> bool:
    """10단계 관문 (a): 새 어댑터를 기본 가중치에 합친 모델이 9단계 fp16 점검과 같은 것을 디코딩하는지."""
    (summary, rows), (base_summary, base_rows) = _check_rows(path), _check_rows(recorded)
    if not summary.get("adapter") or base_summary.get("adapter"):
        raise SystemExit("gate10 compares a run with an adapter against the recorded run without one")
    for key in ("precision", "set", "limit"):
        if summary.get(key) != base_summary.get(key):
            raise SystemExit(f"the two runs differ in {key}: {summary.get(key)} and {base_summary.get(key)}")
    if list(rows) != list(base_rows):
        raise SystemExit("the two runs did not decode the same utterances in the same order")
    differ = [i for i in rows if rows[i]["hypothesis"] != base_rows[i]["hypothesis"]]
    ids = [i for i, row in base_rows.items() if row["length"]]
    edits, lengths = arrays(rows, ids)
    base_edits, _ = arrays(base_rows, ids)
    shown = f" ({', '.join(differ[:5])})" if differ else ""
    print(
        f"{len(rows)} utterances: with {summary['adapter']} {error_rate(edits, lengths) * 100:.2f}%"
        f", recorded {error_rate(base_edits, lengths) * 100:.2f}%"
        f"; hypotheses that differ {len(differ)}{shown}"
    )
    print(f"판정: {'통과 (발화 단위로 같다)' if not differ else '실패'}")
    return not differ


def read_steps(path: str) -> dict[int, dict]:
    """train_qwen의 steps.jsonl에서 스텝 줄만 스텝 번호별로 읽는다(이벤트 줄은 뺀다).

    이어 한 실행이 같은 스텝을 두 번 기록하면 나중 것을 쓴다.
    """
    steps = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            if "event" not in entry:
                steps[entry["step"]] = entry
    return steps


def read_memwatch(path: str) -> list[dict]:
    """스모크 중 5초마다 남긴 GPU 메모리 표본 CSV(저장소 밖에 둔 감시 스크립트가 쓴다)를 읽는다.

    MiB 카운터 열은 float로 바꾸고, 카운터 값이 비어 있던 칸은 None으로 둔다.
    """
    with open(path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    keys = ("shared_mb", "dedicated_mb", "smi_used_mb", "smi_util")
    return [
        {**row, **{k: float(row[k]) if row.get(k) not in (None, "") else None for k in keys}} for row in rows
    ]


# smoke10 명령(10단계 관문 (c)): 스모크 50스텝의 스텝 기록과 메모리 표본을 규칙의 일곱 조건에 대 보고 통과
# 여부를 낸다. 같은 기록으로 본 학습의 스텝 수(1,500 또는 1,000)도 정한다.
# 스텝 시간 중앙값은 규칙대로 21번째 스텝부터, 가장 긴 스텝을 빼고 낸다. 기록된 결과는 13.70초 × 1,500 =
# 5.71시간이라 1,500스텝이었다(docs/experiments.md 10단계 결과).
def smoke10(steps_path: str, mem_path: str, expected_steps: int = 50) -> dict:
    """10단계 관문 (c): 스모크 실행을 등록한 한계에 비춰 보고, 학습할 스텝 수를 정한다."""
    steps = read_steps(steps_path)
    samples = read_memwatch(mem_path)
    shared = [row["shared_mb"] for row in samples if row["shared_mb"] is not None]
    used = [row["smi_used_mb"] for row in samples if row["smi_used_mb"] is not None]
    if not steps or not shared or not used:
        raise SystemExit("smoke10 needs the step log and GPU memory samples with both counters")
    late_skips = sorted(s for s, e in steps.items() if e["skipped"] and s > STAGE10_SKIP_GRACE)
    nonfinite = sorted(s for s, e in steps.items() if not math.isfinite(e["loss"]))
    longest = [s for s, e in steps.items() if e["longest"]]
    peak = max(e["peak_allocated_mb"] or 0 for e in steps.values())  # CPU 실행에서는 None
    times = [e["seconds"] for s, e in steps.items() if s > STAGE10_SKIP_GRACE and not e["longest"]]
    median = statistics.median(times) if times else float("nan")
    hours = median * STAGE10_STEPS / 3600
    all_skips = sum(e["skipped"] for e in steps.values())
    # 규칙 순서대로: 1 손실 유한, 2 21번째 스텝부터 건너뜀 5개 이하, 3 가장 긴 스텝을 돌았음, 4 50스텝 모두, 5
    # 공유 메모리, 6 nvidia-smi, 7 torch 최대 할당.
    checks = [
        (f"1. 모든 손실이 유한 (유한하지 않은 스텝 {nonfinite or '없음'})", not nonfinite),
        (
            f"2. GradScaler 건너뜀이 {STAGE10_SKIP_GRACE}스텝 뒤에 {STAGE10_MAX_SKIPS}개 이하"
            f" ({len(late_skips)}개, 전체 {all_skips}개)",
            len(late_skips) <= STAGE10_MAX_SKIPS,
        ),
        (f"3. 가장 긴 발화의 스텝을 돌았다 ({longest or '없음'})", bool(longest)),
        (
            f"4. {expected_steps}스텝을 모두 돌았다 ({len(steps)}스텝)",
            sorted(steps) == list(range(1, expected_steps + 1)),
        ),
        (
            f"5. 공유 GPU 메모리 최대 {max(shared):,.0f} MiB <= {STAGE10_MAX_SHARED_MB}",
            max(shared) <= STAGE10_MAX_SHARED_MB,
        ),
        (f"6. nvidia-smi 최대 {max(used):,.0f} MiB < {STAGE10_MAX_GPU_MB:,}", max(used) < STAGE10_MAX_GPU_MB),
        (
            f"7. torch 최대 할당 {peak:,} MiB <= {STAGE10_MAX_ALLOCATED_MB:,}",
            peak <= STAGE10_MAX_ALLOCATED_MB,
        ),
    ]
    for label, ok in checks:
        print(f"{label}: {'만족' if ok else '불만족'}")
    passed = all(ok for _, ok in checks)
    max_steps = STAGE10_STEPS if hours <= STAGE10_HOURS else STAGE10_FEWER_STEPS
    print(
        f"스텝 시간 중앙값 ({STAGE10_SKIP_GRACE + 1}스텝부터, 가장 긴 스텝 제외) {median:.2f}초"
        f" x {STAGE10_STEPS:,} = {hours:.2f}시간 -> {max_steps:,}스텝"
    )
    print(f"판정: {'통과' if passed else '실패'}")
    return {"passed": passed, "max_steps": max_steps, "median_step_seconds": median, "hours": hours}


# pick10 명령(10단계 체크포인트 고르기): 500/1,000/1,500스텝 어댑터로 zeroth-val500 전체를 잰 파일들에서 맞춘
# CER이 가장 낮은 것을 고른다. 같으면 이른 스텝(파일은 스텝 순서로 받는다).
# 맞춘 CER로 고르는 이유: Qwen은 한글 숫자를 붙여 쓰는 일이 있어, 판정과 같은 채점으로
# 고른다(docs/experiments.md 10단계 '체크포인트 고르기'). --limit 파일과 발화가 다른 파일은 거부한다.
def pick10(paths: list[str]) -> str:
    """10단계: zeroth-val500 실행의 맞춘 CER이 가장 낮은 체크포인트. 같으면 가장 이른 것(파일은 스텝
    순서로 받는다)."""
    loaded = []
    for path in paths:
        summary, rows = _check_rows(path)
        if summary.get("limit"):
            raise SystemExit(f"{path}: measured with --limit, not the whole set")
        loaded.append((path, summary, {i: rescored(row) for i, row in rows.items() if row["length"]}))
    ids = sorted(loaded[0][2])
    if any(sorted(rows) != ids for _, _, rows in loaded):
        raise SystemExit("the checkpoint runs did not decode the same utterances")
    best, best_rate = paths[0], float("inf")
    for path, summary, rows in loaded:
        edits, lengths = arrays(rows, ids)
        rate = error_rate(edits, lengths)
        low, high = interval(edits, lengths)
        print(f"  {summary.get('adapter')}: {rate * 100:.2f}% [{low * 100:.2f}, {high * 100:.2f}] ({path})")
        if rate < best_rate:
            best, best_rate = path, rate
    print(f"고른 체크포인트: {best}")
    return best


# 10단계 판정 표(docs/experiments.md 10단계): 조건 1(Zeroth에서 N만큼)과 조건 2(FLEURS 한국어에서 N보다
# 낫다)의 조합으로 이름을 정한다. 기록된 결과는 조건 1만 만족한 'N을 유지한다 (다른 도메인 이득 없음)'이다.
def verdict10_label(zeroth_ok: bool, fleurs_ok: bool) -> str:
    """조건 1(Zeroth에서 N과 비교)과 조건 2(FLEURS 한국어에서 N과 비교)로 정하는 10단계 판정."""
    if zeroth_ok and fleurs_ok:
        return "정확도 기준으로 N을 대신하는 범용 모델이다"
    if zeroth_ok:
        return "N을 유지한다 (다른 도메인 이득 없음)"
    if fleurs_ok:
        return "다른 도메인용 기준 모델로만 쓴다"
    return "N을 유지한다"


# 리포트의 요약만 읽는다(10단계 보고의 개수 항목용).
def _summary(report: str, set_name: str) -> dict:
    return json.loads((REPORTS / report / f"{set_name}.json").read_text(encoding="utf-8"))["summary"]


# verdict10 명령(10단계 판정): QN(학습한 Qwen, raw 출력)을 N과 비교한다. 조건 1은 Zeroth 맞춘 CER 상한 <
# +0.10%p, 조건 2는 FLEURS 한국어 원래 CER 상한 < 0.
# 조건 2는 학습 전 Qwen이 이미 N보다 낮아 쉽게 만족될 수 있어, 학습 전 Qwen 대비 망각(상한 ≤ +1.0%p)을 보고
# 줄로 미리 정해 함께 찍는다(판정은 바꾸지 않는다). 나머지(학습 전 Qwen·다른 모델과의 비교, 숫자 분리, 표기
# 무관 CER, 영어, 전화 음질, fixed, 되풀이, 요약 개수)는 보고만 한다.
def verdict10(candidate: str, n: str, untrained: str, others: list[str]) -> str:
    """10단계: LoRA로 학습한 Qwen(raw 출력)을 Zeroth(맞춘 CER)와 FLEURS 한국어에서 N과 비교한다.

    함께 보고하는 것: 학습 전 Qwen에 대한 망각 줄, 다른 모델들, 숫자 유무 분리, 표기 무관 채점, 영어, 전화
    채널, 공식 반복 제거, 되풀이.
    """
    zeroth = compare("zeroth-val", candidate, n, harmonized=True)
    fleurs = compare("fleurs-ko-val", candidate, n)
    forgetting = compare("fleurs-ko-val", candidate, untrained)
    show(f"zeroth-val harmonized {candidate} vs {n}", zeroth)
    show(f"fleurs-ko-val {candidate} vs {n}", fleurs)
    show(f"fleurs-ko-val {candidate} vs {untrained} (forgetting)", forgetting)
    print("reported, no verdict:")
    show(
        f"zeroth-val harmonized vs {untrained}", compare("zeroth-val", candidate, untrained, harmonized=True)
    )
    for reference in (n, untrained, *others):
        part = compare("zeroth-val", candidate, reference, without_digits=True, harmonized=True)
        show(f"zeroth-val harmonized, no digit utterances, vs {reference}", part)
    for reference in others:
        show(
            f"zeroth-val harmonized vs {reference}",
            compare("zeroth-val", candidate, reference, harmonized=True),
        )
        show(f"fleurs-ko-val vs {reference}", compare("fleurs-ko-val", candidate, reference))
    for label, flag in (("reference has digit", True), ("reference has none", False)):
        for reference in (n, untrained):
            part = compare("fleurs-ko-val", candidate, reference, reference_digits=flag)
            if part["utterances"]:
                show(f"fleurs-ko-val {label} vs {reference}", part)
    for reference in (n, untrained):
        agnostic = compare("fleurs-ko-val", candidate, reference, scoring="agnostic")
        show(f"fleurs-ko-val agnostic vs {reference}", agnostic)
        show(f"fleurs-en-val (WER) vs {reference}", compare("fleurs-en-val", candidate, reference))
    # 전화 음질(판정 없음)은 두 리포트가 모두 있을 때만 비교한다.
    phone = "zeroth-val@telephone"
    if all((REPORTS / r / f"{phone}.json").exists() for r in (candidate, untrained)):
        show(f"{phone} harmonized vs {untrained}", compare(phone, candidate, untrained, harmonized=True))
    else:
        print(f"{phone}: not measured")
    fixed = f"{candidate}-fixed"
    print(f"official repetition fix ({fixed} vs {candidate}):")
    for label, set_name, scoring in (
        ("zeroth-val harmonized", "zeroth-val", "harmonized"),
        ("fleurs-ko-val", "fleurs-ko-val", None),
        ("fleurs-en-val (WER)", "fleurs-en-val", None),
    ):
        show(f"  {label}", compare(set_name, fixed, candidate, scoring=scoring))
    print("loops (more edits than reference length):")
    for set_name, scoring in STAGE8_SETS.items():
        loops(set_name, [n, untrained, candidate], scoring=scoring)
    print("summaries (hypotheses with a digit, token limit reached, non-finite logits):")
    for set_name in STAGE8_SETS:
        for report in (untrained, candidate):
            summary = _summary(report, set_name)
            counts = [
                summary.get(k) for k in ("hypotheses_with_digit", "hit_token_limit", "nonfinite_utterances")
            ]
            print(f"  {set_name} {report}: {counts}")
    # 판정에 쓰는 것은 짝지은 구간의 상한뿐이다(점 추정이 아니라 구간으로 판정).
    zeroth_ok = zeroth["delta_interval"][1] < STAGE10_ZEROTH_MAX_ABOVE_N
    fleurs_ok = fleurs["delta_interval"][1] < 0
    forgetting_high = forgetting["delta_interval"][1]
    within = forgetting_high <= STAGE10_FORGETTING_MAX
    limit = STAGE10_ZEROTH_MAX_ABOVE_N * 100
    print(
        f"1. zeroth-val 맞춘 CER {candidate} - {n} 상한 < +{limit:.1f}%p: {'만족' if zeroth_ok else '불만족'}"
    )
    print(f"2. fleurs-ko-val CER {candidate} - {n} 상한 < 0: {'만족' if fleurs_ok else '불만족'}")
    print(
        f"보고: fleurs-ko-val CER {candidate} - {untrained} 상한 {forgetting_high * 100:+.2f}%p"
        f", 망각 상한 +{STAGE10_FORGETTING_MAX * 100:.1f}%p {'안' if within else '밖'}"
    )
    label = verdict10_label(zeroth_ok, fleurs_ok)
    print(f"판정: {label}")
    return label


# --- 11단계: QN 어댑터를 줄여 합치기 (규칙 2026-10-15) ---


# gate11 명령(11단계 관문): 스케일을 정해 합친 실행(--out 파일)이 기록된 파일과 발화 단위로 같은지 본다. (a)는
# 스케일 1.0을 10단계 체크포인트 고르기 파일(val-500 440발화)과, (b)는 스케일 0을 9단계 fp16 점검 파일(앞
# 100발화)과 비교한다. 정밀도·묶음·limit과 발화 순서가 같아야 비교하고, 가설이 하나라도 다르면 실패다. 새
# 파일에는 어댑터와 스케일이 있어야 한다(합치기 경로를 시험하는 관문이다).
def gate11(path: str, recorded: str) -> bool:
    """11단계 관문: 스케일을 정해 합친 실행이 기록된 실행과 같은 것을 디코딩하는지."""
    (summary, rows), (base_summary, base_rows) = _check_rows(path), _check_rows(recorded)
    if not summary.get("adapter") or not summary.get("adapter_scale"):
        raise SystemExit("gate11 needs a run merged with an adapter at a given scale")
    for key in ("precision", "set", "limit"):
        if summary.get(key) != base_summary.get(key):
            raise SystemExit(f"the two runs differ in {key}: {summary.get(key)} and {base_summary.get(key)}")
    if list(rows) != list(base_rows):
        raise SystemExit("the two runs did not decode the same utterances in the same order")
    differ = [i for i in rows if rows[i]["hypothesis"] != base_rows[i]["hypothesis"]]
    ids = [i for i, row in base_rows.items() if row["length"]]
    edits, lengths = arrays(rows, ids)
    base_edits, _ = arrays(base_rows, ids)
    shown = f" ({', '.join(differ[:5])})" if differ else ""
    print(
        f"{len(rows)} utterances: {summary['adapter']} at {summary['adapter_scale']}"
        f" {error_rate(edits, lengths) * 100:.2f}%, recorded {error_rate(base_edits, lengths) * 100:.2f}%"
        f"; hypotheses that differ {len(differ)}{shown}"
    )
    print(f"판정: {'통과 (발화 단위로 같다)' if not differ else '실패'}")
    return not differ


# blocks 명령: 비교 하나의 차이, 발화 단위 구간, 블록(Zeroth 화자, FLEURS 문장) 단위 구간과 블록 하나 빼기
# 범위를 찍는다. 11단계 판정 줄에 함께 보고하는 것이고 판정에는 쓰지 않는다(2026-10-15 추가 분석과 같은
# 방법).
def blocks(set_name: str, candidate: str, baseline: str, scoring: str | None, without_digits: bool) -> dict:
    """비교 하나의 블록 단위 구간(보고만)."""
    result = compare(
        set_name, candidate, baseline, without_digits=without_digits, scoring=scoring, with_blocks=True
    )
    left_out = result["leave_one_out"]
    low, high = min(left_out.values()), max(left_out.values())
    print(
        f"{set_name} {candidate} - {baseline}: {result['delta'] * 100:+.2f}%p"
        f" utterance {_points(result['delta_interval'])} block {_points(result['block_interval'])}"
        f" ({result['blocks']} blocks), leave one out {low * 100:+.2f} .. {high * 100:+.2f}"
    )
    return result


# --- 12단계: N을 CTranslate2 int8_float16으로 (규칙 2026-10-15) ---


# scripts/speed.py가 쓴 speed.json에서 라벨 → 실행 기록. 2026-10-29부터 speed.py는 실행이 하나 끝날 때마다
# 그때까지의 기록을 complete: false로 다시 쓰고, 끝까지 돌면 complete: true로 쓴다. 중간에 멈춘 파일(false)은
# 거부한다: 회차가 빠지면 판정 2의 '두 번 평균'·'두 번 모두'를 볼 수 없고, 빠진 실행 대신 다른 값이 쓰이면
# 안 된다. 키가 없는 그 전 파일(2026-10-18 측정)은 모든 실행이 끝난 뒤에만 썼으므로 끝까지 돈 것으로 본다.
def _speed_runs(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if data.get("preset") != "ct2-int8":
        raise SystemExit(f"{path}: not a ct2-int8 run of scripts/speed.py")
    if not data.get("complete", True):
        raise SystemExit(f"{path}: an unfinished run of scripts/speed.py (complete: false); measure again")
    return {run["label"]: run for run in data["runs"]}


# 두 번 잰 값의 평균과, 두 번이 10% 넘게 다른지(규칙: 다르면 적는다).
def _two_rounds(values: list[float]) -> tuple[float, bool]:
    mean = sum(values) / len(values)
    return mean, (max(values) - min(values)) / mean > 0.10 if mean else False


# verdict12 명령(12단계 판정): 조건 1은 Zeroth 맞춘 CER에서 int8 − fp16(기록) 상한 ≤ +0.10%p, 조건 2는
# 속도 파일에서 nvidia-smi 증가(두 번 평균)가 20% 넘게 작거나, 한 발화 지연 중앙값이 두 번 모두 5% 넘게 짧은
# 것. 함께 보고: 원래 CER, FLEURS 한국어(새로 잰 fp16과 int8), 되풀이, 가설이 다른 발화, Transformers 배치
# 1과의 배속 비율, 환경 확인(속도 실행의 fp16 val-500 오류율이 기록과 같은지), 쉬는 상태였는지.
def verdict12(speed_path: str, candidate: str = "turbo-n-ct2-int8", baseline: str = "turbo-n-ct2") -> str:
    """12단계: int8_float16이 정확도를 지키면서 메모리나 지연을 줄이는지."""
    # 속도 파일을 먼저 읽는다: 다른 프리셋이거나 끝까지 돌지 않은 파일이면 정확도 비교(부트스트랩)를 하기 전에
    # 멈춘다. 읽기만 하고 아무것도 찍지 않으므로 맞는 파일의 출력 순서는 그대로다.
    runs = _speed_runs(speed_path)
    zeroth = compare("zeroth-val", candidate, baseline, harmonized=True)
    show(f"zeroth-val harmonized {candidate} vs {baseline}", zeroth)
    print("reported, no verdict:")
    show("zeroth-val original CER", compare("zeroth-val", candidate, baseline))
    show("fleurs-ko-val", compare("fleurs-ko-val", candidate, baseline))
    for set_name, scoring in (("zeroth-val", "harmonized"), ("fleurs-ko-val", None)):
        base_rows, cand_rows = load_rows(baseline, set_name), load_rows(candidate, set_name)
        differ = sum(cand_rows[i]["hypothesis"] != base_rows[i]["hypothesis"] for i in base_rows)
        print(f"{set_name}: hypotheses that differ {differ} of {len(base_rows)}")
        loops(set_name, [baseline, candidate], scoring=scoring)
        for report in (baseline, candidate):
            summary = _summary(report, set_name)
            counts = [summary.get(k) for k in ("compute_type", "empty_hypotheses", "decoded_again")]
            print(f"  {report}: compute type, empty hypotheses, decoded again {counts}")

    rounds = (1, 2)
    memory, latency, speed = {}, {}, {}
    for name in ("hf-b1", "ct2-fp16", "ct2-int8"):
        picked = [runs[f"{name}-{n}"] for n in rounds]
        speed[name] = _two_rounds([run["summary"]["audio_seconds_per_second"] for run in picked])
        memory[name] = _two_rounds([run["gpu_memory_added_mb"] for run in picked])
        latency[name] = [run["summary"]["batch_seconds_median"] for run in picked]
        print(
            f"{name}: speed {speed[name][0]:.1f}x, nvidia-smi +{memory[name][0]:.0f} MiB, latency median"
            f" {latency[name]} s, gpu util {[run['gpu_util_avg_during'] for run in picked]}%"
            f", cpu {[run.get('cpu_percent_avg_during') for run in picked]}%"
            + (" (rounds differ by more than 10%)" if speed[name][1] or memory[name][1] else "")
        )
    print(f"ct2-fp16 / hf-b1 speed ratio {speed['ct2-fp16'][0] / speed['hf-b1'][0]:.2f}")
    recorded = _summary(baseline, "zeroth-val500")["error_rate"]
    measured = [runs[f"ct2-fp16-{n}"]["summary"]["error_rate"] for n in rounds]
    print(f"environment check: ct2-fp16 val-500 error rate {measured}, recorded {recorded}")
    with open(speed_path, encoding="utf-8") as file:
        print(f"idle reached before the runs: {json.load(file).get('idle_reached')}")

    accurate = zeroth["delta_interval"][1] <= STAGE12_MAX_INCREASE
    smaller = memory["ct2-int8"][0] < (1 - STAGE12_MEMORY_CUT) * memory["ct2-fp16"][0]
    faster = all(
        latency["ct2-int8"][n] < (1 - STAGE12_LATENCY_CUT) * latency["ct2-fp16"][n]
        for n in range(len(rounds))
    )
    if not accurate:
        label = "정확도를 잃는다 (float16 유지)"
    elif smaller or faster:
        label = "int8_float16을 서빙 기본으로 권한다"
    else:
        label = "정확도는 같지만 이득이 없다 (float16 유지)"
    limit = f"+{STAGE12_MAX_INCREASE * 100:.2f}%p"
    print(f"1. zeroth-val 맞춘 CER 상한 {limit} 이하: {'만족' if accurate else '불만족'}")
    print(
        f"2. 메모리 {STAGE12_MEMORY_CUT:.0%} 넘게 감소 {'예' if smaller else '아니오'}"
        f", 지연 중앙값 두 번 모두 {STAGE12_LATENCY_CUT:.0%} 넘게 감소 {'예' if faster else '아니오'}"
        f": {'만족' if smaller or faster else '불만족'}"
    )
    print(f"판정: {label}")
    return label


# --- 민감도 분석: 화자·문장 단위 부트스트랩 (규칙 2026-10-15, docs/experiments.md '추가 분석') ---


# 조건 이름 → (낮은 끝, 높은 끝)을 받아 조건이 성립하는지. 기록된 규칙의 문장 그대로다.
ROBUST_CONDITIONS = {
    "contains 0": lambda low, high: low <= 0 <= high,
    "excludes 0": lambda low, high: not low <= 0 <= high,
    "upper < 0": lambda low, high: high < 0,
    "upper <= +0.1%p": lambda low, high: high <= STAGE8_MAX_INCREASE,
    "upper < +0.10%p": lambda low, high: high < STAGE10_ZEROTH_MAX_ABOVE_N,
    "upper <= +1.0%p": lambda low, high: high <= STAGE10_FORGETTING_MAX,
    "verdict9": lambda low, high: verdict9_label(low, high) == "다른 도메인에서 더 나은 기준 모델이다",
}
# 구간이 판정·선택 조건에 들어간 validation 비교 전부(규칙의 표와 같은 순서·번호). 칸: 번호, 단계,
# 후보, 기준, 묶음, compare에 넘길 인자, 조건 이름, 기록된 조건 결과. 5단계 리포트는 같은 묶음의 원래 CER이다.
# 8단계 조건 2는 두 모델 × 두 후보 × 세 비교 = 12개, 출력 숫자 변환은 같은 리포트를 후처리 전후로 비교한다.
ROBUST_COMPARISONS = [
    (1, "2단계 선택", "small-b", "small-a", "zeroth-val", {}, "contains 0", True),
    (2, "3단계 선택", "turbo-l2", "turbo-l1", "zeroth-val", {}, "excludes 0", True),
    (
        3,
        "3단계 조건 1",
        "turbo-l2",
        "whisper-large-v3-turbo",
        "zeroth-val",
        {"without_digits": True},
        "upper < 0",
        True,
    ),
    (4, "4단계 조건 1", "turbo-t", "turbo-l2", "zeroth-val@telephone", {}, "upper < 0", True),
    (
        5,
        "4단계 조건 1",
        "turbo-t",
        "turbo-l2",
        "zeroth-val@telephone",
        {"without_digits": True},
        "upper < 0",
        True,
    ),
    (6, "5단계", "small-a-ct2-fallback", "small-a-ct2", "zeroth-val", {}, "upper <= +0.1%p", True),
    (7, "5단계", "small-b-ct2-fallback", "small-b-ct2", "zeroth-val", {}, "upper <= +0.1%p", True),
    (8, "7단계 조건 1", "turbo-n2", "turbo-n", "zeroth-val", {"scoring": "harmonized"}, "upper < 0", False),
    *[
        (
            9 + 6 * m + 3 * a + k,
            "8단계 조건 2",
            f"{model}-{arm}",
            model,
            set_name,
            kwargs,
            "upper <= +0.1%p",
            True,
        )
        for m, model in enumerate(("turbo-n", "turbo-n2"))
        for a, arm in enumerate(("d1a", "d1"))
        for k, (set_name, kwargs) in enumerate(
            (
                ("zeroth-val", {"scoring": "harmonized"}),
                ("zeroth-val", {"scoring": "harmonized", "without_digits": True}),
                ("fleurs-ko-val", {}),
            )
        )
    ],
    (
        21,
        "8단계 출력 숫자 변환 조건 1",
        "turbo-n",
        "turbo-n",
        "fleurs-ko-val",
        {"scoring": "itn", "baseline_scoring": None},
        "upper < 0",
        True,
    ),
    (22, "9단계", "qwen3-asr-1.7b", "whisper-large-v3-turbo", "fleurs-ko-val", {}, "verdict9", True),
    (
        23,
        "10단계 조건 1",
        "qwen3-asr-1.7b-qn",
        "turbo-n",
        "zeroth-val",
        {"scoring": "harmonized"},
        "upper < +0.10%p",
        True,
    ),
    (24, "10단계 조건 2", "qwen3-asr-1.7b-qn", "turbo-n", "fleurs-ko-val", {}, "upper < 0", False),
    (
        25,
        "10단계 망각 보고 줄",
        "qwen3-asr-1.7b-qn",
        "qwen3-asr-1.7b",
        "fleurs-ko-val",
        {},
        "upper <= +1.0%p",
        False,
    ),
]


# 구간 하나를 %p로 찍는 모양.
def _points(pair: tuple[float, float]) -> str:
    return f"[{pair[0] * 100:+.2f}, {pair[1] * 100:+.2f}]"


# 비교 하나의 민감도 분석 결과. 기록된 조건 결과(recorded)는 발화 단위 구간으로 다시 계산한 것과도 맞춰
# 본다(다르면 리포트나 코드가 기록과 달라진 것이라 멈춘다). 블록 단위 조건이 기록과 다르면 표시한다:
# 기록이 만족인데 블록 단위에서 만족하지 않으면 '가리지 못함'.
def robust_row(number, stage, candidate, baseline, set_name, kwargs, condition, recorded) -> dict:
    result = compare(set_name, candidate, baseline, with_blocks=True, **kwargs)
    holds = ROBUST_CONDITIONS[condition]
    utterance_ok = holds(*result["delta_interval"])
    if utterance_ok != recorded:
        raise SystemExit(f"{number}: the utterance-level interval no longer gives the recorded result")
    block_ok = holds(*result["block_interval"])
    if block_ok == recorded:
        mark = "같음"
    elif recorded:
        mark = "블록 단위에서는 가리지 못함"
    else:
        mark = "블록 단위에서는 다름"
    left_out = result["leave_one_out"]
    low_block = min(left_out, key=left_out.get)
    high_block = max(left_out, key=left_out.get)
    return {
        **result,
        "number": number,
        "stage": stage,
        "comparison": f"{candidate} - {baseline}",
        "set": set_name,
        "kwargs": kwargs,
        "condition": condition,
        "recorded": recorded,
        "block_ok": block_ok,
        "mark": mark,
        "leave_one_out_range": (left_out[low_block], low_block, left_out[high_block], high_block),
    }


# robust 명령: 규칙의 비교 25개를 차례로 계산해 한 줄씩 찍는다. 차이(점 추정), 발화 단위 구간, 블록 단위 구간,
# 블록 수, Zeroth의 화자 하나 빼기 범위(최솟값·최댓값과 그때 뺀 화자), 조건 결과(기록 → 블록 단위).
# FLEURS는 블록(문장)이 129개라 하나 빼기는 찍지 않는다(규칙). 판정은 바꾸지 않는다.
def robust() -> list[dict]:
    """기록된 판정에 쓰인 구간을 화자·문장 단위 부트스트랩으로 다시 계산한다(민감도 분석)."""
    rows = []
    for spec in ROBUST_COMPARISONS:
        row = robust_row(*spec)
        rows.append(row)
        options = ", ".join(f"{k}={v}" for k, v in row["kwargs"].items())
        line = (
            f"{row['number']:2d} {row['stage']} | {row['comparison']} | {row['set']}"
            + (f" ({options})" if options else "")
            + f" | {row['delta'] * 100:+.2f}%p utterance {_points(row['delta_interval'])}"
            f" block {_points(row['block_interval'])} ({row['blocks']} blocks)"
        )
        if row["set"].startswith("zeroth"):
            low, low_block, high, high_block = row["leave_one_out_range"]
            line += f" | leave one out {low * 100:+.2f} (-{low_block}) .. {high * 100:+.2f} (-{high_block})"
        recorded = "만족" if row["recorded"] else "불만족"
        block = "만족" if row["block_ok"] else "불만족"
        line += f" | {row['condition']}: {recorded} -> {block} ({row['mark']})"
        print(line)
    changed = [row["number"] for row in rows if row["mark"] != "같음"]
    print(f"블록 단위에서 기록과 다른 조건: {changed if changed else '없음'} (판정은 바꾸지 않는다)")
    return rows


# --- Zeroth 문장 겹침 (2026-10-24, docs/experiments.md '추가 확인: Zeroth 문장 겹침') ---

# 기본 정의: 학습 묶음 zeroth-train과, Zeroth·FLEURS 한국어의 validation·test 묶음. 숫자를 바꾼 학습 묶음
# zeroth-train-itn(6·7단계 N·N2, 10단계 QN)은 zeroth-train과 발화가 같고 정답의 숫자 표기만
# 다르다(prepare_itn). --write는 이 정의일 때만 쓴다. 커밋된 파일과 README가 적은 수가 같은 정의를 가리키게
# 하려는 것이다.
OVERLAP_TRAIN = "zeroth-train"
OVERLAP_SETS = ("zeroth-val", "zeroth-val500", "zeroth-test", "fleurs-ko-val", "fleurs-ko-test")
# 파생 파일에 적는 두 기준의 설명(파일 안의 데이터라 영어로 짧게 둔다. digit_utterances의 rule 칸과 같다).
OVERLAP_KEY_NOTES = {
    "raw": "the reference text as prepare stored it (stripped)",
    "normalized": "text_norm.normalize(text, language), the scoring normalization (splits.sentence_key)",
}


# 묶음 목록을 읽어 splits.sentence_overlap으로 센 뒤, 파생 파일에 쓸 기록을 만든다(쓰지는 않는다).
# 입력: 학습 묶음 이름과 평가 묶음 이름들(data/cache/sets/<이름>.json, store.load_set으로 읽음). 학습 묶음과
# 언어가 다른 묶음은 문장 키를 비교할 수 없어 거부한다.
# 출력(dict): train_set·language·command(다시 만드는 명령)·keys(두 기준의 설명)·per_utterance_key, train(학습
# 묶음 요약), sets[이름](발화·화자 수, 기준마다 요약 수, 그리고 정규화 기준의 not_in_train과 train_reads(발화
# ID순)). 정답 글은 넣지 않는다. 발화별 수는 정규화 기준 하나만 둔다(원문 기준은 요약 수로 비교한다).
def overlap_record(train_name: str, set_names: list[str]) -> dict:
    """학습 묶음과 평가 묶음들의 문장 겹침 기록(파생 파일의 내용)."""
    language, train = load_set(train_name, CACHE)
    loaded = {}
    for name in set_names:
        set_language, utterances = load_set(name, CACHE)
        if set_language != language:
            raise SystemExit(
                f"{name}: language {set_language!r} differs from {train_name} ({language!r});"
                " sentence keys are not comparable"
            )
        loaded[name] = utterances
    result = sentence_overlap(train, loaded, language)
    sets = {}
    for name, entry in result["sets"].items():
        per_utterance = entry["normalized"]
        sets[name] = {
            "utterances": entry["utterances"],
            "speakers": entry["speakers"],
            "speakers_in_train": entry["speakers_in_train"],
            # 기준마다 요약 수만(발화별 목록은 아래에 정규화 기준으로 한 번).
            **{
                kind: {k: v for k, v in entry[kind].items() if k not in ("train_reads", "not_in_train")}
                for kind in OVERLAP_KEYS
            },
            "not_in_train": per_utterance["not_in_train"],
            "train_reads": dict(sorted(per_utterance["train_reads"].items())),
        }
    command = f"python -m whisper_ko_ft.analyze overlap --train {train_name} --sets {' '.join(set_names)}"
    return {
        "train_set": train_name,
        "language": language,
        "command": command,
        "keys": OVERLAP_KEY_NOTES,
        "per_utterance_key": "normalized",
        "train": result["train"],
        "sets": sets,
    }


# overlap 명령: 문장 겹침을 묶음마다 한 줄씩 찍고, --write면 기록을 reports/sentence_overlap/zeroth.json에
# 쓴다(reports/의 다른 폴더처럼 커밋하는 파생 파일. 리포트가 아니라 tests/test_recorded_reports.py의 다시
# 채점 대상에서 뺀다). 다른 하위 명령과 달리 리포트가 아니라 묶음 목록을 읽는다(학습 문장은 리포트에 없다).
# 왜 필요한가: 화자 기준 분리는 화자만 떼므로, Zeroth validation·test 문장이 학습에 다른 화자의 녹음으로
# 들어 있는지를 숫자로 남겨야 Zeroth 수치를 '처음 듣는 화자가 학습에서 본 문장을 읽는 조건'으로 바르게 읽을
# 수 있다.
# 주의: 측정이 아니라 데이터 성질의 셈이라 판정은 없고 기록된 수치를 바꾸지 않는다. 데이터가 같으면 몇 번을
# 다시 써도 같은 파일이 나온다(무작위 요소 없음). 학습에 없는 발화가 5개 이하면 ID를 함께 찍는다.
def overlap(
    train_name: str = OVERLAP_TRAIN,
    set_names: list[str] | tuple[str, ...] = OVERLAP_SETS,
    write: bool = False,
) -> dict:
    """학습 묶음과 평가 묶음들의 문장·화자 겹침을 찍는다. `write`면 파생 파일로 남긴다."""
    if write and (train_name, tuple(set_names)) != (OVERLAP_TRAIN, OVERLAP_SETS):
        raise SystemExit("--write records only the default --train and --sets (the committed definition)")
    record = overlap_record(train_name, list(set_names))
    train = record["train"]
    print(
        f"{train_name}: {train['utterances']} utterances, {train['speakers']} speakers"
        + "".join(
            f"; {kind}: {train[kind]['sentences']} sentences, read up to {train[kind]['max_reads']} times,"
            f" {train[kind]['repeated_sentences']} read more than once"
            for kind in OVERLAP_KEYS
        )
    )
    for name, entry in record["sets"].items():
        lines = [
            f"{name}: {entry['utterances']} utterances, {entry['speakers']} speakers"
            f" ({entry['speakers_in_train']} also in {train_name})"
        ]
        for kind in OVERLAP_KEYS:
            counts = entry[kind]
            lines.append(
                f"{kind}: utterances whose sentence is in {train_name} {counts['utterances_in_train']}"
                f" of {entry['utterances']}, sentences {counts['sentences_in_train']}"
                f" of {counts['sentences']}, repeated within the set {counts['repeated_sentences']}"
                f" sentences ({counts['repeated_utterances']} utterances)"
            )
        missing = entry["not_in_train"]
        lines.append(
            f"sentence not in {train_name} (normalized): {len(missing)} of {entry['utterances']} utterances"
            + (f" ({' '.join(missing)})" if 0 < len(missing) <= 5 else "")
        )
        print("\n  ".join(lines))
    if write:
        target = REPORTS / "sentence_overlap" / "zeroth.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"wrote {target}")
    return record


# table --bootstrap-samples의 값 검사: 1 이상의 정수만 받는다(표본이 없으면 구간을 낼 수 없다). argparse가
# 'argument --bootstrap-samples: …' 오류로 계산 전에 멈추도록 ArgumentTypeError를 낸다.
def _sample_count(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


# 명령줄 진입점. 하위 명령마다 인자를 정하고 해당 함수를 부른다. 명령과 단계: digits·table·loops(공통),
# verdict(2·3단계), verdict6(6단계), changes·verdict8·itn8(8단계), gate9·verdict9(9단계),
# gate10·smoke10·pick10·verdict10(10단계), gate11·blocks(11단계), verdict12(12단계), robust(민감도 분석),
# overlap(문장 겹침, 리포트가 아니라 묶음 목록을 읽는 유일한 명령).
def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    digits = commands.add_parser("digits")
    digits.add_argument("--set", required=True, dest="set_name")
    digits.add_argument("--force", action="store_true")
    tab = commands.add_parser("table")
    tab.add_argument("--set", required=True, dest="set_name")
    tab.add_argument("--reports", nargs="+", required=True)
    tab.add_argument("--baseline")
    tab.add_argument("--no-digit-split", action="store_true", help="FLEURS 묶음에는 숫자 발화 목록이 없다")
    tab.add_argument("--harmonized", action="store_true", help="itn.harmonize 뒤에 다시 채점한다(한국어)")
    tab.add_argument(
        "--scoring",
        choices=list(SCORINGS),
        help=(
            "저장된 글을 다른 방식으로 다시 채점한다. 한국어 채점(harmonized·itn·itn-harmonized·itn-latin·"
            "agnostic)은 Zeroth·FLEURS 한국어 묶음에, en-whisper(영어 표준 정규화, 판정 없는 보조 지표)는"
            " FLEURS 영어 묶음에만 쓴다"
        ),
    )
    tab.add_argument(
        "--bootstrap-samples",
        type=_sample_count,
        default=BOOTSTRAP_SAMPLES,
        help=(
            f"오류율 구간과 차이 구간의 부트스트랩 표본 수(기본 {BOOTSTRAP_SAMPLES:,}, seed"
            f" {BOOTSTRAP_SEED}, 1 이상). 기록된 구간과 판정은 모두 기본값이다. 판정 문턱에 가까운"
            " 구간 끝이 표본 수에 따라 얼마나 움직이는지 사후에 볼 때 쓴다(예: 100000). 표본을 행"
            " 묶음으로 나눠 뽑으므로 표본 수를 키워도 시간만 늘고 메모리는 거의 늘지 않는다. 기본값이"
            " 아니면 묶음 머리 줄에 표본 수를 적는다"
        ),
    )
    tab.add_argument(
        "--decimals",
        type=int,
        choices=range(7),
        default=2,
        metavar="0-6",
        help="오류율·구간·차이(%%, %%p)의 소수 자릿수(기본 2). 상대 변화는 늘 한 자리다",
    )
    ver = commands.add_parser("verdict")
    ver.add_argument("--baseline", required=True)
    ver.add_argument("--candidate", required=True)
    ver.add_argument("--no-interval-rule", action="store_true", help="2단계 규칙(구간 조건 없음)")
    ver6 = commands.add_parser("verdict6")
    ver6.add_argument("--baseline", required=True)
    ver6.add_argument("--previous", required=True, help="원래 표기로 학습한 3단계 모델")
    ver6.add_argument("--candidate", required=True)
    loop = commands.add_parser("loops")
    loop.add_argument("--set", required=True, dest="set_name")
    loop.add_argument("--reports", nargs="+", required=True)
    loop.add_argument("--harmonized", action="store_true", help="itn.harmonize 뒤에 다시 채점한다(한국어)")
    loop.add_argument("--scoring", choices=list(SCORINGS))
    chg = commands.add_parser("changes")
    chg.add_argument("--set", required=True, dest="set_name")
    chg.add_argument("--base", required=True, help="기록된 greedy 리포트")
    chg.add_argument("--reports", nargs="+", required=True)
    chg.add_argument("--scoring", choices=list(SCORINGS))
    ver8 = commands.add_parser("verdict8")
    ver8.add_argument("--models", nargs="+", required=True, help="greedy 리포트들. 후보는 <model>-<arm>")
    ver8.add_argument("--arms", nargs="+", required=True, help="선호 순서(단순한 것부터)")
    itn = commands.add_parser("itn8")
    itn.add_argument("--model", required=True)
    itn.add_argument("--baseline", required=True)
    itn.add_argument("--fleurs", default="fleurs-ko-val")
    itn.add_argument("--zeroth", default="zeroth-val")
    gate = commands.add_parser("gate9")
    gate.add_argument("--fp16", required=True, help="fp16 실행의 raw 리포트 파일")
    gate.add_argument("--fp32", required=True, help="fp32 실행의 raw 리포트 파일")
    ver9 = commands.add_parser("verdict9")
    ver9.add_argument("--candidate", required=True, help="raw 출력. <candidate>-fixed도 보고한다")
    ver9.add_argument("--baseline", required=True)
    ver9.add_argument("--others", nargs="*", default=[], help="나란히 보일 파인튜닝한 모델들")
    gate_a = commands.add_parser("gate10")
    gate_a.add_argument("--file", required=True, help="새 어댑터 실행의 raw 리포트 파일")
    gate_a.add_argument("--recorded", required=True, help="9단계 fp16 점검 파일")
    smoke = commands.add_parser("smoke10")
    smoke.add_argument("--steps", required=True, help="outputs/<스모크 실행>/steps.jsonl")
    smoke.add_argument("--mem", required=True, help="스모크 실행의 메모리 감시(memwatch) CSV")
    smoke.add_argument("--expected-steps", type=int, default=50)
    pick = commands.add_parser("pick10")
    pick.add_argument(
        "--files", nargs="+", required=True, help="zeroth-val500 raw 리포트 파일들(스텝 순서로)"
    )
    ver10 = commands.add_parser("verdict10")
    ver10.add_argument("--candidate", required=True, help="raw 출력. <candidate>-fixed도 보고한다")
    ver10.add_argument("--n", required=True, help="후보가 대신할 모델")
    ver10.add_argument("--untrained", required=True, help="학습 전 기본 모델")
    ver10.add_argument("--others", nargs="*", default=[])
    commands.add_parser(
        "robust", help="기록된 판정의 구간을 화자·문장 단위 부트스트랩으로 다시 본다(민감도 분석)"
    )
    gate_s = commands.add_parser("gate11", help="11단계 관문: 스케일을 정해 합친 실행이 기록과 같은지")
    gate_s.add_argument("--file", required=True, help="스케일을 정해 합친 실행의 raw 리포트 파일(--out)")
    gate_s.add_argument("--recorded", required=True, help="비교할 기록 파일(10단계 고르기 또는 9단계 점검)")
    blk = commands.add_parser("blocks", help="비교 하나의 블록 단위 구간(보고만)")
    blk.add_argument("--set", required=True, dest="set_name")
    blk.add_argument("--candidate", required=True)
    blk.add_argument("--baseline", required=True)
    blk.add_argument("--scoring", choices=list(SCORINGS))
    blk.add_argument("--without-digits", action="store_true", help="Zeroth의 숫자 발화를 뺀다")
    ver12 = commands.add_parser("verdict12", help="12단계 판정: CTranslate2 int8_float16")
    ver12.add_argument(
        "--speed",
        required=True,
        help="scripts/speed.py --preset ct2-int8이 쓴 speed.json(complete: false인 중간 기록은 거부한다)",
    )
    ver12.add_argument("--candidate", default="turbo-n-ct2-int8")
    ver12.add_argument("--baseline", default="turbo-n-ct2")
    over = commands.add_parser(
        "overlap",
        help="학습 묶음과 평가 묶음의 문장·화자 겹침을 센다(리포트가 아니라 prepare의 묶음 목록을 읽는다)",
        description=(
            "학습 묶음(--train)과 평가 묶음들(--sets)이 같은 문장을 얼마나 함께 갖는지를 원문 기준과"
            " 채점 정규화 기준(text_norm.normalize)으로 세고, 화자 겹침도 센다. data/cache/sets의 묶음"
            " 목록을 읽으므로 먼저 prepare를 돌려야 한다. 판정은 없고 기록된 수치를 바꾸지 않는다."
        ),
    )
    over.add_argument(
        "--train",
        default=OVERLAP_TRAIN,
        help=f"학습 묶음 이름(기본 {OVERLAP_TRAIN}, data/cache/sets/<이름>.json)",
    )
    over.add_argument(
        "--sets",
        nargs="+",
        default=list(OVERLAP_SETS),
        help=(
            "평가 묶음 이름들(기본은 Zeroth·FLEURS 한국어의 validation·test 다섯 개). 학습 묶음과"
            " 언어가 같아야 한다"
        ),
    )
    over.add_argument(
        "--write",
        action="store_true",
        help=(
            "발화 ID → 학습 낭독 횟수와 요약 수를 reports/sentence_overlap/zeroth.json에 쓴다(정답 글은"
            " 넣지 않는다). 기본 --train·--sets일 때만 쓴다"
        ),
    )
    args = parser.parse_args()

    if args.command == "digits":
        write_digits(args.set_name, args.force)
    elif args.command == "verdict":
        verdict(args.baseline, args.candidate, not args.no_interval_rule)
    elif args.command == "verdict6":
        verdict6(args.baseline, args.previous, args.candidate)
    elif args.command == "loops":
        loops(args.set_name, args.reports, args.harmonized, args.scoring)
    elif args.command == "changes":
        changes(args.set_name, args.base, args.reports, args.scoring)
    elif args.command == "verdict8":
        verdict8(args.models, args.arms)
    elif args.command == "itn8":
        itn8(args.model, args.baseline, args.fleurs, args.zeroth)
    elif args.command == "gate9":
        gate9(args.fp16, args.fp32)
    elif args.command == "verdict9":
        verdict9(args.candidate, args.baseline, args.others)
    elif args.command == "gate10":
        gate10(args.file, args.recorded)
    elif args.command == "smoke10":
        smoke10(args.steps, args.mem, args.expected_steps)
    elif args.command == "pick10":
        pick10(args.files)
    elif args.command == "verdict10":
        verdict10(args.candidate, args.n, args.untrained, args.others)
    elif args.command == "robust":
        robust()
    elif args.command == "gate11":
        gate11(args.file, args.recorded)
    elif args.command == "blocks":
        blocks(args.set_name, args.candidate, args.baseline, args.scoring, args.without_digits)
    elif args.command == "verdict12":
        verdict12(args.speed, args.candidate, args.baseline)
    elif args.command == "overlap":
        overlap(args.train, args.sets, args.write)
    else:
        table(
            args.set_name,
            args.reports,
            args.baseline,
            not args.no_digit_split,
            args.harmonized,
            args.scoring,
            samples=args.bootstrap_samples,
            decimals=args.decimals,
        )


if __name__ == "__main__":
    main()
