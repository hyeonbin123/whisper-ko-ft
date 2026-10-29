"""docs/experiments.md의 수치를 커밋된 리포트에서 지금 코드로 다시 계산한다.

text_norm, metrics, itn, jiwer의 변경이 이 수치를 움직인다면 의도한 변경이어야 한다: 여기의 기대값을 고치고
그 이유를 docs/experiments.md에 적는다.
"""

# [테스트 묶음] 커밋된 리포트(reports/)의 수치를 지금 코드로 다시 계산해 기록과 같은지 확인한다(위 docstring).
# 원래 채점은 리포트에 저장된 발화별 편집 수·길이와 요약 오류율을, 맞춘 CER과 출력 후처리 채점은 저장하지
# 않으므로 총합을 숫자로 고정해 비교한다.
# [왜 중요한가] docs/experiments.md와 README의 수치는 모두 이 리포트에서 나왔다. 정규화·채점·숫자 변환 코드나
# jiwer 버전이 바뀌면 같은 리포트에서 다른 수치가 나와 문서와 코드가 어긋난다. 그런 변경은 의도한 것일 때만
# 고정값을 고치고 이유를 문서에 남기게 한다.
# [범위] 모델이나 GPU 없이 저장된 글만 다시 채점하므로 CI에서도 돈다.

import json

import numpy as np
import pytest

from whisper_ko_ft import analyze
from whisper_ko_ft.metrics import error_rate, score
from whisper_ko_ft.paths import REPORTS

# FILES: reports/<이름>/<묶음>.json 전부. 리포트가 새로 커밋되면 자동으로 검사 대상이 된다. 리포트가 아닌
# 파생 파일 폴더는 뺀다: 숫자 발화 목록(digit_utterances), 문장 겹침(sentence_overlap, analyze overlap이 쓰고
# tests/test_analyze.py가 수를 고정한다), 13단계 문장 hold-out 정의(sentence_holdout, prepare_heldout이 쓰고
# tests/test_prepare_heldout.py가 수를 고정한다). 셋 다 요약(summary)과 발화별 정답·가설이 없다.
DERIVED = {"digit_utterances", "sentence_overlap", "sentence_holdout"}
FILES = sorted(p for p in REPORTS.glob("*/*.json") if p.parent.name not in DERIVED)


# 리포트를 하나도 못 찾으면 아래 검사가 아무것도 하지 않고 통과하므로, 먼저 있는지 확인한다.
def test_reports_are_found():
    assert FILES


# 리포트마다: 발화별로 정답·가설을 다시 채점해 저장된 (편집 수, 길이)와 같고, 정답이 빈 발화를 뺀 묶음
# 오류율이 요약의 error_rate(소수 다섯째 자리)와 같아야 한다. 리포트 이름으로 테스트 ID를 붙여 어느 리포트가
# 깨졌는지 바로 보이게 한다.
@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_stored_scores_are_reproduced(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    language = "ko" if data["summary"]["metric"] == "cer" else "en"
    for row in data["utterances"]:
        scored = score(row["reference"], row["hypothesis"], language)
        got = (scored.edits, scored.length) if scored else (0, 0)
        assert got == (row["edits"], row["length"]), row["id"]
    kept = [row for row in data["utterances"] if row["length"]]
    rate = error_rate(np.array([r["edits"] for r in kept]), np.array([r["length"] for r in kept]))
    assert round(rate, 5) == data["summary"]["error_rate"]


# 맞춘 CER(6·7단계와 2026-10-02 N 확인부터 쓴 Zeroth 지표)의 총합 고정. 맞춘 CER은 리포트에 저장하지 않고
# 부를 때마다 다시 채점한다. 반올림한 오류율로 비교하면 그 값을 움직이지 못할 만큼 작은 변화가 묻히므로 편집
# 수와 정답 길이의 합을 그대로 비교한다. 각 줄 끝의 퍼센트는 문서에 적힌 값이다. 새 단계의 리포트를 커밋할
# 때마다 이 목록에 줄을 더했다.
@pytest.mark.parametrize(
    ("report", "set_name", "edits", "length"),
    [
        ("whisper-large-v3-turbo", "zeroth-val", 3370, 93621),  # 3.60%
        ("turbo-l2", "zeroth-val", 1203, 93621),  # 1.28%
        ("turbo-n", "zeroth-val", 1172, 93621),  # 1.25%
        ("turbo-n2", "zeroth-val", 1324, 93621),  # 1.41%
        ("whisper-large-v3-turbo", "zeroth-test", 814, 19272),  # 4.22%
        ("turbo-l2", "zeroth-test", 394, 19272),  # 2.04%
        ("turbo-n", "zeroth-test", 517, 19272),  # 2.68%
        ("turbo-n2", "zeroth-test", 387, 19272),  # 2.01%
        # 2026-10-02: CTranslate2로 잰 N, 재시도 끔과 켬
        ("turbo-n-ct2", "zeroth-val", 1203, 93621),  # 1.28%
        ("turbo-n-ct2-fallback", "zeroth-val", 1203, 93621),  # 1.28%
        ("turbo-n-ct2", "zeroth-test", 373, 19272),  # 1.94%
        ("turbo-n-ct2-fallback", "zeroth-test", 373, 19272),  # 1.94%
        # 2026-10-02: val-500 속도 확인, transformers 배치 1과 CTranslate2
        ("turbo-n-b1", "zeroth-val500", 320, 18321),  # 1.75%
        ("turbo-n-ct2", "zeroth-val500", 316, 18321),  # 1.72%
        # 2026-10-04, 8단계: 재시도 디코딩(D1a를 고름. D1도 같은 가설을 냈다)
        ("turbo-n-d1a", "zeroth-val", 1172, 93621),  # 1.25%
        ("turbo-n-d1", "zeroth-val", 1172, 93621),  # 1.25%
        ("turbo-n2-d1a", "zeroth-val", 1119, 93621),  # 1.20%
        ("turbo-n2-d1", "zeroth-val", 1119, 93621),  # 1.20%
        ("turbo-n-d1a", "zeroth-test", 381, 19272),  # 1.98%
        # 2026-10-04, 9단계: 학습 전 Qwen3-ASR-1.7B, raw와 공식 반복 제거(바뀐 것 없음)
        ("qwen3-asr-1.7b", "zeroth-val", 3231, 93621),  # 3.45%
        ("qwen3-asr-1.7b-fixed", "zeroth-val", 3231, 93621),  # 3.45%
        ("qwen3-asr-1.7b", "zeroth-test", 817, 19272),  # 4.24%
        ("qwen3-asr-1.7b-fixed", "zeroth-test", 817, 19272),  # 4.24%
        # 2026-10-05, 10단계: Qwen3-ASR-1.7B + LoRA(QN, 체크포인트 1500), raw와 공식 반복 제거(바뀐 것
        # 없음). QN과 학습 전 모델의 전화 채널. 체크포인트 고르기
        ("qwen3-asr-1.7b-qn", "zeroth-val", 864, 93621),  # 0.92%
        ("qwen3-asr-1.7b-qn-fixed", "zeroth-val", 864, 93621),  # 0.92%
        ("qwen3-asr-1.7b-qn", "zeroth-test", 293, 19272),  # 1.52%
        ("qwen3-asr-1.7b-qn-fixed", "zeroth-test", 293, 19272),  # 1.52%
        ("qwen3-asr-1.7b-qn", "zeroth-val@telephone", 1254, 93621),  # 1.34%
        ("qwen3-asr-1.7b", "zeroth-val@telephone", 3566, 93621),  # 3.81%
        ("qwen3-asr-1.7b-qn-steps", "zeroth-val500-step500", 552, 18321),  # 3.01%
        ("qwen3-asr-1.7b-qn-steps", "zeroth-val500-step1000", 281, 18321),  # 1.53%
        ("qwen3-asr-1.7b-qn-steps", "zeroth-val500-step1500", 213, 18321),  # 1.16%
        # 2026-10-18, 11단계: QN 갱신량을 줄여 합친 것. 판정 후보 0.5(raw와 fixed), 보고 팔, 관문 (a)(s =
        # 1.0이 10단계 체크포인트 고르기 파일과 같다)
        ("qwen3-asr-1.7b-qn-s050", "zeroth-val", 1386, 93621),  # 1.48%
        ("qwen3-asr-1.7b-qn-s050-fixed", "zeroth-val", 1386, 93621),  # 1.48%
        ("qwen3-asr-1.7b-qn-s050", "zeroth-test", 433, 19272),  # 2.25%
        ("qwen3-asr-1.7b-qn-s050-fixed", "zeroth-test", 433, 19272),  # 2.25%
        ("qwen3-asr-1.7b-qn-s025", "zeroth-val", 2291, 93621),  # 2.45%
        ("qwen3-asr-1.7b-qn-s075", "zeroth-val", 969, 93621),  # 1.04%
        ("qwen3-asr-1.7b-qn-lm050", "zeroth-val", 1354, 93621),  # 1.45%
        ("qwen3-asr-1.7b-qn-step1000", "zeroth-val", 1042, 93621),  # 1.11%
        ("qwen3-asr-1.7b-qn-scale-gate", "zeroth-val500-s100", 213, 18321),  # 1.16%
        # 2026-10-18, 12단계: N을 CTranslate2 int8_float16으로(기록된 float16은 위의 turbo-n-ct2 1,203개)
        ("turbo-n-ct2-int8", "zeroth-val", 1254, 93621),  # 1.34%
    ],
)
# analyze.load_rows로 맞춘 채점을 다시 해서 총합이 고정값과 같은지.
def test_recorded_harmonized_totals(report, set_name, edits, length):
    rows = analyze.load_rows(report, set_name, harmonized=True)
    got_edits, got_lengths = analyze.arrays(rows, sorted(rows))
    assert (int(got_edits.sum()), int(got_lengths.sum())) == (edits, length)


# 8단계 출력 숫자 변환(2026-10-04) 판정(itn8)의 근거가 된 총합(부를 때마다 다시 채점한다): N과 기준선
# turbo의 FLEURS 한국어 후처리 CER, 후처리해도 맞춘 CER이 그대로인 N의 Zeroth.
@pytest.mark.parametrize(
    ("report", "set_name", "scoring", "edits", "length"),
    [
        ("turbo-n", "fleurs-ko-val", "itn", 512, 10419),  # 4.91%(후처리 전 5.28%)
        ("whisper-large-v3-turbo", "fleurs-ko-val", "itn", 492, 10419),  # 4.72%(후처리 전 4.75%)
        ("turbo-n", "zeroth-val", "itn-harmonized", 1172, 93621),  # 1.25%, 맞춘 CER과 같다
    ],
)
# 지정한 채점 방식으로 다시 채점한 총합이 고정값과 같은지.
def test_recorded_post_processed_totals(report, set_name, scoring, edits, length):
    rows = analyze.load_rows(report, set_name, scoring=scoring)
    got_edits, got_lengths = analyze.arrays(rows, sorted(rows))
    assert (int(got_edits.sum()), int(got_lengths.sum())) == (edits, length)
