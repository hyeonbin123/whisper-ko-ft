"""Tables from the reports: digit utterances, error rates with intervals, paired comparisons.

Usage:
    uv run python -m whisper_ko_ft.analyze digits --set zeroth-val
    uv run python -m whisper_ko_ft.analyze table --set zeroth-val --reports whisper-small small-a \
        --baseline whisper-small

    uv run python -m whisper_ko_ft.analyze verdict --baseline whisper-small --candidate small-a

`digits` fixes the list of "digit utterances" of a set (docs/experiments.md, "숫자 표기"): utterances where
either base model wrote an Arabic digit. It is written once and not overwritten without --force.
`verdict` applies the verdict rules of stages 2 and 3 to validation reports (the stage 3 interval condition is
on unless --no-interval-rule).
`verdict6` applies the stage 6 rules. `--harmonized` scores the stored references and hypotheses again after
`itn.harmonize`, so that either notation of a number gets the same score (docs/experiments.md, stage 6).
`loops` lists the utterances with more edits than reference characters (repetition loops), per report.

Stage 8: `--scoring` scores the stored texts again in other ways (SCORINGS; the reports are never changed).
`changes` shows what a fallback decoding changed against the recorded greedy report, `verdict8` applies the
decoding rules and `itn8` the rules of the output post-processing.

Stage 9 (another base model, Qwen3-ASR): `gate9` decides fp16 or fp32 from two runs on the first 100
utterances of zeroth-val500, `verdict9` applies the other-domain rule and prints what is reported with it.

Stage 10 (LoRA on Qwen3-ASR): `gate10` checks that a fresh adapter (B=0) reproduces the stage 9 fp16 check,
`smoke10` reads a smoke run's step log and GPU memory samples against the gates, `pick10` picks the checkpoint
by harmonized CER on zeroth-val500, and `verdict10` applies the rules against N.
"""

# [파일 역할] 커밋된 리포트(reports/)만 읽어 표와 판정을 내는 분석 도구. 모델을 다시 돌리지 않는다.
# docs/experiments.md의 수치와 판정 문장은 대부분 이 파일의 하위 명령 출력이다.
# [왜 필요한가] 이 프로젝트는 판정 규칙을 재기 전에 적고 커밋한다. 규칙을 코드(상수와 verdict 함수)로 옮겨
# 두면, 결과를 본 뒤 기준을 슬쩍 바꾸거나 손으로 계산하다 틀리는 일을 막고, 누구든 같은 리포트에서 같은 판정을
# 다시 낼 수 있다.
# [주요 구성]
# - 채점 방식: 원래 채점, 맞춘 CER(harmonized, 6단계), 출력 숫자 변환(itn, 8단계), 글자 이름 변환(itn-latin),
#   표기 무관(agnostic). 리포트의 글은 바꾸지 않고 채점할 때만 적용한다(SCORINGS).
# - 공통 도구: load_rows(리포트 읽기와 검사), compare(두 리포트의 짝지은 차이와 구간), table(표), loops(되풀이
#   발화), changes(재시도가 바꾼 것).
# - 단계별 판정: verdict(2·3단계), verdict6, verdict8·itn8, gate9·verdict9, gate10·smoke10·pick10·verdict10.
# [연결] metrics(채점·구간), itn·latin(표기 변환), paths.REPORTS를 쓴다. tests/test_recorded_reports.py가
# load_rows·arrays로 커밋된 리포트의 맞춘 CER 총합을 고정한다.

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics

# numpy: 발화별 편집 수·길이 배열.
import numpy as np

# itn·latin: 다시 채점할 때 쓰는 표기 변환. metrics: 채점, 오류율, 부트스트랩 구간과 짝지은 차이 구간.
# paths.REPORTS: 리포트 폴더.
from whisper_ko_ft.itn import harmonize, to_digits
from whisper_ko_ft.latin import letters_to_latin
from whisper_ko_ft.metrics import difference_interval, error_rate, interval, score
from whisper_ko_ft.paths import REPORTS

# BASE_MODELS: '숫자 발화' 목록을 정할 때 보는 기본 모델 두 개(1단계 기준선).
BASE_MODELS = ("whisper-small", "whisper-large-v3-turbo")
# 2·3단계 판정 기준(docs/experiments.md 2단계 '판정 규칙'): 같은 도메인은 Zeroth validation CER이 기준선보다
# 상대 20% 이상 낮고(-0.20), 숫자 발화를 빼고도 상대 10% 이상 낮아야 한다(-0.10). 다른 도메인은 FLEURS 한국어
# validation CER이 절대 1.0%p(0.010)를 넘게 나빠지지 않아야 한다. 이 1.0%p는 6·7단계와 10단계 망각 줄에도
# 그대로 쓰인다.
# Verdict rules (docs/experiments.md, "판정 규칙"): relative CER change on Zeroth validation, overall and
# without digit utterances, and the largest allowed absolute CER increase on FLEURS Korean validation.
IN_DOMAIN_RELATIVE = -0.20
IN_DOMAIN_NO_DIGITS_RELATIVE = -0.10
OUT_OF_DOMAIN_MAX_INCREASE = 0.010
# 6단계: N의 맞춘 CER이 3단계 L2보다 0.3%p를 넘게 높으면 같은 도메인 이득을 잃은 것으로 본다.
# Stage 6: the harmonized CER may be at most this much above the stage 3 model's.
STAGE6_MAX_ABOVE_L2 = 0.003
# 8단계: 재시도 후보는 어느 오류율도 짝지은 구간 상한 기준 +0.1%p를 넘게 올리면 안 되고, 출력 숫자 변환은
# Zeroth 맞춘 CER을 +0.05%p를 넘게 올리면 안 된다(docs/experiments.md 8단계 판정 규칙).
# Stage 8: a fallback arm may raise no rate by more than this (upper end of the paired 95% interval), and the
# output post-processing may raise the harmonized Zeroth CER by at most STAGE8_ITN_MAX_INCREASE.
STAGE8_MAX_INCREASE = 0.001
STAGE8_ITN_MAX_INCREASE = 0.0005
# 8단계 판정에 쓰는 validation 세 묶음과 채점: Zeroth는 맞춘 채점, FLEURS 한국어·영어는 원래 채점. 9·10단계의
# 되풀이 보고도 같은 묶음을 쓴다.
# The sets of the stage 8 decoding rules, each with the scoring its loops and rates are counted in.
STAGE8_SETS = {"zeroth-val": "harmonized", "fleurs-ko-val": None, "fleurs-en-val": None}
# 9단계 fp16 점검: fp16과 fp32의 원래 CER 차이가 ±0.20%p 안이고 빈 가설·넘침이 없어야 fp16으로
# 잰다(docs/experiments.md 9단계 'fp16 점검').
# Stage 9: fp16 is used when its raw CER on the first 100 utterances of zeroth-val500 is within this of the
# fp32 run's (either direction), with no empty hypothesis and no utterance with non-finite logits.
STAGE9_FP16_MAX_GAP = 0.002
# 10단계 스모크 관문(docs/experiments.md 10단계 '관문' (c)): 앞 20스텝(GradScaler가 배율을 맞추는 구간)은
# 건너뜀을 세지 않고 그 뒤 5개까지 허용, 학습 프로세스의 공유 GPU 메모리 512MiB 이하, nvidia-smi 카드 전체
# 10,752MiB 미만, torch 최대 할당 7,168MiB 이하. 스텝 시간 중앙값 × 1,500이 8시간을 넘으면 1,000스텝으로
# 줄인다.
# 공유 메모리 기준이 0이 아니라 512MiB인 이유(옆 주석): 9단계 디코딩 프로세스는 넘치기 전에도 이 카운터가 약
# 230MB였다.
# Stage 10. Smoke gates: GradScaler skips after the first STAGE10_SKIP_GRACE steps, the Windows "GPU Process
# Memory" shared usage of the training process (stage 9 decoding sat at ~230 MB until it spilled to 3.4 GB),
# nvidia-smi memory.used of the whole card, torch's peak allocation, and the time budget of the full run.
STAGE10_SKIP_GRACE = 20
STAGE10_MAX_SKIPS = 5
STAGE10_MAX_SHARED_MB = 512
STAGE10_MAX_GPU_MB = 10_752  # 10.5 GiB, as nvidia-smi counts (MiB)
STAGE10_MAX_ALLOCATED_MB = 7_168  # 7 GiB
STAGE10_HOURS = 8.0
STAGE10_STEPS, STAGE10_FEWER_STEPS = 1_500, 1_000
# 10단계 판정: 조건 1은 Zeroth 맞춘 CER에서 QN − N 구간 상한 < +0.10%p. 망각 보고 줄은 FLEURS 한국어에서 QN −
# 학습 전 Qwen 상한 ≤ +1.0%p(판정은 바꾸지 않는 보고용).
# _DIGIT: 정답에 아라비아 숫자가 있는지(FLEURS를 숫자 유무로 나눌 때).
# Verdict: QN may be at most this much above N on Zeroth (upper end of the paired 95% interval of the
# harmonized CER); the reported forgetting line allows this much above the untrained Qwen on FLEURS Korean.
STAGE10_ZEROTH_MAX_ABOVE_N = 0.001
STAGE10_FORGETTING_MAX = 0.010
_DIGIT = re.compile(r"[0-9]")


# 채점 방식 다섯 가지. 모두 (정답, 가설)을 받아 바꾼 (정답, 가설)을 돌려주고, 저장된 리포트는 바꾸지 않는다.
# harmonized(6단계 맞춘 CER): 양쪽에 같은 숫자 표기 통일을 적용해 '오 월'과 '5월'이 같은 점수를 받게 한다.
def _harmonized(reference: str, hypothesis: str) -> tuple[str, str]:
    """Stage 6: one number notation on both sides."""
    return harmonize(reference), harmonize(hypothesis)


# itn(8단계 출력 후처리): 가설의 숫자 낱말만 아라비아 숫자로 바꾸고 정답은 그대로 둔다. 서비스에서 출력에
# 후처리를 붙였을 때의 원래 CER이다.
def _itn(reference: str, hypothesis: str) -> tuple[str, str]:
    """Stage 8 output post-processing: spelled numbers in the hypothesis become digits; reference as is."""
    return reference, to_digits(hypothesis)


# itn-harmonized: 후처리한 가설을 맞춘 채점으로 다시 잰다. 후처리가 맞춘 채점에서 다른 것을 바꾸지 않는지 보는
# 데 쓴다(itn8의 조건 2).
def _itn_harmonized(reference: str, hypothesis: str) -> tuple[str, str]:
    """The post-processed hypothesis under the stage 6 scoring."""
    return _harmonized(*_itn(reference, hypothesis))


# itn-latin(판정 없음): 숫자 후처리에 더해 한글 글자 이름을 라틴 글자로 바꾼 가설.
def _itn_latin(reference: str, hypothesis: str) -> tuple[str, str]:
    """Stage 8, no verdict: the post-processing plus letter names as Latin letters."""
    return reference, letters_to_latin(to_digits(hypothesis))


# agnostic(판정 없는 보조 지표, 표기 무관 CER): 숫자와 글자 이름을 양쪽 모두 같은 표기로 맞춘다.
def _agnostic(reference: str, hypothesis: str) -> tuple[str, str]:
    """Stage 8 auxiliary metric: numbers and letter names in one notation on both sides."""
    return letters_to_latin(harmonize(reference)), letters_to_latin(harmonize(hypothesis))


# SCORINGS: --scoring 이름 → 변환 함수. 모두 한국어 CER용이다(load_rows가 영어 묶음에서는 거부한다).
# How the stored reference and hypothesis are rewritten before Korean CER is computed again.
SCORINGS = {
    "harmonized": _harmonized,
    "itn": _itn,
    "itn-harmonized": _itn_harmonized,
    "itn-latin": _itn_latin,
    "agnostic": _agnostic,
}


# 리포트 하나를 읽어 발화 ID → 행 dict로 돌려준다. 정답이 빈 행(길이 0)은 뺀다. scoring을 주면 행마다 다시
# 채점한 편집 수·길이로 바꾼다(harmonized=True는 scoring='harmonized'와 같다).
# 거부하는 경우(규칙을 지키게 하는 검사): 영어 묶음에 한국어용 채점을 주면, --limit으로 일부만 잰 리포트(커밋
# a29a975), 발화 ID가 겹치는 리포트(1단계에서 FLEURS 행이 빠진 오류를 되풀이하지 않게).
# 호출 위치: table, compare, loops, changes, write_digits, tests/test_recorded_reports.py.
def load_rows(
    report: str, set_name: str, harmonized: bool = False, scoring: str | None = None
) -> dict[str, dict]:
    """Scored utterances of a report by id; `harmonized` is `scoring="harmonized"`."""
    scoring = "harmonized" if harmonized else scoring
    if scoring and set_name.startswith("fleurs-en"):
        raise SystemExit(f"{set_name}: the {scoring!r} scoring is for Korean sets")
    data = json.loads((REPORTS / report / f"{set_name}.json").read_text(encoding="utf-8"))
    if data["summary"].get("limit"):
        raise SystemExit(f"{report}/{set_name}: measured with --limit, not the whole set; measure it again")
    rows = {row["id"]: row for row in data["utterances"] if row["length"]}
    if len(rows) != sum(1 for row in data["utterances"] if row["length"]):
        raise SystemExit(f"{report}/{set_name}: utterance ids are not unique; measure it again")
    if scoring:
        rows = {i: rescored(row, scoring) for i, row in rows.items()}
    return rows


# 행 하나를 SCORINGS[scoring]으로 바꾼 글로 한국어 CER을 다시 계산해 edits·length만 바꾼 새 행을 돌려준다.
# 원래 정답·가설 글은 그대로 남는다.
def rescored(row: dict, scoring: str = "harmonized") -> dict:
    """The row scored again (Korean CER) after SCORINGS[scoring]; the stored texts stay as they are."""
    scored = score(*SCORINGS[scoring](row["reference"], row["hypothesis"]), "ko")
    return {**row, "edits": scored.edits, "length": scored.length}


# 묶음의 확정된 '숫자 발화' ID 집합(reports/digit_utterances/<묶음>.json). 목록은 깨끗한 음성에서 정했으므로,
# 'zeroth-val@telephone'처럼 채널이 붙은 묶음도 '@' 앞 이름의 목록을 쓴다.
def digit_ids(set_name: str) -> set[str]:
    """The list is defined on clean audio; "zeroth-val@telephone" uses the list of "zeroth-val"."""
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
# 모든 리포트에 공통으로 있는 발화만 써서 같은 발화끼리 비교한다.
def table(
    set_name: str,
    reports: list[str],
    baseline: str | None,
    split_digits: bool,
    harmonized: bool = False,
    scoring: str | None = None,
) -> None:
    loaded = {name: load_rows(name, set_name, harmonized, scoring) for name in reports}
    shared = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    subsets = {"all": shared}
    if split_digits:
        digits = digit_ids(set_name)
        subsets["no digits"] = [i for i in shared if i not in digits]
    for label, ids in subsets.items():
        print(f"\n{set_name} / {label}: {len(ids)} utterances")
        for name, rows in loaded.items():
            edits, lengths = arrays(rows, ids)
            low, high = interval(edits, lengths)
            line = (
                f"  {name:28s} {error_rate(edits, lengths) * 100:6.2f}%  [{low * 100:.2f}, {high * 100:.2f}]"
            )
            if baseline and name != baseline:
                base_edits, _ = arrays(loaded[baseline], ids)
                base_rate = error_rate(base_edits, lengths)
                d_low, d_high = difference_interval(edits, base_edits, lengths)
                delta = error_rate(edits, lengths) - base_rate
                line += (
                    f"  vs {baseline}: {delta * 100:+.2f}%p [{d_low * 100:+.2f}, {d_high * 100:+.2f}]"
                    f", relative {delta / base_rate * 100:+.1f}%"
                )
            print(line)


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
) -> dict:
    """Rates of two reports over their shared utterances and the paired interval of the difference.

    `without_digits` drops the fixed digit utterances of a Zeroth set. `reference_digits` keeps only the
    utterances whose reference has (True) or lacks (False) an Arabic digit, for sets like FLEURS whose
    references write numbers as digits. `baseline_scoring` scores the baseline differently from the candidate
    (stage 8: a report post-processed against itself); both must leave the reference lengths equal.
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
    return {
        "utterances": len(ids),
        "baseline": base,
        "candidate": cand,
        "delta": cand - base,
        "relative": (cand - base) / base if base else float("nan"),
        "delta_interval": difference_interval(cand_edits, base_edits, lengths),
    }


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
        # Reported since stage 3, not part of the verdict (docs/experiments.md, stage 2 post-hoc analysis).
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
        # Stage 3: not an improvement when the paired interval of the no-digit difference includes 0.
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
    """Utterances with more edits than reference characters (a phrase repeated to the end), per report."""
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
    """Stage 6: harmonized CER on Zeroth validation against the base model and the stage 3 model."""
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
    """What a fallback decoding changed against a recorded greedy report (stage 8), per report.

    Decoded again: the report's final temperature is above 0. A hypothesis that changed without being decoded
    again means the first (greedy) pass did not repeat the recorded one.
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
    """The stage 8 conditions for one fallback arm: reports `<model>-<arm>` against the greedy `<model>`."""
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
    """Stage 8: the first arm (simplest first) that stops the recorded loops without costing accuracy."""
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
    """Stage 8 output post-processing: `model` post-processed against itself; the rest is reported."""
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
    """Summary and rows (by id) of a stage 9 fp16-check file, a raw report of a --limit run."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return data["summary"], {row["id"]: row for row in data["utterances"]}


# gate9 명령(9단계 fp16 점검): 앞 100발화를 fp16(GPU)과 fp32(CPU)로 디코딩한 두 파일을 비교한다. 두 파일이
# 정밀도 표시와 발화 순서까지 맞는지 먼저 확인하고, 원래 CER 차이가 ±0.20%p 안이며 fp16 쪽에 빈 가설과 넘침
# 발화가 없으면 'fp16을 쓴다'. 아니면 fp32로 잰다. 가설이 다른 발화 수도 보고한다.
# CPU와 GPU의 계산 차이도 fp16 쪽 차이로 세게 되므로 점검이 그만큼 엄격해질 뿐이다(docs/experiments.md 9단계).
def gate9(fp16: str, fp32: str) -> bool:
    """Stage 9: True when fp16 may be used (docs/experiments.md, 9단계 "fp16 점검")."""
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
    """Stage 9 rule on the paired 95% interval of (candidate - base turbo) on FLEURS Korean validation."""
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
    """Stage 9: the candidate's raw output against the base turbo on FLEURS Korean validation (original CER).

    Everything else is reported: the Zeroth numbers next to the base turbo and the fine-tuned models, other
    scorings, English, loops, and what the official repetition fix (`<candidate>-fixed`) changes.
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
    """Stage 10 gate (a): a fresh adapter merged into the base decodes what the stage 9 fp16 check did."""
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


# train_qwen의 steps.jsonl에서 스텝 줄만 스텝 번호별로 읽는다(이벤트 줄은 뺀다). 이어 한 실행이 같은 스텝을 두
# 번 기록하면 나중 것을 쓴다.
def read_steps(path: str) -> dict[int, dict]:
    """Step entries of a train_qwen log by step; a resumed run logs some steps twice, the later one counts."""
    steps = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            if "event" not in entry:
                steps[entry["step"]] = entry
    return steps


# 스모크 중 5초마다 남긴 GPU 메모리 표본 CSV(저장소 밖의 감시 스크립트가 쓴다)를 읽는다. 숫자 열은 float로
# 바꾸고, 카운터가 비어 있던 칸은 None으로 둔다.
def read_memwatch(path: str) -> list[dict]:
    """GPU memory samples (work/stage10/memwatch.ps1): MiB as numbers, None where a counter was missing."""
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
    """Stage 10 gate (c): the smoke run against the registered limits, and the number of steps to train."""
    steps = read_steps(steps_path)
    samples = read_memwatch(mem_path)
    shared = [row["shared_mb"] for row in samples if row["shared_mb"] is not None]
    used = [row["smi_used_mb"] for row in samples if row["smi_used_mb"] is not None]
    if not steps or not shared or not used:
        raise SystemExit("smoke10 needs the step log and GPU memory samples with both counters")
    late_skips = sorted(s for s, e in steps.items() if e["skipped"] and s > STAGE10_SKIP_GRACE)
    nonfinite = sorted(s for s, e in steps.items() if not math.isfinite(e["loss"]))
    longest = [s for s, e in steps.items() if e["longest"]]
    peak = max(e["peak_allocated_mb"] or 0 for e in steps.values())  # None on the CPU
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
    """Stage 10: the checkpoint whose zeroth-val500 run has the lowest harmonized CER; on a tie the earliest
    (the files are given in step order)."""
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
    """Stage 10 verdict from condition 1 (Zeroth against N) and condition 2 (FLEURS Korean against N)."""
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
    """Stage 10: the LoRA-trained Qwen (raw output) against N on Zeroth (harmonized) and FLEURS Korean.

    Reported with it: the forgetting line against the untrained Qwen, the other models, digit splits, the
    notation-agnostic scoring, English, the telephone channel, the official repetition fix and loops.
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


# 명령줄 진입점. 하위 명령마다 인자를 정하고 해당 함수를 부른다. 명령과 단계: digits·table·loops(공통),
# verdict(2·3단계), verdict6(6단계), changes·verdict8·itn8(8단계), gate9·verdict9(9단계),
# gate10·smoke10·pick10·verdict10(10단계).
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
    tab.add_argument("--no-digit-split", action="store_true", help="FLEURS sets have no digit list")
    tab.add_argument("--harmonized", action="store_true", help="score again after itn.harmonize (Korean)")
    tab.add_argument(
        "--scoring", choices=list(SCORINGS), help="score the stored texts again (Korean, stage 8)"
    )
    ver = commands.add_parser("verdict")
    ver.add_argument("--baseline", required=True)
    ver.add_argument("--candidate", required=True)
    ver.add_argument("--no-interval-rule", action="store_true", help="stage 2 rules (no interval condition)")
    ver6 = commands.add_parser("verdict6")
    ver6.add_argument("--baseline", required=True)
    ver6.add_argument("--previous", required=True, help="the stage 3 model trained on the original notation")
    ver6.add_argument("--candidate", required=True)
    loop = commands.add_parser("loops")
    loop.add_argument("--set", required=True, dest="set_name")
    loop.add_argument("--reports", nargs="+", required=True)
    loop.add_argument("--harmonized", action="store_true", help="score again after itn.harmonize (Korean)")
    loop.add_argument("--scoring", choices=list(SCORINGS))
    chg = commands.add_parser("changes")
    chg.add_argument("--set", required=True, dest="set_name")
    chg.add_argument("--base", required=True, help="the recorded greedy report")
    chg.add_argument("--reports", nargs="+", required=True)
    chg.add_argument("--scoring", choices=list(SCORINGS))
    ver8 = commands.add_parser("verdict8")
    ver8.add_argument("--models", nargs="+", required=True, help="greedy reports; arms are <model>-<arm>")
    ver8.add_argument("--arms", nargs="+", required=True, help="in order of preference (simplest first)")
    itn = commands.add_parser("itn8")
    itn.add_argument("--model", required=True)
    itn.add_argument("--baseline", required=True)
    itn.add_argument("--fleurs", default="fleurs-ko-val")
    itn.add_argument("--zeroth", default="zeroth-val")
    gate = commands.add_parser("gate9")
    gate.add_argument("--fp16", required=True, help="raw report file of the fp16 run")
    gate.add_argument("--fp32", required=True, help="raw report file of the fp32 run")
    ver9 = commands.add_parser("verdict9")
    ver9.add_argument("--candidate", required=True, help="raw output; <candidate>-fixed is reported")
    ver9.add_argument("--baseline", required=True)
    ver9.add_argument("--others", nargs="*", default=[], help="fine-tuned models shown next to it")
    gate_a = commands.add_parser("gate10")
    gate_a.add_argument("--file", required=True, help="raw report file of the fresh-adapter run")
    gate_a.add_argument("--recorded", required=True, help="the stage 9 fp16 check file")
    smoke = commands.add_parser("smoke10")
    smoke.add_argument("--steps", required=True, help="outputs/<smoke run>/steps.jsonl")
    smoke.add_argument("--mem", required=True, help="the memwatch CSV of the smoke run")
    smoke.add_argument("--expected-steps", type=int, default=50)
    pick = commands.add_parser("pick10")
    pick.add_argument(
        "--files", nargs="+", required=True, help="zeroth-val500 raw report files, in step order"
    )
    ver10 = commands.add_parser("verdict10")
    ver10.add_argument("--candidate", required=True, help="raw output; <candidate>-fixed is reported")
    ver10.add_argument("--n", required=True, help="the model it would replace")
    ver10.add_argument("--untrained", required=True, help="the base model before training")
    ver10.add_argument("--others", nargs="*", default=[])
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
    else:
        table(
            args.set_name, args.reports, args.baseline, not args.no_digit_split, args.harmonized, args.scoring
        )


if __name__ == "__main__":
    main()
