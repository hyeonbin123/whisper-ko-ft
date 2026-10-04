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
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from whisper_ko_ft.itn import harmonize, to_digits
from whisper_ko_ft.latin import letters_to_latin
from whisper_ko_ft.metrics import difference_interval, error_rate, interval, score
from whisper_ko_ft.paths import REPORTS

BASE_MODELS = ("whisper-small", "whisper-large-v3-turbo")
# Verdict rules (docs/experiments.md, "판정 규칙"): relative CER change on Zeroth validation, overall and
# without digit utterances, and the largest allowed absolute CER increase on FLEURS Korean validation.
IN_DOMAIN_RELATIVE = -0.20
IN_DOMAIN_NO_DIGITS_RELATIVE = -0.10
OUT_OF_DOMAIN_MAX_INCREASE = 0.010
# Stage 6: the harmonized CER may be at most this much above the stage 3 model's.
STAGE6_MAX_ABOVE_L2 = 0.003
# Stage 8: a fallback arm may raise no rate by more than this (upper end of the paired 95% interval), and the
# output post-processing may raise the harmonized Zeroth CER by at most STAGE8_ITN_MAX_INCREASE.
STAGE8_MAX_INCREASE = 0.001
STAGE8_ITN_MAX_INCREASE = 0.0005
# The sets of the stage 8 decoding rules, each with the scoring its loops and rates are counted in.
STAGE8_SETS = {"zeroth-val": "harmonized", "fleurs-ko-val": None, "fleurs-en-val": None}
# Stage 9: fp16 is used when its raw CER on the first 100 utterances of zeroth-val500 is within this of the
# fp32 run's (either direction), with no empty hypothesis and no utterance with non-finite logits.
STAGE9_FP16_MAX_GAP = 0.002
_DIGIT = re.compile(r"[0-9]")


def _harmonized(reference: str, hypothesis: str) -> tuple[str, str]:
    """Stage 6: one number notation on both sides."""
    return harmonize(reference), harmonize(hypothesis)


def _itn(reference: str, hypothesis: str) -> tuple[str, str]:
    """Stage 8 output post-processing: spelled numbers in the hypothesis become digits; reference as is."""
    return reference, to_digits(hypothesis)


def _itn_harmonized(reference: str, hypothesis: str) -> tuple[str, str]:
    """The post-processed hypothesis under the stage 6 scoring."""
    return _harmonized(*_itn(reference, hypothesis))


def _itn_latin(reference: str, hypothesis: str) -> tuple[str, str]:
    """Stage 8, no verdict: the post-processing plus letter names as Latin letters."""
    return reference, letters_to_latin(to_digits(hypothesis))


def _agnostic(reference: str, hypothesis: str) -> tuple[str, str]:
    """Stage 8 auxiliary metric: numbers and letter names in one notation on both sides."""
    return letters_to_latin(harmonize(reference)), letters_to_latin(harmonize(hypothesis))


# How the stored reference and hypothesis are rewritten before Korean CER is computed again.
SCORINGS = {
    "harmonized": _harmonized,
    "itn": _itn,
    "itn-harmonized": _itn_harmonized,
    "itn-latin": _itn_latin,
    "agnostic": _agnostic,
}


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


def rescored(row: dict, scoring: str = "harmonized") -> dict:
    """The row scored again (Korean CER) after SCORINGS[scoring]; the stored texts stay as they are."""
    scored = score(*SCORINGS[scoring](row["reference"], row["hypothesis"]), "ko")
    return {**row, "edits": scored.edits, "length": scored.length}


def digit_ids(set_name: str) -> set[str]:
    """The list is defined on clean audio; "zeroth-val@telephone" uses the list of "zeroth-val"."""
    path = REPORTS / "digit_utterances" / f"{set_name.split('@')[0]}.json"
    return set(json.loads(path.read_text(encoding="utf-8"))["ids"])


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


def arrays(rows: dict[str, dict], ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([rows[i]["edits"] for i in ids]), np.array([rows[i]["length"] for i in ids])


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


def show(label: str, result: dict) -> None:
    low, high = result["delta_interval"]
    print(
        f"{label:26s} {result['baseline'] * 100:5.2f}% -> {result['candidate'] * 100:5.2f}%"
        f"  {result['delta'] * 100:+.2f}%p [{low * 100:+.2f}, {high * 100:+.2f}]"
        f"  relative {result['relative'] * 100:+.1f}%  ({result['utterances']} utterances)"
    )


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


def _loop_ids(rows: dict[str, dict]) -> set[str]:
    return {i for i, row in rows.items() if row["edits"] > row["length"]}


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


def verdict8(models: list[str], arms: list[str]) -> str | None:
    """Stage 8: the first arm (simplest first) that stops the recorded loops without costing accuracy."""
    results = {arm: stage8_arm(models, arm) for arm in arms}
    if not all(result["tested"] for result in results.values()):
        print("판정: 하지 않는다 (기록된 되풀이가 첫 디코딩에서 나오지 않아 재시도를 시험하지 못했다)")
        raise SystemExit("no verdict: a recorded loop was not decoded again")
    picked = next((arm for arm in arms if results[arm]["stops"] and results[arm]["harmless"]), None)
    print(f"판정: {picked}를 고른다" if picked else "판정: 재시도를 쓰지 않는다 (greedy 그대로)")
    return picked


def itn8_label(fleurs_upper: float, zeroth_delta: float) -> str:
    if zeroth_delta > STAGE8_ITN_MAX_INCREASE:
        return "출력 숫자 변환은 같은 도메인을 해친다"
    if fleurs_upper < 0:
        return "출력 숫자 변환을 쓴다"
    return "출력 숫자 변환은 이득이 없다"


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


def _check_rows(path: str) -> tuple[dict, dict[str, dict]]:
    """Summary and rows (by id) of a stage 9 fp16-check file, a raw report of a --limit run."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return data["summary"], {row["id"]: row for row in data["utterances"]}


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


def verdict9_label(low: float, high: float) -> str:
    """Stage 9 rule on the paired 95% interval of (candidate - base turbo) on FLEURS Korean validation."""
    if high < 0:
        return "다른 도메인에서 더 나은 기준 모델이다"
    if low > 0:
        return "다른 도메인에서 기준선보다 나쁘다"
    return "다른 도메인에서 차이를 가리지 못했다"


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
    else:
        table(
            args.set_name, args.reports, args.baseline, not args.no_digit_split, args.harmonized, args.scoring
        )


if __name__ == "__main__":
    main()
