"""Tables from the reports: digit utterances, error rates with intervals, paired comparisons.

Usage:
    uv run python -m whisper_ko_ft.analyze digits --set zeroth-val
    uv run python -m whisper_ko_ft.analyze table --set zeroth-val --reports whisper-small small-a \
        --baseline whisper-small

    uv run python -m whisper_ko_ft.analyze verdict --baseline whisper-small --candidate small-a

`digits` fixes the list of "digit utterances" of a set (docs/experiments.md, "숫자 표기"): utterances where
either base model wrote an Arabic digit. It is written once and not overwritten without --force.
`verdict` applies the verdict rules of stages 2 and 3 to validation reports.
`verdict6` applies the stage 6 rules. `--harmonized` scores the stored references and hypotheses again after
`itn.harmonize`, so that either notation of a number gets the same score (docs/experiments.md, stage 6).
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from whisper_ko_ft.itn import harmonize
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
_DIGIT = re.compile(r"[0-9]")


def load_rows(report: str, set_name: str, harmonized: bool = False) -> dict[str, dict]:
    data = json.loads((REPORTS / report / f"{set_name}.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in data["utterances"] if row["length"]}
    if len(rows) != sum(1 for row in data["utterances"] if row["length"]):
        raise SystemExit(f"{report}/{set_name}: utterance ids are not unique; measure it again")
    if harmonized:
        rows = {i: rescored(row) for i, row in rows.items()}
    return rows


def rescored(row: dict) -> dict:
    """The row scored again (Korean CER) with one number notation on both sides."""
    scored = score(harmonize(row["reference"]), harmonize(row["hypothesis"]), "ko")
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
    set_name: str, reports: list[str], baseline: str | None, split_digits: bool, harmonized: bool = False
) -> None:
    loaded = {name: load_rows(name, set_name, harmonized) for name in reports}
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
) -> dict:
    """Rates of two reports over their shared utterances and the paired interval of the difference.

    `without_digits` drops the fixed digit utterances of a Zeroth set. `reference_digits` keeps only the
    utterances whose reference has (True) or lacks (False) an Arabic digit, for sets like FLEURS whose
    references write numbers as digits.
    """
    cand_rows = load_rows(candidate, set_name, harmonized)
    base_rows = load_rows(baseline, set_name, harmonized)
    ids = sorted(set(cand_rows) & set(base_rows))
    if without_digits:
        digits = digit_ids(set_name)
        ids = [i for i in ids if i not in digits]
    if reference_digits is not None:
        ids = [i for i in ids if bool(_DIGIT.search(base_rows[i]["reference"])) == reference_digits]
    if not ids:
        return {"utterances": 0}
    cand_edits, lengths = arrays(cand_rows, ids)
    base_edits, _ = arrays(base_rows, ids)
    base, cand = error_rate(base_edits, lengths), error_rate(cand_edits, lengths)
    return {
        "utterances": len(ids),
        "baseline": base,
        "candidate": cand,
        "delta": cand - base,
        "relative": (cand - base) / base,
        "delta_interval": difference_interval(cand_edits, base_edits, lengths),
    }


def verdict(baseline: str, candidate: str) -> str:
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
        overall["relative"] <= IN_DOMAIN_RELATIVE and no_digits["relative"] <= IN_DOMAIN_NO_DIGITS_RELATIVE
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
    ver = commands.add_parser("verdict")
    ver.add_argument("--baseline", required=True)
    ver.add_argument("--candidate", required=True)
    ver6 = commands.add_parser("verdict6")
    ver6.add_argument("--baseline", required=True)
    ver6.add_argument("--previous", required=True, help="the stage 3 model trained on the original notation")
    ver6.add_argument("--candidate", required=True)
    args = parser.parse_args()

    if args.command == "digits":
        write_digits(args.set_name, args.force)
    elif args.command == "verdict":
        verdict(args.baseline, args.candidate)
    elif args.command == "verdict6":
        verdict6(args.baseline, args.previous, args.candidate)
    else:
        table(args.set_name, args.reports, args.baseline, not args.no_digit_split, args.harmonized)


if __name__ == "__main__":
    main()
