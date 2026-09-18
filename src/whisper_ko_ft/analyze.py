"""Tables from the reports: digit utterances, error rates with intervals, paired comparisons.

Usage:
    uv run python -m whisper_ko_ft.analyze digits --set zeroth-val
    uv run python -m whisper_ko_ft.analyze table --set zeroth-val --reports whisper-small small-a \
        --baseline whisper-small

`digits` fixes the list of "digit utterances" of a set (docs/experiments.md, "숫자 표기"): utterances where
either base model wrote an Arabic digit. It is written once and not overwritten without --force.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from whisper_ko_ft.metrics import difference_interval, error_rate, interval
from whisper_ko_ft.paths import REPORTS

BASE_MODELS = ("whisper-small", "whisper-large-v3-turbo")


def load_rows(report: str, set_name: str) -> dict[str, dict]:
    data = json.loads((REPORTS / report / f"{set_name}.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in data["utterances"] if row["length"]}
    if len(rows) != sum(1 for row in data["utterances"] if row["length"]):
        raise SystemExit(f"{report}/{set_name}: utterance ids are not unique; measure it again")
    return rows


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


def table(set_name: str, reports: list[str], baseline: str | None, split_digits: bool) -> None:
    loaded = {name: load_rows(name, set_name) for name in reports}
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
    args = parser.parse_args()

    if args.command == "digits":
        write_digits(args.set_name, args.force)
    else:
        table(args.set_name, args.reports, args.baseline, not args.no_digit_split)


if __name__ == "__main__":
    main()
