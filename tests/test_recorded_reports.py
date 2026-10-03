"""The numbers in docs/experiments.md are recomputed from the committed reports with the current code.

A change to text_norm, metrics, itn or jiwer that moves them must be deliberate: update the expected
values here and write the reason in docs/experiments.md.
"""

import json

import numpy as np
import pytest

from whisper_ko_ft import analyze
from whisper_ko_ft.metrics import error_rate, score
from whisper_ko_ft.paths import REPORTS

FILES = sorted(p for p in REPORTS.glob("*/*.json") if p.parent.name != "digit_utterances")


def test_reports_are_found():
    assert FILES


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


# Harmonized CER (stages 6 and 7, and the 2026-10-02 checks of N) is not stored; it is scored again on
# every call. Pinned as totals, so that a change too small to move the rounded rate still fails. The
# comment is the rate in the docs.
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
        # 2026-10-02: N in CTranslate2, fallback off and on
        ("turbo-n-ct2", "zeroth-val", 1203, 93621),  # 1.28%
        ("turbo-n-ct2-fallback", "zeroth-val", 1203, 93621),  # 1.28%
        ("turbo-n-ct2", "zeroth-test", 373, 19272),  # 1.94%
        ("turbo-n-ct2-fallback", "zeroth-test", 373, 19272),  # 1.94%
        # 2026-10-02: the speed check on val-500, transformers batch 1 and CTranslate2
        ("turbo-n-b1", "zeroth-val500", 320, 18321),  # 1.75%
        ("turbo-n-ct2", "zeroth-val500", 316, 18321),  # 1.72%
    ],
)
def test_recorded_harmonized_totals(report, set_name, edits, length):
    rows = analyze.load_rows(report, set_name, harmonized=True)
    got_edits, got_lengths = analyze.arrays(rows, sorted(rows))
    assert (int(got_edits.sum()), int(got_lengths.sum())) == (edits, length)
