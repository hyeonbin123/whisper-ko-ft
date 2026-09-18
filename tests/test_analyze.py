import json

import pytest

from whisper_ko_ft import analyze


def write_report(root, name, set_name, rows):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{set_name}.json").write_text(
        json.dumps({"summary": {}, "utterances": rows}), encoding="utf-8"
    )


def rows(edits, digit_flags=None, length=10):
    flags = digit_flags or [False] * len(edits)
    return [
        {"id": f"u{n}", "edits": e, "length": length, "has_digit": flag, "reference": "문장"}
        for n, (e, flag) in enumerate(zip(edits, flags, strict=True))
    ]


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "REPORTS", tmp_path)
    return tmp_path


def test_duplicate_ids_are_refused(reports):
    write_report(reports, "m", "fleurs-ko-val", [{"id": "1", "edits": 0, "length": 5}] * 2)
    with pytest.raises(SystemExit, match="not unique"):
        analyze.load_rows("m", "fleurs-ko-val")


def test_digit_list_is_the_union_of_both_base_models_and_is_not_overwritten(reports):
    write_report(reports, "whisper-small", "zeroth-val", rows([1, 1, 1], [True, False, False]))
    write_report(reports, "whisper-large-v3-turbo", "zeroth-val", rows([1, 1, 1], [False, True, False]))

    analyze.write_digits("zeroth-val", force=False)

    assert analyze.digit_ids("zeroth-val") == {"u0", "u1"}
    assert analyze.digit_ids("zeroth-val@telephone") == {"u0", "u1"}
    with pytest.raises(SystemExit, match="fixed at stage 1"):
        analyze.write_digits("zeroth-val", force=False)


def prepare_verdict(reports, zeroth_candidate, fleurs_candidate):
    digits = [True, False, False, False]
    write_report(reports, "whisper-small", "zeroth-val", rows([4, 2, 2, 2], digits))
    write_report(reports, "whisper-large-v3-turbo", "zeroth-val", rows([0, 0, 0, 0]))
    analyze.write_digits("zeroth-val", force=False)
    write_report(reports, "cand", "zeroth-val", rows(zeroth_candidate))
    write_report(reports, "whisper-small", "fleurs-ko-val", rows([1, 1], length=100))
    write_report(reports, "cand", "fleurs-ko-val", rows(fleurs_candidate, length=100))


def test_verdict_general_improvement(reports):
    prepare_verdict(reports, zeroth_candidate=[1, 1, 1, 1], fleurs_candidate=[1, 2])  # +0.5%p elsewhere
    assert analyze.verdict("whisper-small", "cand") == "범용 개선"


def test_verdict_domain_only_when_the_other_domain_gets_worse(reports):
    prepare_verdict(reports, zeroth_candidate=[1, 1, 1, 1], fleurs_candidate=[3, 2])  # +1.5%p elsewhere
    assert analyze.verdict("whisper-small", "cand") == "도메인 전용"


def test_verdict_no_effect_when_the_gain_is_only_digit_notation(reports):
    # Overall -40%, but every gain is in the digit utterance: the rest is unchanged.
    prepare_verdict(reports, zeroth_candidate=[0, 2, 2, 2], fleurs_candidate=[1, 1])
    assert analyze.verdict("whisper-small", "cand") == "효과 없음"


def test_compare_can_split_by_digits_in_the_reference(reports):
    base = [
        {"id": "a", "edits": 1, "length": 10, "reference": "1978년"},
        {"id": "b", "edits": 1, "length": 10, "reference": "올해"},
    ]
    cand = [
        {"id": "a", "edits": 6, "length": 10, "reference": "1978년"},
        {"id": "b", "edits": 1, "length": 10, "reference": "올해"},
    ]
    write_report(reports, "base", "fleurs-ko-val", base)
    write_report(reports, "cand", "fleurs-ko-val", cand)

    with_digit = analyze.compare("fleurs-ko-val", "cand", "base", reference_digits=True)
    without = analyze.compare("fleurs-ko-val", "cand", "base", reference_digits=False)

    assert (with_digit["utterances"], round(with_digit["delta"], 3)) == (1, 0.5)
    assert (without["utterances"], without["delta"]) == (1, 0.0)
