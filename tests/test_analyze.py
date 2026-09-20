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


def notation_rows(hypotheses):
    return [
        {"id": f"u{n}", "edits": 9, "length": 9, "reference": "이천 십 팔 년 오 월", "hypothesis": h}
        for n, h in enumerate(hypotheses)
    ]


def test_harmonized_scoring_ignores_the_notation_of_numbers(reports):
    write_report(reports, "cand", "zeroth-val", notation_rows(["2018년 5월", "이천 십 팔 년 오 월"]))
    loaded = analyze.load_rows("cand", "zeroth-val", harmonized=True)
    assert [row["edits"] for row in loaded.values()] == [0, 0]
    assert analyze.load_rows("cand", "zeroth-val")["u0"]["edits"] == 9  # the stored score is untouched


def prepare_verdict6(reports, candidate_edits, fleurs_candidate):
    def spelled(edits):
        return [{**row, "reference": "문장", "hypothesis": "문장"} for row in rows(edits, length=100)]

    # harmonized scoring scores the texts again, so the texts carry the errors here
    def with_errors(edits):
        return [
            {**row, "reference": "가" * 100, "hypothesis": "가" * (100 - e)}
            for row, e in zip(spelled(edits), edits, strict=True)
        ]

    write_report(reports, "base", "zeroth-val", with_errors([10, 10]))
    write_report(reports, "l2", "zeroth-val", with_errors([2, 2]))
    write_report(reports, "cand", "zeroth-val", with_errors(candidate_edits))
    write_report(reports, "base", "fleurs-ko-val", rows([1, 1], length=100))
    write_report(reports, "cand", "fleurs-ko-val", rows(fleurs_candidate, length=100))


def test_verdict6_general_when_both_rules_hold(reports):
    prepare_verdict6(reports, candidate_edits=[2, 2], fleurs_candidate=[1, 2])
    assert analyze.verdict6("base", "l2", "cand") == "표기를 바꿔 학습하면 범용으로 쓸 수 있다"


def test_verdict6_notation_is_not_enough_when_the_other_domain_gets_worse(reports):
    prepare_verdict6(reports, candidate_edits=[2, 2], fleurs_candidate=[3, 3])
    assert analyze.verdict6("base", "l2", "cand") == "표기만으로는 해결되지 않는다"


def test_verdict6_gain_lost_when_clearly_above_the_previous_model(reports):
    prepare_verdict6(reports, candidate_edits=[3, 3], fleurs_candidate=[1, 1])  # +1.0%p above l2
    assert analyze.verdict6("base", "l2", "cand") == "표기를 바꾸면 같은 도메인의 이득을 잃는다"
