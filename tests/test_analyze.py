import json

import pytest

from whisper_ko_ft import analyze
from whisper_ko_ft.metrics import score


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


def test_a_limit_run_is_refused(reports):
    folder = reports / "m"
    folder.mkdir()
    (folder / "zeroth-val.json").write_text(
        json.dumps({"summary": {"limit": 100}, "utterances": rows([1, 1])}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="--limit"):
        analyze.load_rows("m", "zeroth-val")


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


def test_verdict_needs_the_no_digit_interval_to_exclude_zero(reports):
    # No digits: relative -16.7%, but one utterance gets worse, so the paired interval includes 0.
    prepare_verdict(reports, zeroth_candidate=[0, 0, 2, 3], fleurs_candidate=[1, 1])
    assert analyze.verdict("whisper-small", "cand") == "효과 없음"
    assert analyze.verdict("whisper-small", "cand", require_interval=False) == "범용 개선"  # stage 2 rules


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


def test_loops_counts_utterances_with_more_edits_than_reference_characters(reports, capsys):
    looped = "가나다" + "라니" * 40  # a repeated syllable pair, as in 105_003_0478
    utterances = [
        {"id": "a", "edits": 80, "length": 9, "reference": "이천 십 팔 년 오 월", "hypothesis": looped},
        {"id": "b", "edits": 9, "length": 9, "reference": "이천 십 팔 년 오 월", "hypothesis": "2018년 5월"},
        {"id": "c", "edits": 0, "length": 2, "reference": "문장", "hypothesis": "문장"},
    ]
    write_report(reports, "m", "zeroth-val", utterances)

    assert analyze.loops("zeroth-val", ["m"]) == {"m": ["a"]}
    assert analyze.loops("zeroth-val", ["m"], harmonized=True) == {"m": ["a"]}  # b is notation only
    assert "1 of 3" in capsys.readouterr().out


def scored_rows(pairs, **extra):
    """Rows whose stored edits are the original scoring of their texts, as evaluate writes them."""
    out = []
    for n, (reference, hypothesis) in enumerate(pairs):
        scored = score(reference, hypothesis, "ko")
        out.append(
            {
                "id": f"u{n}",
                "reference": reference,
                "hypothesis": hypothesis,
                "edits": scored.edits,
                "length": scored.length,
                **extra,
            }
        )
    return out


def test_output_itn_rewrites_the_hypothesis_only(reports):
    pairs = [
        ("1978년 5월에 열렸다", "천 구백 칠십 팔 년 오 월에 열렸다"),  # FLEURS writes digits
        ("천 구백 칠십 팔 년", "1978년"),  # a spelled-out reference stays spelled out
    ]
    write_report(reports, "m", "fleurs-ko-val", scored_rows(pairs))

    itn = analyze.load_rows("m", "fleurs-ko-val", scoring="itn")
    assert itn["u0"]["edits"] == 0
    assert itn["u1"]["edits"] == analyze.load_rows("m", "fleurs-ko-val")["u1"]["edits"] > 0
    both = analyze.load_rows("m", "fleurs-ko-val", scoring="itn-harmonized")
    assert [row["edits"] for row in both.values()] == [0, 0]


def test_notation_agnostic_scoring_also_reads_letter_names(reports):
    pairs = [("faa는 50%를 줄였다", "에프 에이에이는 오십 퍼센트를 줄였다")]
    write_report(reports, "m", "fleurs-ko-val", scored_rows(pairs))
    assert analyze.load_rows("m", "fleurs-ko-val", scoring="harmonized")["u0"]["edits"] > 0
    assert analyze.load_rows("m", "fleurs-ko-val", scoring="agnostic")["u0"]["edits"] == 0
    latin = analyze.load_rows("m", "fleurs-ko-val", scoring="itn-latin")["u0"]
    assert latin["hypothesis"] == pairs[0][1]  # the stored text is never changed
    assert latin["edits"] < analyze.load_rows("m", "fleurs-ko-val")["u0"]["edits"]


def test_korean_rescoring_is_refused_for_english_sets(reports):
    write_report(reports, "m", "fleurs-en-val", rows([1]))
    with pytest.raises(SystemExit, match="Korean"):
        analyze.load_rows("m", "fleurs-en-val", scoring="itn")


def test_compare_refuses_different_reference_lengths(reports):
    write_report(reports, "m", "fleurs-ko-val", scored_rows([("faa는 50%를", "faa는 50%를")]))
    with pytest.raises(SystemExit, match="reference"):
        analyze.compare("fleurs-ko-val", "m", "m", scoring="agnostic", baseline_scoring=None)


def loop_rows(edits, temperatures=None):
    """Zeroth rows scored again from their texts: `e` inserted characters after a 100-character reference."""
    temperatures = temperatures or [None] * len(edits)
    out = []
    for n, (e, t) in enumerate(zip(edits, temperatures, strict=True)):
        row = {"id": f"u{n}", "reference": "가" * 100, "hypothesis": "가" * 100 + "나" * e, "edits": e}
        row["length"] = 100
        if t is not None:
            row["temperature"], row["fallback_exhausted"] = t, False
        out.append(row)
    return out


def fleurs_rows(edits, temperatures=None):
    temperatures = temperatures or [None] * len(edits)
    out = []
    for n, (e, t) in enumerate(zip(edits, temperatures, strict=True)):
        row = {"id": f"u{n}", "reference": "문장", "hypothesis": f"가설 {e}", "edits": e, "length": 100}
        if t is not None:
            row["temperature"], row["fallback_exhausted"] = t, False
        out.append(row)
    return out


def prepare_stage8(reports, arms):
    """Recorded greedy reports of models n and n2 (n2 loops on u3 of zeroth-val) and the arms' reports.

    `arms` maps an arm to {(model, set): (edits, temperatures)}; sets not given repeat the greedy report.
    """
    (reports / "digit_utterances").mkdir()
    (reports / "digit_utterances" / "zeroth-val.json").write_text(
        json.dumps({"ids": ["u0"]}), encoding="utf-8"
    )
    greedy = {
        ("n", "zeroth-val"): [1, 1, 1, 1],
        ("n2", "zeroth-val"): [1, 1, 1, 150],
        ("n", "fleurs-ko-val"): [1, 1, 1, 1],
        ("n2", "fleurs-ko-val"): [1, 1, 1, 1],
        ("n", "fleurs-en-val"): [1, 1],
        ("n2", "fleurs-en-val"): [1, 1],
    }
    for (model, set_name), edits in greedy.items():
        make = loop_rows if set_name == "zeroth-val" else fleurs_rows
        write_report(reports, model, set_name, make(edits))
    for arm, changed in arms.items():
        for (model, set_name), edits in greedy.items():
            make = loop_rows if set_name == "zeroth-val" else fleurs_rows
            edits, temperatures = changed.get((model, set_name), (edits, [0.0] * len(edits)))
            write_report(reports, f"{model}-{arm}", set_name, make(edits, temperatures))


STOPPED = ([1, 1, 1, 2], [0.0, 0.0, 0.0, 0.2])  # n2's loop decoded again at 0.2 and gone


def test_verdict8_picks_the_simpler_arm_when_it_passes(reports):
    stopped = {("n2", "zeroth-val"): STOPPED}
    prepare_stage8(reports, {"d1a": stopped, "d1": stopped})
    assert analyze.verdict8(["n", "n2"], ["d1a", "d1"]) == "d1a"


def test_verdict8_takes_the_next_arm_when_the_first_costs_accuracy(reports):
    worse = {("n2", "zeroth-val"): STOPPED, ("n", "fleurs-ko-val"): ([1, 3, 1, 3], [0.0, 0.4, 0.0, 0.4])}
    prepare_stage8(reports, {"d1a": worse, "d1": {("n2", "zeroth-val"): STOPPED}})
    assert analyze.verdict8(["n", "n2"], ["d1a", "d1"]) == "d1"


def test_verdict8_keeps_greedy_when_a_loop_is_left(reports):
    still = {("n2", "zeroth-val"): ([1, 1, 1, 150], [0.0, 0.0, 0.0, 1.0])}  # decoded again, looped every time
    new_loop = {("n2", "zeroth-val"): STOPPED, ("n", "fleurs-en-val"): ([1, 150], [0.0, 0.0])}
    prepare_stage8(reports, {"d1a": still, "d1": new_loop})
    assert analyze.verdict8(["n", "n2"], ["d1a", "d1"]) is None


def test_verdict8_makes_no_verdict_when_the_recorded_loop_is_not_decoded_again(reports):
    # The first (greedy) pass did not loop on u3 this time, so the fallback was never put to the test.
    gone = {("n2", "zeroth-val"): ([1, 1, 1, 2], [0.0, 0.0, 0.0, 0.0])}
    prepare_stage8(reports, {"d1a": gone, "d1": gone})
    with pytest.raises(SystemExit, match="no verdict"):
        analyze.verdict8(["n", "n2"], ["d1a", "d1"])


def test_changes_separates_decoding_again_from_other_changes(reports, capsys):
    write_report(reports, "base", "fleurs-ko-val", fleurs_rows([1, 150, 1]))
    changed = fleurs_rows([1, 2, 3], [0.0, 0.2, 0.0])  # u1 decoded again, u2 changed without it
    changed[0]["hypothesis"] = "가설 1"
    write_report(reports, "arm", "fleurs-ko-val", changed)
    found = analyze.changes("fleurs-ko-val", "base", ["arm"])["arm"]
    assert found["redecoded"] == ["u1"]
    assert found["changed_not_redecoded"] == ["u2"]
    assert found["changed_outside_loops"] == ["u2"]
    assert (found["loops_before"], found["loops_after"]) == (["u1"], [])


@pytest.mark.parametrize(
    ("fleurs_upper", "zeroth_delta", "label"),
    [
        (-0.001, 0.0, "출력 숫자 변환을 쓴다"),
        (0.0, 0.0, "출력 숫자 변환은 이득이 없다"),
        (-0.001, 0.0006, "출력 숫자 변환은 같은 도메인을 해친다"),
    ],
)
def test_itn8_label(fleurs_upper, zeroth_delta, label):
    assert analyze.itn8_label(fleurs_upper, zeroth_delta) == label


def test_itn8_on_reports(reports):
    words = ["이", "삼", "사", "오", "육"]
    fleurs = [(f"{1972 + n}년 {n + 3}월", f"천 구백 칠십 {words[n]} 년 {words[n + 1]} 월") for n in range(4)]
    zeroth = [("천 구백 칠십 팔 년 오 월", "1978년 5월")] * 4
    for name in ("n", "base"):
        write_report(reports, name, "fleurs-ko-val", scored_rows(fleurs))
        write_report(reports, name, "zeroth-val", scored_rows(zeroth))
    assert analyze.itn8("n", "base") == "출력 숫자 변환을 쓴다"


def gate_file(path, precision, edits, hypotheses=None, nonfinite=None, device="cuda"):
    """A stage 9 fp16-check file: a raw report of the first utterances of zeroth-val500 (limit 100)."""
    hypotheses = hypotheses or ["가설"] * len(edits)
    nonfinite = nonfinite or [False] * len(edits)
    utterances = [
        {"id": f"u{n}", "reference": "문장", "hypothesis": h, "edits": e, "length": 250}
        | {"nonfinite_logits": bad}
        for n, (e, h, bad) in enumerate(zip(edits, hypotheses, nonfinite, strict=True))
    ]
    summary = {"precision": precision, "device": device, "set": "zeroth-val500", "limit": len(edits)}
    path.write_text(json.dumps({"summary": summary, "utterances": utterances}), encoding="utf-8")
    return str(path)


def test_gate9_keeps_fp16_when_it_matches_fp32(tmp_path):
    fp32 = gate_file(tmp_path / "fp32.json", "fp32", [2, 2, 2, 2], device="cpu")
    assert analyze.gate9(gate_file(tmp_path / "a.json", "fp16", [2, 2, 2, 2]), fp32)
    # 1,000 reference characters: one edit is 0.1%p, in either direction.
    assert analyze.gate9(gate_file(tmp_path / "b.json", "fp16", [3, 2, 2, 2]), fp32)
    assert analyze.gate9(gate_file(tmp_path / "c.json", "fp16", [1, 2, 2, 2]), fp32)
    assert not analyze.gate9(gate_file(tmp_path / "d.json", "fp16", [5, 2, 2, 2]), fp32)  # +0.3%p
    assert not analyze.gate9(gate_file(tmp_path / "e.json", "fp16", [0, 1, 2, 2]), fp32)  # -0.3%p


@pytest.mark.parametrize(
    "broken",
    [
        {"hypotheses": ["가설", "", "가설", "가설"]},  # an empty output
        {"nonfinite": [False, False, True, False]},  # overflowing logits
    ],
)
def test_gate9_falls_back_to_fp32_on_empty_outputs_or_overflow(tmp_path, broken):
    fp32 = gate_file(tmp_path / "fp32.json", "fp32", [2, 2, 2, 2], device="cpu")
    assert not analyze.gate9(gate_file(tmp_path / "fp16.json", "fp16", [2, 2, 2, 2], **broken), fp32)


def test_gate9_refuses_files_it_cannot_pair(tmp_path):
    fp32 = gate_file(tmp_path / "fp32.json", "fp32", [2, 2, 2, 2], device="cpu")
    with pytest.raises(SystemExit, match="precision"):
        analyze.gate9(fp32, fp32)
    with pytest.raises(SystemExit, match="utterances"):
        analyze.gate9(gate_file(tmp_path / "fp16.json", "fp16", [2, 2, 2]), fp32)


@pytest.mark.parametrize(
    ("low", "high", "label"),
    [
        (-0.012, -0.001, "다른 도메인에서 더 나은 기준 모델이다"),
        (-0.012, 0.0, "다른 도메인에서 차이를 가리지 못했다"),
        (0.0, 0.004, "다른 도메인에서 차이를 가리지 못했다"),
        (0.001, 0.009, "다른 도메인에서 기준선보다 나쁘다"),
    ],
)
def test_verdict9_label(low, high, label):
    assert analyze.verdict9_label(low, high) == label


def prepare_stage9(reports, fleurs_candidate):
    """Base turbo, N and the candidate's two outputs on the three validation sets."""
    (reports / "digit_utterances").mkdir()
    (reports / "digit_utterances" / "zeroth-val.json").write_text(
        json.dumps({"ids": ["u0"]}), encoding="utf-8"
    )
    zeroth = {"turbo": [4, 4, 4, 4], "n": [1, 1, 1, 1], "cand": [2, 2, 2, 150], "cand-fixed": [2, 2, 2, 20]}
    fleurs = {"turbo": [5] * 4, "n": [6] * 4, "cand": fleurs_candidate, "cand-fixed": fleurs_candidate}
    for name in zeroth:
        write_report(reports, name, "zeroth-val", loop_rows(zeroth[name]))
        write_report(reports, name, "fleurs-ko-val", fleurs_rows(fleurs[name]))
        write_report(reports, name, "fleurs-en-val", fleurs_rows([1, 1]))


@pytest.mark.parametrize(
    ("fleurs_candidate", "label"),
    [
        ([3, 3, 3, 3], "다른 도메인에서 더 나은 기준 모델이다"),
        ([5, 5, 5, 5], "다른 도메인에서 차이를 가리지 못했다"),
        ([3, 7, 3, 7], "다른 도메인에서 차이를 가리지 못했다"),
        ([7, 7, 7, 7], "다른 도메인에서 기준선보다 나쁘다"),
    ],
)
def test_verdict9_on_reports(reports, capsys, fleurs_candidate, label):
    prepare_stage9(reports, fleurs_candidate)
    assert analyze.verdict9("cand", "turbo", ["n"]) == label
    out = capsys.readouterr().out
    assert "cand-fixed" in out and "zeroth-val harmonized" in out and "fleurs-en-val" in out
