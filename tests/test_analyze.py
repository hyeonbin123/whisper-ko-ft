# [테스트 묶음] analyze.py의 리포트 읽기 검사, 채점 방식, 단계별 판정 규칙을 작은 가짜 리포트로 확인한다. 실제
# reports/ 대신 임시 폴더에 몇 발화짜리 리포트를 써 두고 판정 함수가 규칙대로 이름을 붙이는지 본다.
# [왜 중요한가] 판정 규칙은 재기 전에 문서에 적고 커밋한 약속이다. 코드가 그 약속과 다르면(예: 3단계 구간
# 조건을 빠뜨린 버그, 커밋 037ad21에서 고침) 결과를 본 뒤 기준이 바뀐 것과 같은 일이 된다. 판정 표의 모든
# 칸(만족/불만족 조합)과 경계값, 판정하지 않는 경우를 테스트로 고정한다.
# [구성] 공통(load_rows 검사, 숫자 발화 목록, compare, table) → 2·3단계 verdict → 6단계 맞춘 CER과 verdict6 →
# 되풀이와 8단계 채점·verdict8·changes·itn8 → 9단계 gate9·verdict9 → 10단계 gate10·pick10·smoke10·verdict10 →
# 민감도 분석(robust) → 11·12단계 → Zeroth 문장 겹침(overlap, 커밋된 파생 파일의 수 고정) → 영어 보조 채점
# en-whisper(예시, 커밋된 FLEURS 영어 리포트의 총합 고정).

import inspect
import json
import sys

import pytest

from whisper_ko_ft import analyze
from whisper_ko_ft.metrics import score
from whisper_ko_ft.paths import REPORTS


# 임시 리포트 폴더에 <이름>/<묶음>.json을 쓴다(요약은 비워 둔다).
def write_report(root, name, set_name, rows):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{set_name}.json").write_text(
        json.dumps({"summary": {}, "utterances": rows}), encoding="utf-8"
    )


# 발화별 편집 수만 정한 가짜 행 목록. 정답 길이는 모두 같고, digit_flags로 가설의 숫자 여부(has_digit)를
# 정한다.
def rows(edits, digit_flags=None, length=10):
    flags = digit_flags or [False] * len(edits)
    return [
        {"id": f"u{n}", "edits": e, "length": length, "has_digit": flag, "reference": "문장"}
        for n, (e, flag) in enumerate(zip(edits, flags, strict=True))
    ]


# analyze.REPORTS를 임시 폴더로 바꾸는 fixture. 실제 커밋된 리포트를 건드리지 않는다.
@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "REPORTS", tmp_path)
    return tmp_path


# 발화 ID가 겹치는 리포트는 거부한다. 1단계에서 FLEURS의 같은 문장 ID 때문에 표에서 행이 빠졌던 오류를 다시
# 만들지 않게 한다.
def test_duplicate_ids_are_refused(reports):
    write_report(reports, "m", "fleurs-ko-val", [{"id": "1", "edits": 0, "length": 5}] * 2)
    with pytest.raises(SystemExit, match="not unique"):
        analyze.load_rows("m", "fleurs-ko-val")


# --limit으로 일부만 잰 리포트는 판정에 쓰지 않는다(커밋 a29a975).
def test_a_limit_run_is_refused(reports):
    folder = reports / "m"
    folder.mkdir()
    (folder / "zeroth-val.json").write_text(
        json.dumps({"summary": {"limit": 100}, "utterances": rows([1, 1])}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="--limit"):
        analyze.load_rows("m", "zeroth-val")


# 숫자 발화 목록은 기본 모델 두 개 중 하나라도 가설에 숫자가 있는 발화의 합집합이고, 채널 묶음(@telephone)도
# 같은 목록을 쓰며, 한 번 정한 목록은 --force 없이 다시 쓰지 않는다.
def test_digit_list_is_the_union_of_both_base_models_and_is_not_overwritten(reports):
    write_report(reports, "whisper-small", "zeroth-val", rows([1, 1, 1], [True, False, False]))
    write_report(reports, "whisper-large-v3-turbo", "zeroth-val", rows([1, 1, 1], [False, True, False]))

    analyze.write_digits("zeroth-val", force=False)

    assert analyze.digit_ids("zeroth-val") == {"u0", "u1"}
    assert analyze.digit_ids("zeroth-val@telephone") == {"u0", "u1"}
    with pytest.raises(SystemExit, match="fixed at stage 1"):
        analyze.write_digits("zeroth-val", force=False)


# 2·3단계 판정용 리포트: 기준선 small(첫 발화만 숫자 발화), 후보, FLEURS 한국어 두 모델. 숫자 발화 목록도
# 만든다.
def prepare_verdict(reports, zeroth_candidate, fleurs_candidate):
    digits = [True, False, False, False]
    write_report(reports, "whisper-small", "zeroth-val", rows([4, 2, 2, 2], digits))
    write_report(reports, "whisper-large-v3-turbo", "zeroth-val", rows([0, 0, 0, 0]))
    analyze.write_digits("zeroth-val", force=False)
    write_report(reports, "cand", "zeroth-val", rows(zeroth_candidate))
    write_report(reports, "whisper-small", "fleurs-ko-val", rows([1, 1], length=100))
    write_report(reports, "cand", "fleurs-ko-val", rows(fleurs_candidate, length=100))


# 같은 도메인에서 크게 좋아지고(전체·숫자 발화 제외 모두) 다른 도메인 악화가 1.0%p 안이면 '범용 개선'.
def test_verdict_general_improvement(reports):
    prepare_verdict(reports, zeroth_candidate=[1, 1, 1, 1], fleurs_candidate=[1, 2])  # 다른 도메인 +0.5%p
    assert analyze.verdict("whisper-small", "cand") == "범용 개선"


# 같은 도메인은 좋아졌지만 FLEURS 한국어가 1.5%p 나빠지면 '도메인 전용'(2·3단계의 실제 판정과 같은 칸).
def test_verdict_domain_only_when_the_other_domain_gets_worse(reports):
    prepare_verdict(reports, zeroth_candidate=[1, 1, 1, 1], fleurs_candidate=[3, 2])  # 다른 도메인 +1.5%p
    assert analyze.verdict("whisper-small", "cand") == "도메인 전용"


# 전체로는 좋아져도 이득이 모두 숫자 발화(표기)에서 나오면 '효과 없음'. 숫자 발화 제외 조건이 있는 이유다.
def test_verdict_no_effect_when_the_gain_is_only_digit_notation(reports):
    # 전체로는 -40%지만 이득은 모두 숫자 발화에서 나오고 나머지는 그대로다.
    prepare_verdict(reports, zeroth_candidate=[0, 2, 2, 2], fleurs_candidate=[1, 1])
    assert analyze.verdict("whisper-small", "cand") == "효과 없음"


# 3단계 구간 조건: 숫자 발화 제외의 상대 변화가 기준을 넘어도 짝지은 구간이 0을 포함하면 개선이 아니다.
# require_interval=False(2단계 규칙)에서는 같은 데이터가 '범용 개선'이 된다.
def test_verdict_needs_the_no_digit_interval_to_exclude_zero(reports):
    # 숫자 발화 제외: 상대 -16.7%지만 한 발화가 나빠져 짝지은 구간이 0을 포함한다.
    prepare_verdict(reports, zeroth_candidate=[0, 0, 2, 3], fleurs_candidate=[1, 1])
    assert analyze.verdict("whisper-small", "cand") == "효과 없음"
    assert analyze.verdict("whisper-small", "cand", require_interval=False) == "범용 개선"  # 2단계 규칙


# FLEURS를 정답에 숫자가 있는 발화와 없는 발화로 나눠 비교할 수 있다. 악화가 숫자 있는 발화에만 있는
# 경우(2단계에서 본 모양)를 그대로 보여 주는지.
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


# 기준선 오류율이 0인 부분집합에서도 table이 멈추지 않는다. 상대 변화는 compare처럼 nan으로 찍고, 차이(%p)와
# 구간은 그대로 찍는다. 고치기 전에는 상대 변화를 0으로 나눠 ZeroDivisionError로 멈췄다.
def test_table_prints_nan_relative_change_when_the_baseline_makes_no_errors(reports, capsys):
    write_report(reports, "base", "fleurs-ko-val", rows([0, 0]))
    write_report(reports, "cand", "fleurs-ko-val", rows([1, 0]))

    analyze.table("fleurs-ko-val", ["base", "cand"], "base", split_digits=False)

    line = next(x for x in capsys.readouterr().out.splitlines() if x.strip().startswith("cand"))
    assert "vs base: +5.00%p" in line and "relative +nan%" in line


# 명령줄 인자를 바꿔 analyze.main을 한 번 부른다.
def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["analyze", *args])
    analyze.main()


TABLE_ARGS = (
    "table",
    "--set",
    "fleurs-ko-val",
    "--reports",
    "base",
    "cand",
    "--baseline",
    "base",
    "--no-digit-split",
)


# table --bootstrap-samples(2026-10-29): 표의 구간(모델마다의 오류율 구간과 기준선과의 차이 구간)이 모두
# 그 표본 수로 계산된다. 주지 않으면 기록과 같은 1,000번이다. 묶음 머리 줄에는 기본값이 아닐 때만 표본 수를
# 덧붙인다(기본 출력은 예전과 한 글자도 같다).
@pytest.mark.parametrize(("extra", "samples"), [((), 1_000), (("--bootstrap-samples", "10000"), 10_000)])
def test_table_passes_the_bootstrap_samples(reports, monkeypatch, capsys, extra, samples):
    write_report(reports, "base", "fleurs-ko-val", rows([1, 0, 2, 1]))
    write_report(reports, "cand", "fleurs-ko-val", rows([0, 0, 1, 1]))
    seen = []

    # 구간 함수를 감싸 실제로 받은 표본 수(위치·키워드 어느 쪽으로 넘겨도, 안 넘기면 기본값)를 적어 둔다.
    def spy(function):
        def wrapped(*args, **kwargs):
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            seen.append((function.__name__, bound.arguments["samples"]))
            return function(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(analyze, "interval", spy(analyze.interval))
    monkeypatch.setattr(analyze, "difference_interval", spy(analyze.difference_interval))
    run_cli(monkeypatch, *TABLE_ARGS, *extra)

    assert seen == [("interval", samples), ("interval", samples), ("difference_interval", samples)]
    header = capsys.readouterr().out.splitlines()[1]
    added = "" if samples == 1_000 else ", bootstrap 10,000 samples (seed 0)"
    assert header == "fleurs-ko-val / all: 4 utterances" + added


# table --decimals: 오류율·구간·차이를 그 자릿수로 찍는다(상대 변화는 그대로 한 자리). 문턱에 가까운 구간 끝을
# 읽을 때 쓴다. 기본 2자리의 모양은 바로 위 테스트들이 지킨다.
def test_table_prints_the_requested_decimals(reports, monkeypatch, capsys):
    write_report(reports, "base", "fleurs-ko-val", rows([0, 0]))
    write_report(reports, "cand", "fleurs-ko-val", rows([1, 0]))

    run_cli(monkeypatch, *TABLE_ARGS, "--decimals", "3")

    out = capsys.readouterr().out.splitlines()
    base = next(x for x in out if x.strip().startswith("base"))
    cand = next(x for x in out if x.strip().startswith("cand"))
    assert base.endswith("  0.000%  [0.000, 0.000]")
    assert cand.endswith("  5.000%  [0.000, 10.000]  vs base: +5.000%p [+0.000, +10.000], relative +nan%")


# 표본이 없는 부트스트랩과 범위 밖 자릿수는 계산 전에 명령줄에서 그 인자의 오류로 거부한다('알 수 없는 인자'가
# 아니라 'argument <옵션>:' 오류).
@pytest.mark.parametrize(
    ("option", "value"),
    [("--bootstrap-samples", "0"), ("--bootstrap-samples", "-5"), ("--decimals", "-1"), ("--decimals", "7")],
)
def test_table_refuses_bad_bootstrap_options(reports, monkeypatch, capsys, option, value):
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, *TABLE_ARGS, option, value)
    assert stop.value.code == 2 and f"argument {option}:" in capsys.readouterr().err


# 정답은 Zeroth식('이천 십 팔 년 오 월')이고 가설만 다른 행들. 저장된 편집 수는 일부러 9로 둔다.
def notation_rows(hypotheses):
    return [
        {"id": f"u{n}", "edits": 9, "length": 9, "reference": "이천 십 팔 년 오 월", "hypothesis": h}
        for n, h in enumerate(hypotheses)
    ]


# 맞춘 CER은 '2018년 5월'과 '이천 십 팔 년 오 월'을 같게 보아 둘 다 오류 0이다. 원래 채점(저장된 값)은 그대로
# 남는다.
def test_harmonized_scoring_ignores_the_notation_of_numbers(reports):
    write_report(reports, "cand", "zeroth-val", notation_rows(["2018년 5월", "이천 십 팔 년 오 월"]))
    loaded = analyze.load_rows("cand", "zeroth-val", harmonized=True)
    assert [row["edits"] for row in loaded.values()] == [0, 0]
    assert analyze.load_rows("cand", "zeroth-val")["u0"]["edits"] == 9  # 저장된 점수는 바뀌지 않는다


# 6단계 판정용 리포트. 맞춘 채점은 글을 다시 채점하므로 정답·가설 글 자체에 오류 수만큼 차이를 넣는다. 기준선
# 10%, L2 2%, 후보는 인자로 정한다.
def prepare_verdict6(reports, candidate_edits, fleurs_candidate):
    def spelled(edits):
        return [{**row, "reference": "문장", "hypothesis": "문장"} for row in rows(edits, length=100)]

    # 맞춘 채점은 글을 다시 채점하므로, 여기서는 글 자체가 오류를 담는다
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


# 기준선보다 크게 낮고 L2와 같으며 다른 도메인 악화가 작으면 '범용으로 쓸 수 있다'(6단계의 실제 판정).
def test_verdict6_general_when_both_rules_hold(reports):
    prepare_verdict6(reports, candidate_edits=[2, 2], fleurs_candidate=[1, 2])
    assert analyze.verdict6("base", "l2", "cand") == "표기를 바꿔 학습하면 범용으로 쓸 수 있다"


# 다른 도메인 악화가 2%p면 '표기만으로는 해결되지 않는다'.
def test_verdict6_notation_is_not_enough_when_the_other_domain_gets_worse(reports):
    prepare_verdict6(reports, candidate_edits=[2, 2], fleurs_candidate=[3, 3])
    assert analyze.verdict6("base", "l2", "cand") == "표기만으로는 해결되지 않는다"


# L2보다 1.0%p 높으면(허용 0.3%p 초과) '같은 도메인의 이득을 잃는다'.
def test_verdict6_gain_lost_when_clearly_above_the_previous_model(reports):
    prepare_verdict6(reports, candidate_edits=[3, 3], fleurs_candidate=[1, 1])  # l2보다 +1.0%p
    assert analyze.verdict6("base", "l2", "cand") == "표기를 바꾸면 같은 도메인의 이득을 잃는다"


# 되풀이 발화는 '편집 수 > 정답 길이'로 센다. 숫자 표기만 다른 발화(b)는 정답 길이를 넘지 않아 되풀이가
# 아니고, 맞춘 채점으로 다시 세도 되풀이는 a 하나다.
def test_loops_counts_utterances_with_more_edits_than_reference_characters(reports, capsys):
    looped = "가나다" + "라니" * 40  # 105_003_0478처럼 두 음절을 끝까지 되풀이한다
    utterances = [
        {"id": "a", "edits": 80, "length": 9, "reference": "이천 십 팔 년 오 월", "hypothesis": looped},
        {"id": "b", "edits": 9, "length": 9, "reference": "이천 십 팔 년 오 월", "hypothesis": "2018년 5월"},
        {"id": "c", "edits": 0, "length": 2, "reference": "문장", "hypothesis": "문장"},
    ]
    write_report(reports, "m", "zeroth-val", utterances)

    assert analyze.loops("zeroth-val", ["m"]) == {"m": ["a"]}
    assert analyze.loops("zeroth-val", ["m"], harmonized=True) == {"m": ["a"]}  # b는 표기 차이뿐이다
    assert "1 of 3" in capsys.readouterr().out


def scored_rows(pairs, **extra):
    """evaluate가 쓰는 것처럼 저장된 편집 수가 글의 원래 채점과 같은 행들(다시 채점한 값과 비교용)."""
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


# 출력 숫자 변환(itn)은 가설만 바꾼다. 숫자로 쓰인 FLEURS 정답에 대해 한글 숫자 가설은 오류 0이 되고, 정답이
# 한글 숫자인 발화는 그대로 틀린다. itn-harmonized는 양쪽을 맞춰 둘 다 0이다.
def test_output_itn_rewrites_the_hypothesis_only(reports):
    pairs = [
        ("1978년 5월에 열렸다", "천 구백 칠십 팔 년 오 월에 열렸다"),  # FLEURS는 아라비아 숫자로 적는다
        ("천 구백 칠십 팔 년", "1978년"),  # 한글로 풀어 쓴 정답은 그대로 남는다
    ]
    write_report(reports, "m", "fleurs-ko-val", scored_rows(pairs))

    itn = analyze.load_rows("m", "fleurs-ko-val", scoring="itn")
    assert itn["u0"]["edits"] == 0
    assert itn["u1"]["edits"] == analyze.load_rows("m", "fleurs-ko-val")["u1"]["edits"] > 0
    both = analyze.load_rows("m", "fleurs-ko-val", scoring="itn-harmonized")
    assert [row["edits"] for row in both.values()] == [0, 0]


# 표기 무관 채점은 숫자와 '%'뿐 아니라 글자 이름('에프 에이에이' → FAA)까지 맞춰 오류 0이 된다. 맞춘
# 채점만으로는 오류가 남고, itn-latin은 가설만 바꿔 오류를 줄인다. 저장된 가설 글은 어떤 채점에서도 바뀌지
# 않는다.
def test_notation_agnostic_scoring_also_reads_letter_names(reports):
    pairs = [("faa는 50%를 줄였다", "에프 에이에이는 오십 퍼센트를 줄였다")]
    write_report(reports, "m", "fleurs-ko-val", scored_rows(pairs))
    assert analyze.load_rows("m", "fleurs-ko-val", scoring="harmonized")["u0"]["edits"] > 0
    assert analyze.load_rows("m", "fleurs-ko-val", scoring="agnostic")["u0"]["edits"] == 0
    latin = analyze.load_rows("m", "fleurs-ko-val", scoring="itn-latin")["u0"]
    assert latin["hypothesis"] == pairs[0][1]  # 저장된 글은 절대 바뀌지 않는다
    assert latin["edits"] < analyze.load_rows("m", "fleurs-ko-val")["u0"]["edits"]


# 한국어용 다시 채점을 영어 묶음에 주면 거부한다.
def test_korean_rescoring_is_refused_for_english_sets(reports):
    write_report(reports, "m", "fleurs-en-val", rows([1]))
    with pytest.raises(SystemExit, match="Korean"):
        analyze.load_rows("m", "fleurs-en-val", scoring="itn")


# 영어 리포트의 행: evaluate처럼 등록 정규화('en', 단어 단위)로 채점한 편집 수와 정답 길이를 저장한다.
def english_rows(pairs):
    out = []
    for n, (reference, hypothesis) in enumerate(pairs):
        scored = score(reference, hypothesis, "en")
        out.append(
            {
                "id": f"u{n}",
                "reference": reference,
                "hypothesis": hypothesis,
                "edits": scored.edits,
                "length": scored.length,
            }
        )
    return out


# 영어 보조 채점 en-whisper(docs/experiments.md '보조 지표' 절): 숫자 표기('7'과 'seven'), '%'와 'percent',
# 영미 철자 차이가 0 편집이 되고, 정답 길이는 정규화한 글의 단어 수로 다시 센다('10,000'은 등록 정규화에서
# 두 단어, 표준 정규화에서 한 단어). 등록 채점(scoring 없음)의 값과 저장된 글은 그대로다.
def test_en_whisper_scoring_matches_english_notation(reports):
    pairs = [
        ("He took 7 wickets", "He took seven wickets"),
        ("10,000 years ago", "ten thousand years ago"),
        ("100% of the vote", "100 percent of the vote"),
        ("good behaviour", "good behavior"),
    ]
    write_report(reports, "m", "fleurs-en-val", english_rows(pairs))
    registered = analyze.load_rows("m", "fleurs-en-val")
    assert all(row["edits"] > 0 for row in registered.values())
    aux = analyze.load_rows("m", "fleurs-en-val", scoring="en-whisper")
    assert [row["edits"] for row in aux.values()] == [0, 0, 0, 0]
    assert (registered["u1"]["length"], aux["u1"]["length"]) == (4, 3)
    assert aux["u0"]["hypothesis"] == pairs[0][1]  # 저장된 글은 절대 바뀌지 않는다


# en-whisper는 표준 정규화의 출력을 그대로 단어 단위로 센다(Whisper의 WER 절차). 등록 정규화를 덧씌우지
# 않으므로 숫자에 붙은 '%'·'$'는 남아 '100%'와 '100'은 한 단어가 다르다. 등록 채점은 기호를 지워 둘을 같게
# 본다.
def test_en_whisper_keeps_the_symbols_the_standard_normalizer_keeps(reports):
    pairs = [("100%", "100"), ("$5", "5"), ("2.5 percent", "2.5%")]
    write_report(reports, "m", "fleurs-en-val", english_rows(pairs))
    registered = analyze.load_rows("m", "fleurs-en-val")
    assert [row["edits"] for row in registered.values()] == [0, 0, 1]
    aux = analyze.load_rows("m", "fleurs-en-val", scoring="en-whisper")
    assert [(row["edits"], row["length"]) for row in aux.values()] == [(1, 1), (1, 1), (0, 1)]


# 표준 정규화로 정답이 비는 발화(군말만 있는 경우)는 공통 규칙의 '정답이 빈 발화'처럼 보조 채점에서 뺀다.
# 등록 채점에서는 그대로 남는다.
def test_en_whisper_drops_a_reference_that_normalizes_to_nothing(reports):
    pairs = [("Hmm.", "Hmm"), ("7 wickets", "seven wickets")]
    write_report(reports, "m", "fleurs-en-val", english_rows(pairs))
    assert set(analyze.load_rows("m", "fleurs-en-val")) == {"u0", "u1"}
    assert set(analyze.load_rows("m", "fleurs-en-val", scoring="en-whisper")) == {"u1"}


# 영어용 보조 채점을 한국어 묶음(Zeroth, FLEURS 한국어)에 주면 거부한다.
@pytest.mark.parametrize("set_name", ["zeroth-val", "fleurs-ko-val"])
def test_en_whisper_is_refused_for_korean_sets(reports, set_name):
    write_report(reports, "m", set_name, rows([1]))
    with pytest.raises(SystemExit, match="English"):
        analyze.load_rows("m", set_name, scoring="en-whisper")


# 규칙의 명령(table --scoring en-whisper --baseline)으로 표기만 다른 두 리포트를 비교하면 등록 채점에서는
# 차이가 나고 보조 채점에서는 +0.00%p가 된다.
def test_table_reports_the_en_whisper_difference(reports, capsys):
    references = ["he took 7 wickets", "100% of the vote"]
    write_report(reports, "base", "fleurs-en-val", english_rows([(r, r) for r in references]))
    spelled = ["he took seven wickets", "100 percent of the vote"]
    write_report(reports, "cand", "fleurs-en-val", english_rows(list(zip(references, spelled, strict=True))))

    analyze.table("fleurs-en-val", ["base", "cand"], "base", split_digits=False)
    registered = capsys.readouterr().out
    assert "vs base: +25.00%p" in registered  # 정답 8단어 중 편집 2개(치환 1, 삽입 1)
    analyze.table("fleurs-en-val", ["base", "cand"], "base", split_digits=False, scoring="en-whisper")
    aux = capsys.readouterr().out
    assert "vs base: +0.00%p" in aux


# 후보와 기준선을 서로 다른 채점으로 비교할 때 정답 길이가 달라지면(여기서는 '%' 처리 차이) 짝지은 비교를
# 거부한다.
def test_compare_refuses_different_reference_lengths(reports):
    write_report(reports, "m", "fleurs-ko-val", scored_rows([("faa는 50%를", "faa는 50%를")]))
    with pytest.raises(SystemExit, match="reference"):
        analyze.compare("fleurs-ko-val", "m", "m", scoring="agnostic", baseline_scoring=None)


# e > 100이면 되풀이 발화가 된다. temperatures를 주면 재시도 기록을 붙인다.
def loop_rows(edits, temperatures=None):
    """8~10단계 테스트용 Zeroth 행: 100글자 정답 뒤에 `e`글자를 덧붙인 가설.

    글에서 다시 채점해도(맞춘 채점 포함) 편집 수가 e로 나온다.
    """
    temperatures = temperatures or [None] * len(edits)
    out = []
    for n, (e, t) in enumerate(zip(edits, temperatures, strict=True)):
        row = {"id": f"u{n}", "reference": "가" * 100, "hypothesis": "가" * 100 + "나" * e, "edits": e}
        row["length"] = 100
        if t is not None:
            row["temperature"], row["fallback_exhausted"] = t, False
        out.append(row)
    return out


# FLEURS용 행. 원래 채점 그대로 쓰므로 편집 수를 직접 넣는다(길이 100).
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
    """8단계 판정용 리포트: 모델 n과 n2의 기록된 greedy 리포트와 후보들의 리포트.

    기록된 greedy에서는 n2의 zeroth-val u3만 되풀이(150 > 100)한다. `arms`는 후보 이름 →
    {(모델, 묶음): (편집 수, temperature 목록)}이며, 주지 않은 묶음은 greedy 리포트를 그대로(temperature 0)
    되풀이한다.
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


# n2의 되풀이 발화가 temperature 0.2로 다시 디코딩되어 되풀이가 없어진 결과.
STOPPED = ([1, 1, 1, 2], [0.0, 0.0, 0.0, 0.2])  # n2의 되풀이가 0.2에서 다시 디코딩되어 사라짐


# 두 후보가 모두 통과하면 더 단순한 D1a를 고른다.
def test_verdict8_picks_the_simpler_arm_when_it_passes(reports):
    stopped = {("n2", "zeroth-val"): STOPPED}
    prepare_stage8(reports, {"d1a": stopped, "d1": stopped})
    assert analyze.verdict8(["n", "n2"], ["d1a", "d1"]) == "d1a"


# D1a가 되풀이는 막았지만 다른 묶음의 정확도를 잃으면(구간 상한 +0.1%p 초과) 다음 후보 D1을 고른다.
def test_verdict8_takes_the_next_arm_when_the_first_costs_accuracy(reports):
    worse = {("n2", "zeroth-val"): STOPPED, ("n", "fleurs-ko-val"): ([1, 3, 1, 3], [0.0, 0.4, 0.0, 0.4])}
    prepare_stage8(reports, {"d1a": worse, "d1": {("n2", "zeroth-val"): STOPPED}})
    assert analyze.verdict8(["n", "n2"], ["d1a", "d1"]) == "d1"


# 되풀이가 남거나(마지막 temperature까지 되풀이) 새 되풀이가 생기면 어느 후보도 고르지 않고 greedy를 그대로
# 쓴다(None).
def test_verdict8_keeps_greedy_when_a_loop_is_left(reports):
    still = {("n2", "zeroth-val"): ([1, 1, 1, 150], [0.0, 0.0, 0.0, 1.0])}  # 다시 디코딩했지만 매번 되풀이
    new_loop = {("n2", "zeroth-val"): STOPPED, ("n", "fleurs-en-val"): ([1, 150], [0.0, 0.0])}
    prepare_stage8(reports, {"d1a": still, "d1": new_loop})
    assert analyze.verdict8(["n", "n2"], ["d1a", "d1"]) is None


# 기록된 되풀이 발화가 첫 디코딩에서 되풀이하지 않아 다시 디코딩되지 않았으면 재시도를 시험하지 못한 것이므로
# 판정하지 않고 멈춘다.
def test_verdict8_makes_no_verdict_when_the_recorded_loop_is_not_decoded_again(reports):
    # 이번에는 첫(greedy) 디코딩이 u3에서 되풀이하지 않아, 재시도가 한 번도 시험되지 않았다.
    gone = {("n2", "zeroth-val"): ([1, 1, 1, 2], [0.0, 0.0, 0.0, 0.0])}
    prepare_stage8(reports, {"d1a": gone, "d1": gone})
    with pytest.raises(SystemExit, match="no verdict"):
        analyze.verdict8(["n", "n2"], ["d1a", "d1"])


# changes가 다시 디코딩한 발화(u1)와, 다시 디코딩하지 않았는데 가설이 바뀐 발화(u2)를 구분하고, 되풀이 전후를
# 맞게 세는지.
def test_changes_separates_decoding_again_from_other_changes(reports, capsys):
    write_report(reports, "base", "fleurs-ko-val", fleurs_rows([1, 150, 1]))
    changed = fleurs_rows([1, 2, 3], [0.0, 0.2, 0.0])  # u1은 다시 디코딩됨, u2는 그 없이 바뀜
    changed[0]["hypothesis"] = "가설 1"
    write_report(reports, "arm", "fleurs-ko-val", changed)
    found = analyze.changes("fleurs-ko-val", "base", ["arm"])["arm"]
    assert found["redecoded"] == ["u1"]
    assert found["changed_not_redecoded"] == ["u2"]
    assert found["changed_outside_loops"] == ["u2"]
    assert (found["loops_before"], found["loops_after"]) == (["u1"], [])


# itn8 판정 이름의 경계: FLEURS 구간 상한이 0보다 작아야 '쓴다'(0이면 '이득이 없다'), Zeroth 맞춘 CER이
# +0.05%p를 넘으면 FLEURS와 관계없이 '해친다'.
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


# 리포트로 itn8 전체를 돌리면, FLEURS에서는 한글 숫자 가설이 후처리로 맞게 되고 Zeroth 맞춘 CER은 그대로라
# '출력 숫자 변환을 쓴다'.
def test_itn8_on_reports(reports):
    words = ["이", "삼", "사", "오", "육"]
    fleurs = [(f"{1972 + n}년 {n + 3}월", f"천 구백 칠십 {words[n]} 년 {words[n + 1]} 월") for n in range(4)]
    zeroth = [("천 구백 칠십 팔 년 오 월", "1978년 5월")] * 4
    for name in ("n", "base"):
        write_report(reports, name, "fleurs-ko-val", scored_rows(fleurs))
        write_report(reports, name, "zeroth-val", scored_rows(zeroth))
    assert analyze.itn8("n", "base") == "출력 숫자 변환을 쓴다"


def gate_file(path, precision, edits, hypotheses=None, nonfinite=None, device="cuda"):
    """9단계 fp16 점검 파일 흉내: zeroth-val500 앞 발화들을 --out으로 쓴 raw 리포트(limit 100).

    발화마다 정답 길이는 250글자이고, 정밀도·장치·빈 가설·넘침을 인자로 정한다.
    """
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


# fp16과 fp32의 CER 차이가 ±0.20%p 안(편집 하나가 0.1%p)이면 fp16을 쓰고, 어느 방향이든 0.3%p면 fp32로 잰다.
def test_gate9_keeps_fp16_when_it_matches_fp32(tmp_path):
    fp32 = gate_file(tmp_path / "fp32.json", "fp32", [2, 2, 2, 2], device="cpu")
    assert analyze.gate9(gate_file(tmp_path / "a.json", "fp16", [2, 2, 2, 2]), fp32)
    # 정답이 1,000글자라 편집 하나가 어느 방향으로든 0.1%p다.
    assert analyze.gate9(gate_file(tmp_path / "b.json", "fp16", [3, 2, 2, 2]), fp32)
    assert analyze.gate9(gate_file(tmp_path / "c.json", "fp16", [1, 2, 2, 2]), fp32)
    assert not analyze.gate9(gate_file(tmp_path / "d.json", "fp16", [5, 2, 2, 2]), fp32)  # +0.3%p
    assert not analyze.gate9(gate_file(tmp_path / "e.json", "fp16", [0, 1, 2, 2]), fp32)  # -0.3%p


# CER이 같아도 fp16 쪽에 빈 가설이나 넘침 발화가 하나라도 있으면 fp32로 잰다.
@pytest.mark.parametrize(
    "broken",
    [
        {"hypotheses": ["가설", "", "가설", "가설"]},  # 빈 출력
        {"nonfinite": [False, False, True, False]},  # 넘친 로짓
    ],
)
def test_gate9_falls_back_to_fp32_on_empty_outputs_or_overflow(tmp_path, broken):
    fp32 = gate_file(tmp_path / "fp32.json", "fp32", [2, 2, 2, 2], device="cpu")
    assert not analyze.gate9(gate_file(tmp_path / "fp16.json", "fp16", [2, 2, 2, 2], **broken), fp32)


# 정밀도 표시가 fp16·fp32 짝이 아니거나 발화가 다르면 비교 자체를 거부한다.
def test_gate9_refuses_files_it_cannot_pair(tmp_path):
    fp32 = gate_file(tmp_path / "fp32.json", "fp32", [2, 2, 2, 2], device="cpu")
    with pytest.raises(SystemExit, match="precision"):
        analyze.gate9(fp32, fp32)
    with pytest.raises(SystemExit, match="utterances"):
        analyze.gate9(gate_file(tmp_path / "fp16.json", "fp16", [2, 2, 2]), fp32)


# 9단계 판정 이름의 경계: 상한이 0이면 '더 나은'이 아니고(상한 < 0이어야 한다), 하한이 0이면 '나쁘다'가
# 아니다.
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
    """9단계 판정용 리포트: 세 validation 묶음에 기준선 turbo, N, 후보의 두 출력(raw와 fixed).

    fixed는 raw의 되풀이를 줄인 모양으로 둔다.
    """
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


# FLEURS 한국어에서 후보가 확실히 낮으면 '더 나은 기준 모델', 같거나 발화마다 엇갈리면 '가리지 못했다', 확실히
# 높으면 '나쁘다'. 보고 항목(fixed, Zeroth 맞춘 CER, 영어)이 출력에 있는지도 본다.
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


# --- 10단계: Qwen3-ASR에 LoRA ---


def adapter_file(
    path,
    hypotheses,
    adapter="outputs/qwen-qn-init/checkpoint-0",
    precision="fp16",
    limit=100,
    set_name="zeroth-val500",
    references=None,
):
    """10단계 --out raw 리포트 흉내: 관문 (a)의 B=0 점검(limit 100) 또는 체크포인트의 zeroth-val500 전체.

    정답·가설 글을 실제로 채점해 편집 수를 넣는다.
    """
    references = references or ["가" * 100] * len(hypotheses)
    utterances = []
    for n, (reference, hypothesis) in enumerate(zip(references, hypotheses, strict=True)):
        scored = score(reference, hypothesis, "ko")
        utterances.append(
            {
                "id": f"u{n}",
                "reference": reference,
                "hypothesis": hypothesis,
                "edits": scored.edits,
                "length": scored.length,
                "nonfinite_logits": False,
            }
        )
    summary = {"precision": precision, "device": "cuda", "set": set_name, "limit": limit, "adapter": adapter}
    path.write_text(json.dumps({"summary": summary, "utterances": utterances}), encoding="utf-8")
    return str(path)


# 관문 (a)는 가설이 하나라도 다르면 실패다(오류율이 비슷한 것으로는 부족하다).
def test_gate10_passes_only_when_every_hypothesis_is_reproduced(tmp_path):
    recorded = adapter_file(tmp_path / "fp16.json", ["가" * 100, "가" * 99], adapter=None)
    same = adapter_file(tmp_path / "same.json", ["가" * 100, "가" * 99])
    one_off = adapter_file(tmp_path / "off.json", ["가" * 100, "가" * 98])
    assert analyze.gate10(same, recorded)
    assert not analyze.gate10(one_off, recorded)


# 어댑터가 있는 실행과 없는 기록의 짝이 아니거나, 발화 수·정밀도가 다르면 비교를 거부한다.
def test_gate10_refuses_files_it_cannot_pair(tmp_path):
    recorded = adapter_file(tmp_path / "fp16.json", ["가"] * 3, adapter=None)
    with pytest.raises(SystemExit, match="adapter"):
        analyze.gate10(recorded, recorded)
    with pytest.raises(SystemExit, match="utterances"):
        analyze.gate10(adapter_file(tmp_path / "short.json", ["가"] * 2), recorded)
    with pytest.raises(SystemExit, match="precision"):
        analyze.gate10(adapter_file(tmp_path / "fp32.json", ["가"] * 3, precision="fp32"), recorded)


# 체크포인트 고르기는 맞춘 CER로 한다. 숫자 표기만 다른 1,000·1,500스텝은 둘 다 오류 0이라 동률이고 이른
# 1,000스텝이 이긴다. --limit 파일은 거부한다.
def test_pick10_takes_the_lowest_harmonized_cer_and_the_earlier_step_on_a_tie(tmp_path):
    references = ["이천 십 팔 년 가나다라마바사"] * 2
    files = [
        adapter_file(
            tmp_path / "s500.json",
            ["2018년 가나다라마바", "2018년 가나다라마바사"],
            adapter="outputs/qwen-qn/checkpoint-500",
            limit=None,
            references=references,
        ),
        adapter_file(
            tmp_path / "s1000.json",
            ["2018년 가나다라마바사"] * 2,  # 아라비아 숫자: harmonize를 거치면 맞다
            adapter="outputs/qwen-qn/checkpoint-1000",
            limit=None,
            references=references,
        ),
        adapter_file(
            tmp_path / "s1500.json",
            ["이천 십 팔 년 가나다라마바사"] * 2,
            adapter="outputs/qwen-qn/checkpoint-1500",
            limit=None,
            references=references,
        ),
    ]
    assert analyze.pick10(files) == files[1]  # 1,000과 1,500이 0으로 같아 이른 스텝이 이긴다
    with pytest.raises(SystemExit, match="limit"):
        analyze.pick10([adapter_file(tmp_path / "limited.json", ["가"], limit=1)])


# steps.jsonl 흉내(한 줄에 한 항목).
def steps_file(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return str(path)


# 스모크 50스텝 기록 흉내. 25번째가 가장 긴 스텝(30초, 메모리 +500MiB)이고, 건너뜀·손실·스텝 수·메모리를 바꿔
# 각 관문을 일부러 실패시킬 수 있다.
def smoke_steps(path, count=50, seconds=6.0, skipped=(), loss=None, peak=5000, longest_at=25, extra=()):
    entries = [{"event": "start", "first_step": 1}]
    for step in range(1, count + 1):
        entries.append(
            {
                "step": step,
                "loss": loss if loss and step == 30 else 0.5,
                "skipped": step in skipped,
                "seconds": 30.0 if step == longest_at else seconds,
                "longest": step == longest_at,
                "peak_allocated_mb": peak + (500 if step == longest_at else 0),
            }
        )
    return steps_file(path, [*entries, *extra])


# 메모리 감시 CSV 흉내. 표본 하나에만 지정한 공유 메모리와 nvidia-smi 값을 넣는다.
def mem_file(path, shared=230, smi=9000):
    rows = ["time,shared_mb,dedicated_mb,smi_used_mb,smi_util,paused"]
    rows += [f"12:00:0{n},{shared if n == 5 else 230},8000,{smi if n == 5 else 8800},90,0" for n in range(10)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


# 깨끗한 스모크는 통과하고 1,500스텝을 유지한다. 스텝 시간 중앙값에서 가장 긴 스텝은 빠진다.
def test_smoke10_passes_a_clean_run_and_keeps_1500_steps(tmp_path):
    result = analyze.smoke10(smoke_steps(tmp_path / "s.jsonl"), mem_file(tmp_path / "m.csv"))
    assert result["passed"] and result["max_steps"] == 1500
    assert result["median_step_seconds"] == 6.0  # 가장 긴 스텝은 빠진다


# 일곱 조건을 하나씩 어긴 경우마다 해당 번호 줄이 '불만족'으로 찍히고 전체가 실패한다: 손실 nan, 20스텝 뒤
# 건너뜀 6개, 가장 긴 스텝 없음, 49스텝, 공유 메모리 3,400MiB(9단계에서 넘쳤을 때의 크기), nvidia-smi
# 10,752MiB(경계값은 실패), torch 7,200MiB.
@pytest.mark.parametrize(
    ("steps", "mem", "failed"),
    [
        ({"loss": float("nan")}, {}, "1."),
        ({"skipped": (1, 2, 3, 21, 22, 23, 24, 25, 26)}, {}, "2."),  # 20스텝 뒤에 6개
        ({"longest_at": None}, {}, "3."),
        ({"count": 49}, {}, "4."),
        ({}, {"shared": 3400}, "5."),
        ({}, {"smi": 10752}, "6."),
        ({"peak": 6700}, {}, "7."),  # 가장 긴 스텝에서 7,200 MiB
    ],
)
def test_smoke10_fails_each_gate(tmp_path, capsys, steps, mem, failed):
    result = analyze.smoke10(smoke_steps(tmp_path / "s.jsonl", **steps), mem_file(tmp_path / "m.csv", **mem))
    assert not result["passed"]
    assert any(line.startswith(failed) and "불만족" in line for line in capsys.readouterr().out.splitlines())


# 앞 20스텝의 건너뜀은 세지 않는다. 스텝당 20초면 1,500스텝이 8.3시간이라 1,000스텝으로 줄인다.
def test_smoke10_allows_skips_in_the_first_20_steps_and_cuts_slow_runs_to_1000_steps(tmp_path):
    result = analyze.smoke10(
        smoke_steps(tmp_path / "s.jsonl", skipped=range(1, 21), seconds=20.0), mem_file(tmp_path / "m.csv")
    )
    assert result["passed"] and result["max_steps"] == 1000  # 20초 x 1,500 = 8.3시간


# 10단계 판정 표의 네 칸이 docs/experiments.md의 문장과 같은지.
@pytest.mark.parametrize(
    ("zeroth_ok", "fleurs_ok", "label"),
    [
        (True, True, "정확도 기준으로 N을 대신하는 범용 모델이다"),
        (True, False, "N을 유지한다 (다른 도메인 이득 없음)"),
        (False, True, "다른 도메인용 기준 모델로만 쓴다"),
        (False, False, "N을 유지한다"),
    ],
)
def test_verdict10_label(zeroth_ok, fleurs_ok, label):
    assert analyze.verdict10_label(zeroth_ok, fleurs_ok) == label


def prepare_stage10(reports, zeroth_candidate, fleurs_candidate):
    """10단계 판정용 리포트: validation 세 묶음에 기준선 turbo, N, 학습 전 Qwen, 후보.

    학습 전 Qwen과 후보는 raw와 fixed 둘 다 쓰고, 두 모델의 전화 음질 판(zeroth-val@telephone)도 쓴다.
    """
    (reports / "digit_utterances").mkdir()
    (reports / "digit_utterances" / "zeroth-val.json").write_text(
        json.dumps({"ids": ["u0"]}), encoding="utf-8"
    )
    zeroth = {"turbo": [4] * 4, "n": [1] * 4, "qwen": [3] * 4, "cand": zeroth_candidate}
    fleurs = {"turbo": [5] * 4, "n": [6] * 4, "qwen": [4] * 4, "cand": fleurs_candidate}
    for name in zeroth:
        names = (name, f"{name}-fixed") if name in ("cand", "qwen") else (name,)
        for report in names:
            write_report(reports, report, "zeroth-val", loop_rows(zeroth[name]))
            write_report(reports, report, "fleurs-ko-val", fleurs_rows(fleurs[name]))
            write_report(reports, report, "fleurs-en-val", fleurs_rows([1, 1]))
    for name in ("qwen", "cand"):
        write_report(reports, name, "zeroth-val@telephone", loop_rows([5] * 4))


# 조건 1·2의 네 조합이 판정 표대로 나오고, 망각 보고 줄(학습 전 Qwen 대비 상한 ≤ +1.0%p)이 '안/밖'으로 맞게
# 찍히는지. 실제 10단계 결과는 둘째 줄과 같은 모양(조건 1만 만족, 망각 허용 폭 밖)이었다. 전화 음질·fixed·영어
# 보고도 출력에 있어야 한다.
@pytest.mark.parametrize(
    ("zeroth_candidate", "fleurs_candidate", "label", "forgetting"),
    [
        ([1, 1, 1, 1], [3, 3, 3, 3], "정확도 기준으로 N을 대신하는 범용 모델이다", "안"),
        ([1, 1, 1, 1], [6, 6, 6, 6], "N을 유지한다 (다른 도메인 이득 없음)", "밖"),
        ([2, 2, 2, 2], [3, 3, 3, 3], "다른 도메인용 기준 모델로만 쓴다", "안"),
        ([2, 2, 2, 2], [7, 7, 7, 7], "N을 유지한다", "밖"),
    ],
)
def test_verdict10_on_reports(reports, capsys, zeroth_candidate, fleurs_candidate, label, forgetting):
    prepare_stage10(reports, zeroth_candidate, fleurs_candidate)
    assert analyze.verdict10("cand", "n", "qwen", ["turbo"]) == label
    out = capsys.readouterr().out
    assert f"망각 상한 +1.0%p {forgetting}" in out
    assert "zeroth-val@telephone" in out and "cand-fixed" in out and "fleurs-en-val" in out


# --- 민감도 분석: 화자·문장 단위 부트스트랩 (robust) ---


# 블록 이름: Zeroth는 화자(ID의 첫 '_' 앞), FLEURS는 문장(ID의 '#' 앞). 그 밖의 묶음은 규칙이 없어 거부한다.
def test_blocks_are_speakers_for_zeroth_and_sentences_for_fleurs():
    assert analyze.block_of("zeroth-val", "152_003_1752") == "152"
    assert analyze.block_of("zeroth-val@telephone", "152_003_1752") == "152"
    assert analyze.block_of("fleurs-ko-val", "1563#2") == analyze.block_of("fleurs-ko-val", "1563") == "1563"
    with pytest.raises(ValueError):
        analyze.block_of("other", "1")


# compare에 with_blocks를 주면 같은 발화에서 블록 단위 구간과 하나 빼기 값이 붙고, 기본 결과는 그대로다.
def test_compare_with_blocks_adds_block_results_and_keeps_the_rest(reports):
    ids = [f"{s}_003_{n:04d}" for s in ("101", "102", "103") for n in range(4)]
    cand = [
        {"id": i, "edits": 1 + (n % 3 == 0), "length": 10, "reference": "문장"} for n, i in enumerate(ids)
    ]
    base = [{"id": i, "edits": 1, "length": 10, "reference": "문장"} for i in ids]
    write_report(reports, "c", "zeroth-val", cand)
    write_report(reports, "b", "zeroth-val", base)
    plain = analyze.compare("zeroth-val", "c", "b")
    blocked = analyze.compare("zeroth-val", "c", "b", with_blocks=True)
    assert {k: blocked[k] for k in plain} == plain
    assert blocked["blocks"] == 3 and set(blocked["leave_one_out"]) == {"101", "102", "103"}


# 조건 이름이 기록된 규칙의 문장대로 판단하는지(경계값 포함).
def test_robust_conditions_follow_the_recorded_rules():
    conditions = analyze.ROBUST_CONDITIONS
    assert conditions["contains 0"](-0.001, 0.0) and not conditions["contains 0"](0.0001, 0.002)
    assert conditions["excludes 0"](-0.005, -0.001) and not conditions["excludes 0"](-0.001, 0.0)
    assert conditions["upper <= +0.1%p"](-0.01, 0.001) and not conditions["upper <= +0.1%p"](0.0, 0.0011)
    assert conditions["upper < +0.10%p"](-0.01, 0.0009) and not conditions["upper < +0.10%p"](0.0, 0.001)
    assert conditions["upper <= +1.0%p"](0.0, 0.010) and not conditions["upper <= +1.0%p"](0.0, 0.0101)
    assert conditions["verdict9"](-0.01, -0.001) and not conditions["verdict9"](-0.01, 0.0)


# 규칙의 비교 목록: 번호가 1~25로 빠짐없이 한 번씩, 조건 이름은 모두 정의되어 있다.
def test_robust_comparisons_cover_the_registered_list():
    numbers = [spec[0] for spec in analyze.ROBUST_COMPARISONS]
    assert numbers == list(range(1, 26))
    assert all(spec[6] in analyze.ROBUST_CONDITIONS for spec in analyze.ROBUST_COMPARISONS)


# --- 11단계 관문과 블록 보고, 12단계 판정 ---


# 11단계 관문: 스케일을 정해 합친 실행이 기록된 실행과 가설이 모두 같아야 통과한다. 어댑터와 스케일이 없는
# 파일은 관문의 대상이 아니라 거부한다.
def test_gate11_passes_only_when_every_hypothesis_is_reproduced(tmp_path):
    recorded = adapter_file(tmp_path / "step1500.json", ["가" * 100, "가" * 99])
    scaled = adapter_file(tmp_path / "s100.json", ["가" * 100, "가" * 99])
    data = json.loads(open(scaled, encoding="utf-8").read())
    data["summary"]["adapter_scale"] = {"audio_tower": 1.0, "language_model": 1.0}
    open(scaled, "w", encoding="utf-8").write(json.dumps(data))
    assert analyze.gate11(scaled, recorded)
    off = json.loads(open(scaled, encoding="utf-8").read())
    off["utterances"][1]["hypothesis"] = "가" * 98
    open(tmp_path / "off.json", "w", encoding="utf-8").write(json.dumps(off))
    assert not analyze.gate11(str(tmp_path / "off.json"), recorded)
    with pytest.raises(SystemExit, match="scale"):
        analyze.gate11(recorded, recorded)


# blocks: 비교 하나에 블록 단위 구간과 하나 빼기 값이 붙는다(보고만).
def test_blocks_reports_one_comparison(reports, capsys):
    ids = [f"{s}_003_{n:04d}" for s in ("101", "102") for n in range(3)]
    write_report(
        reports, "c", "zeroth-val", [{"id": i, "edits": 2, "length": 10, "reference": "문장"} for i in ids]
    )
    write_report(
        reports, "b", "zeroth-val", [{"id": i, "edits": 1, "length": 10, "reference": "문장"} for i in ids]
    )
    result = analyze.blocks("zeroth-val", "c", "b", None, False)
    assert result["blocks"] == 2 and result["block_interval"] == pytest.approx((0.1, 0.1))
    assert "2 blocks" in capsys.readouterr().out


# 12단계 판정용 가짜 speed.json: 실행마다 배속, 한 발화 지연 중앙값, nvidia-smi 증가.
def speed_file(path, fp16, int8):
    runs = []
    for n in (1, 2):
        for label, (memory, latency) in (
            ("hf-b1", (2000, (0.4, 0.4))),
            ("ct2-fp16", fp16),
            ("ct2-int8", int8),
        ):
            summary = {
                "audio_seconds_per_second": 20.0,
                "batch_seconds_median": latency[n - 1],
                "error_rate": 0.1,
            }
            runs.append(
                {
                    "label": f"{label}-{n}",
                    "gpu_memory_added_mb": memory,
                    "gpu_util_avg_during": 40,
                    "summary": summary,
                }
            )
    path.write_text(json.dumps({"preset": "ct2-int8", "idle_reached": False, "runs": runs}), encoding="utf-8")
    return str(path)


# 12단계 판정 표: 정확도를 지키고 메모리가 20% 넘게 줄거나 지연이 두 번 모두 5% 넘게 줄면 권하고, 정확도만
# 지키면 '이득 없음', 정확도를 잃으면 메모리와 관계없이 'float16 유지'.
@pytest.mark.parametrize(
    ("int8_edits", "int8_speed", "label"),
    [
        (1, (1500, (0.40, 0.40)), "int8_float16을 서빙 기본으로 권한다"),  # 메모리 −25%
        (1, (1900, (0.37, 0.37)), "int8_float16을 서빙 기본으로 권한다"),  # 지연 −7.5%
        (1, (1900, (0.37, 0.39)), "정확도는 같지만 이득이 없다 (float16 유지)"),  # 한 번만 5% 넘게
        (3, (1500, (0.30, 0.30)), "정확도를 잃는다 (float16 유지)"),
    ],
)
def test_verdict12_follows_the_table(reports, tmp_path, int8_edits, int8_speed, label):
    reference = "가나다라마바사아자차"  # 10글자. 맞춘 채점이 다시 채점하므로 글로 오류 수를 정한다

    def utterances(ids, edits):
        hypothesis = reference[: 10 - edits] + "카" * edits
        return [
            {"id": i, "edits": edits, "length": 10, "reference": reference, "hypothesis": hypothesis}
            for i in ids
        ]

    for name, edits in (("fp", 1), ("int8", int8_edits)):
        write_report(reports, name, "zeroth-val", utterances([f"101_003_{n:04d}" for n in range(20)], edits))
        write_report(reports, name, "fleurs-ko-val", utterances([str(n) for n in range(5)], 1))
    (reports / "fp" / "zeroth-val500.json").write_text(
        json.dumps({"summary": {"error_rate": 0.1}, "utterances": []}), encoding="utf-8"
    )
    speed = speed_file(tmp_path / "speed.json", (2000, (0.40, 0.40)), int8_speed)
    assert analyze.verdict12(speed, "int8", "fp") == label


# 끝까지 돌지 않은 speed.json(complete: false. 2026-10-29부터 scripts/speed.py가 실행마다 다시 쓰는 부분
# 기록)은 정확도 리포트를 읽기 전에 거부한다. 두 회차가 다 없으면 '두 번 모두'를 볼 수 없다.
# complete: true와 키가 없는 그 전 파일(위 판정 표 테스트)은 받는다.
def test_verdict12_refuses_an_incomplete_speed_file(reports, tmp_path):
    path = tmp_path / "speed.json"
    speed_file(path, (2000, (0.40, 0.40)), (1500, (0.40, 0.40)))
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "complete": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match="complete: false"):
        analyze.verdict12(str(path), "int8", "fp")  # 리포트가 없어도 그보다 먼저 멈춘다

    path.write_text(json.dumps({**data, "complete": True}), encoding="utf-8")
    assert len(analyze._speed_runs(str(path))) == 6


# --- Zeroth 문장 겹침 (overlap) ---


# 임시 data/cache/sets에 prepare와 같은 꼴의 묶음 목록을 쓴다. 겹침 셈은 ID·화자·정답만 보므로 오디오 위치는
# 아무 값이나 둔다. utterances는 (ID, 화자, 정답) 목록이다.
def write_set(cache, name, language, utterances):
    folder = cache / "sets"
    folder.mkdir(parents=True, exist_ok=True)
    entries = [
        {"id": i, "speaker": s, "text": t, "store": "s", "offset": 0, "length": 1} for i, s, t in utterances
    ]
    (folder / f"{name}.json").write_text(
        json.dumps({"language": language, "utterances": entries}, ensure_ascii=False), encoding="utf-8"
    )


# analyze.CACHE를 임시 폴더로 바꾸고 학습 묶음과 기본 평가 묶음 다섯 개, 영어 묶음 하나를 쓴다. 학습: 두
# 화자가 '가나다'를 세 번(한 번은 문장부호만 다름), '라마바'를 한 번 읽음. zeroth-val: '가나다'와 처음 보는
# 문장.
@pytest.fixture
def overlap_sets(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(analyze, "CACHE", cache)
    train = [("a_1", "a", "가나다"), ("a_2", "a", "가나다"), ("b_1", "b", "가나다."), ("b_2", "b", "라마바")]
    write_set(cache, "zeroth-train", "ko", train)
    write_set(cache, "zeroth-val", "ko", [("v_1", "v", "가나다"), ("v_2", "v", "처음 보는 문장")])
    write_set(cache, "zeroth-val500", "ko", [("v_1", "v", "가나다")])
    write_set(cache, "zeroth-test", "ko", [("w_1", "w", "라마바")])
    write_set(cache, "fleurs-ko-val", "ko", [("1", "", "다른 글")])
    write_set(cache, "fleurs-ko-test", "ko", [("2", "", "다른 글"), ("2#2", "", "다른 글")])
    write_set(cache, "fleurs-en-val", "en", [("3", "", "other text")])
    return cache


# overlap은 리포트가 아니라 prepare가 만든 묶음 목록을 읽어 수를 찍고, 파일에 쓸 기록과 같은 dict를 돌려준다.
# 발화별 낭독 횟수와 학습에 없는 발화는 정규화 기준(문장 키)이다. --write 없이는 아무것도 쓰지 않는다.
def test_overlap_reads_the_prepared_sets_and_prints_the_counts(reports, overlap_sets, capsys):
    record = analyze.overlap("zeroth-train", ["zeroth-val", "zeroth-test"])
    out = capsys.readouterr().out

    assert (record["train"]["raw"]["sentences"], record["train"]["normalized"]["sentences"]) == (3, 2)
    val = record["sets"]["zeroth-val"]
    assert (val["raw"]["utterances_in_train"], val["normalized"]["utterances_in_train"]) == (1, 1)
    assert val["train_reads"] == {"v_1": 3, "v_2": 0} and val["not_in_train"] == ["v_2"]
    assert record["sets"]["zeroth-test"]["train_reads"] == {"w_1": 1}
    assert "zeroth-val: 2 utterances" in out and "v_2" in out
    assert not (reports / "sentence_overlap").exists()


# 학습 묶음과 언어가 다른 묶음은 문장 키가 달라 비교하지 않는다.
def test_overlap_refuses_a_set_in_another_language(reports, overlap_sets):
    with pytest.raises(SystemExit, match="language"):
        analyze.overlap("zeroth-train", ["fleurs-en-val"])


# --write는 기본 정의(학습 zeroth-train, 묶음 다섯 개)일 때만 reports/sentence_overlap/zeroth.json에 쓴다.
# 다른 인자로 쓰면 커밋된 파일이 README가 적은 정의와 달라지므로 거부한다. 쓴 파일은 돌려준 기록과 같고 정답
# 글이 들어 있지 않다(발화 ID와 수만).
def test_overlap_writes_only_the_default_definition_without_text(reports, overlap_sets):
    with pytest.raises(SystemExit, match="--write"):
        analyze.overlap("zeroth-train", ["zeroth-val"], write=True)
    assert not (reports / "sentence_overlap").exists()

    record = analyze.overlap(write=True)
    text = (reports / "sentence_overlap" / "zeroth.json").read_text(encoding="utf-8")
    assert json.loads(text) == record
    assert list(record["sets"]) == list(analyze.OVERLAP_SETS)
    assert record["train_set"] == "zeroth-train" and record["per_utterance_key"] == "normalized"
    assert all(word not in text for word in ("가나다", "라마바", "처음", "다른 글"))
    fleurs = record["sets"]["fleurs-ko-test"]
    assert fleurs["speakers"] == 0 and fleurs["normalized"]["repeated_utterances"] == 2


# 커밋된 파생 파일 reports/sentence_overlap/zeroth.json의 수를 고정한다. README 한계 절과
# docs/experiments.md '추가 확인: Zeroth 문장 겹침'이 이 수를 적는다. 데이터나 문장 키가 바뀌어 파일을 다시
# 쓴 결과가 달라지면 여기서 드러난다. 값은 `analyze overlap --write`(2026-10-24)의 출력이다. 원문 기준과
# 정규화 기준의 수가 같다.
OVERLAP_FILE = REPORTS / "sentence_overlap" / "zeroth.json"


# 학습 묶음: 20,049발화, 화자 95명, 고유 문장 3,001개(가장 많이 읽힌 문장 18번, 두 번 이상 읽힌 문장 2,977개).
def test_recorded_sentence_overlap_train():
    data = json.loads(OVERLAP_FILE.read_text(encoding="utf-8"))
    assert data["train_set"] == "zeroth-train"
    assert (data["language"], data["per_utterance_key"]) == ("ko", "normalized")
    train = data["train"]
    assert (train["utterances"], train["speakers"]) == (20049, 95)
    for kind in ("raw", "normalized"):
        assert (train[kind]["sentences"], train[kind]["max_reads"], train[kind]["repeated_sentences"]) == (
            3001,
            18,
            2977,
        )


# 평가 묶음마다: 발화, 화자와 그중 학습 화자, 고유 문장과 그중 학습에 있는 문장, 문장이 학습에 있는 발화,
# 학습에 없는 발화 ID. 발화별 낭독 횟수가 요약 수·학습에 없는 발화 목록과 맞는지도 본다.
@pytest.mark.parametrize(
    ("set_name", "utterances", "speakers", "sentences", "sentences_in_train", "in_train", "not_in_train"),
    [
        ("zeroth-val", 2214, 10, 1642, 1640, 2212, ["152_003_2851", "217_003_0624"]),
        ("zeroth-val500", 440, 10, 379, 379, 440, []),
        ("zeroth-test", 457, 10, 423, 422, 456, ["121_003_1564"]),
        # FLEURS는 화자 칸이 비어 화자 수가 0이다(화자 겹침 확인이 아니다). 학습 문장과 겹치는 것은 없다.
        ("fleurs-ko-val", 226, 0, 129, 0, 0, None),
        ("fleurs-ko-test", 382, 0, 270, 0, 0, None),
    ],
)
def test_recorded_sentence_overlap_sets(
    set_name, utterances, speakers, sentences, sentences_in_train, in_train, not_in_train
):
    entry = json.loads(OVERLAP_FILE.read_text(encoding="utf-8"))["sets"][set_name]
    assert (entry["utterances"], entry["speakers"], entry["speakers_in_train"]) == (utterances, speakers, 0)
    for kind in ("raw", "normalized"):
        summary = entry[kind]
        assert (summary["sentences"], summary["sentences_in_train"]) == (sentences, sentences_in_train)
        assert summary["utterances_in_train"] == in_train
    reads = entry["train_reads"]
    assert len(reads) == utterances
    assert sum(1 for n in reads.values() if n) == in_train
    assert entry["not_in_train"] == sorted(i for i, n in reads.items() if not n)
    if not_in_train is not None:
        assert entry["not_in_train"] == not_in_train


# 영어 보조 지표 en-whisper(docs/experiments.md '보조 지표' 절)로 커밋된 FLEURS 영어 리포트를 다시 채점한 편집
# 수와 정답 단어 수의 총합을 고정한다. 리포트에는 저장하지 않고 부를 때마다 다시 채점하므로, transformers의
# EnglishTextNormalizer나 사전 사본이 바뀌어 값이 움직이면 여기서 드러난다. 각 줄 끝의 퍼센트는 문서에 적은
# 보조 지표 WER이고 괄호 안은 등록 지표다. 값은 규칙의 명령(`analyze table --scoring en-whisper`)의 2026-10-24
# 출력이다. 두 묶음 모두 정규화한 정답이 빈 발화는 없어 발화 수는 등록 지표와 같다(394, 647).
@pytest.mark.parametrize(
    ("report", "set_name", "edits", "length"),
    [
        ("whisper-small", "fleurs-en-val", 561, 8298),  # 6.76% (등록 7.35%)
        ("small-a", "fleurs-en-val", 603, 8298),  # 7.27% (8.22%)
        ("whisper-large-v3-turbo", "fleurs-en-val", 388, 8298),  # 4.68% (5.50%)
        ("turbo-l2", "fleurs-en-val", 417, 8298),  # 5.03% (5.96%)
        ("turbo-n", "fleurs-en-val", 420, 8298),  # 5.06% (6.09%)
        ("qwen3-asr-1.7b", "fleurs-en-val", 306, 8298),  # 3.69% (4.40%)
        ("qwen3-asr-1.7b-qn", "fleurs-en-val", 309, 8298),  # 3.72% (4.91%)
        ("qwen3-asr-1.7b-qn-s050", "fleurs-en-val", 294, 8298),  # 3.54% (4.42%)
        ("whisper-small", "fleurs-en-test", 951, 14557),  # 6.53% (7.08%)
        ("small-a", "fleurs-en-test", 978, 14557),  # 6.72% (7.63%)
        ("whisper-large-v3-turbo", "fleurs-en-test", 635, 14557),  # 4.36% (4.95%)
        ("turbo-l2", "fleurs-en-test", 643, 14557),  # 4.42% (5.23%)
        ("turbo-n", "fleurs-en-test", 647, 14557),  # 4.44% (5.31%)
        ("qwen3-asr-1.7b", "fleurs-en-test", 478, 14557),  # 3.28% (3.91%)
        ("qwen3-asr-1.7b-qn", "fleurs-en-test", 497, 14557),  # 3.41% (4.76%)
        ("qwen3-asr-1.7b-qn-s050", "fleurs-en-test", 472, 14557),  # 3.24% (4.06%)
    ],
)
def test_recorded_en_whisper_totals(report, set_name, edits, length):
    rows = analyze.load_rows(report, set_name, scoring="en-whisper")
    got_edits, got_lengths = analyze.arrays(rows, sorted(rows))
    assert (int(got_edits.sum()), int(got_lengths.sum())) == (edits, length)
