# [테스트 묶음] 13단계 문장 hold-out(docs/experiments.md 13단계)을 확인한다: splits의 H 고르기 규칙
# (heldout_count·heldout_sentences), prepare_heldout이 만드는 학습·고르기 묶음과 판정·보고 발화 묶음, 학습 전
# 관문(문장·화자 겹침 0), 명령(main)이 쓰는 묶음 파일과 정의 파일, 그리고 커밋된 정의 파일
# reports/sentence_holdout/zeroth.json의 수.
# [왜 중요한가] 13단계는 '학습에서 본 적 없는 문장'에서의 이득을 잰다. 판정 묶음의 문장이 학습 묶음에 한
# 발화라도 남으면(예: 숫자를 바꾼 ITN 묶음을 바뀐 글로 걸러 원래 문장이 빠져나감) 재려는 것 자체가 무너진다.
# 그래서 H가 입력 순서와 무관하게 해시 순서로만 정해지는지, ITN 묶음을 원래 정답의 키로 거르는지, 관문이 문장·
# 화자 겹침을 실제로 잡는지를 작은 가짜 묶음으로 고정한다. 커밋된 정의 파일은 데이터 없이(CI) 커밋된 리포트의
# 정답 글과 문장 겹침 파일(reports/sentence_overlap/zeroth.json)로 다시 맞춰 본다.

import hashlib
import json
import re
import sys

import pytest

from whisper_ko_ft import analyze, prepare_heldout
from whisper_ko_ft.paths import REPORTS
from whisper_ko_ft.splits import (
    HOLDOUT_DEFINITION,
    heldout_count,
    heldout_sentences,
    sentence_hash,
    sentence_key,
)
from whisper_ko_ft.store import load_set

# --- H 고르기 (splits.heldout_count, heldout_sentences) ---


# 개수는 15% × K를 반올림한 것이다(.5는 올림). 정수 계산이라 4.5 같은 경계에서도 부동소수점·은행가 반올림에
# 흔들리지 않는다. 실제 Zeroth의 K = 3,004는 451개다.
@pytest.mark.parametrize(
    ("total", "count"),
    [(0, 0), (1, 0), (3, 0), (4, 1), (10, 2), (30, 5), (40, 6), (3004, 451)],
)
def test_heldout_count_rounds_fifteen_percent_half_up(total, count):
    assert heldout_count(total) == count


# 입력 순서를 뒤집고 중복을 넣어도 같은 문장이 같은 순서로 나온다(고유 키 수 K로 개수를 정한다).
def test_heldout_sentences_do_not_depend_on_input_order_or_duplicates():
    keys = [f"문장{n}" for n in range(40)]
    picked = heldout_sentences(keys)
    assert picked == heldout_sentences(list(reversed(keys)) + keys)
    assert len(picked) == heldout_count(40) == 6


# 고른 문장은 키의 SHA-256(UTF-8) 16진수가 가장 작은 것들이고 그 순서대로다. 사전 순이나 입력 순서가 아니다.
def test_heldout_sentences_are_the_smallest_hashes():
    keys = [f"key {n}" for n in range(100)]
    picked = heldout_sentences(keys)
    assert [sentence_hash(k) for k in picked] == sorted(sentence_hash(k) for k in keys)[:15]
    assert picked != sorted(keys)[:15]
    assert sentence_hash("가") == hashlib.sha256("가".encode()).hexdigest()


# --- 가짜 묶음 ---

# 가짜 문장 30개(숫자·라틴 글자 없이 한글만, 정규화 키가 서로 다름). VAL_ONLY·TEST_ONLY는 학습에 없는
# 문장이다.
SENTENCES = [f"가짜 문장 {chr(0xAC00 + 28 * n)}{chr(0xB098 + 28 * n)} 끝말" for n in range(30)]
VAL_ONLY = "검증에만 있는 문장"
TEST_ONLY = "시험에만 있는 문장"
SOURCE_COUNTS = ("zeroth-train", "zeroth-val", "zeroth-val500", "zeroth-test")


# prepare가 쓰는 묶음 항목 하나. 겹침 셈과 hold-out은 ID·화자·정답만 보므로 오디오 위치는 아무 값이나 둔다
# (length는 시간 합계에 쓰인다. 기본 1초).
def entry(utterance_id, speaker, text, store="zeroth_train", length=16_000):
    return {
        "id": utterance_id,
        "speaker": speaker,
        "text": text,
        "store": store,
        "offset": 0,
        "length": length,
    }


# prepare·prepare_itn이 만드는 여섯 묶음의 가짜. 학습: 화자 셋이 30문장을 읽고 앞 20문장은 한 번 더
# 읽음(50발화). validation: 화자 둘이 30문장을 한 번씩 + 학습에 없는 문장(31발화), val-500은 그 앞 12발화.
# test: 화자 둘이 30문장 + 학습에 없는 문장. ITN 묶음은 ID·순서·나머지 칸이 같고 글만 바뀐다(여기서는 끝에 '
# 바뀜'을 붙여 원래 정답과 키가 늘 다르게 한다). extra는 학습 묶음에 더할 (ID, 화자, 원래 정답, ITN 글).
def fake_sources(extra=()):
    train = [entry(f"t{1 + n % 3}_003_{n:04d}", f"t{1 + n % 3}", s) for n, s in enumerate(SENTENCES)]
    train += [
        entry(f"t{1 + (n + 1) % 3}_003_{n + 100:04d}", f"t{1 + (n + 1) % 3}", s)
        for n, s in enumerate(SENTENCES[:20])
    ]
    itn_texts = [u["text"] + " 바뀜" for u in train]
    for utterance_id, speaker, original, itn_text in extra:
        train.append(entry(utterance_id, speaker, original))
        itn_texts.append(itn_text)
    val = [entry(f"v{1 + n % 2}_003_{n:04d}", f"v{1 + n % 2}", s) for n, s in enumerate(SENTENCES)]
    val.append(entry("v1_003_0900", "v1", VAL_ONLY))
    test = [
        entry(f"w{1 + n % 2}_003_{n:04d}", f"w{1 + n % 2}", s, store="zeroth_test")
        for n, s in enumerate(SENTENCES)
    ]
    test.append(entry("w1_003_0900", "w1", TEST_ONLY, store="zeroth_test"))
    val500 = val[:12]
    return {
        "zeroth-train": train,
        "zeroth-val": val,
        "zeroth-val500": val500,
        "zeroth-test": test,
        "zeroth-train-itn": [{**u, "text": t} for u, t in zip(train, itn_texts, strict=True)],
        "zeroth-val500-itn": [{**u, "text": u["text"] + " 바뀜"} for u in val500],
    }


# 가짜 묶음의 H(문장 키 집합). 학습·validation·test 키의 합집합은 32개라 5개다.
def held_keys(sources):
    return set(prepare_heldout.holdout(sources)["held_out"])


# --- 묶음 만들기 (prepare_heldout.holdout) ---


# 학습·고르기 묶음은 ITN 묶음에서 '원래 정답'의 키가 H에 든 발화를 뺀 것이다. ITN 글 자체는 H 키와 겹치지
# 않으므로(끝에 ' 바뀜'), 바뀐 글로 거르는 잘못된 구현이면 아무것도 빠지지 않아 이 테스트가 잡는다. 남은
# 발화는 ITN 글 그대로이고 원래 순서다.
def test_training_sets_drop_held_out_sentences_by_the_original_reference():
    sources = fake_sources()
    result = prepare_heldout.holdout(sources)
    held = set(result["held_out"])
    assert result["sentences"] == 32 and len(held) == heldout_count(32) == 5

    for new, (stored, base) in prepare_heldout.TRAINING_SETS.items():
        original = {u["id"]: sentence_key(u["text"]) for u in sources[base]}
        expected = [u for u in sources[stored] if original[u["id"]] not in held]
        assert result["sets"][new] == expected
        assert result["removed"][new] == len(sources[stored]) - len(expected)
        assert not any(sentence_key(u["text"]) in held for u in sources[stored])
    # 학습 문장 30개 중 적어도 세 개가 H에 들어(학습에 없는 문장은 둘뿐) 학습 묶음에서 실제로 발화가 빠진다.
    assert result["removed"]["zeroth-train-itn-h"] > 0


# validation·test는 H 소속으로만 unseen(H 문장)·seen(나머지)으로 나뉜다: 둘은 겹치지 않고 합치면 원래 묶음이며
# 원래 순서다. 학습에 아예 없는 문장의 발화는 어느 쪽에 들든 never_trained에 적힌다(N도 보지 못한 발화).
def test_validation_and_test_are_split_by_the_held_out_sentences():
    sources = fake_sources()
    result = prepare_heldout.holdout(sources)
    held = set(result["held_out"])
    subsets = result["subsets"]
    assert set(subsets) == set(prepare_heldout.SUBSETS)
    for of, only in (("zeroth-val", VAL_ONLY), ("zeroth-test", TEST_ONLY)):
        unseen, seen = subsets[f"{of}-unseen"], subsets[f"{of}-seen"]
        assert unseen == [u for u in sources[of] if sentence_key(u["text"]) in held]
        assert seen == [u for u in sources[of] if sentence_key(u["text"]) not in held]
        assert unseen and seen
        lonely = next(u["id"] for u in sources[of] if u["text"] == only)
        where = f"{of}-unseen" if sentence_key(only) in held else f"{of}-seen"
        assert result["never_trained"][where] == [lonely]
        other = f"{of}-seen" if where.endswith("unseen") else f"{of}-unseen"
        assert result["never_trained"][other] == []


# 깨끗한 분리에서 관문은 모두 0이다. 고르기 묶음(val-500)과 validation unseen은 같은 validation 화자라 화자가
# 겹치지만(허용, 기록만 한다) 학습 묶음과는 화자도 0이다.
def test_the_gate_is_zero_on_a_clean_split():
    result = prepare_heldout.holdout(fake_sources())
    gate = result["gate"]
    for name in prepare_heldout.JUDGED:
        assert gate["zeroth-train-itn-h"][name] == {"reference": 0, "harmonized": 0, "speakers": 0}
        counts = gate["zeroth-val500-itn-h"][name]
        assert (counts["reference"], counts["harmonized"]) == (0, 0)
    assert gate["zeroth-val500-itn-h"]["zeroth-val-unseen"]["speakers"] > 0
    assert prepare_heldout.gate_problems(gate) == []


# 판정 묶음의 화자가 학습 묶음에도 있으면 관문이 막는다(화자 기준 분리가 깨진 데이터). 더한 학습 발화는 H 밖
# 문장이라 문장 겹침은 그대로 0이다.
def test_the_gate_refuses_a_judged_speaker_in_the_training_set():
    held = held_keys(fake_sources())
    free = next(s for s in SENTENCES if sentence_key(s) not in held)
    extra = [(f"{s}_003_0800", s, free, free + " 바뀜") for s in ("v1", "v2", "w1", "w2")]
    result = prepare_heldout.holdout(fake_sources(extra))
    for name in prepare_heldout.JUDGED:
        counts = result["gate"]["zeroth-train-itn-h"][name]
        assert counts["speakers"] > 0 and (counts["reference"], counts["harmonized"]) == (0, 0)
    problems = prepare_heldout.gate_problems(result["gate"])
    assert problems and all("speakers" in p for p in problems)


# 원래 정답은 H 밖인데 학습에 들어가는 글(ITN)이 H 문장과 같은 발화: 원래 정답 키로는 남지만, 저장된 글을 맞춘
# 채점(itn.harmonize)으로 바꾼 키로는 판정 문장과 겹친다. 표기만 다른 같은 문장이 학습에 남는 경우를 막는
# 둘째 키다.
def test_the_gate_also_compares_the_harmonized_stored_text():
    held = held_keys(fake_sources())
    hidden = next(s for s in SENTENCES if sentence_key(s) in held)
    free = next(s for s in SENTENCES if sentence_key(s) not in held)
    result = prepare_heldout.holdout(fake_sources([("t9_003_0001", "t9", free, hidden)]))
    counts = result["gate"]["zeroth-train-itn-h"]["zeroth-val-unseen"]
    assert (counts["reference"], counts["harmonized"], counts["speakers"]) == (0, 1, 0)
    assert any("harmonized" in p for p in prepare_heldout.gate_problems(result["gate"]))


# 입력 묶음 검사: 원래 묶음 넷의 발화 수가 기록과 같고, ITN 묶음은 원래 묶음과 ID·순서·글 밖의 칸이 같아야
# 한다. 다르면(다른 데이터, 다시 만든 ITN 묶음의 순서가 바뀜 등) 무엇이 다른지 적는다.
def test_the_sources_must_be_the_recorded_sets():
    sources = fake_sources()
    expected = {name: len(sources[name]) for name in SOURCE_COUNTS}
    assert prepare_heldout.check_sources(sources, expected) == []

    problems = prepare_heldout.check_sources(sources, {**expected, "zeroth-test": 457})
    assert len(problems) == 1 and "zeroth-test" in problems[0]
    shuffled = {**sources, "zeroth-train-itn": list(reversed(sources["zeroth-train-itn"]))}
    assert any("zeroth-train-itn" in p for p in prepare_heldout.check_sources(shuffled, expected))
    first, *rest = sources["zeroth-val500-itn"]
    moved = {**sources, "zeroth-val500-itn": [{**first, "offset": 5}, *rest]}
    assert any("zeroth-val500-itn" in p for p in prepare_heldout.check_sources(moved, expected))
    twice = {**sources, "zeroth-val": [*sources["zeroth-val"], sources["zeroth-val"][0]]}
    assert any(
        "not unique" in p for p in prepare_heldout.check_sources(twice, {**expected, "zeroth-val": 32})
    )


# --- 명령 (prepare_heldout.main) ---


# 임시 data/cache/sets에 가짜 묶음 여섯 개를 쓰고 prepare_heldout의 CACHE·REPORTS·기록 발화 수를 그 가짜에
# 맞춘다. 돌려주는 것: (묶음 폴더, 정의 파일 경로, 묶음).
@pytest.fixture
def workspace(tmp_path, monkeypatch):
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    sources = fake_sources()
    (cache / "sets").mkdir(parents=True)
    for name, utterances in sources.items():
        payload = {"language": "ko", "utterances": utterances}
        (cache / "sets" / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    monkeypatch.setattr(prepare_heldout, "CACHE", cache)
    monkeypatch.setattr(prepare_heldout, "REPORTS", reports)
    monkeypatch.setattr(prepare_heldout, "EXPECTED_UTTERANCES", {n: len(sources[n]) for n in SOURCE_COUNTS})
    return cache / "sets", reports.joinpath(*HOLDOUT_DEFINITION), sources


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prepare_heldout", *args])
    prepare_heldout.main()


# 정의 파일이 없으면 --write-definition 없이는 아무것도 쓰지 않는다. --write-definition이면 정의 파일과
# 학습·고르기 묶음 두 개를 쓴다: 원래 묶음 여섯 개는 바이트까지 그대로이고, 묶음 파일은 store.load_set으로
# 읽히며 그 SHA-256이 정의 파일에 있다. 정의 파일에는 정답 글이 없다(한글이 한 글자도 없음). 같은 데이터로
# 다시 돌리면(플래그 없이) 정의 파일과 같다고 찍고 같은 바이트를 쓴다.
def test_main_writes_the_definition_and_the_two_training_sets(workspace, monkeypatch, capsys):
    sets, definition, sources = workspace
    before = {p.name: p.read_bytes() for p in sets.glob("*.json")}
    with pytest.raises(SystemExit, match="--write-definition"):
        run_main(monkeypatch)
    assert sorted(p.name for p in sets.glob("*.json")) == sorted(before) and not definition.exists()

    run_main(monkeypatch, "--write-definition")
    text = definition.read_text(encoding="utf-8")
    record = json.loads(text)
    assert not re.search("[가-힣]", text)
    written = sorted(set(p.name for p in sets.glob("*.json")) - set(before))
    assert written == ["zeroth-train-itn-h.json", "zeroth-val500-itn-h.json"]
    assert all((sets / name).read_bytes() == data for name, data in before.items())
    for name in prepare_heldout.TRAINING_SETS:
        data = (sets / f"{name}.json").read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sets"][name]["sha256"]
        language, utterances = load_set(name, sets.parent)
        assert language == "ko" and len(utterances) == record["sets"][name]["utterances"]
    assert record["held_out"] == 5 and len(record["held_out_sha256"]) == 5
    # 설명 칸(문자열)과 수 칸이 같은 이름으로 덮어쓰이지 않는다: 관문 설명과 관문 수가 따로 남는다.
    assert set(prepare_heldout.NOTES) <= set(record) and not set(prepare_heldout.NOTES) & {"gate", "sets"}
    assert isinstance(record["gate_note"], str) and set(record["gate"]) == set(prepare_heldout.TRAINING_SETS)
    assert record["subsets"]["zeroth-val-unseen"]["ids"] == sorted(
        record["subsets"]["zeroth-val-unseen"]["ids"]
    )
    first = {name: (sets / f"{name}.json").read_bytes() for name in prepare_heldout.TRAINING_SETS}

    run_main(monkeypatch)
    assert "matches" in capsys.readouterr().out
    assert {name: (sets / f"{name}.json").read_bytes() for name in prepare_heldout.TRAINING_SETS} == first


# 다시 계산한 정의가 커밋된 정의와 다르면 묶음을 쓰기 전에 멈춘다(기록한 hold-out과 다른 묶음으로 학습하지
# 않게). 예: 학습에 남는 발화의 ITN 글 하나가 바뀌면(숫자 변환 규칙을 고쳐 prepare_itn을 다시 돌린 경우) 학습
# 묶음 파일의 SHA-256이 달라진다. validation unseen이 바뀌는 경우(H 문장을 읽은 발화가 늘어남)도 같다. 관문을
# 어기는 데이터도 아무것도 쓰기 전에 멈춘다.
def test_main_stops_before_writing_when_the_definition_or_the_gate_differs(workspace, monkeypatch):
    sets, definition, sources = workspace
    run_main(monkeypatch, "--write-definition")
    first = {name: (sets / f"{name}.json").read_bytes() for name in prepare_heldout.TRAINING_SETS}
    recorded = definition.read_bytes()
    held = held_keys(sources)

    def write(name, utterances):
        payload = {"language": "ko", "utterances": utterances}
        (sets / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    kept = next(n for n, u in enumerate(sources["zeroth-train"]) if sentence_key(u["text"]) not in held)
    itn = [dict(u) for u in sources["zeroth-train-itn"]]
    itn[kept]["text"] += " 다시"
    write("zeroth-train-itn", itn)
    with pytest.raises(SystemExit, match="differs"):
        run_main(monkeypatch)
    write("zeroth-train-itn", sources["zeroth-train-itn"])

    hidden = next(s for s in SENTENCES if sentence_key(s) in held)
    write("zeroth-val", [*sources["zeroth-val"][:-1], {**sources["zeroth-val"][-1], "text": hidden}])
    with pytest.raises(SystemExit, match="differs"):
        run_main(monkeypatch)
    assert {name: (sets / f"{name}.json").read_bytes() for name in prepare_heldout.TRAINING_SETS} == first
    assert definition.read_bytes() == recorded

    # 원래 묶음 여섯 개를 다시 쓰되 validation 화자 둘이 학습에도 H 밖 문장을 하나씩 읽게 한다(H는 처음과
    # 같다).
    free = next(s for s in SENTENCES if sentence_key(s) not in held)
    shared = fake_sources([("v1_003_0800", "v1", free, free), ("v2_003_0800", "v2", free, free)])
    for name, utterances in shared.items():
        write(name, utterances)
    monkeypatch.setattr(prepare_heldout, "EXPECTED_UTTERANCES", {n: len(shared[n]) for n in SOURCE_COUNTS})
    with pytest.raises(SystemExit, match="gate"):
        run_main(monkeypatch, "--write-definition")
    assert definition.read_bytes() == recorded
    assert {name: (sets / f"{name}.json").read_bytes() for name in prepare_heldout.TRAINING_SETS} == first


# --- 커밋된 정의 파일 reports/sentence_holdout/zeroth.json (데이터 없이 CI에서도 돈다) ---

# 값은 `prepare_heldout --write-definition`(2026-10-29)의 출력이고 docs/experiments.md 13단계 '묶음'이 이 수를
# 적는다: 3,004문장 중 451문장을 떼면 학습 17,025발화 39.33시간, validation unseen 347발화(화자 9명), test
# unseen 74발화.
DEFINITION = REPORTS.joinpath(*HOLDOUT_DEFINITION)


def recorded_definition():
    return json.loads(DEFINITION.read_text(encoding="utf-8"))


# 머리 수(K, 떼는 수, 해시 목록)와 두 학습 묶음, 네 발화 묶음의 수, 학습에 없던 문장의 발화, 관문.
def test_recorded_definition_counts():
    data = recorded_definition()
    assert (data["percent"], data["sentences"], data["held_out"]) == (15, 3004, 451)
    assert data["held_out"] == heldout_count(data["sentences"])
    hashes = data["held_out_sha256"]
    assert hashes == sorted(hashes) and len(set(hashes)) == 451
    assert all(re.fullmatch("[0-9a-f]{64}", h) for h in hashes)

    sets = {
        name: (e["from"], e["utterances"], e["removed"], e["hours"], e["speakers"])
        for name, e in data["sets"].items()
    }
    assert sets == {
        "zeroth-train-itn-h": ("zeroth-train-itn", 17025, 3024, 39.33, 95),
        "zeroth-val500-itn-h": ("zeroth-val500-itn", 379, 61, 0.92, 10),
    }
    subsets = data["subsets"]
    counts = {name: (e["of"], e["utterances"], e["speakers"], len(e["ids"])) for name, e in subsets.items()}
    assert counts == {
        "zeroth-val-unseen": ("zeroth-val", 347, 9, 347),
        "zeroth-val-seen": ("zeroth-val", 1867, 10, 1867),
        "zeroth-test-unseen": ("zeroth-test", 74, 10, 74),
        "zeroth-test-seen": ("zeroth-test", 383, 10, 383),
    }
    assert {name: e["never_trained"] for name, e in subsets.items()} == {
        "zeroth-val-unseen": [],
        "zeroth-val-seen": ["152_003_2851", "217_003_0624"],
        "zeroth-test-unseen": ["121_003_1564"],
        "zeroth-test-seen": [],
    }
    assert prepare_heldout.gate_problems(data["gate"]) == []
    assert data["gate"]["zeroth-train-itn-h"]["zeroth-val-unseen"] == {
        "reference": 0,
        "harmonized": 0,
        "speakers": 0,
    }


# validation·test 발화가 정의대로 나뉘었는지 커밋된 리포트의 정답 글로 다시 본다: unseen과 seen은 겹치지 않고
# 합치면 그 묶음의 리포트 발화 전부이며, 발화의 문장 키 해시가 H에 들면 unseen이다. H 밖 해시는 모두 H의 가장
# 큰 해시보다 크다(H가 해시 순서의 앞부분이라는 정의).
@pytest.mark.parametrize("set_name", ["zeroth-val", "zeroth-test"])
def test_recorded_subsets_follow_the_held_out_sentences(set_name):
    data = recorded_definition()
    held = set(data["held_out_sha256"])
    last = max(held)
    rows = json.loads((REPORTS / "whisper-large-v3-turbo" / f"{set_name}.json").read_text(encoding="utf-8"))
    unseen = set(data["subsets"][f"{set_name}-unseen"]["ids"])
    seen = set(data["subsets"][f"{set_name}-seen"]["ids"])
    assert unseen.isdisjoint(seen) and unseen | seen == {row["id"] for row in rows["utterances"]}
    for row in rows["utterances"]:
        digest = sentence_hash(sentence_key(row["reference"]))
        assert (row["id"] in unseen) == (digest in held), row["id"]
        assert digest in held or digest > last


# 학습에 아예 없던 문장의 발화(never_trained)는 문장 겹침 파일(reports/sentence_overlap/zeroth.json, analyze
# overlap)이 적은 '학습에 없는 발화'와 같다. 두 파일이 같은 문장 키로 만들어졌는지의 교차 확인이다.
def test_recorded_never_trained_matches_the_sentence_overlap_file():
    data = recorded_definition()
    overlap = json.loads((REPORTS / "sentence_overlap" / "zeroth.json").read_text(encoding="utf-8"))
    for of in ("zeroth-val", "zeroth-test"):
        never = (
            data["subsets"][f"{of}-unseen"]["never_trained"] + data["subsets"][f"{of}-seen"]["never_trained"]
        )
        assert sorted(never) == overlap["sets"][of]["not_in_train"]


# analyze verdict13은 같은 위치의 정의 파일을 읽는다(경로 상수를 한곳에서 가져온다).
def test_analyze_reads_the_definition_prepare_heldout_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "REPORTS", tmp_path)
    target = tmp_path.joinpath(*HOLDOUT_DEFINITION)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"subsets": {}}), encoding="utf-8")
    assert analyze.holdout_definition() == {"subsets": {}}
