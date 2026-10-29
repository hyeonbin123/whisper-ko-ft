"""13단계 문장 hold-out 묶음을 쓴다(docs/experiments.md 13단계).

사용법:
    uv run python -m whisper_ko_ft.prepare_heldout                     # 정의 파일과 견주고 묶음을 쓴다
    uv run python -m whisper_ko_ft.prepare_heldout --write-definition  # 정의 파일을 쓴다(규칙을 정할 때)

prepare·prepare_itn이 만든 data/cache/sets의 zeroth-train·zeroth-val·zeroth-val500·zeroth-test와
zeroth-train-itn·zeroth-val500-itn을 읽어, 세 묶음(zeroth-train·zeroth-val·zeroth-test) 문장 키 합집합의
SHA-256 순 앞 15%(H)를 학습에서 뺀 두 묶음을 같은 폴더에 쓴다:
    zeroth-train-itn-h    학습. zeroth-train-itn에서 원래 정답이 H 문장인 발화를 뺀 것
    zeroth-val500-itn-h   체크포인트 고르기. zeroth-val500-itn에서 같은 규칙으로 뺀 것
판정·보고에 쓰는 발화(zeroth-val·zeroth-test에서 H 문장을 읽은 unseen 발화와 나머지 seen 발화)는 묶음
파일로 쓰지 않고 정의 파일 reports/sentence_holdout/zeroth.json에 발화 ID로 적는다(정답 글은 넣지 않는다).
기존 묶음은 바꾸지 않는다.
"""

# [파일 역할] 13단계(처음 보는 문장에서의 같은 도메인 이득)의 데이터 준비. 원래 묶음에서 H를 정하고, N과 같은
# 설정으로 다시 학습할 학습·고르기 묶음 두 개와, 판정·보고 발화 ID·H 키의 해시·관문 결과를 담은 정의 파일을
# 만든다. 정의 파일은 커밋하고, analyze verdict13이 그 발화 ID로 기존 리포트에서 unseen·seen 발화를 고른다.
# [왜 필요한가] Zeroth의 validation·test 문장은 거의 모두 학습 묶음에 다른 화자의 녹음으로 들어 있어(docs/
# experiments.md '추가 확인: Zeroth 문장 겹침'), 지금까지의 Zeroth 수치는 '처음 듣는 화자가 학습에서 본 문장을
# 읽는' 조건이다. 일부 문장을 학습에서 통째로 빼야 처음 보는 문장의 이득을 따로 잴 수 있다.
# [설계 이유]
# - H는 해시 정렬로만 정한다(splits.heldout_sentences). 무작위 요소가 없어 누가 돌려도 같고, 문장 난도와
#   무관하게 나뉜다.
# - ITN 묶음(숫자를 아라비아 숫자로 바꾼 정답)은 같은 발화 ID의 원래 정답으로 문장 키를 만든다. 바뀐 글로 키를
#   만들면 숫자가 바뀐 문장(학습 20,049문장 중 4,703문장)은 H 키와 달라 빠지지 않고 학습에 남는다.
# - 판정·보고 발화를 묶음 파일로 쓰지 않는다: 판정은 기존 zeroth-val·zeroth-test 리포트에서 발화 ID로 고르므로
#   파일이 필요 없고, evaluate는 이름이 '-test'로 끝나는 묶음만 --allow-test로 막으므로 'zeroth-test-unseen'
#   같은 파일은 'test는 단계마다 한 번' 보호를 비켜 갈 수 있다.
# - 관문(학습 전): 학습·고르기 묶음과 판정 발화(validation unseen, test unseen)의 문장 겹침을 원래 정답 키와,
#   저장된 글에 맞춘 채점(itn.harmonize)을 적용한 키 두 가지로 세고, 학습 묶음과의 화자 겹침도 센다. 하나라도
#   0이 아니면 아무것도 쓰지 않고 멈춘다. 둘째 키는 표기만 다른 같은 문장이 학습 글로 남는 경우를 막는다(판정
#   지표가 맞춘 CER이다). 셈은 splits.sentence_overlap(analyze overlap의 문장 겹침 셈)을 그대로 쓴다.
# - 정의 파일과 견주기: 플래그 없이 돌리면 다시 계산한 정의가 커밋된 정의 파일과 같을 때만 묶음을 쓴다. 학습
#   묶음 파일의 SHA-256까지 정의에 있으므로, GPU에서 학습하기 직전에 돌리면 기록한 hold-out과 바이트까지 같은
#   묶음으로 학습하는지가 확인된다(데이터·ITN 묶음·코드가 바뀌면 멈춘다).
# [연결] splits(H 규칙, 문장 키, 겹침 셈), itn.harmonize(관문의 둘째 키), prepare.EXPECTED_UTTERANCES(원래
# 묶음의 기록된 발화 수), store(Utterance, 표본화율), paths(CACHE, REPORTS). 쓴 묶음은 train --train-set
# zeroth-train-itn-h --eval-set zeroth-val500-itn-h가 store.load_set으로 읽는다.

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence

# itn.harmonize: 관문의 둘째 키(맞춘 채점의 표기 통일). paths: 묶음 폴더(CACHE)와 정의 파일 폴더(REPORTS).
# 테스트가 이 모듈의 CACHE·REPORTS·EXPECTED_UTTERANCES를 임시 값으로 바꿀 수 있게 이름으로 가져와 부를 때
# 읽는다. prepare.EXPECTED_UTTERANCES: 원래 묶음의 기록된 발화 수(docs/experiments.md 1단계 결과). splits: H
# 규칙과 겹침 셈. store: 발화 레코드(겹침 셈의 입력)와 표본화율(시간 합계).
from whisper_ko_ft.itn import harmonize
from whisper_ko_ft.paths import CACHE, REPORTS
from whisper_ko_ft.prepare import EXPECTED_UTTERANCES
from whisper_ko_ft.splits import (
    HELDOUT_PERCENT,
    HOLDOUT_DEFINITION,
    heldout_sentences,
    sentence_hash,
    sentence_key,
    sentence_overlap,
)
from whisper_ko_ft.store import SAMPLE_RATE, Utterance

# ORIGINAL_SETS: 원래 정답(문장 키를 만드는 글)이 든 묶음 넷. ITN_SETS: 숫자를 바꾼 묶음 → 같은 발화의 원래
# 묶음(prepare_itn이 글만 바꿔 쓴 사본). KEY_SETS: H를 정하는 문장 키 합집합의 묶음(val-500은 zeroth-val의
# 일부라 넣지 않는다).
ORIGINAL_SETS = ("zeroth-train", "zeroth-val", "zeroth-val500", "zeroth-test")
ITN_SETS = {"zeroth-train-itn": "zeroth-train", "zeroth-val500-itn": "zeroth-val500"}
SOURCE_SETS = (*ORIGINAL_SETS, *ITN_SETS)
KEY_SETS = ("zeroth-train", "zeroth-val", "zeroth-test")
# TRAINING_SETS: 새로 쓰는 학습·고르기 묶음 → (글을 그대로 가져올 ITN 묶음, 원래 정답을 가져올 묶음).
# TRAIN_SET: 관문에서 화자 겹침도 0이어야 하는 학습 묶음(고르기 묶음은 validation 화자의 발화라 판정 발화와
# 화자가 겹친다. N의 고르기 묶음 val-500도 그랬다).
TRAINING_SETS = {
    "zeroth-train-itn-h": ("zeroth-train-itn", "zeroth-train"),
    "zeroth-val500-itn-h": ("zeroth-val500-itn", "zeroth-val500"),
}
TRAIN_SET = "zeroth-train-itn-h"
# SUBSETS: 판정·보고 발화 묶음 → (원래 묶음, H 문장 발화인가). unseen은 H 문장을 읽은 발화(N은 그 문장을
# 학습에서 봤고 N-H는 못 봄), seen은 나머지(둘 다 봄. 학습에 아예 없던 문장은 never_trained에 따로 적는다).
# JUDGED: 관문에서 학습·고르기 묶음과 문장이 겹치면 안 되는 발화 묶음(판정의 validation unseen, 참고의 test
# unseen).
SUBSETS = {
    "zeroth-val-unseen": ("zeroth-val", True),
    "zeroth-val-seen": ("zeroth-val", False),
    "zeroth-test-unseen": ("zeroth-test", True),
    "zeroth-test-seen": ("zeroth-test", False),
}
JUDGED = ("zeroth-val-unseen", "zeroth-test-unseen")
# 정의 파일에 적는 설명(파일 안의 데이터라 analyze overlap의 keys 칸처럼 영어로 짧게 둔다). 이름이 수 칸(gate,
# sets 등)과 겹치면 definition에서 한쪽이 덮어쓰이므로 관문 설명은 gate_note로 둔다.
NOTES = {
    "rule": "docs/experiments.md stage 13 (sentence hold-out)",
    "command": "python -m whisper_ko_ft.prepare_heldout --write-definition",
    "key": (
        "splits.sentence_key(reference) = text_norm.normalize(text, 'ko'); the -itn sets use the original"
        " reference of the same utterance id"
    ),
    "order": (
        "sha256(key.encode('utf-8')).hexdigest() ascending over the keys of zeroth-train, zeroth-val and"
        " zeroth-test; held out: the first (K * percent + 50) // 100"
    ),
    "gate_note": (
        "per training set and judged subset: utterances whose sentence is in the training set by the"
        " reference key and by normalize(itn.harmonize(stored text)), and speakers in common"
    ),
}


# data/cache/sets에서 원래 묶음 넷과 ITN 묶음 둘을 읽어 묶음 이름 → 발화 항목(dict) 목록을 돌려준다. 항목은
# prepare가 쓴 그대로(id·speaker·text·store·offset·length)라 쓸 때도 그대로 옮긴다. 없거나 한국어가 아닌
# 묶음은 무엇을 먼저 돌릴지 적고 멈춘다.
def load_sources() -> dict[str, list[dict]]:
    """원래 묶음 넷과 ITN 묶음 둘의 발화 항목."""
    sources: dict[str, list[dict]] = {}
    for name in SOURCE_SETS:
        path = CACHE / "sets" / f"{name}.json"
        if not path.exists():
            raise SystemExit(
                f"{path} is missing; run python -m whisper_ko_ft.prepare and"
                " python -m whisper_ko_ft.prepare_itn"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["language"] != "ko":
            raise SystemExit(f"{name}: language {data['language']!r}, expected 'ko'")
        sources[name] = data["utterances"]
    return sources


# 발화 항목에서 글을 뺀 나머지(ID·화자·오디오 위치). ITN 묶음이 원래 묶음과 글만 다른지 견줄 때 쓴다.
def _without_text(utterance: Mapping) -> dict:
    return {key: value for key, value in utterance.items() if key != "text"}


# 입력 묶음이 기록된 묶음인지 본다. 다른 점의 목록을 돌려주고 모두 같으면 빈 목록이다.
# - 묶음마다 발화 ID가 겹치지 않는다(겹치면 ID로 원래 정답을 찾는 일이 틀어진다).
# - 원래 묶음 넷의 발화 수가 기록(expected, 기본은 prepare.EXPECTED_UTTERANCES)과 같다.
# - ITN 묶음은 원래 묶음과 ID·순서·오디오 위치가 같고 글만 다르다(prepare_itn의 사본). 다르면 N이 학습한
#   묶음과 다른 것을 거르는 셈이다.
# - val-500은 zeroth-val의 일부다.
def check_sources(sources: Mapping[str, Sequence[dict]], expected: Mapping[str, int]) -> list[str]:
    """기록된 묶음과 다른 점의 목록. 모두 같으면 빈 목록."""
    problems: list[str] = []
    for name in SOURCE_SETS:
        ids = [u["id"] for u in sources[name]]
        if len(set(ids)) != len(ids):
            problems.append(f"{name}: utterance ids are not unique")
    for name in ORIGINAL_SETS:
        count = len(sources[name])
        if count != expected[name]:
            problems.append(f"{name}: {count:,} utterances, recorded {expected[name]:,}")
    for itn, base in ITN_SETS.items():
        if [_without_text(u) for u in sources[itn]] != [_without_text(u) for u in sources[base]]:
            problems.append(
                f"{itn}: differs from {base} in more than the reference text (ids, order or audio)"
            )
    if not {u["id"] for u in sources["zeroth-val500"]} <= {u["id"] for u in sources["zeroth-val"]}:
        problems.append("zeroth-val500: not part of zeroth-val")
    return problems


# 발화 항목들을 글만 바꾼 Utterance 목록으로 만든다(splits.sentence_overlap의 입력). texts는 항목과 같은
# 순서다.
def _utterances(entries: Sequence[dict], texts: Sequence[str]) -> list[Utterance]:
    return [Utterance(**{**u, "text": text}) for u, text in zip(entries, texts, strict=True)]


# 관문 하나: 학습(또는 고르기) 묶음과 판정 발화 묶음들의 겹침. original은 발화 ID → 원래 정답(학습 묶음의 ITN
# 글 대신 키를 만들 글). 돌려주는 것: 판정 묶음 이름 → {reference: 원래 정답 키로 학습 묶음에 문장이 있는 판정
# 발화 수, harmonized: 저장된 글에 itn.harmonize를 적용한 키로 같은 수, speakers: 학습 묶음과 같은 화자 수}.
def _gate(
    training: Sequence[dict], original: Mapping[str, str], judged: Mapping[str, Sequence[dict]]
) -> dict:
    by_reference = sentence_overlap(
        _utterances(training, [original[u["id"]] for u in training]),
        {name: _utterances(entries, [u["text"] for u in entries]) for name, entries in judged.items()},
    )
    by_harmonized = sentence_overlap(
        _utterances(training, [harmonize(u["text"]) for u in training]),
        {
            name: _utterances(entries, [harmonize(u["text"]) for u in entries])
            for name, entries in judged.items()
        },
    )
    return {
        name: {
            "reference": by_reference["sets"][name]["normalized"]["utterances_in_train"],
            "harmonized": by_harmonized["sets"][name]["normalized"]["utterances_in_train"],
            "speakers": by_reference["sets"][name]["speakers_in_train"],
        }
        for name in judged
    }


# hold-out 전체를 계산한다(파일을 읽거나 쓰지 않는 순수 함수라 가짜 묶음으로 테스트한다). 입력은
# load_sources의 결과(check_sources를 통과한 것). 돌려주는 것(dict):
# - sentences: 문장 키 합집합의 크기 K. held_out: H 키 목록(해시 순).
# - sets: 새 학습·고르기 묶음 이름 → 남긴 ITN 항목(원래 순서, ITN 글 그대로). removed: 묶음마다 뺀 발화 수.
# - subsets: 판정·보고 묶음 이름 → 항목(원래 순서). never_trained: 그중 문장이 zeroth-train에 아예 없는 발화
#   ID(정렬). N도 보지 못한 발화라 unseen이면 M에, seen이면 C에 섞인다(규칙은 H 소속으로만 나눈다).
# - gate: 새 묶음 이름 → _gate의 결과.
def holdout(sources: Mapping[str, Sequence[dict]]) -> dict:
    """원래 묶음들에서 H, 학습·고르기 묶음, 판정·보고 발화, 관문 수를 계산한다."""
    keys = {name: [sentence_key(u["text"]) for u in sources[name]] for name in ORIGINAL_SETS}
    union = {key for name in KEY_SETS for key in keys[name]}
    held_out = heldout_sentences(union)
    held = set(held_out)

    sets: dict[str, list[dict]] = {}
    removed: dict[str, int] = {}
    originals: dict[str, dict[str, str]] = {}
    for new, (stored, base) in TRAINING_SETS.items():
        originals[new] = {u["id"]: u["text"] for u in sources[base]}
        base_keys = dict(zip((u["id"] for u in sources[base]), keys[base], strict=True))
        sets[new] = [u for u in sources[stored] if base_keys[u["id"]] not in held]
        removed[new] = len(sources[stored]) - len(sets[new])

    subsets = {
        name: [u for u, key in zip(sources[of], keys[of], strict=True) if (key in held) == unseen]
        for name, (of, unseen) in SUBSETS.items()
    }
    trained = set(keys["zeroth-train"])
    never_trained = {
        name: sorted(u["id"] for u in entries if sentence_key(u["text"]) not in trained)
        for name, entries in subsets.items()
    }
    judged = {name: subsets[name] for name in JUDGED}
    gate = {new: _gate(sets[new], originals[new], judged) for new in TRAINING_SETS}
    return {
        "sentences": len(union),
        "held_out": held_out,
        "sets": sets,
        "removed": removed,
        "subsets": subsets,
        "never_trained": never_trained,
        "gate": gate,
    }


# 관문 결과에서 어긴 것의 목록(모두 지키면 빈 목록). 학습·고르기 묶음 모두 판정 발화와 문장 겹침이 두 키 모두
# 0이어야 하고, 학습 묶음(TRAIN_SET)은 화자 겹침도 0이어야 한다.
def gate_problems(gate: Mapping[str, Mapping[str, Mapping[str, int]]]) -> list[str]:
    """관문(문장·화자 겹침 0)을 어긴 것의 목록."""
    problems: list[str] = []
    for new, entries in gate.items():
        for name, counts in entries.items():
            if counts["reference"] or counts["harmonized"]:
                problems.append(
                    f"{new}: {name} utterances whose sentence is in it: {counts['reference']} by the"
                    f" reference key, {counts['harmonized']} by the harmonized key"
                )
            if new == TRAIN_SET and counts["speakers"]:
                problems.append(f"{new}: shares {counts['speakers']} speakers with {name}")
    return problems


# 묶음 파일의 바이트. prepare.write_set과 같은 꼴(언어와 발화 목록, ensure_ascii=False, 줄바꿈 없음)이라
# store.load_set이 그대로 읽는다. 정의 파일에 이 바이트의 SHA-256을 적는다.
def set_bytes(utterances: Sequence[dict]) -> bytes:
    """묶음 파일(data/cache/sets/<이름>.json)에 쓸 바이트."""
    return json.dumps({"language": "ko", "utterances": list(utterances)}, ensure_ascii=False).encode("utf-8")


# 음성 길이 합(시간, 소수 둘째 자리)과 화자 수. prepare.write_set이 찍는 수와 같은 계산이다.
def _hours(entries: Sequence[dict]) -> float:
    return round(sum(u["length"] for u in entries) / SAMPLE_RATE / 3600, 2)


def _speakers(entries: Sequence[dict]) -> int:
    return len({u["speaker"] for u in entries if u["speaker"]})


# 정의 파일의 내용(커밋한다). 정답 글은 넣지 않고 H 키의 SHA-256, 묶음마다 수와 파일 SHA-256, 판정·보고 발화
# ID(정렬), 학습에 없던 문장의 발화, 관문 수를 둔다. files는 새 묶음 이름 → set_bytes의 결과.
def definition(result: Mapping, files: Mapping[str, bytes]) -> dict:
    """hold-out 계산 결과를 정의 파일에 쓸 기록으로 만든다."""
    return {
        **NOTES,
        "percent": HELDOUT_PERCENT,
        "sentences": result["sentences"],
        "held_out": len(result["held_out"]),
        "held_out_sha256": sorted(sentence_hash(key) for key in result["held_out"]),
        "sets": {
            name: {
                "from": TRAINING_SETS[name][0],
                "utterances": len(entries),
                "removed": result["removed"][name],
                "hours": _hours(entries),
                "speakers": _speakers(entries),
                "sha256": hashlib.sha256(files[name]).hexdigest(),
            }
            for name, entries in result["sets"].items()
        },
        "subsets": {
            name: {
                "of": SUBSETS[name][0],
                "utterances": len(entries),
                "hours": _hours(entries),
                "speakers": _speakers(entries),
                "never_trained": result["never_trained"][name],
                "ids": sorted(u["id"] for u in entries),
            }
            for name, entries in result["subsets"].items()
        },
        "gate": result["gate"],
    }


# 정의의 수를 사람이 읽게 찍는다(docs/experiments.md 13단계 '묶음'의 수가 이 출력이다).
def show(record: Mapping) -> None:
    print(
        f"sentences {record['sentences']:,}, held out {record['held_out']}"
        f" (the first {record['percent']}% by SHA-256 of the sentence key)"
    )
    for name, entry in record["sets"].items():
        print(
            f"set {name}: {entry['utterances']:,} utterances, {entry['hours']:.2f} h, {entry['speakers']}"
            f" speakers ({entry['removed']:,} removed from {entry['from']})"
        )
    for name, entry in record["subsets"].items():
        never = entry["never_trained"]
        print(
            f"subset {name}: {entry['utterances']:,} utterances, {entry['hours']:.2f} h, {entry['speakers']}"
            f" speakers" + (f"; sentence not in zeroth-train: {' '.join(never)}" if never else "")
        )
    for new, entries in record["gate"].items():
        for name, counts in entries.items():
            print(
                f"gate {new} / {name}: shared sentences by the reference key {counts['reference']}, by the"
                f" harmonized key {counts['harmonized']}; shared speakers {counts['speakers']}"
            )


# 명령줄 진입점. 흐름: 원래 묶음 읽기 → 기록과 견주기 → hold-out 계산 → 관문 → 정의를
# 쓰거나(--write-definition) 커밋된 정의와 견주기 → 학습·고르기 묶음 쓰기. 쓰기 전에 모든 검사를 하므로 멈추면
# 아무 파일도 바뀌지 않는다.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "13단계 문장 hold-out: 원래 정답의 문장 키를 SHA-256 순으로 정렬한 앞 15%(H)를 학습에서 뺀"
            " 학습 묶음 zeroth-train-itn-h와 고르기 묶음 zeroth-val500-itn-h를 data/cache/sets에 쓴다(기존"
            " 묶음은 그대로)"
        ),
        epilog=(
            "판정·보고 발화(zeroth-val·zeroth-test의 unseen·seen)는 정의 파일"
            " reports/sentence_holdout/zeroth.json에 발화 ID로만 적는다. 플래그 없이 돌리면 다시 계산한"
            " 정의가 커밋된 정의 파일과 같을 때만 묶음을 쓴다(학습 전에 돌려 기록한 hold-out과 같은 묶음인지"
            " 확인한다). 관문(학습·고르기 묶음과 판정 발화의 문장 겹침 0, 학습 묶음과의 화자 겹침 0)을 어기면"
            " 아무것도 쓰지 않는다."
        ),
    )
    parser.add_argument(
        "--write-definition",
        action="store_true",
        help=(
            "정의 파일을 쓴다(있으면 바꿔 쓴다). 규칙을 정할 때 한 번 쓰고 커밋한다. 없이 돌리면 커밋된"
            " 정의와 견주기만 한다"
        ),
    )
    args = parser.parse_args()

    sources = load_sources()
    problems = check_sources(sources, EXPECTED_UTTERANCES)
    if problems:
        raise SystemExit(
            "the source sets differ from the recorded ones:\n  "
            + "\n  ".join(problems)
            + "\nrun python -m whisper_ko_ft.prepare and prepare_itn again (the hold-out is defined on them)"
        )
    result = holdout(sources)
    problems = gate_problems(result["gate"])
    if problems:
        raise SystemExit("the gate failed, nothing written:\n  " + "\n  ".join(problems))
    files = {name: set_bytes(entries) for name, entries in result["sets"].items()}
    record = definition(result, files)
    show(record)

    target = REPORTS.joinpath(*HOLDOUT_DEFINITION)
    if args.write_definition:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"wrote {target}")
    elif not target.exists():
        raise SystemExit(
            f"{target} is missing; pass --write-definition to write it and commit it before training"
        )
    else:
        committed = json.loads(target.read_text(encoding="utf-8"))
        if committed != record:
            changed = sorted(
                key for key in committed.keys() | record.keys() if committed.get(key) != record.get(key)
            )
            raise SystemExit(
                f"the hold-out computed from data/cache/sets differs from {target} ({', '.join(changed)});"
                " nothing written. The data, the -itn sets or the code are not the recorded ones"
            )
        print(f"definition matches {target}")
    for name, data in files.items():
        (CACHE / "sets" / f"{name}.json").write_bytes(data)
        print(f"wrote set {name}")


if __name__ == "__main__":
    main()
