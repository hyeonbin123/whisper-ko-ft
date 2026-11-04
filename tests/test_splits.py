# [테스트 묶음] splits.py의 화자 기준 분리 규칙(docs/experiments.md '데이터 분리')과, 학습 묶음과 평가 묶음의
# 문장 겹침을 세는 순수 함수(sentence_overlap, docs/experiments.md '추가 확인: Zeroth 문장 겹침'), 그리고 측정
# 하네스가 --allow-test를 요구할 묶음을 가리는 이름 규칙(is_test_set)을 확인한다.
# [왜 중요한가] validation 화자가 실행마다 달라지면 단계끼리 서로 다른 validation으로 비교하게 된다. 분리가
# 입력 순서나 중복에 흔들리지 않고 규칙대로 해시 순서를 따르는지 고정한다. 문장 겹침은 README의 해석 단서가
# 기대는 수라, 원문 기준과 정규화 기준의 셈이 모두 정의대로 나오는지 작은 가짜 묶음으로 고정한다. test 묶음
# 이름 규칙은 'test는 단계마다 한 번'(같은 문서 '데이터 분리'의 test 줄)을 지키는 장치라, test 묶음에서 만든
# 이름이 그 보호를 비켜 가지 않는지 고정한다.

import pytest

from whisper_ko_ft.splits import (
    is_test_set,
    sentence_key,
    sentence_overlap,
    val500_ids,
    validation_speakers,
)
from whisper_ko_ft.store import Utterance


# 입력 순서를 뒤집고 중복을 넣어도 같은 10명이 나와야 한다(화자 105명, Zeroth train과 같은 수).
def test_validation_speakers_do_not_depend_on_input_order():
    speakers = [str(n) for n in range(100, 205)]
    picked = validation_speakers(speakers)
    assert picked == validation_speakers(reversed(speakers + speakers))
    assert len(picked) == 10 and len(set(picked)) == 10


# 해시 순서로 고르므로 단순히 ID가 작은 10명이 아니어야 한다. ID 정렬로 바뀌는 회귀를 잡는다.
def test_validation_speakers_are_not_simply_the_smallest_ids():
    speakers = [str(n) for n in range(100, 205)]
    assert validation_speakers(speakers) != sorted(speakers)[:10]


# 화자 이름순, 그 안에서 발화 ID순으로 앞의 per_speaker개를 고르는지. 입력은 일부러 역순으로 준다.
def test_val500_takes_first_ids_per_speaker():
    entries = [{"speaker": s, "id": f"{s}_{n:03d}"} for s in ("b", "a") for n in range(5, 0, -1)]
    assert val500_ids(entries, per_speaker=2) == ["a_001", "a_002", "b_001", "b_002"]


# --- 문장 겹침 (sentence_overlap) ---


# 가짜 발화 하나. 겹침 셈은 ID·화자·정답 글만 보므로 오디오 위치는 아무 값이나 둔다.
def utt(utterance_id, speaker, text):
    return Utterance(id=utterance_id, speaker=speaker, text=text, store="s", offset=0, length=1)


# 학습 묶음: 화자 셋(t1, t2, t3)이 문장 셋을 1번, 2번, 4번 읽은 7발화.
TRAIN = [
    utt("t1_1", "t1", "첫째 문장"),
    utt("t1_2", "t1", "둘째 문장"),
    utt("t2_1", "t2", "둘째 문장"),
    utt("t1_3", "t1", "학교에 간다"),
    utt("t2_2", "t2", "학교에 간다"),
    utt("t3_1", "t3", "학교에 간다"),
    utt("t3_2", "t3", "학교에 간다"),
]
# 평가 묶음: 학습에 없는 화자 둘(v1, v2). 문장은 둘이고 그중 하나('학교에 간다')만 학습에 있다. 그 문장은 두
# 화자가 읽었는데, v2의 정답은 띄어쓰기·문장부호만 다르다(원문으로는 다른 문장, 정규화하면 같은 문장).
VAL = [
    utt("v1_1", "v1", "학교에 간다"),
    utt("v2_1", "v2", "학교에,  간다!"),
    utt("v1_2", "v1", "새로운 문장"),
]


# 문장 키는 채점과 같은 정규화다: 띄어쓰기·문장부호·대소문자 차이는 같은 문장으로 본다.
def test_sentence_key_is_the_scoring_normalization():
    assert sentence_key("학교에,  간다!") == sentence_key("학교에 간다") == "학교에간다"
    assert sentence_key("Hello, World", "en") == "hello world"


# 위 가짜 묶음의 모든 수를 고정한다: 학습 쪽(발화·화자·고유 문장·가장 많이 읽힌 횟수·두 번 이상 읽힌 문장과
# 그 발화), 평가 쪽(화자와 학습 화자 겹침, 고유 문장, 학습에 있는 문장·발화, 묶음 안 반복 문장과 그 발화,
# 발화별 학습 낭독 횟수, 학습에 없는 발화). 원문 기준과 정규화 기준이 v2_1 한 발화에서만 갈린다.
def test_sentence_overlap_counts_on_a_small_fake_split():
    result = sentence_overlap(TRAIN, {"val": VAL})

    train = result["train"]
    assert (train["utterances"], train["speakers"]) == (7, 3)
    for kind in ("raw", "normalized"):
        assert train[kind] == {
            "sentences": 3,
            "max_reads": 4,
            "repeated_sentences": 2,
            "repeated_utterances": 6,
        }

    val = result["sets"]["val"]
    assert (val["utterances"], val["speakers"], val["speakers_in_train"]) == (3, 2, 0)
    assert val["normalized"] == {
        "sentences": 2,
        "sentences_in_train": 1,
        "utterances_in_train": 2,
        "repeated_sentences": 1,
        "repeated_utterances": 2,
        "train_reads": {"v1_1": 4, "v2_1": 4, "v1_2": 0},
        "not_in_train": ["v1_2"],
    }
    assert val["raw"] == {
        "sentences": 3,
        "sentences_in_train": 1,
        "utterances_in_train": 1,
        "repeated_sentences": 0,
        "repeated_utterances": 0,
        "train_reads": {"v1_1": 4, "v2_1": 0, "v1_2": 0},
        "not_in_train": ["v1_2", "v2_1"],
    }


# 화자 겹침은 실제로 센다(위 테스트의 0이 '늘 0'이 아님을 확인). FLEURS처럼 화자 칸이 빈 발화는 화자로 세지
# 않으므로 빈 화자끼리 겹친 것으로 보지 않는다. 학습 문장과 하나도 겹치지 않는 묶음은 학습 쪽 수가 모두 0이고,
# 묶음 안의 반복(같은 문장의 두 녹음)은 따로 센다.
def test_sentence_overlap_counts_shared_speakers_and_ignores_empty_ones():
    shared = [utt("t1_9", "t1", "새로운 문장"), utt("v3_1", "v3", "다른 문장")]
    fleurs_like = [utt("1001", "", "전혀 다른 글"), utt("1001#2", "", "전혀 다른 글")]
    train = [*TRAIN, utt("x_1", "", "첫째 문장")]
    result = sentence_overlap(train, {"shared": shared, "fleurs": fleurs_like})

    assert result["train"]["speakers"] == 3
    assert (result["sets"]["shared"]["speakers"], result["sets"]["shared"]["speakers_in_train"]) == (2, 1)
    fleurs = result["sets"]["fleurs"]
    assert (fleurs["speakers"], fleurs["speakers_in_train"]) == (0, 0)
    assert fleurs["normalized"]["utterances_in_train"] == 0
    assert fleurs["normalized"]["not_in_train"] == ["1001", "1001#2"]
    assert (fleurs["normalized"]["repeated_sentences"], fleurs["normalized"]["repeated_utterances"]) == (1, 2)


# 발화 ID가 겹치는 묶음은 발화별 낭독 횟수가 조용히 덮어써지므로 거부한다(1단계 FLEURS ID 오류와 같은 종류).
def test_sentence_overlap_refuses_duplicate_ids():
    with pytest.raises(ValueError, match="not unique"):
        sentence_overlap(TRAIN, {"val": [utt("v1_1", "v1", "가"), utt("v1_1", "v1", "나")]})


# 입력 순서를 바꿔도 수가 같다(셈은 순서와 무관해야 한다). 학습에 없는 발화 목록은 ID순이다.
def test_sentence_overlap_does_not_depend_on_input_order():
    forward = sentence_overlap(TRAIN, {"val": VAL})
    backward = sentence_overlap(list(reversed(TRAIN)), {"val": list(reversed(VAL))})
    assert forward == backward


# --- test 묶음 이름 (is_test_set) ---


# 세 측정 하네스가 --allow-test를 요구할 묶음을 이름으로 가린다. data/cache/sets의 묶음 12개 중 test는
# 셋뿐이고 나머지 아홉(validation·학습·고르기·13단계 hold-out 묶음)은 test가 아니다. 그 밖에 test 묶음에서
# 만든 이름('zeroth-test-unseen' 같은 꼴. 이름이 '-test'로 끝나지 않는다)과 대소문자만 다른 이름(Windows 파일
# 시스템은 같은 묶음 파일을 연다)도 test로 본다. 하이픈 뒤의 'test'만 보므로 낱말 끝에 test가 든 이름은
# 걸리지 않는다.
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("zeroth-test", True),
        ("fleurs-ko-test", True),
        ("fleurs-en-test", True),
        ("zeroth-test-unseen", True),
        ("zeroth-test-seen", True),
        ("Zeroth-Test", True),
        ("ZEROTH-TEST-UNSEEN", True),
        ("zeroth-train", False),
        ("zeroth-train-itn", False),
        ("zeroth-train-itn-h", False),
        ("zeroth-val", False),
        ("zeroth-val500", False),
        ("zeroth-val500-itn", False),
        ("zeroth-val500-itn-h", False),
        ("fleurs-ko-val", False),
        ("fleurs-en-val", False),
        ("zeroth-latest", False),
        ("contest", False),
    ],
)
def test_test_sets_are_told_apart_by_name(name, expected):
    assert is_test_set(name) is expected
