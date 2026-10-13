# [테스트 묶음] splits.py의 화자 기준 분리 규칙(docs/experiments.md '데이터 분리')을 확인한다.
# [왜 중요한가] validation 화자가 실행마다 달라지면 단계끼리 서로 다른 validation으로 비교하게 된다. 분리가
# 입력 순서나 중복에 흔들리지 않고 규칙대로 해시 순서를 따르는지 고정한다.

from whisper_ko_ft.splits import val500_ids, validation_speakers


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
