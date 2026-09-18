from whisper_ko_ft.splits import val500_ids, validation_speakers


def test_validation_speakers_do_not_depend_on_input_order():
    speakers = [str(n) for n in range(100, 205)]
    picked = validation_speakers(speakers)
    assert picked == validation_speakers(reversed(speakers + speakers))
    assert len(picked) == 10 and len(set(picked)) == 10


def test_validation_speakers_are_not_simply_the_smallest_ids():
    speakers = [str(n) for n in range(100, 205)]
    assert validation_speakers(speakers) != sorted(speakers)[:10]


def test_val500_takes_first_ids_per_speaker():
    entries = [{"speaker": s, "id": f"{s}_{n:03d}"} for s in ("b", "a") for n in range(5, 0, -1)]
    assert val500_ids(entries, per_speaker=2) == ["a_001", "a_002", "b_001", "b_002"]
