import json

import numpy as np

from whisper_ko_ft.store import AudioStore, Utterance, load_set


def test_store_reads_the_slice_of_one_utterance(tmp_path):
    samples = np.arange(-5, 5, dtype=np.int16)
    (tmp_path / "toy.int16").write_bytes(samples.tobytes())
    utterance = Utterance(id="u", speaker="s", text="t", store="toy", offset=3, length=4)

    audio = AudioStore("toy", root=tmp_path).read(utterance)

    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio * 32768, [-2, -1, 0, 1])


def test_load_set_returns_language_and_utterances(tmp_path):
    entry = {"id": "u", "speaker": "s", "text": "안녕", "store": "toy", "offset": 0, "length": 16000}
    (tmp_path / "sets").mkdir()
    (tmp_path / "sets" / "toy-val.json").write_text(
        json.dumps({"language": "ko", "utterances": [entry]}, ensure_ascii=False), encoding="utf-8"
    )

    language, utterances = load_set("toy-val", root=tmp_path)

    assert language == "ko"
    assert utterances[0].text == "안녕" and utterances[0].seconds == 1.0
