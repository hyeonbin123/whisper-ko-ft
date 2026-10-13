# [테스트 묶음] store.py(오디오 저장소와 묶음 목록 읽기)를 임시 폴더의 작은 파일로 확인한다. 실제 데이터 없이
# 돈다. tmp_path는 pytest가 테스트마다 주는 임시 폴더다.
# [왜 중요한가] offset·length 계산이 한 칸만 어긋나도 모든 발화가 다른 소리를 읽게 되는데, 학습·평가는 오류
# 없이 돈다.

import json

import numpy as np

from whisper_ko_ft.store import AudioStore, Utterance, load_set


# int16 값 -5~4를 저장해 두고 offset 3부터 4개를 읽으면 -2, -1, 0, 1이 나와야 한다. 결과가 float32이고 32768로
# 나눈 값인지도 본다.
def test_store_reads_the_slice_of_one_utterance(tmp_path):
    samples = np.arange(-5, 5, dtype=np.int16)
    (tmp_path / "toy.int16").write_bytes(samples.tobytes())
    utterance = Utterance(id="u", speaker="s", text="t", store="toy", offset=3, length=4)

    audio = AudioStore("toy", root=tmp_path).read(utterance)

    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio * 32768, [-2, -1, 0, 1])


# 묶음 JSON에서 언어와 Utterance 목록을 읽고, 한국어 정답과 seconds(16,000표본 = 1초)가 맞는지 본다.
def test_load_set_returns_language_and_utterances(tmp_path):
    entry = {"id": "u", "speaker": "s", "text": "안녕", "store": "toy", "offset": 0, "length": 16000}
    (tmp_path / "sets").mkdir()
    (tmp_path / "sets" / "toy-val.json").write_text(
        json.dumps({"language": "ko", "utterances": [entry]}, ensure_ascii=False), encoding="utf-8"
    )

    language, utterances = load_set("toy-val", root=tmp_path)

    assert language == "ko"
    assert utterances[0].text == "안녕" and utterances[0].seconds == 1.0
