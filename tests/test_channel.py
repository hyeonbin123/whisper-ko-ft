# [테스트 묶음] channel.py(4단계 전화 음질 흉내)가 의도대로 소리를 바꾸는지 확인한다: 길이·dtype 유지, 말소리
# 대역(1 kHz) 보존, 대역 밖(100 Hz, 6 kHz) 제거, 4 kHz 위 성분 없음, mu-law 256단계 양자화, 너무 짧은 입력
# 처리.
# [왜 중요한가] 이 변환으로 만든 전화 음질 수치가 4단계 판정과 10단계 전화 음질 보고에 쓰인다. 변환이 조용히
# 틀리면(대역이 좁아지지 않거나 길이가 달라지면) 전화 음질 실험 전체가 다른 조건을 잰 것이 된다.
# [방법] 실제 음성 대신 순음(사인파)과 백색 잡음을 넣고 주파수별 에너지를 직접 재서, 결과가 분명한 경우만
# 검사한다.

import numpy as np

from whisper_ko_ft.channel import mu_law_roundtrip, telephone
from whisper_ko_ft.store import SAMPLE_RATE


# 주어진 주파수의 순음(진폭 0.5, float32)을 만든다.
def tone(frequency: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


# FFT로 [low, high) Hz 대역의 에너지(파워 스펙트럼 합)를 잰다.
def band_energy(samples: np.ndarray, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum())


# 출력이 float32이고 입력과 길이가 같은지. 16k → 8k → 16k 리샘플로 길이가 달라질 수 있어 잘라 맞추는 부분을
# 1초가 아닌 길이(0.73초)로 검사한다.
def test_length_and_dtype_are_kept():
    out = telephone(tone(1000, 0.73))
    assert out.dtype == np.float32 and len(out) == int(SAMPLE_RATE * 0.73)


# 말소리 대역 안의 1 kHz 순음은 거의 그대로 남아야 한다(원래와 상관계수 0.98 초과). 필터가 파형을 시간축으로
# 밀었다면 상관이 낮아지므로 영위상 필터링도 함께 확인된다.
def test_speech_band_tone_survives():
    original, out = tone(1000), telephone(tone(1000))
    assert np.corrcoef(original, out)[0, 1] > 0.98


# 대역 밖 순음(100 Hz는 300 Hz 아래, 6 kHz는 3.4 kHz 위)은 에너지가 원래의 2% 아래로 줄어야 한다.
def test_tones_outside_the_band_are_removed():
    for frequency in (100, 6000):
        out = telephone(tone(frequency))
        assert band_energy(out, 0, 8000) < 0.02 * band_energy(tone(frequency), 0, 8000)


# 백색 잡음을 넣어도 4.2 kHz 위에는 거의 아무것도 남지 않아야 한다(대역 안 에너지의 0.1% 미만). 대역 제한과 8
# kHz 다운샘플이 실제로 고역을 없앴는지 본다.
def test_nothing_is_left_above_4khz():
    rng = np.random.default_rng(0)
    out = telephone(rng.normal(0, 0.1, SAMPLE_RATE).astype(np.float32))
    assert band_energy(out, 4200, 8000) < 0.001 * band_energy(out, 300, 3400)


# mu-law 왕복 결과는 서로 다른 값이 256개 이하여야 하고(8비트), 작은 소리(0.01)는 오차 0.002 안에서 보존되어야
# 한다. mu-law가 작은 진폭을 촘촘히 양자화한다는 성질을 확인한다.
def test_mu_law_uses_at_most_256_levels_and_keeps_quiet_samples():
    ramp = np.linspace(-1, 1, 100_000).astype(np.float32)
    restored = mu_law_roundtrip(ramp)
    assert len(np.unique(restored)) <= 256
    quiet = np.float32(0.01)
    assert abs(mu_law_roundtrip(np.array([quiet]))[0] - quiet) < 0.002


# 64표본보다 짧은 입력은 영위상 필터의 덧대기 길이가 모자라므로 오류 없이 그대로 돌려줘야 한다.
def test_very_short_input_is_returned_unchanged():
    short = np.zeros(10, dtype=np.float32)
    assert np.array_equal(telephone(short), short)
