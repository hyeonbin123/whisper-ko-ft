import numpy as np

from whisper_ko_ft.channel import mu_law_roundtrip, telephone
from whisper_ko_ft.store import SAMPLE_RATE


def tone(frequency: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def band_energy(samples: np.ndarray, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum())


def test_length_and_dtype_are_kept():
    out = telephone(tone(1000, 0.73))
    assert out.dtype == np.float32 and len(out) == int(SAMPLE_RATE * 0.73)


def test_speech_band_tone_survives():
    original, out = tone(1000), telephone(tone(1000))
    assert np.corrcoef(original, out)[0, 1] > 0.98


def test_tones_outside_the_band_are_removed():
    for frequency in (100, 6000):
        out = telephone(tone(frequency))
        assert band_energy(out, 0, 8000) < 0.02 * band_energy(tone(frequency), 0, 8000)


def test_nothing_is_left_above_4khz():
    rng = np.random.default_rng(0)
    out = telephone(rng.normal(0, 0.1, SAMPLE_RATE).astype(np.float32))
    assert band_energy(out, 4200, 8000) < 0.001 * band_energy(out, 300, 3400)


def test_mu_law_uses_at_most_256_levels_and_keeps_quiet_samples():
    ramp = np.linspace(-1, 1, 100_000).astype(np.float32)
    restored = mu_law_roundtrip(ramp)
    assert len(np.unique(restored)) <= 256
    quiet = np.float32(0.01)
    assert abs(mu_law_roundtrip(np.array([quiet]))[0] - quiet) < 0.002


def test_very_short_input_is_returned_unchanged():
    short = np.zeros(10, dtype=np.float32)
    assert np.array_equal(telephone(short), short)
