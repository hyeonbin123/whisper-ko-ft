"""Telephone channel simulation: 300-3400 Hz band, 8 kHz sampling, 8-bit mu-law (G.711 style).

Input and output are 16 kHz float32 so the result goes through the same feature extractor. There is no
added noise, packet loss or handset response; docs/experiments.md lists this as a limit of stage 4.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from whisper_ko_ft.store import SAMPLE_RATE

MU = 255.0
_BAND = butter(4, [300, 3400], btype="bandpass", fs=SAMPLE_RATE, output="sos")


def mu_law_roundtrip(samples: np.ndarray) -> np.ndarray:
    """Compress to 8-bit mu-law codes and expand back."""
    clipped = np.clip(samples, -1.0, 1.0)
    compressed = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    codes = np.round((compressed + 1.0) / 2.0 * 255.0)  # 256 levels
    restored = codes / 255.0 * 2.0 - 1.0
    return (np.sign(restored) * np.expm1(np.abs(restored) * np.log1p(MU)) / MU).astype(np.float32)


def telephone(samples: np.ndarray) -> np.ndarray:
    """16 kHz audio as it would sound after a narrowband telephone channel, still at 16 kHz."""
    if len(samples) < 64:  # too short for the zero-phase filter's padding
        return samples.astype(np.float32)
    band_limited = sosfiltfilt(_BAND, samples)
    narrow = resample_poly(band_limited, 1, 2)  # 8 kHz
    wide = resample_poly(mu_law_roundtrip(narrow), 2, 1)  # back to 16 kHz
    return wide[: len(samples)].astype(np.float32)


CHANNELS = {"telephone": telephone}
