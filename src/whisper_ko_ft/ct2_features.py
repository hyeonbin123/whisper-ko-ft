"""Log-mel settings of a CTranslate2 Whisper folder for faster-whisper.

faster-whisper reads the feature settings from preprocessor_config.json in the model folder and uses 80 mel
bins when the file is missing. whisper-large-v3 and large-v3-turbo take 128. transformers 5 saves the feature
extractor inside processor_config.json instead, so a merged or fine-tuned folder has no
preprocessor_config.json, ct2-transformers-converter had nothing to copy, and the turbo model got 80-bin
features ("expected (1, 128, 3000), but got (1, 80, 3000)", 2026-09-20).

`merge_adapter` writes preprocessor_config.json next to the merged model (convert it with
--copy_files tokenizer.json preprocessor_config.json), and `evaluate_ct2` takes the number of mel bins from
the converted model itself, so a folder converted without the file is read correctly too.
"""

from __future__ import annotations

import json
from pathlib import Path

PREPROCESSOR_CONFIG = "preprocessor_config.json"
# ct2-transformers-converter --copy_files for a folder written by merge_adapter.
COPY_FILES = ("tokenizer.json", PREPROCESSOR_CONFIG)
# The arguments of faster_whisper.feature_extractor.FeatureExtractor.
FEATURE_KEYS = ("feature_size", "sampling_rate", "hop_length", "chunk_length", "n_fft")


def write_preprocessor_config(feature_extractor, num_mel_bins: int, folder: Path) -> Path:
    """Save a WhisperFeatureExtractor as the flat preprocessor_config.json that faster-whisper reads.

    `num_mel_bins` is the model's (config.num_mel_bins); a feature extractor that does not fit it is refused.
    """
    if feature_extractor.feature_size != num_mel_bins:
        raise ValueError(
            f"the feature extractor makes {feature_extractor.feature_size} mel bins, "
            f"but the model takes {num_mel_bins}"
        )
    target = Path(folder) / PREPROCESSOR_CONFIG
    text = json.dumps(feature_extractor.to_dict(), indent=2, sort_keys=True) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def feature_kwargs(folder: Path, n_mels: int) -> dict:
    """FeatureExtractor arguments for a converted model whose encoder takes `n_mels` mel bins.

    The settings come from the folder's preprocessor_config.json when it has one, and its number of mel bins
    must agree with the model. Without the file, faster-whisper's defaults are kept except the mel bins.
    """
    path = Path(folder) / PREPROCESSOR_CONFIG
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    kwargs = {key: config[key] for key in FEATURE_KEYS if key in config}
    size = kwargs.setdefault("feature_size", n_mels)
    if size != n_mels:
        raise ValueError(f"{path} has feature_size={size}, but the model takes {n_mels} mel bins")
    return kwargs
