import json
import sys

import pytest
from transformers import WhisperConfig, WhisperFeatureExtractor

from whisper_ko_ft.ct2_features import (
    COPY_FILES,
    PREPROCESSOR_CONFIG,
    feature_kwargs,
    write_preprocessor_config,
)

# whisper-large-v3 and large-v3-turbo take 128 mel bins, whisper-small 80.
TURBO_MEL_BINS = 128
SMALL_MEL_BINS = 80


def test_merged_folder_gets_the_128_bin_config_that_the_converter_copies(tmp_path):
    config = WhisperConfig(num_mel_bins=TURBO_MEL_BINS)  # no weights are loaded
    extractor = WhisperFeatureExtractor(feature_size=TURBO_MEL_BINS)

    written = write_preprocessor_config(extractor, config.num_mel_bins, tmp_path)

    assert written.name == PREPROCESSOR_CONFIG and PREPROCESSOR_CONFIG in COPY_FILES
    saved = json.loads(written.read_text(encoding="utf-8"))
    assert saved["feature_size"] == TURBO_MEL_BINS  # flat, as faster-whisper reads it
    assert feature_kwargs(tmp_path, TURBO_MEL_BINS)["feature_size"] == TURBO_MEL_BINS


def test_a_feature_extractor_that_does_not_fit_the_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="128"):
        write_preprocessor_config(WhisperFeatureExtractor(), TURBO_MEL_BINS, tmp_path)
    assert not (tmp_path / PREPROCESSOR_CONFIG).exists()


@pytest.mark.parametrize("n_mels", [TURBO_MEL_BINS, SMALL_MEL_BINS])
def test_a_folder_without_the_config_uses_the_model_mel_bins(tmp_path, n_mels):
    # The folder converted on 2026-09-20 had only tokenizer.json, and faster-whisper used 80 bins for turbo.
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert feature_kwargs(tmp_path, n_mels) == {"feature_size": n_mels}


def test_a_config_that_disagrees_with_the_model_is_refused(tmp_path):
    write_preprocessor_config(WhisperFeatureExtractor(), SMALL_MEL_BINS, tmp_path)
    with pytest.raises(ValueError, match="128 mel bins"):
        feature_kwargs(tmp_path, TURBO_MEL_BINS)


class Reached(Exception):
    """The model was loaded and the set was about to be read."""


class FakeWhisperModel:
    """faster_whisper.WhisperModel for a turbo folder without preprocessor_config.json, without weights."""

    def __init__(self, path, device, compute_type):
        from faster_whisper.feature_extractor import FeatureExtractor

        self.model = type("Ct2Whisper", (), {"n_mels": TURBO_MEL_BINS})()
        self.feature_extractor = FeatureExtractor()  # what faster-whisper builds: 80 bins
        FakeWhisperModel.last = self


def test_evaluate_ct2_uses_128_mel_bins_for_turbo_before_reading_audio(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper")  # the ct2 dependency group; CI does not install it
    from whisper_ko_ft import evaluate_ct2

    def stop(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(evaluate_ct2, "load_set", stop)
    monkeypatch.setattr(sys, "argv", ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x"])

    with pytest.raises(Reached):
        evaluate_ct2.main()
    assert FakeWhisperModel.last.feature_extractor.mel_filters.shape[0] == TURBO_MEL_BINS


def test_evaluate_ct2_no_report_skips_the_existing_report_check(tmp_path, monkeypatch, capsys):
    pytest.importorskip("faster_whisper")
    from whisper_ko_ft import evaluate_ct2

    def stop(*args, **kwargs):
        raise Reached

    (tmp_path / "n").mkdir()
    (tmp_path / "n" / "x.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate_ct2, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_ct2, "git_commit", lambda: "test")
    monkeypatch.setattr(evaluate_ct2, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(evaluate_ct2, "load_set", stop)
    args = ["evaluate_ct2", "--model", str(tmp_path), "--name", "n", "--set", "x"]

    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        evaluate_ct2.main()
    assert "--overwrite" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*args, "--no-report"])  # speed runs: the report is not replaced
    with pytest.raises(Reached):
        evaluate_ct2.main()
