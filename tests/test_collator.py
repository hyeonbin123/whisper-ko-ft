from types import SimpleNamespace

import numpy as np
import torch

from whisper_ko_ft.train import Collator

START, PAD = 1, 0


class FakeFeatureExtractor:
    def __call__(self, audio, **kwargs):
        batch = len(audio)
        return SimpleNamespace(
            input_features=torch.zeros(batch, 80, 3000), attention_mask=torch.ones(batch, 3000)
        )


class FakeTokenizer:
    def __init__(self, with_start: bool) -> None:
        self.with_start = with_start

    def __call__(self, texts, **kwargs):
        rows = [([START] if self.with_start else []) + [ord(c) for c in text] for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.tensor([row + [PAD] * (width - len(row)) for row in rows])
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows])
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


def collate(with_start: bool) -> dict:
    processor = SimpleNamespace(feature_extractor=FakeFeatureExtractor(), tokenizer=FakeTokenizer(with_start))
    rows = [{"audio": np.zeros(16000, dtype=np.float32), "text": text} for text in ("abc", "a")]
    return Collator(processor, decoder_start_token_id=START)(rows)


def test_padding_is_ignored_by_the_loss():
    labels = collate(with_start=False)["labels"]
    assert labels.tolist() == [[97, 98, 99], [97, -100, -100]]


def test_start_token_is_dropped_because_the_model_adds_it():
    labels = collate(with_start=True)["labels"]
    assert labels.tolist() == [[97, 98, 99], [97, -100, -100]]


def test_features_and_mask_are_passed_through():
    batch = collate(with_start=False)
    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["attention_mask"].shape == (2, 3000)
