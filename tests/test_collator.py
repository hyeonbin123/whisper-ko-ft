# [테스트 묶음] train.Collator가 Whisper 학습 입력(특징, attention mask, 라벨)을 바르게 만드는지 확인한다.
# 진짜 처리기 대신 가짜 특징 추출기·토크나이저를 써서 모델을 내려받지 않고 CPU에서 바로 돈다.
# [왜 중요한가] 라벨이 틀리면(패딩이 손실에 들어가거나 시작 토큰이 두 번 들어가면) 학습은 오류 없이 돌면서
# 결과만 나빠진다. 이런 조용한 버그를 막는다.

from types import SimpleNamespace

import numpy as np
import torch

from whisper_ko_ft.train import Collator

# START: 가짜 디코더 시작 토큰. PAD: 가짜 패딩 토큰.
START, PAD = 1, 0


# 가짜 특징 추출기: 멜 80 × 3,000프레임(30초) 크기의 0 특징과 마스크를 돌려준다.
class FakeFeatureExtractor:
    def __call__(self, audio, **kwargs):
        batch = len(audio)
        return SimpleNamespace(
            input_features=torch.zeros(batch, 80, 3000), attention_mask=torch.ones(batch, 3000)
        )


# 가짜 토크나이저: 글자마다 유니코드 번호를 토큰으로 쓰고 오른쪽을 PAD로 채운다. with_start면 맨 앞에 시작
# 토큰을 붙여 Whisper 토크나이저의 접두 토큰을 흉내 낸다.
class FakeTokenizer:
    def __init__(self, with_start: bool) -> None:
        self.with_start = with_start

    def __call__(self, texts, **kwargs):
        rows = [([START] if self.with_start else []) + [ord(c) for c in text] for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.tensor([row + [PAD] * (width - len(row)) for row in rows])
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows])
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


# 정답 'abc'와 'a' 두 행으로 콜레이터를 한 번 돌린다.
def collate(with_start: bool) -> dict:
    processor = SimpleNamespace(feature_extractor=FakeFeatureExtractor(), tokenizer=FakeTokenizer(with_start))
    rows = [{"audio": np.zeros(16000, dtype=np.float32), "text": text} for text in ("abc", "a")]
    return Collator(processor, decoder_start_token_id=START)(rows)


# 패딩 위치는 -100이 되어 손실에서 빠져야 한다.
def test_padding_is_ignored_by_the_loss():
    labels = collate(with_start=False)["labels"]
    assert labels.tolist() == [[97, 98, 99], [97, -100, -100]]


# 토크나이저가 시작 토큰을 붙여도 라벨에서는 떼어 내야 한다. 모델이 라벨을 밀어 디코더 입력을 만들 때 시작
# 토큰을 스스로 붙이므로, 남기면 두 번 들어간다.
def test_start_token_is_dropped_because_the_model_adds_it():
    labels = collate(with_start=True)["labels"]
    assert labels.tolist() == [[97, 98, 99], [97, -100, -100]]


# 특징과 attention mask가 모양 그대로 모델 입력으로 넘어가는지.
def test_features_and_mask_are_passed_through():
    batch = collate(with_start=False)
    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["attention_mask"].shape == (2, 3000)
