"""테스트용 작은 무작위 Qwen3-ASR: 실제 클래스와 모듈 이름을 쓰고, 내려받을 가중치는 없다."""

# [파일 역할] 테스트용 작은 무작위 Qwen3-ASR. transformers의 실제 클래스와 같은 모듈 이름(audio_tower,
# language_model, lm_head 등)을 가지되 층과 차원을 아주 작게 해, 4GB 가중치를 내려받지 않고 CPU에서 순식간에
# 돈다.
# [왜 필요한가] train_qwen의 LoRA 정규식, 손실 계산, 학습 루프와 evaluate_qwen의 디코딩·넘침 감시·어댑터
# 병합을 실제 모델과 같은 코드 경로로 시험하려는 것이다.
# [연결] tests 폴더에 __init__.py가 없어 pytest가 이 폴더를 import 경로에 넣으므로, test_evaluate_qwen.py와
# test_train_qwen.py가 'from tiny_qwen import …'로 가져온다.

import torch
from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

# 작은 모델의 특수 토큰 번호: 오디오 자리 표시(AUDIO), 패딩(PAD), 끝(END). 끝 토큰 목록에 PAD와 END를 함께
# 둔다.
AUDIO = 100  # 작은 모델의 오디오 자리 표시 토큰
PAD, END = 101, 102


# 오디오 인코더 2층, 언어 모델 2층, 어휘 120개의 설정. 멜 수는 실제 모델과 같은 128이라 실제 처리기의 특징
# 모양을 그대로 받을 수 있다.
def tiny_config() -> Qwen3ASRConfig:
    return Qwen3ASRConfig(
        audio_config={
            "d_model": 16,
            "encoder_layers": 2,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "num_mel_bins": 128,
            "output_dim": 32,
            "downsample_hidden_size": 8,
        },
        text_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 64,
            "vocab_size": 120,
        },
        audio_token_id=AUDIO,
        pad_token_id=PAD,
        eos_token_id=[PAD, END],
    )


# seed 0으로 초기화한 작은 모델을 원하는 dtype으로 만든다. 같은 시드라 부를 때마다 같은 가중치가 나온다.
def tiny_model(dtype=torch.float32) -> Qwen3ASRForConditionalGeneration:
    torch.manual_seed(0)
    return Qwen3ASRForConditionalGeneration(tiny_config()).to(dtype)
