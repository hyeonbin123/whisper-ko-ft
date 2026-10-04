"""A tiny random Qwen3-ASR for tests: the real classes and module names, no weights to download."""

import torch
from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

AUDIO = 100  # the tiny model's audio placeholder token
PAD, END = 101, 102


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


def tiny_model(dtype=torch.float32) -> Qwen3ASRForConditionalGeneration:
    torch.manual_seed(0)
    return Qwen3ASRForConditionalGeneration(tiny_config()).to(dtype)
