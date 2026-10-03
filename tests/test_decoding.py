"""Stage 8: the temperature fallback of transformers' Whisper generate, and how the harness records it."""

import pytest
import torch
from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration
from transformers.generation.utils import GenerateEncoderDecoderOutput

from whisper_ko_ft import evaluate

TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
EOS = 89


class Recorder:
    """Stands in for the model: keeps the arguments of generate."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[5, 6], [7, EOS]])


class Ids:
    """Stands in for the processor: token ids as text, end-of-text dropped."""

    def batch_decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return [" ".join(str(t) for t in row.tolist() if t != EOS) for row in ids]


def test_greedy_decoding_passes_the_recorded_arguments():
    # Stages 1 to 7 ("디코딩" in docs/experiments.md): without --fallback a report is measured the same way.
    model = Recorder()
    settings = evaluate.decoding_settings(None)
    texts, decisions = evaluate.transcribe(model, Ids(), "features", "mask", "ko", settings)
    assert model.kwargs == {
        "input_features": "features",
        "attention_mask": "mask",
        "language": "ko",
        "task": "transcribe",
        "num_beams": 1,
        "max_new_tokens": 256,
        "return_timestamps": False,
    }
    assert texts == ["5 6", "7"]
    assert decisions is None


def test_fallback_settings_are_the_registered_candidates():
    assert evaluate.decoding_settings("ratio") == {
        "temperature": TEMPERATURES,
        "compression_ratio_threshold": 1.35,
    }
    assert evaluate.decoding_settings("ratio-logprob") == {
        "temperature": TEMPERATURES,
        "compression_ratio_threshold": 1.35,
        "logprob_threshold": -1.0,
    }
    lowered = evaluate.decoding_settings("ratio", compression_ratio_threshold=1.0)  # smoke runs
    assert lowered["compression_ratio_threshold"] == 1.0
    with pytest.raises(ValueError):
        evaluate.decoding_settings("ratio", logprob_threshold=-0.5)  # D1a has no log-probability threshold
    with pytest.raises(ValueError):
        evaluate.decoding_settings(None, compression_ratio_threshold=1.0)


def test_sequences_from_a_tensor_or_a_generate_output():
    ids = torch.tensor([[1, 2]])
    assert evaluate.sequences(ids) is ids
    assert evaluate.sequences(GenerateEncoderDecoderOutput(sequences=ids)) is ids
    assert evaluate.sequences({"sequences": ids}) is ids


def test_final_temperatures_follow_the_order_of_the_fallback_calls():
    calls = [
        (0.0, False),
        (0.0, True),
        (0.0, False),
        (0.0, True),  # first pass: utterances 1 and 3 are decoded again
        (0.2, True),
        (0.2, False),  # utterance 1 once more
        (0.4, False),
    ]
    assert evaluate.final_temperatures(calls, 4, TEMPERATURES) == [
        (0.0, False),
        (0.4, False),
        (0.0, False),
        (0.2, False),
    ]


def test_final_temperatures_mark_utterances_still_failing_at_the_last_temperature():
    calls = [(0.0, True), (0.0, False), (1.0, True)]
    assert evaluate.final_temperatures(calls, 2, (0.0, 1.0)) == [(1.0, True), (0.0, False)]


@pytest.mark.parametrize(
    "calls",
    [
        [(0.0, False)],  # fewer calls than utterances
        [(0.0, True), (0.0, False), (0.2, False), (0.2, False)],  # more calls than utterances left
        [(0.0, True), (0.0, False)],  # decoded again at 0.0 but never at a higher temperature
        [(0.0, True), (0.2, False)],  # two temperatures inside one pass
    ],
)
def test_final_temperatures_refuse_calls_they_cannot_place(calls):
    with pytest.raises(RuntimeError):
        evaluate.final_temperatures(calls, 2, TEMPERATURES)


def tiny_whisper() -> WhisperForConditionalGeneration:
    """A randomly initialized Whisper small enough to decode 256 tokens on a CPU in a moment."""
    torch.manual_seed(0)
    config = WhisperConfig(
        vocab_size=100,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=1500,
        max_target_positions=300,
        pad_token_id=EOS,
        bos_token_id=EOS,
        eos_token_id=EOS,
        decoder_start_token_id=90,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    model = WhisperForConditionalGeneration(config).eval()
    model.generation_config = GenerationConfig(
        decoder_start_token_id=90,
        eos_token_id=EOS,
        pad_token_id=EOS,
        bos_token_id=EOS,
        lang_to_id={"<|ko|>": 91},
        task_to_id={"transcribe": 92},
        no_timestamps_token_id=93,
        is_multilingual=True,
    )
    return model


def inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.randn(batch, 8, 3000, generator=torch.Generator().manual_seed(1))
    return features, torch.ones(batch, 3000, dtype=torch.long)


def test_the_trace_follows_transformers_on_a_tiny_model():
    model = tiny_whisper()
    features, mask = inputs(4)
    greedy, _ = evaluate.transcribe(model, Ids(), features, mask, "ko", evaluate.decoding_settings(None))

    def decide(
        seek_sequence, seek_outputs, index, logits_processor, generation_config, vocab_size, temperature
    ):
        # utterances 1 and 3 fail the first pass, everything passes after that
        return temperature == 0.0 and index in (1, 3), False

    model._need_fallback = decide
    settings = evaluate.decoding_settings("ratio")
    trace = evaluate.FallbackTrace(model, settings)
    texts, decisions = evaluate.transcribe(model, Ids(), features, mask, "ko", settings, trace)

    assert decisions == [(0.0, False), (0.2, False), (0.0, False), (0.2, False)]
    assert (texts[0], texts[2]) == (greedy[0], greedy[2])  # the first pass is the greedy decoding


@pytest.mark.parametrize("fallback", ["ratio", "ratio-logprob"])
def test_the_real_thresholds_on_a_tiny_model(fallback):
    model = tiny_whisper()
    features, mask = inputs(3)
    logprob = fallback == "ratio-logprob"

    # Thresholds no decoding can meet: every utterance is decoded at every temperature, the last one is kept.
    never = evaluate.decoding_settings(
        fallback, compression_ratio_threshold=0.0, logprob_threshold=0.0 if logprob else None
    )
    texts, decisions = evaluate.transcribe(
        model, Ids(), features, mask, "ko", never, evaluate.FallbackTrace(model, never)
    )
    assert decisions == [(1.0, True)] * 3
    assert len(texts) == 3

    # Thresholds every decoding meets: nothing is decoded again and the result is the greedy one.
    always = evaluate.decoding_settings(
        fallback, compression_ratio_threshold=1e9, logprob_threshold=-1e9 if logprob else None
    )
    trace = evaluate.FallbackTrace(model, always)
    texts, decisions = evaluate.transcribe(model, Ids(), features, mask, "ko", always, trace)
    greedy, _ = evaluate.transcribe(model, Ids(), features, mask, "ko", evaluate.decoding_settings(None))
    assert decisions == [(0.0, False)] * 3
    assert texts == greedy


def test_counts_for_the_summary():
    rows = [
        {"hypothesis": "가나", "edits": 0, "length": 2, "temperature": 0.0, "fallback_exhausted": False},
        {"hypothesis": "라니" * 30, "edits": 70, "length": 5, "temperature": 1.0, "fallback_exhausted": True},
        {"hypothesis": "", "edits": 3, "length": 3, "temperature": 0.4, "fallback_exhausted": False},
        {"hypothesis": "", "edits": 0, "length": 0},  # empty reference: not scored, not counted
    ]
    assert evaluate.decoding_counts(rows, fallback=True) == {
        "redecoded": 2,
        "fallback_exhausted": 1,
        "empty_hypotheses": 1,
        "loops": 1,
    }
    greedy = [
        {k: v for k, v in row.items() if k not in ("temperature", "fallback_exhausted")} for row in rows
    ]
    assert evaluate.decoding_counts(greedy, fallback=False) == {
        "redecoded": None,
        "fallback_exhausted": None,
        "empty_hypotheses": 1,
        "loops": 1,
    }
