"""Stage 8: the temperature fallback of transformers' Whisper generate, and how the harness records it."""

# [테스트 묶음] 8단계의 재시도 디코딩을 하네스가 어떻게 넘기고 기록하는지 확인한다: 재시도 없는 greedy 인자가
# 1~7단계와 같은지, 등록한 후보(D1a, D1)의 설정, generate 출력 모양, 재시도 결정 기록(final_temperatures,
# FallbackTrace), 요약 개수.
# [왜 중요한가] transformers는 어떤 발화를 다시 디코딩했는지 돌려주지 않아 비공개 메서드를 감싸 기록한다. 이
# 기록이 틀리면 8단계 판정('되풀이 발화가 다시 디코딩됐는가')이 틀린다. 또 --fallback을 빼면 앞 단계와 똑같이
# 재야 기존 리포트와 비교할 수 있다.
# [방법] 앞부분은 가짜 모델·처리기로 인자만 보고, 뒷부분은 아주 작은 무작위 Whisper를 CPU에서 실제
# transformers generate로 돌려 재시도 경로를 확인한다(docs/experiments.md 8단계 '재기 전에 아는 것').

import pytest
import torch
from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration
from transformers.generation.utils import GenerateEncoderDecoderOutput

from whisper_ko_ft import evaluate

# 등록한 temperature 순서(evaluate.TEMPERATURES와 같아야 한다). EOS: 작은 모델의 끝 토큰 번호.
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
EOS = 89


# 모델 대역: generate에 넘어온 인자를 그대로 저장하고 고정된 토큰을 돌려준다.
class Recorder:
    """Stands in for the model: keeps the arguments of generate."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[5, 6], [7, EOS]])


# 처리기 대역: 토큰 번호를 공백으로 이은 글로 바꾸고 끝 토큰은 뺀다(skip_special_tokens가 켜져 있어야 한다).
class Ids:
    """Stands in for the processor: token ids as text, end-of-text dropped."""

    def batch_decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return [" ".join(str(t) for t in row.tolist() if t != EOS) for row in ids]


# --fallback 없이 transcribe가 generate에 넘기는 인자가 정확히 공통 규칙의 greedy 설정뿐인지(temperature 같은
# 추가 인자가 없다). 재시도 기능을 넣은 뒤에도 1~7단계 리포트와 같은 조건으로 잰다는 보장이다.
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


# 후보 설정이 docs/experiments.md 8단계에 적은 값과 같은지(1.35, −1.0, temperature 6단계). 스모크용 기준값
# 바꾸기는 되지만, D1a에 로그확률 기준을 주거나 재시도 없이 기준값만 주면 거부한다.
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


# generate가 텐서를 돌려주든 출력 객체(또는 dict)를 돌려주든 같은 토큰 텐서를 꺼낸다.
def test_sequences_from_a_tensor_or_a_generate_output():
    ids = torch.tensor([[1, 2]])
    assert evaluate.sequences(ids) is ids
    assert evaluate.sequences(GenerateEncoderDecoderOutput(sequences=ids)) is ids
    assert evaluate.sequences({"sequences": ids}) is ids


# 결정 순서 해석: 첫 temperature에서 네 발화 중 1·3번이 기준을 넘고, 0.2에서는 1번만 다시 넘고, 0.4에서 통과.
# 결과는 발화별 마지막 temperature(0, 0.4, 0, 0.2)다.
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


# 마지막 temperature에서도 기준을 넘은 발화는 fallback_exhausted=True로 표시된다(transformers는 그 마지막
# 결과를 쓴다).
def test_final_temperatures_mark_utterances_still_failing_at_the_last_temperature():
    calls = [(0.0, True), (0.0, False), (1.0, True)]
    assert evaluate.final_temperatures(calls, 2, (0.0, 1.0)) == [(1.0, True), (0.0, False)]


# 기록이 순서 가정과 맞지 않는 네 경우(결정 수가 모자람, 남은 발화보다 많음, 다시 디코딩한다면서 더 높은
# temperature가 없음, 한 단계 안에 temperature가 섞임)는 RuntimeError로 거부해야 한다. 틀린 기록을 조용히
# 리포트에 남기지 않게 한다.
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


# 무작위로 초기화한 아주 작은 Whisper(어휘 100, 층 1개, 멜 8개). 실제 generate 경로를 CPU에서 순식간에
# 돌리려고 쓴다. 언어(ko)·task 토큰 번호를 generation_config에 넣어 언어 고정 디코딩이 되게 한다. seed 0이라
# 항상 같은 모델이다.
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


# 고정 시드의 무작위 특징(멜 8 × 3,000프레임)과 전부 1인 attention mask.
def inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.randn(batch, 8, 3000, generator=torch.Generator().manual_seed(1))
    return features, torch.ones(batch, 3000, dtype=torch.long)


# 실제 transformers generate에서 FallbackTrace가 결정을 바르게 기록하는지. _need_fallback을 '첫 단계에서
# 1·3번만 실패'로 바꿔 두면 기록은 1·3번만 0.2로 나와야 하고, 다시 디코딩하지 않은 0·2번의 가설은 greedy
# 결과와 같아야 한다(짧은 입력에서도 배치의 일부만 다시 디코딩된다는 확인). 대역 함수의 인자 순서는
# transformers 5.17의 _need_fallback과 같다.
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


# 진짜 기준 판단으로 두 극단을 본다. 어떤 디코딩도 통과할 수 없는 기준이면 모든 발화가 마지막
# temperature(1.0)까지 가서 exhausted가 되고, 모든 디코딩이 통과하는 기준이면 아무것도 다시 디코딩하지 않고
# greedy와 같은 가설이 나온다. 로그확률 기준(D1)처럼 점수를 돌려받는 경로도 함께 돈다.
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


# 요약 개수: 다시 디코딩(temperature > 0) 2개, 끝까지 기준을 넘음 1개, 빈 가설 1개, 되풀이(편집 수 > 정답
# 길이) 1개. 정답이 빈 행은 세지 않는다. 재시도를 쓰지 않은 리포트는 재시도 관련 개수가 None이다.
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
