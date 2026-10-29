"""Qwen3-ASR로 묶음을 받아 적고(9단계), 한 번의 디코딩에서 raw와 fixed 두 리포트를 쓴다.

사용법:
    uv run python -m whisper_ko_ft.evaluate_qwen --set zeroth-val
    uv run python -m whisper_ko_ft.evaluate_qwen --model Qwen/Qwen3-ASR-0.6B-hf --set zeroth-val
    uv run python -m whisper_ko_ft.evaluate_qwen --set zeroth-val500 --limit 100 --out <파일>  (fp16 점검)
    uv run python -m whisper_ko_ft.evaluate_qwen --adapter outputs/qwen-qn/checkpoint-1000 --name <이름>
        --set zeroth-val [--channel telephone]  (10단계)

디코딩 설정은 docs/experiments.md(9단계)에 고정되어 있다: 고정한 revision의 모델 전용 Transformers
클래스(transformers >= 5.13), fp16(또는 fp32), SDPA 어텐션, greedy, 공식 프롬프트로 언어
고정(`apply_transcription_request`가 프롬프트를 "language Korean<asr_text>"로 끝낸다), 새 토큰 최대 256개,
배치 16, 그리고 Whisper 하네스가 듣는 것과 같은 오디오 앞 30초.

한 번의 디코딩에서 출력 두 개가 나온다. "raw"는 <asr_text> 표시 뒤에 생성된 글에 아무것도 더 하지 않은
것이고 reports/<이름>/에 간다. "fixed"는 공식 출력 `processor.extract_transcription`이며, 원래 구현의 반복
제거(`_detect_and_fix_repetitions`)까지 거친다. reports/<이름>-fixed/에 간다. Whisper 하네스에는 그런 반복
제거가 없으므로 판정은 raw로 한다.

발화 행에는 생성한 토큰 수, 디코딩이 토큰 한도에 닿았는지, 그 발화의 로짓 중 유한하지 않은 값(fp16 넘침)이
있었는지도 남긴다. CPU 실행은 결코 측정이 아니다: --out(fp16 점검의 fp32 쪽)이나 --no-report가 있어야 한다.

10단계: --adapter는 `evaluate`가 Whisper에 하듯 디코딩 전에 LoRA 어댑터(`train_qwen`)를 fp16 기본 가중치에
합친다. --name이 필요하고, 리포트에 어댑터 폴더와 가중치의 SHA-256을 적는다. --out은 zeroth-val500에서
체크포인트를 재는 실행에도 쓴다. --channel telephone은 앞 30초를 자르기 전에 오디오를 4단계의 전화 채널에
통과시키고, 리포트는 <묶음>@telephone.json으로 간다.

11단계(2026-10-15 규칙): --adapter-scale s는 합치기 전에 LoRA 갱신량을 s배로 줄이거나 키운다(W0 + s·ΔW,
학습 전 가중치와 학습한 가중치의 선형 보간). --adapter-scope는 s를 쓸 tower(all, language, audio)를 고르고
나머지는 1.0으로 둔다. 리포트 요약에 tower별 스케일과 adapter_config.json의 SHA-256이 남는다.
"""

# [파일 역할] 9·10단계의 측정 하네스. 계열이 다른 공개 모델 Qwen3-ASR(오디오 인코더 + Qwen3 언어 모델)로
# 묶음을 받아 적고, evaluate.py와 같은 모양의 리포트를 남긴다. 9단계는 학습 없이(1.7B와 참고용 0.6B), 10단계는
# train_qwen으로 학습한 LoRA 어댑터를 합쳐서 잰다.
# [설계 이유] Whisper 하네스와 같은 조건이 되도록 맞췄다: 같은 묶음과 채점(score_rows), 최대 256토큰, greedy,
# 언어 고정, 앞 30초, 배치 16, 같은 덮어쓰기·test 보호와 커밋 기록. 모델 구조와 입력 방식이 달라 파일을
# 나눴다.
# [raw와 fixed] 공식 파서는 반복 제거 후처리를 자동으로 적용하는데 Whisper 하네스에는 그런 장치가 없다. 그래서
# 한 번의 디코딩에서 후처리 없는 raw와 공식 결과 fixed를 함께 남기고, 판정은 raw로 한다(위 docstring,
# docs/experiments.md 9단계).
# [fp16 점검] RTX 2080 Ti에는 bf16이 없어 bf16으로 배포된 모델을 fp16으로 돌린다. 넘침을 잡으려고 발화마다
# 로짓이 유한한지 기록하고, 앞 100발화를 GPU fp16과 CPU fp32로 각각 디코딩해 analyze gate9로 비교했다. fp32
# 가중치는 약 8.2GB라 다른 프로그램과 나눠 쓰는 11GB GPU 대신 CPU에서 쟀다(docs/experiments.md 9단계 'fp16
# 점검').
# [연결] evaluate.py의 상수와 도우미, provenance의 커밋 기록·어댑터 지문을 가져다 쓰고, train_qwen이
# DEFAULT_MODEL·REGISTERED·LANGUAGES·first_window를 가져가 학습 입력을 평가와 같게 만든다.

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

# numpy: 오디오 배열과 채점 배열. torch: 추론, dtype, 넘침 검사. transformers: 버전 기록(모델 클래스는
# load_model에서 불러온다).
import numpy as np
import torch
import transformers

# channel: 10단계의 --channel telephone. evaluate: Whisper 하네스와 같은 최대 토큰 수·30초 창·채점·배치 지연
# 요약·리포트 글(report_json, NaN이면 ValueError)을 그대로 써서 두 하네스의 조건을 맞춘다. provenance: 측정
# 코드의 커밋 기록과 어댑터 지문(adapter_info. 2026-10-29에 이 파일에서 옮겨 간 함수로, Whisper 하네스와 함께
# 쓴다). metrics, paths, store: 채점, 리포트 위치, 묶음과 오디오.
from whisper_ko_ft.channel import CHANNELS
from whisper_ko_ft.evaluate import (
    MAX_NEW_TOKENS,
    WINDOW_SECONDS,
    batch_seconds_summary,
    report_json,
    score_rows,
)
from whisper_ko_ft.metrics import error_rate, interval
from whisper_ko_ft.paths import REPORTS
from whisper_ko_ft.provenance import adapter_info, git_commit
from whisper_ko_ft.store import SAMPLE_RATE, AudioStore, load_set

# DEFAULT_MODEL: 9·10단계 후보 Qwen3-ASR-1.7B의 Transformers 판(-hf).
# REGISTERED: 모델 → (고정한 revision, 리포트 이름). revision을 고정해 Hub의 모델이 바뀌어도 같은 가중치로
# 다시 잴 수 있게 한다(docs/experiments.md 9단계 '재기 전에 아는 것'). revision은 2026-10-04에 내려받은
# 것이다.
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
REGISTERED = {
    "Qwen/Qwen3-ASR-1.7B-hf": ("bcd2b5b7f32b480ab5790554cfa8347f246a14f3", "qwen3-asr-1.7b"),
    "Qwen/Qwen3-ASR-0.6B-hf": ("7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c", "qwen3-asr-0.6b"),
}
# LANGUAGES: 묶음의 언어 코드 → 공식 프롬프트의 언어 이름. 프롬프트 끝에 'language Korean<asr_text>'를 미리
# 채워 언어를 고정한다(Whisper의 언어 고정에 해당).
# DTYPES: --precision 값 → torch dtype. 측정은 fp16, fp32는 점검용이다.
# GENERATE: greedy(표본 없음, 빔 1), 최대 256토큰. 모델 기본값(512)이 아니라 Whisper 하네스와 같은 값이다.
# MARKER: 받아 적은 글이 시작됨을 알리는 공식 표시. OUTPUTS: 한 디코딩에서 만드는 두 출력.
# WINDOW_SAMPLES: 앞 30초에 해당하는 표본 수.
LANGUAGES = {"ko": "Korean", "en": "English"}
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
GENERATE = {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS}
MARKER = "<asr_text>"
OUTPUTS = ("raw", "fixed")
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE


# 공식 파서의 출력 정리에서 반복 제거만 뺀 것이다. 앞뒤 공백을 지우고, 대화 템플릿의 'assistant' 줄과
# <asr_text> 표시가 있으면 그 뒤만 남긴다. 언어를 고정하면 생성된 부분에 표시가 없어 글 전체가
# 된다(docs/experiments.md 9단계 '디코딩').
# 반복을 지우지 않고 <non_speech> 같은 태그도 남긴다(그 수는 요약에 센다). 호출 위치: transcribe.
def raw_transcription(text: str) -> str:
    """디코딩 출력의 공식 파싱(`_parse_single_output`)에서 반복 제거만 뺀 것."""
    text = text.strip()
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[-1]
    if MARKER in text:
        text = text.split(MARKER, 1)[1]
    return text.strip()


# 앞 30초만 잘라 돌려준다. Qwen3-ASR은 더 긴 입력도 받지만 Whisper 하네스가 앞 30초만 들으므로 같은 소리를
# 듣게 한다. 30초 이하면 복사 없이 그대로 돌려준다.
# 호출 위치: prepare_audio, train_qwen의 콜레이터(학습 입력도 같은 창).
def first_window(audio: np.ndarray) -> np.ndarray:
    """앞 30초. 더 긴 발화에서 Whisper 하네스가 듣는 부분이다."""
    return audio[:WINDOW_SAMPLES] if audio.shape[0] > WINDOW_SAMPLES else audio


def prepare_audio(audio: np.ndarray, channel: str | None) -> np.ndarray:
    """채널 변환(있으면)을 발화 전체에 먼저 적용한 뒤 앞 30초를 자른다.

    Whisper 하네스도 변환 뒤에 특징 추출기가 30초로 자르므로 순서가 같다.
    """
    return first_window(CHANNELS[channel](audio) if channel else audio)


# 발화마다 (첫 끝 토큰 앞까지의 생성 토큰 수, 끝 토큰 없이 한도까지 갔는지)를 돌려준다. 한도에 닿은 발화는
# 되풀이에 빠졌을 가능성이 있어 요약에 센다(hit_token_limit).
# eos_token_id가 하나일 수도 여러 개일 수도 있어 목록으로 맞춘 뒤 torch.isin으로 찾는다.
def generated_lengths(generated: torch.Tensor, eos_token_id) -> list[tuple[int, bool]]:
    """행마다 (첫 끝 토큰 앞까지의 토큰 수, 끝 토큰 대신 토큰 한도에 닿았는지)."""
    eos = torch.tensor(eos_token_id if isinstance(eos_token_id, list | tuple) else [eos_token_id])
    ended = torch.isin(generated.cpu(), eos)
    lengths = []
    for row in ended:
        hits = row.nonzero()
        lengths.append((int(hits[0]), False) if len(hits) else (int(row.shape[0]), True))
    return lengths


# fp16 넘침 감시. 언어 모델 출력층(lm_head)에 forward 훅을 걸어, 디코딩 단계마다 배치의 각 행 로짓에 유한하지
# 않은 값(inf, nan)이 있었는지 OR로 누적한다. take가 배치 하나가 끝날 때 결과를 꺼내고 비운다.
# 왜 필요한가: 넘침이 있었는지는 가설 글만 봐서는 확인할 수 없어 디코딩 중에 직접 본다. 발화마다 기록해 fp16
# 점검의 통과 조건(넘침 발화 0)과 리포트의 nonfinite_utterances에 쓴다.
class NonFiniteWatch:
    """훅으로 배치에서 로짓이 한 번이라도 유한하지 않았던(fp16 넘침) 행을 표시한다."""

    def __init__(self, head: torch.nn.Module) -> None:
        self.rows: torch.Tensor | None = None
        head.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        bad = ~torch.isfinite(output).flatten(1).all(dim=1)
        self.rows = bad if self.rows is None else self.rows | bad

    def take(self, batch_size: int) -> list[bool]:
        rows, self.rows = self.rows, None
        return [False] * batch_size if rows is None else [bool(flag) for flag in rows.tolist()]


# 배치 하나를 디코딩해 발화마다 raw·fixed 출력, 생성 토큰 수, 한도 도달, 넘침 여부를 돌려준다. 호출 위치:
# main, tests/test_evaluate_qwen.py(작은 무작위 Qwen3-ASR).
def transcribe(
    model, processor, audio: list[np.ndarray], language: str, device: str, dtype, watch=None
) -> list[dict]:
    """배치 하나의 두 출력과, 발화마다 토큰 수와 넘침 표시."""
    # 공식 입력 함수로 프롬프트(언어 고정)와 오디오 특징을 만든다. 맥락 프롬프트(system)는 비운다. 멜 특징은
    # 처리기 기본값대로 CPU에서 계산하고 속도에 포함된다(docs/experiments.md 9단계). 그 뒤 모델과 같은
    # 장치·dtype으로 옮긴다.
    inputs = processor.apply_transcription_request(audio, language=LANGUAGES[language]).to(device, dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, **GENERATE)
    # generate 결과에는 프롬프트가 앞에 붙어 있다. 왼쪽 패딩(공식 기본값)이라 모든 행의 프롬프트가 같은
    # 위치에서 끝나므로, 프롬프트 길이 뒤만 잘라 생성된 부분을 얻는다.
    generated = output[:, inputs["input_ids"].shape[1] :]
    texts = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
    # 같은 글을 공식 파서에도 넘겨 fixed 출력(반복 제거 포함)을 만든다. 디코딩은 한 번뿐이다.
    fixed = processor.extract_transcription(texts)
    lengths = generated_lengths(generated, model.generation_config.eos_token_id)
    nonfinite = watch.take(len(texts)) if watch else [False] * len(texts)
    return [
        {
            "raw": raw_transcription(text),
            "fixed": fix,
            "new_tokens": n,
            "hit_token_limit": hit,
            "nonfinite_logits": bad,
        }
        for text, fix, (n, hit), bad in zip(texts, fixed, lengths, nonfinite, strict=True)
    ]


# 같은 발화 행에서 raw와 fixed 리포트 두 개를 만든다. 각 리포트는 자기 가설로 따로 채점한다(score_rows).
# 요약에 더하는 것: 되풀이 발화(편집 수 > 정답 길이), 256토큰 도달, 넘침 발화, 남은 <non_speech> 태그, fixed가
# raw와 다른 발화 수(fixed_changed, 두 리포트에 같은 값). 9단계 결과의 '공식 반복 제거가 바꾼 발화 0'이 이
# 값이다.
def report_payloads(rows: list[dict], language: str, meta: dict) -> dict[str, dict]:
    """출력마다 리포트 하나: 발화는 같고, 각 리포트는 자기 가설로 채점한다."""
    changed = sum(row["raw"] != row["fixed"] for row in rows)
    payloads = {}
    for output in OUTPUTS:
        scored = []
        for row in rows:
            out = {key: row[key] for key in ("id", "speaker", "seconds", "reference")}
            out["hypothesis"] = row[output]
            out.update({key: row[key] for key in ("new_tokens", "hit_token_limit", "nonfinite_logits")})
            scored.append(out)
        skipped = score_rows(scored, language)
        kept = [row for row in scored if row["length"]]
        edits = np.array([row["edits"] for row in kept])
        lengths = np.array([row["length"] for row in kept])
        low, high = interval(edits, lengths)
        summary = {
            **meta,
            "output": output,
            "metric": "cer" if language == "ko" else "wer",
            "utterances": len(kept),
            "skipped_empty_reference": skipped,
            "error_rate": round(error_rate(edits, lengths), 5),
            "interval_95": [round(low, 5), round(high, 5)],
            "hypotheses_with_digit": sum(row.get("has_digit", False) for row in kept),
            "empty_hypotheses": sum(not row["hypothesis"].strip() for row in kept),
            "loops": sum(row["edits"] > row["length"] for row in kept),
            "hit_token_limit": sum(row["hit_token_limit"] for row in kept),
            "nonfinite_utterances": sum(row["nonfinite_logits"] for row in kept),
            "non_speech_tags": sum("<non_speech>" in row["hypothesis"] for row in kept),
            "fixed_changed": changed,
        }
        payloads[output] = {"summary": summary, "utterances": scored}
    return payloads


# 11단계 --adapter-scope: 스케일을 쓸 tower를 LoRA 층 이름의 일부로 고른다. 오디오 인코더 층은 이름에
# '.audio_tower.', 언어 모델 층은 '.language_model.'이 들어 있다(train_qwen.LORA_TARGETS의 두 갈래).
SCOPES = {
    "all": (".audio_tower.", ".language_model."),
    "language": (".language_model.",),
    "audio": (".audio_tower.",),
}


# 11단계: 고른 tower의 LoRA 층마다 PEFT의 set_scale로 scaling을 scale·alpha/r로 바꾼다. 합치기
# (merge_and_unload)는 이 scaling으로 갱신량 B·A·scaling을 계산해 더하므로, 합친 가중치가 W0 + scale·ΔW가
# 된다(PEFT 0.21.0 LoraLayer.set_scale, get_delta_weight). 돌려주는 것은 스케일을 바꾼 층 수다. 한 층도
# 맞지 않으면 조용히 넘어가지 않고 멈춘다(층 이름 규칙이 바뀐 경우).
def scale_adapter(peft_model, scale: float, scope: str) -> int:
    """고른 tower의 LoRA 층 scaling을 처음 값(alpha/r)의 scale배로 바꾸고, 바꾼 층 수를 돌려준다."""
    from peft.tuners.lora import LoraLayer

    scaled = 0
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer) and any(part in name for part in SCOPES[scope]):
            for adapter_name in list(module.scaling):
                module.set_scale(adapter_name, scale)
            scaled += 1
    if not scaled:
        raise ValueError(f"no LoRA layer matched the {scope!r} scope")
    return scaled


# 10단계: train_qwen이 저장한 LoRA 어댑터를 기본 가중치에 합친다. 기본 모델의 dtype(fp16) 그대로 합치므로,
# Whisper의 N을 evaluate --adapter로 잴 때와 같은 방식이다(docs/experiments.md 10단계 '디코딩과 리포트').
# 11단계: scale을 주면(None이 아니면) 합치기 전에 scale_adapter로 고른 tower의 갱신량을 scale배로 한다.
# None이면 10단계와 똑같은 경로다. scale 1.0을 직접 주면 scaling을 같은 값(alpha/r)으로 다시 쓰므로 합친
# 가중치가 같아야 한다(11단계 관문 (a)가 이 경로를 확인한다).
# peft는 어댑터가 있을 때만 불러온다.
def merge_adapter(model, adapter: str, scale: float | None = None, scope: str = "all"):
    """LoRA 어댑터를 가중치에 합친 기본 모델(기본 모델의 dtype 그대로). scale이 있으면 갱신량을 scale배로."""
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(model, adapter)
    scaled = 0
    if scale is not None:
        scaled = scale_adapter(peft_model, scale, scope)
        print(f"adapter scale {scale} on {scaled} LoRA layers ({scope})", flush=True)  # 실행 로그에 남긴다
    merged = peft_model.merge_and_unload()
    # 스케일을 바꾼 LoRA 층 수를 합친 모델에 붙여 둔다(main이 요약의 adapter_scaled_layers로 옮긴다).
    merged.adapter_scaled_layers = scaled
    return merged


# 리포트 요약의 어댑터 항목(어댑터 폴더, 가중치·설정의 SHA-256)은 provenance.adapter_info다. 2026-10-15부터
# 이 파일에 있던 함수를 2026-10-29에 옮겼고(Whisper 하네스 evaluate도 같은 지문을 적게 되어), 키와 값의 꼴은
# 그대로다(위 import).


# 요약에 남길 tower별 스케일(어댑터가 없으면 None). scale이 None이면 두 tower 모두 1.0(10단계와 같은
# 합치기)이고, 주면 scope에 든 tower는 scale, 나머지는 1.0이다.
def adapter_scale_info(adapter: str | None, scale: float | None, scope: str) -> dict | None:
    if adapter is None:
        return None
    towers = {"audio_tower": ".audio_tower.", "language_model": ".language_model."}
    return {
        tower: scale if scale is not None and part in SCOPES[scope] else 1.0 for tower, part in towers.items()
    }


# 고정한 revision으로 처리기와 모델을 불러와 장치에 올린다. 어텐션은 SDPA(FlashAttention 2는 쓰지 않는다,
# docs/experiments.md 9단계). 모델 클래스는 transformers 5.13부터 있어 이 함수 안에서 불러온다. --adapter면
# 합친 뒤 올린다.
def load_model(
    model_id: str,
    revision: str,
    dtype,
    device: str,
    adapter: str | None = None,
    adapter_scale: float | None = None,
    adapter_scope: str = "all",
):
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=dtype, attn_implementation="sdpa"
    )
    if adapter:
        model = merge_adapter(model, adapter, adapter_scale, adapter_scope)
    return model.to(device).eval(), processor


# 명령줄 진입점. 주요 인자: --model(기본 1.7B), --revision, --set, --name(raw 리포트 이름, fixed는
# <name>-fixed), --adapter(10단계), --adapter-scale·--adapter-scope(11단계), --channel,
# --precision(fp16/fp32), --device(cuda/cpu), --batch-size(기본 16), --limit, --allow-test, --no-report,
# --out(raw만 지정한 파일에 쓰기: fp16 점검과 체크포인트 고르기), --overwrite.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face 모델 ID(-hf 체크포인트)")
    parser.add_argument("--revision", help="기본값은 등록한 모델의 고정 revision")
    parser.add_argument("--set", required=True, dest="set_name")
    parser.add_argument("--name", help="raw 출력의 리포트 이름. fixed는 <이름>-fixed에 간다")
    parser.add_argument(
        "--adapter", help="--model에 합칠 LoRA 어댑터 폴더(10단계 train_qwen이 저장한 것. --name 필요)"
    )
    parser.add_argument(
        "--adapter-scale",
        type=float,
        help="11단계: 합치기 전에 LoRA 갱신량을 이 배수로(0이면 학습 전, 1이면 그대로). --adapter 필요",
    )
    parser.add_argument(
        "--adapter-scope",
        choices=list(SCOPES),
        default="all",
        help="--adapter-scale을 쓸 tower(all, language: 언어 모델만, audio: 오디오 인코더만). 나머지는 1.0",
    )
    parser.add_argument(
        "--channel", choices=list(CHANNELS), help="오디오를 채널 흉내(전화 음질 등)에 통과시킨다"
    )
    parser.add_argument("--precision", choices=list(DTYPES), default="fp16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="앞의 N개 발화만 잰다(fp16 점검, 스모크, 속도)")
    parser.add_argument("--allow-test", action="store_true", help="*-test 묶음을 잴 때 필요하다")
    parser.add_argument("--no-report", action="store_true", help="요약만 찍고 리포트는 쓰지 않는다")
    parser.add_argument("--out", help="raw 리포트 하나만 이 파일에 쓴다(fp16 점검)")
    parser.add_argument("--overwrite", action="store_true", help="이미 있는 리포트를 바꿔 쓴다")
    args = parser.parse_args()

    # evaluate.py와 같은 규칙 보호: test 묶음은 --allow-test, 어댑터로 잴 때는 --name 필수(기준 모델의
    # 리포트와 섞이지 않게).
    if args.set_name.endswith("-test") and not args.allow_test:
        parser.error("test sets are measured once per stage; pass --allow-test when the stage is done")
    if args.adapter and not args.name:
        parser.error("--adapter needs --name (the default is the base model's report folder)")
    # 11단계: 스케일은 어댑터를 합칠 때만 뜻이 있고, 음수 배수는 규칙에 없다.
    if args.adapter_scale is not None and not args.adapter:
        parser.error("--adapter-scale needs --adapter")
    if args.adapter_scale is not None and args.adapter_scale < 0:
        parser.error("--adapter-scale must not be negative")
    if args.adapter_scope != "all" and args.adapter_scale is None:
        parser.error("--adapter-scope needs --adapter-scale")
    # 등록한 모델이면 고정한 revision과 리포트 이름을 쓰고, 명령줄 값이 있으면 그것이 우선한다. 등록하지 않은
    # 모델은 둘 다 직접 줘야 한다.
    revision, name = REGISTERED.get(args.model, (args.revision, args.name))
    revision, name = args.revision or revision, args.name or name
    if not revision or not name:
        parser.error("a model that is not registered needs --revision and --name")
    if args.out and args.no_report:
        parser.error("--out writes a report; drop --no-report")
    # CPU 실행은 측정 조건이 아니므로 reports/에 리포트를 남기지 못하게 한다. fp16 점검의 fp32 쪽(--out)이나
    # 확인용(--no-report)만 허용한다.
    if args.device == "cpu" and not (args.out or args.no_report):
        parser.error("a CPU run is not the measured condition; pass --out (the fp16 check) or --no-report")
    # 쓸 파일 정하기: --out이면 raw 하나만 그 파일에, --no-report면 쓰지 않음, 그 밖에는 reports/<name>/와
    # reports/<name>-fixed/에 두 리포트(채널이 있으면 <묶음>@telephone.json). 이미 있으면 --overwrite 없이
    # 멈춘다.
    if args.out:
        targets = {"raw": Path(args.out)}
    elif args.no_report:
        targets = {}
    else:
        suffix = f"@{args.channel}" if args.channel else ""
        targets = {"raw": REPORTS / name / f"{args.set_name}{suffix}.json"}
        targets["fixed"] = REPORTS / f"{name}-fixed" / f"{args.set_name}{suffix}.json"
    for target in targets.values():
        if target.exists() and not args.overwrite:
            parser.error(f"{target} exists; pass --overwrite to measure it again")
    commit = git_commit()  # 측정하는 코드의 커밋을, 코드가 바뀔 수 있기 전에 기록한다

    # 묶음을 읽고, 어댑터 정보(SHA-256)를 커밋처럼 디코딩 전에 읽어 둔 뒤 모델을 올리고 출력층에 넘침 감시를
    # 건다. 발화가 없는 묶음은 모델을 올리기 전에 멈춘다(Whisper 하네스와 같은 보호).
    language, utterances = load_set(args.set_name)
    if not utterances:
        parser.error(
            f"set {args.set_name} has no utterances; rebuild the sets with `python -m whisper_ko_ft.prepare`"
        )
    if args.limit:
        utterances = utterances[: args.limit]
    dtype, cuda = DTYPES[args.precision], args.device == "cuda"
    adapter = adapter_info(args.adapter)  # 커밋처럼 디코딩 전에 읽어 둔다
    model, processor = load_model(
        args.model, revision, dtype, args.device, args.adapter, args.adapter_scale, args.adapter_scope
    )
    watch = NonFiniteWatch(model.lm_head)
    stores: dict[str, AudioStore] = {}

    rows: list[dict] = []
    batch_seconds: list[float] = []
    # GPU면 피크 메모리 통계를 비운다. 시계는 모델을 띄운 뒤부터 잰다.
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for begin in range(0, len(utterances), args.batch_size):
        batch_started = time.perf_counter()
        batch = utterances[begin : begin + args.batch_size]
        # 발화를 읽어 채널 변환과 앞 30초 자르기를 한 뒤 배치로 디코딩한다.
        audio = [
            prepare_audio(stores.setdefault(u.store, AudioStore(u.store)).read(u), args.channel)
            for u in batch
        ]
        outputs = transcribe(model, processor, audio, language, args.device, dtype, watch)
        batch_seconds.append(time.perf_counter() - batch_started)  # 글로 바꾸는 단계가 GPU 완료를 기다린다
        for utterance, out in zip(batch, outputs, strict=True):
            rows.append(
                {
                    "id": utterance.id,
                    "speaker": utterance.speaker,
                    "seconds": round(utterance.seconds, 2),
                    "reference": utterance.text,
                    **out,
                }
            )
        if (begin // args.batch_size) % 20 == 0:
            print(f"{begin + len(batch):,}/{len(utterances):,}", flush=True)
    if cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - started

    # 두 리포트가 함께 쓰는 요약 항목: 모델·revision·어댑터, 엔진, 묶음·채널·언어, 정밀도·장치·어텐션, 디코딩
    # 설정, 30초 넘는 발화 수, 속도와 배치 지연, 최대 GPU 메모리(GPU일 때), limit, 커밋, 시각,
    # torch·transformers 버전, GPU 이름.
    audio_seconds = sum(row["seconds"] for row in rows)
    meta = {
        "model": args.model,
        "revision": revision,
        **adapter,
        "adapter_scale": adapter_scale_info(args.adapter, args.adapter_scale, args.adapter_scope),
        "adapter_scaled_layers": getattr(model, "adapter_scaled_layers", None),
        "engine": "transformers qwen3_asr",
        "set": args.set_name,
        "channel": args.channel,
        "language": language,
        "precision": args.precision,
        "device": args.device,
        "attention": model.config._attn_implementation,
        "decoding": {**GENERATE, "language": LANGUAGES[language], "first_seconds": WINDOW_SECONDS},
        "over_30_seconds": sum(row["seconds"] > WINDOW_SECONDS for row in rows),
        "audio_seconds": round(audio_seconds, 1),
        "wall_seconds": round(wall, 1),
        "audio_seconds_per_second": round(audio_seconds / wall, 1),
        **batch_seconds_summary(batch_seconds),
        "batch_size": args.batch_size,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
        "limit": args.limit,
        "commit": commit,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }
    # raw와 fixed 요약을 모두 찍고, 쓰기로 한 것만 파일로 남긴다(LF, 한국어 그대로. report_json은 요약에 NaN이
    # 있으면 ValueError로 멈춰 파일을 남기지 않는다).
    payloads = report_payloads(rows, language, meta)
    for output, payload in payloads.items():
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        if output in targets:
            target = targets[output]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(report_json(payload), encoding="utf-8", newline="\n")
            print(f"wrote {target}")


if __name__ == "__main__":
    main()
