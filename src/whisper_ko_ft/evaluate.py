"""Transcribe a set with a Whisper model and write an error-rate report.

Usage:
    uv run python -m whisper_ko_ft.evaluate --model openai/whisper-small --set zeroth-val
    uv run python -m whisper_ko_ft.evaluate --model outputs/small-a/best --set fleurs-ko-val --name small-a

Decoding settings are fixed in docs/experiments.md ("디코딩"). Test sets need --allow-test, because they
are measured once per stage. The report keeps every reference and hypothesis so rates can be recomputed.
With --channel telephone the report is written as <set>@telephone.json. Reports are committed records: an
existing one is replaced only with --overwrite, and --adapter needs --name.

--fallback turns on the temperature fallback of transformers' Whisper generate (stage 8): a decoding whose
token bytes compress too well (a loop), or with a low mean log-probability ("ratio-logprob" only), is decoded
again at the next temperature. The report then records each utterance's final temperature.
"""

# [파일 역할] 이 프로젝트의 기본 측정 하네스. Whisper 모델(기준 모델, 전체 파인튜닝 체크포인트, 또는 기본 모델
# + LoRA 어댑터)로 묶음 하나를 받아 적고, 발화별 정답·가설·편집 수와 요약(오류율, 95% 구간, 속도, GPU 메모리,
# 측정 코드의 커밋)을 reports/<이름>/<묶음>.json에 남긴다.
# [설계 이유] 측정 규칙을 재기 전에 적어 두는 프로젝트라(docs/experiments.md), 디코딩 설정을 코드에 고정하고
# 규칙을 실수로 어기지 못하게 막는 장치를 넣었다. test 묶음은 --allow-test 없이는 재지 않고(단계마다 한 번),
# 커밋된 리포트는 --overwrite 없이는 덮어쓰지 않으며(커밋 a29a975), 리포트에 측정 코드의 커밋을 남긴다(바뀐
# 코드로 쟀으면 +dirty, 커밋 6ba3c27). 발화별 기록을 남기므로 analyze가 숫자 발화 제외, 맞춘 CER 같은 수치를
# 모델을 다시 돌리지 않고 계산한다.
# [흐름] 인자 확인 → 묶음과 모델 불러오기 → 배치마다 오디오 읽기(필요하면 전화 채널 변환) → GPU에서 로그 멜
# 특징 계산 → generate(greedy, 또는 8단계의 재시도) → 채점 → 요약 출력과 리포트 쓰기.
# [연결] evaluate_qwen이 MAX_NEW_TOKENS·WINDOW_SECONDS·score_rows·batch_seconds_summary·git_commit을,
# evaluate_ct2와 train·train_qwen이 git_commit을, train이 MAX_NEW_TOKENS를 가져다 쓴다.

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

# numpy: 채점 배열과 백분위 계산. torch: GPU 추론, 메모리 측정, 재시도 표본의 시드. transformers: Whisper
# 모델과 처리기(특징 추출기 + 토크나이저).
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# channel: --channel telephone(4단계 전화 음질 변환). metrics: 채점과 구간. paths: 리포트 위치와 저장소 루트.
# store: 묶음 목록과 오디오 저장소.
from whisper_ko_ft.channel import CHANNELS
from whisper_ko_ft.metrics import error_rate, interval, score
from whisper_ko_ft.paths import REPORTS, ROOT
from whisper_ko_ft.store import SAMPLE_RATE, AudioStore, load_set

# MAX_NEW_TOKENS: 발화당 생성할 최대 토큰 수(docs/experiments.md 공통 규칙 '디코딩'). 같은 구절을 되풀이하는
# 발화도 여기서 끊긴다. train(학습 중 평가)과 evaluate_qwen도 이 값을 가져가 하네스끼리 조건을 맞춘다.
# WINDOW_SECONDS: Whisper가 한 번에 듣는 길이(30초). 특징 추출기는 이보다 긴 음성의 앞 30초만 쓰므로 넘는 발화
# 수를 요약에 센다. evaluate_qwen도 같은 길이로 잘라 같은 소리를 듣게 한다.
# _DIGIT: 가설에 아라비아 숫자가 있는지 본다. 1단계에서 '숫자 발화'(기본 모델 둘 중 하나라도 가설에 숫자가
# 있는 발화)를 정할 때 쓴 표시다(docs/experiments.md '숫자 표기').
MAX_NEW_TOKENS = 256
WINDOW_SECONDS = 30
_DIGIT = re.compile(r"[0-9]")

# TEMPERATURES: 재시도할 때 차례로 올릴 temperature. 0은 greedy이고, 기준을 넘은 발화만 다음 값으로 다시
# 디코딩한다.
# FALLBACKS: 8단계 후보. ratio(D1a)는 생성한 토큰 바이트의 zlib 압축비가 1.35를 넘으면(같은 구절이 되풀이되면
# 잘 압축된다), ratio-logprob(D1)은 거기에 더해 평균 로그확률이 −1.0보다 낮을 때도 다시 디코딩한다. 두 값은
# transformers 문서의 권장값이고 바꾸지 않기로 했다(docs/experiments.md 8단계).
# Stage 8 candidates (docs/experiments.md): D1a = "ratio", D1 = "ratio-logprob". The ratio is zlib on the
# token bytes, as transformers computes it (1.35 is its documented value); faster-whisper's 2.4 uses the text.
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
FALLBACKS: dict[str, dict[str, float]] = {
    "ratio": {"compression_ratio_threshold": 1.35},
    "ratio-logprob": {"compression_ratio_threshold": 1.35, "logprob_threshold": -1.0},
}


# --fallback 값을 generate에 넘길 추가 인자로 바꾼다. None(기본)이면 빈 dict라 generate 인자가 1~7단계 greedy
# 측정과 똑같다(tests/test_decoding.py가 확인).
# 두 기준값 인자는 스모크 실행에서 다시 디코딩을 일부러 일으키려고 기준을 바꿀 때만 쓴다. main은 이때
# --no-report를 요구해, 등록하지 않은 기준으로 리포트가 남지 않게 한다.
# 잘못된 조합(재시도 없이 기준값만 주기, 로그확률 기준이 없는 D1a에 로그확률 기준 주기)은 ValueError로
# 거부한다.
def decoding_settings(
    fallback: str | None,
    compression_ratio_threshold: float | None = None,
    logprob_threshold: float | None = None,
) -> dict:
    """Extra generate arguments: none for greedy decoding, the fallback candidate otherwise.

    The two thresholds replace the candidate's values in smoke runs (to force decoding again).
    """
    if fallback is None:
        if compression_ratio_threshold is not None or logprob_threshold is not None:
            raise ValueError("thresholds need --fallback")
        return {}
    settings: dict = {"temperature": TEMPERATURES, **FALLBACKS[fallback]}
    if compression_ratio_threshold is not None:
        settings["compression_ratio_threshold"] = compression_ratio_threshold
    if logprob_threshold is not None:
        if "logprob_threshold" not in settings:
            raise ValueError(f"the {fallback!r} fallback has no log-probability threshold")
        settings["logprob_threshold"] = logprob_threshold
    return settings


# 한 배치에서 기록한 재시도 결정(calls)을 발화별 (마지막 temperature, 마지막 temperature에서도 기준을
# 넘었는지)로 바꾼다.
# calls의 순서: 첫 temperature에서 발화마다 하나씩, 그다음 temperature에서는 아직 기준을 넘는 발화만 배치
# 순서대로 하나씩(위 docstring). 이 순서를 그대로 따라가며 각 발화의 마지막 결정을 채운다.
# 순서가 이 가정과 맞지 않으면(개수가 다르거나, 한 단계 안에 temperature가 섞이거나, 기준을 넘었는데 더 높은
# temperature로 가지 않음) RuntimeError로 멈춘다. transformers 내부 동작이 바뀌었을 때 틀린 기록을 조용히
# 남기지 않으려는 것이다.
def final_temperatures(
    calls: list[tuple[float | None, bool]], batch_size: int, temperatures: Sequence[float]
) -> list[tuple[float, bool]]:
    """(final temperature, still failing at the last temperature) of each utterance of a batch.

    `calls` are the fallback decisions in the order transformers makes them: one per utterance at the first
    temperature, then one per utterance still failing, in batch order, at each higher temperature.
    """
    # 처음에는 모든 발화가 temperature 0, 기준 통과로 시작한다. pending은 아직 다음 단계로 넘어갈 수 있는
    # 발화, position은 calls에서 읽은 위치다.
    final = [(0.0, False)] * batch_size
    pending, position = list(range(batch_size)), 0
    while pending and position < len(calls):
        # 이번 temperature 단계의 결정은 남은 발화 수만큼 이어져 있어야 하고, 모두 같은 temperature여야 한다.
        level = calls[position : position + len(pending)]
        if len(level) != len(pending) or len({t for t, _ in level}) != 1:
            raise RuntimeError(f"fallback calls do not match the batch: {calls}")
        still = []
        for utterance, (temperature, needs) in zip(pending, level, strict=True):
            final[utterance] = (float(temperature or 0.0), needs)
            if needs:
                still.append(utterance)
        position += len(pending)
        pending = still
    # 결정을 다 쓰지 못했거나, 발화가 있는데 결정이 하나도 없으면 맞지 않는다.
    if position != len(calls) or (position == 0 and batch_size):
        raise RuntimeError(f"fallback calls do not match the batch: {calls}")
    # 마지막에도 기준을 넘은 발화는 마지막 temperature(1.0)까지 갔어야 한다. 그보다 낮은 곳에서 멈췄으면
    # 기록이 빠진 것이다.
    for temperature, needs in final:
        if needs and temperature != temperatures[-1]:
            raise RuntimeError(f"decoded again at {temperature} but never at a higher temperature: {calls}")
    return final


# transformers가 어떤 발화를 어느 temperature로 다시 디코딩했는지 기록한다. 리포트의 발화별 temperature와
# fallback_exhausted, 요약의 redecoded가 여기서 나온다.
# 왜 감싸나: Whisper generate는 재시도 판단(_need_fallback)을 밖으로 돌려주지 않는다(위 docstring). 그래서 이
# 모델 객체 하나의 메서드만 감싸 판단 결과를 옆으로 빼낸다. 판단 자체는 원래 메서드가 하므로 디코딩 결과는
# 바뀌지 않는다.
# 주의: 비공개 메서드라 transformers 버전이 바뀌면 인자 위치가 달라질 수 있다. 그때는 final_temperatures의
# 검사가 멈춰 알린다. tests/test_decoding.py가 작은 무작위 Whisper로 실제 경로를 확인한다.
class FallbackTrace:
    """Records which utterances transformers decoded again, and at what temperature.

    Whisper's generate decides per decoding in `_need_fallback` (transformers 5.17, generation_whisper.py) but
    does not return the decisions, so the method is wrapped on this model instance.
    """

    def __init__(self, model, settings: dict) -> None:
        self.temperatures = settings["temperature"]
        self.calls: list[tuple[float | None, bool]] = []
        decide = model._need_fallback

        # 원래 판단을 그대로 돌려주면서 (temperature, 다시 디코딩할지)를 쌓는다. temperature는 키워드로 오지
        # 않으면 7번째 위치 인자다(transformers 5.17의 _need_fallback 인자 순서).
        def traced(*args, **kwargs):
            needs, skip = decide(*args, **kwargs)
            temperature = kwargs["temperature"] if "temperature" in kwargs else args[6]
            self.calls.append((temperature, bool(needs)))
            return needs, skip

        model._need_fallback = traced

    # 배치 하나가 끝날 때 쌓인 결정을 꺼내 비우고 발화별 결과로 바꾼다. 호출 위치: transcribe.
    def take(self, batch_size: int) -> list[tuple[float, bool]]:
        calls, self.calls = self.calls, []
        return final_temperatures(calls, batch_size, self.temperatures)


# generate 결과에서 토큰 ID 텐서를 꺼낸다. 로그확률 기준(D1)처럼 점수를 돌려받는 경로에서는 출력 객체가 오고,
# 그 밖에는 텐서가 온다. 두 경우를 한곳에서 맞춘다.
def sequences(output) -> torch.Tensor:
    """Token ids from generate, which returns a tensor or, with return_dict_in_generate, an output object."""
    return output if isinstance(output, torch.Tensor) else output["sequences"]


# 배치 하나를 받아 적어 (가설 목록, 발화별 마지막 temperature 또는 None)을 돌려준다.
# 설정은 공통 규칙 그대로다: greedy(num_beams=1), 언어 고정, task=transcribe, 타임스탬프 없음, 최대
# 256토큰(docs/experiments.md '디코딩'). settings는 재시도 후보일 때만 temperature와 기준값을 더한다.
# 호출 위치: main, tests/test_decoding.py.
def transcribe(
    model,
    processor,
    features,
    attention_mask,
    language: str,
    settings: dict,
    trace: FallbackTrace | None = None,
) -> tuple[list[str], list[tuple[float, bool]] | None]:
    """Hypotheses of one batch, and each utterance's final temperature when the fallback is traced."""
    # 추론만 하므로 inference_mode로 기울기 기록을 끈다(메모리와 시간 절약).
    with torch.inference_mode():
        output = model.generate(
            input_features=features,
            attention_mask=attention_mask,
            language=language,
            task="transcribe",
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            return_timestamps=False,
            **settings,
        )
    # 특수 토큰(언어, task, 끝 표시)을 빼고 글로 바꾼다.
    texts = processor.batch_decode(sequences(output), skip_special_tokens=True)
    return texts, trace.take(len(texts)) if trace else None


# 리포트 행마다 편집 수(edits)와 정답 길이(length)를 채운다. 정규화한 정답이 빈 행은 0과 0으로 적고 그 수를
# 돌려준다(요약의 skipped_empty_reference). 가설에 아라비아 숫자가 있는지(has_digit)도 적는다.
# 호출 위치: main, evaluate_qwen(raw와 fixed 리포트).
def score_rows(rows: list[dict], language: str) -> int:
    """Edits and reference length of each row (0 and 0 when the normalized reference is empty).

    Returns the number of rows skipped for an empty reference.
    """
    skipped = 0
    for row in rows:
        scored = score(row["reference"], row["hypothesis"], language)
        if scored is None:
            skipped += 1
            row["edits"] = row["length"] = 0
            continue
        row["edits"], row["length"] = scored.edits, scored.length
        row["has_digit"] = bool(_DIGIT.search(row["hypothesis"]))
    return skipped


# 배치마다 걸린 시간의 중앙값과 90번째 백분위. 배치 1이면 한 발화의 지연이다. 9단계 속도 비교(Qwen과 turbo의
# 한 발화 지연)를 위해 두 하네스의 요약에 넣었다(docs/experiments.md 9단계 '속도와 GPU 메모리').
def batch_seconds_summary(seconds: list[float]) -> dict:
    """Median and 90th percentile of the time per batch: the latency of one utterance at batch size 1."""
    if not seconds:
        return {"batch_seconds_median": None, "batch_seconds_p90": None}
    median, p90 = np.percentile(seconds, [50, 90])
    return {"batch_seconds_median": round(float(median), 3), "batch_seconds_p90": round(float(p90), 3)}


# 요약에 넣을 개수: 다시 디코딩한 발화(redecoded), 마지막 temperature까지 기준을 넘은
# 발화(fallback_exhausted), 빈 가설, 되풀이 발화(loops: 편집 수 > 정답 길이). 정답이 빈 행은 세지 않는다.
# 재시도를 쓰지 않은 리포트는 재시도 관련 두 값을 None으로 둔다.
# 되풀이 수는 원래 채점 기준이다. Zeroth의 맞춘 채점 기준 되풀이는 analyze loops --harmonized로 따로 센다.
def decoding_counts(rows: list[dict], fallback: bool) -> dict:
    """Summary counts over the scored utterances: decoded again, loops (more edits than reference), empty."""
    kept = [row for row in rows if row["length"]]
    return {
        "redecoded": sum(row["temperature"] > 0 for row in kept) if fallback else None,
        "fallback_exhausted": sum(row["fallback_exhausted"] for row in kept) if fallback else None,
        "empty_hypotheses": sum(not row["hypothesis"].strip() for row in kept),
        "loops": sum(row["edits"] > row["length"] for row in kept),
    }


# 측정·학습한 코드의 커밋을 리포트와 run.json에 남긴다. 짧은 HEAD 해시이고, src·pyproject.toml·uv.lock이
# HEAD와 다르면 '+dirty'를 붙인다. 리포트나 문서만 바뀐 것은 측정 코드가 아니므로 보지 않는다.
# git이 없으면 'unknown', 커밋이 없으면 'uncommitted'. 호출 위치:
# evaluate·evaluate_ct2·evaluate_qwen·train·train_qwen.
def git_commit(root: Path = ROOT) -> str:
    """Short HEAD hash; '+dirty' when code or dependency files differ from HEAD (not reports or docs)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"],
            cwd=root,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        return "unknown"
    if not head:
        return "uncommitted"
    return f"{head}+dirty" if dirty else head


# 모델을 fp16으로 GPU에 올리고 처리기를 함께 돌려준다. --adapter가 있으면 LoRA를 기본 가중치에
# 합쳐(merge_and_unload) 보통의 Whisper 모델로 잰다. 합치면 추론 구조가 기본 모델과 같아 속도·메모리를 같은
# 조건에서 비교할 수 있다.
# peft는 어댑터가 있을 때만 불러온다. 처리기는 기본 모델(path)의 것을 쓴다.
def load_model(path: str, adapter: str | None) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    model = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float16)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    processor = WhisperProcessor.from_pretrained(path)
    return model.to("cuda").eval(), processor


# 명령줄 진입점. 주요 인자: --model(모델 ID 또는 체크포인트 폴더), --adapter(LoRA), --set(묶음), --name(리포트
# 폴더 이름), --channel, --batch-size(기본 16, 속도 측정은 1), --limit, --allow-test, --no-report,
# --overwrite, --fallback과 --seed(8단계), 스모크용 기준값 둘.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID or a local checkpoint folder")
    parser.add_argument("--adapter", help="LoRA adapter folder to merge into --model")
    parser.add_argument("--set", required=True, dest="set_name")
    parser.add_argument(
        "--name", help="report name; defaults to the last part of --model (required with --adapter)"
    )
    parser.add_argument(
        "--channel", choices=list(CHANNELS), help="pass the audio through a channel simulation"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="only the first N utterances (smoke runs, speed checks)")
    parser.add_argument("--allow-test", action="store_true", help="required for *-test sets")
    parser.add_argument("--no-report", action="store_true", help="print the summary only")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    parser.add_argument("--fallback", choices=list(FALLBACKS), help="temperature fallback (stage 8)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed of the fallback, per batch")
    parser.add_argument(
        "--compression-ratio-threshold", type=float, help="smoke runs only (--no-report): replace 1.35"
    )
    parser.add_argument("--logprob-threshold", type=float, help="smoke runs only (--no-report): replace -1.0")
    args = parser.parse_args()

    # 규칙을 지키게 하는 검사들. test 묶음은 단계마다 한 번만 재므로 --allow-test가 있어야 한다. --adapter로
    # 잴 때 --name이 없으면 기본 이름이 기준 모델의 리포트 폴더가 되어 기준선 리포트와 섞이므로 막는다.
    if args.set_name.endswith("-test") and not args.allow_test:
        parser.error("test sets are measured once per stage; pass --allow-test when the stage is done")
    if args.adapter and not args.name:
        parser.error("--adapter needs --name (the default is the base model's report folder)")
    try:
        settings = decoding_settings(args.fallback, args.compression_ratio_threshold, args.logprob_threshold)
    except ValueError as error:
        parser.error(str(error))
    # 스모크용 기준값으로 잰 결과는 등록한 후보가 아니므로 리포트로 남기지 못하게 한다.
    overridden = args.compression_ratio_threshold is not None or args.logprob_threshold is not None
    if overridden and not args.no_report:
        parser.error("threshold overrides are not registered candidates; pass --no-report (smoke runs)")
    # 리포트 경로: reports/<이름>/<묶음>[@채널].json. 이름 기본값은 --model의 마지막 경로 부분이다. 이미
    # 있으면 --overwrite 없이 멈춘다(커밋된 측정 기록 보호).
    suffix = f"@{args.channel}" if args.channel else ""
    target = REPORTS / (args.name or Path(args.model).name) / f"{args.set_name}{suffix}.json"
    if not args.no_report and target.exists() and not args.overwrite:
        parser.error(f"{target.relative_to(ROOT)} exists; pass --overwrite to measure it again")
    commit = git_commit()  # the code that measures, before any of it can change

    # 묶음을 읽고(--limit이면 앞 N개만), 모델을 올린다. 재시도 후보면 이 모델 객체에 FallbackTrace를 건다.
    language, utterances = load_set(args.set_name)
    if args.limit:
        utterances = utterances[: args.limit]
    stores: dict[str, AudioStore] = {}
    model, processor = load_model(args.model, args.adapter)
    trace = FallbackTrace(model, settings) if args.fallback else None

    # 피크 메모리 통계를 비우고 시계를 시작한다. 모델을 띄우는 시간은 속도에서 뺀다(docs/experiments.md 속도
    # 지표).
    rows: list[dict] = []
    batch_seconds: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for begin in range(0, len(utterances), args.batch_size):
        batch_started = time.perf_counter()
        batch = utterances[begin : begin + args.batch_size]
        # 저장소 이름마다 memmap을 한 번만 연다. --channel이면 음성을 전화 채널 변환(16 kHz 그대로)에
        # 통과시킨다.
        audio = [stores.setdefault(u.store, AudioStore(u.store)).read(u) for u in batch]
        if args.channel:
            audio = [CHANNELS[args.channel](samples) for samples in audio]
        # 로그 멜 특징을 GPU(device="cuda")에서 계산한다. 특징 추출기가 30초로 채우거나 자르므로 긴 발화는 앞
        # 30초만 들어간다. attention mask도 받아 generate에 넘긴다(커밋 96cee32).
        inputs = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True, device="cuda"
        )
        # 재시도는 temperature > 0에서 표본을 뽑는다. 시드를 '--seed + 배치 첫 발화 번호'로 배치마다 정해,
        # 결과가 실행 순서가 아니라 배치 내용에만 달리게 한다(docs/experiments.md 8단계).
        if args.fallback:
            torch.manual_seed(args.seed + begin)  # sampled decodings depend only on the batch, not on the run
        # 모델이 fp16이므로 특징도 fp16으로 바꿔 넘긴다.
        texts, decisions = transcribe(
            model,
            processor,
            inputs.input_features.to("cuda", dtype=torch.float16),
            inputs.attention_mask.to("cuda"),
            language,
            settings,
            trace,
        )
        batch_seconds.append(time.perf_counter() - batch_started)  # decoding the text waits for the GPU
        # 발화별 행: ID, 화자, 길이, 정답, 가설(앞뒤 공백 제거). 재시도 후보면 마지막 temperature와
        # fallback_exhausted를 더한다. 이 행들이 리포트의 utterances가 되어 모든 수치를 나중에 다시 계산하는
        # 근거가 된다.
        for index, (utterance, text) in enumerate(zip(batch, texts, strict=True)):
            row = {
                "id": utterance.id,
                "speaker": utterance.speaker,
                "seconds": round(utterance.seconds, 2),
                "reference": utterance.text,
                "hypothesis": text.strip(),
            }
            if decisions is not None:
                row["temperature"], row["fallback_exhausted"] = decisions[index]
            rows.append(row)
        if (begin // args.batch_size) % 20 == 0:
            print(f"{begin + len(batch):,}/{len(utterances):,}", flush=True)
    # GPU 작업이 모두 끝난 뒤에 시계를 멈춘다.
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

    # 채점하고, 정답이 빈 행을 뺀 뒤 묶음 오류율과 부트스트랩 구간을 낸다.
    skipped = score_rows(rows, language)
    kept = [r for r in rows if r["length"]]
    edits = np.array([r["edits"] for r in kept])
    lengths = np.array([r["length"] for r in kept])
    low, high = interval(edits, lengths)
    audio_seconds = sum(r["seconds"] for r in rows)
    # 리포트 요약. 오류율과 구간 말고도 재현과 해석에 필요한 것을 함께 남긴다: 디코딩 설정과 시드,
    # 재시도·되풀이·빈 가설 수, 30초 넘는 발화 수, 속도(음성 초 ÷ 걸린 초)와 배치 지연, torch 최대 할당
    # 메모리, --limit(analyze는 limit이 있는 리포트를 받지 않는다), 측정 코드의 커밋, 측정 시각, torch 버전,
    # GPU 이름.
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "set": args.set_name,
        "channel": args.channel,
        "language": language,
        "metric": "cer" if language == "ko" else "wer",
        "utterances": len(kept),
        "skipped_empty_reference": skipped,
        "error_rate": round(error_rate(edits, lengths), 5),
        "interval_95": [round(low, 5), round(high, 5)],
        "hypotheses_with_digit": sum(r.get("has_digit", False) for r in kept),
        "fallback": args.fallback,
        "decoding": {**settings, "seed": args.seed} if args.fallback else None,
        **decoding_counts(rows, fallback=bool(args.fallback)),
        "over_30_seconds": sum(r["seconds"] > WINDOW_SECONDS for r in rows),
        "audio_seconds": round(audio_seconds, 1),
        "wall_seconds": round(wall, 1),
        "audio_seconds_per_second": round(audio_seconds / wall, 1),
        **batch_seconds_summary(batch_seconds),
        "batch_size": args.batch_size,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "limit": args.limit,
        "commit": commit,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # --no-report가 아니면 요약과 발화별 행을 JSON으로 쓴다. 한국어를 그대로(ensure_ascii=False) 적고 줄바꿈은
    # LF로 고정한다. 이 파일을 커밋한다.
    if not args.no_report:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "utterances": rows}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
