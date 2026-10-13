"""Transcribe a set with faster-whisper (CTranslate2), with or without its decoding fallback.

Usage (needs `uv sync --group ct2` and a model converted with ct2-transformers-converter):
    uv run python -m whisper_ko_ft.evaluate_ct2 --model outputs/small-a/ct2 --name a-ct2 --set zeroth-val
    (add --fallback, with another --name, to turn on the decoding fallback)

One utterance at a time, greedy, fixed language, no timestamps, no VAD (docs/experiments.md, "5단계").
--fallback turns on faster-whisper's defaults: when the text compresses too well (a repeated phrase) or the
average log probability is low, it decodes again at a higher temperature.
The number of mel bins comes from the converted model (128 for large-v3 and turbo, see `ct2_features`), and
the model is loaded before any audio of the set is read.
"""

# [파일 역할] 5단계(되풀이 오류와 추론 엔진)와 2026-10-02 추가 측정에서 쓴 faster-whisper(CTranslate2) 측정
# 하네스. 변환된 모델로 묶음을 한 발화씩 받아 적고 evaluate.py와 같은 모양의 리포트를 남긴다.
# [왜 따로 두었나] 서비스에서 쓰는 추론 엔진의 재시도 장치가 되풀이 오류를 막는지, 같은 가중치로
# Transformers보다 얼마나 빠른지 보려는 것이다(docs/experiments.md 5단계, '추가 측정'). 엔진과 API가 달라
# evaluate.py와 코드를 나눴고, 채점(metrics)과 커밋 기록(git_commit)은 같은 것을 쓴다.
# [두 조건] '재시도 끔'(NO_FALLBACK)은 Transformers로 잰 앞 단계와 같은 greedy, '재시도 켬'(--fallback)은
# faster-whisper 기본값이다.
# [알려진 차이] '재시도 끔'은 로그확률 기준이 없어 무음 확률만으로 발화를 건너뛸 수 있고, '재시도 켬'은
# 로그확률이 높으면 건너뛰지 않는다. 5단계와 같은 조건을 지키려고 바꾸지 않았고, 요약의 empty_hypotheses로
# 영향을 보인다(docs/experiments.md '추가 측정').

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

# numpy: 채점 배열. torch: 여기서는 CTranslate2가 쓸 CUDA 라이브러리 폴더를 찾는 데만 쓴다(아래).
import numpy as np
import torch

# ct2_features.feature_kwargs: 변환된 모델의 멜 수에 맞춘 특징 설정. evaluate.git_commit: 측정 코드의 커밋
# 기록(evaluate.py와 같은 방식). metrics: 채점. store: 묶음과 오디오.
from whisper_ko_ft.ct2_features import feature_kwargs
from whisper_ko_ft.evaluate import git_commit
from whisper_ko_ft.metrics import error_rate, interval, score
from whisper_ko_ft.paths import REPORTS, ROOT
from whisper_ko_ft.store import AudioStore, load_set

# Windows에서 CTranslate2가 CUDA 12 DLL을 찾을 수 있게, torch 휠에 들어 있는 lib 폴더를 DLL 검색 경로와 PATH
# 앞에 넣는다. 이 설정이 ctranslate2를 불러오기 전에 끝나야 하므로 아래 import들이 모듈 중간에 있고, 그래서
# 줄마다 E402(모듈 중간 import) 검사를 끈다.
# CTranslate2 needs the CUDA 12 libraries; on Windows the ones that ship with torch are used.
_TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.name == "nt":
    os.add_dll_directory(_TORCH_LIB)
    os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ["PATH"]

import ctranslate2  # noqa: E402
import faster_whisper  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.feature_extractor import FeatureExtractor  # noqa: E402

# NO_FALLBACK('재시도 끔'): temperature 0만, 압축비·로그확률 기준 없음. Transformers greedy와 같은 조건이다.
# FALLBACK('재시도 켬'): faster-whisper 기본값. 글자의 압축비가 2.4를 넘거나(같은 구절의 반복) 평균 로그확률이
# −1.0보다 낮으면 temperature를 0.2씩 올려 다시 디코딩한다(docs/experiments.md 5단계). 2.4는 글자 기준이라
# 8단계 transformers의 1.35(토큰 바이트 기준)와 값이 다르다.
NO_FALLBACK = {"temperature": 0.0, "compression_ratio_threshold": None, "log_prob_threshold": None}
FALLBACK = {
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
}


# 명령줄 진입점. --model은 변환된 CTranslate2 폴더, --name은 리포트 이름(필수), --fallback은 '재시도 켬'. test
# 묶음 보호(--allow-test)와 덮어쓰기 보호(--overwrite)는 evaluate.py와 같다. --no-report는 속도만 볼 때 쓴다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="CTranslate2 model folder")
    parser.add_argument("--name", required=True, help="report name")
    parser.add_argument("--set", required=True, dest="set_name")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    parser.add_argument("--no-report", action="store_true", help="print the summary only (speed runs)")
    args = parser.parse_args()
    if args.set_name.endswith("-test") and not args.allow_test:
        parser.error("test sets are measured once per stage; pass --allow-test when the stage is done")
    target = REPORTS / args.name / f"{args.set_name}.json"
    if not args.no_report and target.exists() and not args.overwrite:
        parser.error(f"{target.relative_to(ROOT)} exists; pass --overwrite to measure it again")
    # 측정 코드의 커밋은 모델이나 오디오를 읽기 전에 기록한다(evaluate.py와 같다).
    commit = git_commit()

    # 모델을 오디오보다 먼저 띄운다(위 docstring). 설정 파일의 멜 수가 모델과 맞지 않으면 오디오를 읽기 전에
    # 바로 멈춘다. 2026-09-20 시도는 특징을 다 만든 뒤 첫 발화에서 멈췄다(docs/experiments.md '추가 측정').
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    # 기본 특징 추출기를 변환된 모델의 멜 수(model.model.n_mels, turbo는 128)에 맞춘 것으로 바꾼다.
    # faster-whisper makes 80 mel bins unless the folder has a preprocessor_config.json (ct2_features).
    model.feature_extractor = FeatureExtractor(**feature_kwargs(Path(args.model), model.model.n_mels))
    language, utterances = load_set(args.set_name)
    if args.limit:
        utterances = utterances[: args.limit]
    # 재시도 끔/켬 조건을 고르고, 저장소별 memmap을 열어 둘 dict를 만든다.
    options = FALLBACK if args.fallback else NO_FALLBACK
    stores: dict[str, AudioStore] = {}

    rows: list[dict] = []
    started = time.perf_counter()
    # 한 발화씩 처리한다(배치 없음). 5단계 규칙의 '한 발화씩'이고, 배치 1 속도를 Transformers와 비교하는
    # 조건이기도 하다.
    for n, utterance in enumerate(utterances):
        audio = stores.setdefault(utterance.store, AudioStore(utterance.store)).read(utterance)
        # greedy(beam_size=1), 언어 고정, 타임스탬프 없음, 앞 구간의 글을 다음 디코딩에 이어 주지 않음, VAD
        # 없음(docs/experiments.md 5단계 설정). 재시도 관련 값만 options로 바뀐다.
        segments, _ = model.transcribe(
            audio,
            language=language,
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            **options,
        )
        # segments는 생성기라 끝까지 꺼내야 디코딩이 끝난다. 구간 글을 공백으로 이어 가설 하나로 만든다.
        segments = list(segments)
        text = " ".join(segment.text.strip() for segment in segments)
        # evaluate.py와 같은 채점. 정답이 비면 편집 수와 길이를 0으로 적는다.
        scored = score(utterance.text, text, language)
        # 발화별 행. temperature는 이 발화의 구간 중 가장 높은 temperature이고, 0보다 크면 재시도로 다시
        # 디코딩된 것이다.
        rows.append(
            {
                "id": utterance.id,
                "seconds": round(utterance.seconds, 2),
                "reference": utterance.text,
                "hypothesis": text,
                "edits": scored.edits if scored else 0,
                "length": scored.length if scored else 0,
                "temperature": max((segment.temperature for segment in segments), default=0.0),
            }
        )
        if n % 400 == 0:
            print(f"{n:,}/{len(utterances):,}", flush=True)
    wall = time.perf_counter() - started

    # 정답이 빈 발화를 빼고 묶음 오류율과 부트스트랩 구간을 낸다.
    kept = [r for r in rows if r["length"]]
    edits, lengths = np.array([r["edits"] for r in kept]), np.array([r["length"] for r in kept])
    low, high = interval(edits, lengths)
    audio_seconds = sum(r["seconds"] for r in rows)
    # 요약. more_edits_than_characters는 되풀이 발화(편집 수 > 정답 길이, 5단계 판정의 '정답보다 오류가 많은
    # 발화') 수, decoded_again은 다시 디코딩한 발화 수, empty_hypotheses는 빈 가설 수(위 '알려진 차이'),
    # mel_bins는 실제로 쓴 멜 수다. 엔진 버전도 남긴다.
    # empty_hypotheses는 evaluate.decoding_counts와 같은 기준으로 센다: 채점한 발화(kept)만, 공백만 남은
    # 가설도 빈 가설로. 2026-10-13 전에는 정답이 빈 발화까지 세고 공백 가설은 세지 않았다. 커밋된
    # faster-whisper 리포트에는 정답이 빈 발화도 공백 가설도 없어, 발화 행으로 다시 세면 두 기준 모두 0이다.
    summary = {
        "model": args.model,
        "engine": "faster-whisper",
        "fallback": args.fallback,
        "set": args.set_name,
        "metric": "cer" if language == "ko" else "wer",
        "utterances": len(kept),
        "error_rate": round(error_rate(edits, lengths), 5),
        "interval_95": [round(low, 5), round(high, 5)],
        "more_edits_than_characters": sum(r["edits"] > r["length"] for r in kept),
        "decoded_again": sum(r["temperature"] > 0 for r in rows),
        "empty_hypotheses": sum(not r["hypothesis"].strip() for r in kept),
        "mel_bins": model.feature_extractor.mel_filters.shape[0],
        "audio_seconds": round(audio_seconds, 1),
        "wall_seconds": round(wall, 1),
        "audio_seconds_per_second": round(audio_seconds / wall, 1),
        "limit": args.limit,
        "commit": commit,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "faster_whisper": faster_whisper.__version__,
        "ctranslate2": ctranslate2.__version__,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # --no-report면 요약만 찍고 끝낸다(속도 측정).
    if args.no_report:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "utterances": rows}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
