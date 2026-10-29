"""faster-whisper(CTranslate2)로 묶음을 받아 적는다. 디코딩 재시도(fallback)를 켜거나 끌 수 있다.

사용법(`uv sync --group ct2`와 ct2-transformers-converter로 변환한 모델이 필요하다):
    uv run python -m whisper_ko_ft.evaluate_ct2 --model outputs/small-a/ct2 --name a-ct2 --set zeroth-val
    (디코딩 재시도를 켜려면 다른 --name과 함께 --fallback을 더한다)

한 발화씩, greedy, 언어 고정, 타임스탬프 없음, VAD 없음(docs/experiments.md "5단계").
--fallback은 faster-whisper 기본값을 켠다: 글이 너무 잘 압축되거나(같은 구절의 반복) 평균 로그확률이 낮으면
더 높은 temperature로 다시 디코딩한다.
멜 필터 수는 변환된 모델에서 가져오고(large-v3와 turbo는 128개, `ct2_features` 참고), 모델은 묶음의 오디오를
읽기 전에 띄운다.

12단계(2026-10-15 규칙): --compute-type은 CTranslate2가 모델을 불러올 때 쓸 정밀도다(기본 float16은 기록된
측정과 같다. int8_float16은 가중치를 int8로 양자화하고 나머지 계산은 fp16). 변환을 다시 하지 않고 같은 폴더를
쓴다. --device cpu는 확인용이라 --no-report가 있어야 한다. 요약에 정밀도·장치와 한 발화 지연의 중앙값·90번째
백분위(batch_seconds_median, batch_seconds_p90. evaluate의 배치 1 지연과 같은 키)를 남긴다.

2026-10-29: 요약에 변환 폴더의 지문(model.bin·config.json의 SHA-256)을 남긴다. 같은 경로에 다시 변환한 이력이
있어, 리포트만으로 어느 변환으로 쟀는지 알 수 있게 하려는 것이다. --seed를 주면 모델을 띄우기 전에
`ctranslate2.set_random_seed(seed)`를 한 번 불러, 같은 명령(같은 모델 폴더·묶음·--limit·정밀도·장치)을 다시
돌리면 재시도(--fallback)의 temperature 표본이 같게 한다. evaluate --seed처럼 발화(배치)마다 시드를 다시 정할
수는 없다: set_random_seed는 전역 값만 바꾸고, 난수 생성기는 디코딩하는 작업 스레드가 처음 쓸 때 그때의 값으로
한 번 만든 뒤 다시 시드하지 않는다(CTranslate2 4.8.2 src/random.cc의 thread_local mt19937). GPU의 curand
상태도 작업 스레드마다 한 벌이고, 더 많은 상태가 필요할 때만 그때의 값으로 다시 만든다(src/cuda/random.cu).
2026-10-29 CPU에서 작은 무작위 모델로 확인한 결과도 같다: generate마다 앞에서 set_random_seed를 불러도 표본은
처음 시드에서 이어질 뿐 처음으로 돌아가지 않았다. 그래서 한 실행 안의 표본은 앞 발화들이 쓴 난수에 이어지고,
시드가 같아도 묶음이나 발화 순서가 다르면 표본이 다르다. 기본값(없음)은 기록된 측정과 같은 경로(시드 없음)다.
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

# ct2_features.feature_kwargs: 변환된 모델의 멜 수에 맞춘 특징 설정. evaluate.batch_seconds_summary: 지연의
# 중앙값·90번째 백분위(같은 키로 남겨 두 하네스를 나란히 읽는다). evaluate.report_json: 리포트 글(NaN이면
# ValueError). provenance.git_commit: 측정 코드의 커밋 기록(evaluate.py와 같은 방식).
# provenance.converted_model_hashes: 변환 폴더의 지문(model.bin·config.json SHA-256). metrics: 채점. store:
# 묶음과 오디오.
from whisper_ko_ft.ct2_features import feature_kwargs
from whisper_ko_ft.evaluate import batch_seconds_summary, report_json
from whisper_ko_ft.metrics import error_rate, interval, score
from whisper_ko_ft.paths import REPORTS, ROOT
from whisper_ko_ft.provenance import converted_model_hashes, git_commit
from whisper_ko_ft.store import AudioStore, load_set

# CTranslate2는 CUDA 12 라이브러리가 필요하다. Windows에서는 torch 휠에 들어 있는 것을 쓴다. 그래서
# CTranslate2가 CUDA 12 DLL을 찾을 수 있게 torch의 lib 폴더를 DLL 검색 경로와 PATH 앞에 넣는다. 이 설정이
# ctranslate2를 불러오기 전에 끝나야 하므로 아래 import들이 모듈 중간에 있고, 그래서 줄마다 E402(모듈 중간
# import) 검사를 끈다.
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
# COMPUTE_TYPES: --compute-type 값. float16이 지금까지의 모든 기록(5단계, 2026-10-02)과 같은 기본값이고,
# int8_float16이 12단계의 후보다(compute capability 7.0 이상, RTX 2080 Ti는 7.5). int8·int8_float32·float32는
# CPU 확인용이다(CPU에는 float16 계산이 없다).
COMPUTE_TYPES = ("float16", "int8_float16", "int8", "int8_float32", "float32")
# SEED_MAX: --seed로 받는 가장 큰 값. ctranslate2.set_random_seed는 C++의 unsigned int를 받아 음수와 2**32
# 이상은 TypeError로 거부하고, 2**32 - 1(4294967295)은 CTranslate2가 '시드를 정하지 않음' 표시로 쓰는 값이라
# 받아도 생성기를 std::random_device로 시드한다(4.8.2 src/random.cc의 default_seed. 2026-10-29 CPU에서 이 값을
# 두 번 걸어 표본이 서로 다름을 확인). 요약에는 시드가 적히는데 표본은 실행마다 달라지는 일이 없게, --seed는
# 0~SEED_MAX만 받는다.
SEED_MAX = 2**32 - 2


# 명령줄 진입점. --model은 변환된 CTranslate2 폴더, --name은 리포트 이름(필수), --fallback은 '재시도 켬'. test
# 묶음 보호(--allow-test)와 덮어쓰기 보호(--overwrite)는 evaluate.py와 같다. --no-report는 속도만 볼 때 쓴다.
# --compute-type과 --device는 12단계에서 더했다(기본값은 지금까지의 측정과 같다). --seed는 2026-10-29에
# 더했다(기본 None = 지금까지처럼 시드 없음).
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="CTranslate2 모델 폴더")
    parser.add_argument("--name", required=True, help="리포트 이름(reports/<이름>/)")
    parser.add_argument("--set", required=True, dest="set_name")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        help="재시도(--fallback)의 temperature 표본 시드(0~4294967294). 모델을 띄우기 전에 ctranslate2의 "
        "난수 시드를 한 번 정해, 같은 명령을 다시 돌리면 같은 표본을 뽑는다. CTranslate2는 작업 스레드의 "
        "난수 생성기를 처음 만들 때의 시드로 계속 쓰므로 발화마다 다시 정하지는 못한다(표본은 앞 발화들이 "
        "쓴 난수에 이어진다). 재시도 끔에서는 표본이 없어 영향이 없다. 기본은 기록된 측정과 같이 시드 없음",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="이미 있는 리포트를 바꿔 쓴다")
    parser.add_argument("--no-report", action="store_true", help="요약만 찍고 리포트는 안 쓴다(속도 측정)")
    parser.add_argument(
        "--compute-type",
        choices=COMPUTE_TYPES,
        default="float16",
        help="CTranslate2가 불러올 때 쓸 정밀도(기본 float16은 기록과 같다. 12단계 후보는 int8_float16)",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda", help="cpu는 확인용(--no-report 필요)"
    )
    args = parser.parse_args()
    if args.set_name.endswith("-test") and not args.allow_test:
        parser.error("test sets are measured once per stage; pass --allow-test when the stage is done")
    # CPU 실행은 측정 조건이 아니므로 리포트를 남기지 못하게 한다(evaluate_qwen과 같은 보호).
    if args.device == "cpu" and not args.no_report:
        parser.error("a CPU run is not the measured condition; pass --no-report")
    # CTranslate2가 시드로 쓰지 못하는 값(위 SEED_MAX)은 시드를 걸거나 모델을 띄우기 전에 거부한다.
    if args.seed is not None and not 0 <= args.seed <= SEED_MAX:
        parser.error(f"--seed must be between 0 and {SEED_MAX}")
    target = REPORTS / args.name / f"{args.set_name}.json"
    if not args.no_report and target.exists() and not args.overwrite:
        parser.error(f"{target.relative_to(ROOT)} exists; pass --overwrite to measure it again")
    # 측정 코드의 커밋은 모델이나 오디오를 읽기 전에 기록한다(evaluate.py와 같다).
    commit = git_commit()
    # --seed가 있으면 모델을 띄우기 전에 CTranslate2의 난수 시드를 정한다. 디코딩하는 작업 스레드의 난수
    # 생성기는 처음 쓸 때 그때의 시드로 한 번만 만들어지므로, 이 호출은 모델·첫 디코딩보다 앞서야 하고
    # 한 번이면 된다(뒤에서 다시 불러도 이미 만든 생성기는 그대로다. 위 docstring). 기본(None)은 기록된 측정과
    # 같이 시드를 걸지 않는다(그때 CTranslate2는 생성기마다 std::random_device로 시드한다).
    if args.seed is not None:
        ctranslate2.set_random_seed(args.seed)

    # 모델을 오디오보다 먼저 띄운다(위 docstring). 설정 파일의 멜 수가 모델과 맞지 않으면 오디오를 읽기 전에
    # 바로 멈춘다. 2026-09-20 시도는 특징을 다 만든 뒤 첫 발화에서 멈췄다(docs/experiments.md '추가 측정').
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    # 기본 특징 추출기를 변환된 모델의 멜 수(model.model.n_mels, turbo는 128)에 맞춘 것으로 바꾼다.
    # faster-whisper는 폴더에 preprocessor_config.json이 없으면 멜 80개로 특징을 만든다(ct2_features).
    model.feature_extractor = FeatureExtractor(**feature_kwargs(Path(args.model), model.model.n_mels))
    # 발화가 없는 묶음은 오디오를 읽기 전에 멈춘다(evaluate와 같은 보호. 모델은 위 순서대로 이미 띄웠다).
    language, utterances = load_set(args.set_name)
    if not utterances:
        parser.error(
            f"set {args.set_name} has no utterances; rebuild the sets with `python -m whisper_ko_ft.prepare`"
        )
    if args.limit:
        utterances = utterances[: args.limit]
    # 변환 폴더의 지문: model.bin(turbo는 약 1.6 GB)과 config.json의 SHA-256. 리포트만으로 어느 변환으로
    # 쟀는지 알 수 있게 남긴다(outputs/turbo-n/ct2는 같은 경로에 다시 만든 이력이 있다). 발화 지연과 wall 시계
    # 밖에서 한 번만 계산한다. model.bin은 방금 모델을 띄우며 읽어 보통 운영체제 파일 캐시에 있다(turbo-n/ct2:
    # 캐시에 있으면 약 2초, HDD에서 처음 읽으면 20초 안팎. 2026-10-24에 18초, 10-29에 24초). 시계 밖이라
    # 배속·지연 수치에는 들어가지 않지만 프로세스가 도는 시간은 그만큼 길어져, scripts/speed.py가 실행
    # 전체에서 내는 GPU·CPU 사용률 평균(gpu_util_avg_during 등)에는 이 시간이 들어간다.
    hashes = converted_model_hashes(Path(args.model))
    # 재시도 끔/켬 조건을 고르고, 저장소별 memmap을 열어 둘 dict를 만든다(저장소마다 처음 읽을 때 한 번 연다).
    options = FALLBACK if args.fallback else NO_FALLBACK
    stores: dict[str, AudioStore] = {}

    rows: list[dict] = []
    utterance_seconds: list[float] = []
    started = time.perf_counter()
    # 한 발화씩 처리한다(배치 없음). 5단계 규칙의 '한 발화씩'이고, 배치 1 속도를 Transformers와 비교하는
    # 조건이기도 하다.
    for n, utterance in enumerate(utterances):
        # 한 발화의 지연: 오디오 읽기부터 구간을 끝까지 꺼낼 때까지(채점 제외). evaluate의 배치 시간과 같은
        # 범위다.
        utterance_started = time.perf_counter()
        # 2026-10-29까지 쓴 stores.setdefault(utterance.store, AudioStore(utterance.store))는 기본값 인자를
        # 키가 있어도 먼저 계산해 발화마다 memmap을 새로 열었고, 그 시간이 이 지연 안에 들어갔다. 이제는
        # 저장소가 처음 나올 때만 연다(읽는 표본은 같다).
        if utterance.store not in stores:
            stores[utterance.store] = AudioStore(utterance.store)
        audio = stores[utterance.store].read(utterance)
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
        utterance_seconds.append(time.perf_counter() - utterance_started)
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
    # mel_bins는 실제로 쓴 멜 수다. 엔진 버전도 남긴다. model_sha256·config_sha256(변환 폴더 지문)과
    # seed(--seed 값, 없으면 null)는 2026-10-29에 더한 키라 그 전의 리포트에는 없다.
    # empty_hypotheses는 evaluate.decoding_counts와 같은 기준으로 센다: 채점한 발화(kept)만, 공백만 남은
    # 가설도 빈 가설로. 2026-10-13 전에는 정답이 빈 발화까지 세고 공백 가설은 세지 않았다. 커밋된
    # faster-whisper 리포트에는 정답이 빈 발화도 공백 가설도 없어, 발화 행으로 다시 세면 두 기준 모두 0이다.
    summary = {
        "model": args.model,
        **hashes,
        "engine": "faster-whisper",
        "compute_type": args.compute_type,
        "device": args.device,
        "fallback": args.fallback,
        "seed": args.seed,
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
        **batch_seconds_summary(utterance_seconds),
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
    # report_json은 요약에 NaN(채점한 발화가 없을 때의 0 ÷ 0)이 있으면 ValueError로 멈춰 파일을 남기지 않는다.
    target.write_text(report_json(payload), encoding="utf-8", newline="\n")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
