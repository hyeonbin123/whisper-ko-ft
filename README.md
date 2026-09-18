# whisper-ko-ft

공개 한국어 음성 데이터(Zeroth-Korean, 51.6시간)로 Whisper를 파인튜닝하고, **미리 정한 규칙으로 전후를 측정**하는 프로젝트. 같은 도메인에서 얼마나 좋아지는지뿐 아니라 다른 도메인(FLEURS)과 영어에서 무엇을 잃는지, 전화 음질(8kHz)에서는 어떻게 되는지를 함께 잰다.

진행 중이다. 계획은 [docs/plan.md](docs/plan.md), 측정 규칙과 결과는 [docs/experiments.md](docs/experiments.md).

## 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 0. 기반 | 데이터 받기, 화자 기준 분리, 정규화 규칙, 측정 스크립트 | 완료 |
| 1. 기준선 | whisper-small, whisper-large-v3-turbo를 그대로 쓴 오류율·속도·GPU 메모리 | 완료 |
| 2. small 전체 파인튜닝 | 후보 설정 중 validation으로 선택, 다른 도메인·영어 확인 | 진행 중 |
| 3. turbo LoRA | 같은 절차 | 예정 |
| 4. 전화 음질 | 8kHz·μ-law 조건의 기준선과, 그 조건을 섞어 학습한 모델 | 예정 |
| 5. 정리 | 결과표, 한계, 추론 속도 | 예정 |

## 지금까지의 결과

기준선(모델을 그대로 쓴 경우, validation, RTX 2080 Ti). 대괄호는 95% 부트스트랩 구간.

| 묶음 | whisper-small (244M) | whisper-large-v3-turbo (809M) |
|---|---|---|
| Zeroth validation 2,214발화, CER | 9.62% [9.19, 10.11] | 4.15% [3.88, 4.42] |
| 〃 숫자 발화 705개 제외, CER | 6.03% [5.74, 6.31] | 2.18% [2.00, 2.36] |
| FLEURS 한국어 validation, CER | 8.07% [6.89, 9.34] | 4.75% [3.68, 5.91] |
| FLEURS 영어 validation, WER | 7.35% [6.52, 8.32] | 5.50% [4.71, 6.32] |
| 속도 (배치 16, 음성 초 ÷ 걸린 초) | 103 | 87 |
| GPU 메모리 최대 (추론) | 1,548MB | 2,525MB |

- Zeroth-Korean의 정답은 숫자를 한글로 풀어 쓰고("오 월") Whisper는 아라비아 숫자("5월")로 쓴다. 이 표기 차이가 전체 오류율의 상당 부분이었다(발화의 32%). 그래서 모든 Zeroth 수치를 "전체"와 "숫자 발화 제외" 두 가지로 보고하고, 파인튜닝의 판정도 두 수치 모두에 조건을 건다
- turbo는 크기가 3배지만 배치 추론 속도는 small과 비슷했다

## 실행

준비: NVIDIA GPU(11GB에서 확인), Python 3.11, [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                       # torch는 CUDA 12.8 빌드를 받는다 (약 2.5GB)
uv run python -m whisper_ko_ft.download       # Zeroth-Korean, FLEURS 한국어·영어 (약 4GB)
uv run python -m whisper_ko_ft.prepare        # 16kHz 오디오 저장소와 학습·validation·test 목록 (약 6.5GB)
uv run python -m whisper_ko_ft.evaluate --model openai/whisper-small --set zeroth-val
uv run python -m whisper_ko_ft.analyze table --set zeroth-val --reports whisper-small whisper-large-v3-turbo
uv run python -m whisper_ko_ft.train --run-name small-a --learning-rate 1e-5   # 3,000스텝, 이 PC에서 몇 시간
```

- `prepare`는 Zeroth-Korean train에서 화자 10명을 validation으로 떼어 낸다 (화자 ID의 SHA-256 순, 규칙은 experiments.md)
- `evaluate`는 `reports/<이름>/<묶음>.json`에 요약과 발화별 정답·가설·오류 수를 남긴다. test 묶음은 `--allow-test`를 줘야 잰다 (단계마다 한 번만 재기 위한 장치)

테스트: `uv run pytest`, `uv run ruff check .`

## 데이터와 모델

| 쓰임 | 이름 | 라이선스 | 원본 |
|---|---|---|---|
| 학습·평가 | Zeroth-Korean | CC BY 4.0 | [openslr.org/40](https://openslr.org/40/), [kresnik/zeroth_korean](https://huggingface.co/datasets/kresnik/zeroth_korean) |
| 평가 | FLEURS (한국어, 영어) | CC BY 4.0 | [google/fleurs](https://huggingface.co/datasets/google/fleurs) |
| 모델 | Whisper small, large-v3-turbo | MIT | [openai/whisper-small](https://huggingface.co/openai/whisper-small), [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |

데이터와 학습 결과물은 저장소에 넣지 않는다.
