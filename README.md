# whisper-ko-ft

공개 한국어 음성 데이터(Zeroth-Korean, 51.6시간)로 Whisper를 파인튜닝하고, **미리 정한 규칙으로 전후를 측정**하는 프로젝트. 같은 도메인에서 얼마나 좋아지는지뿐 아니라 다른 도메인(FLEURS)과 영어에서 무엇을 잃는지, 전화 음질(8kHz)에서는 어떻게 되는지를 함께 잰다.

진행 중이다. 계획은 [docs/plan.md](docs/plan.md), 측정 규칙과 결과는 [docs/experiments.md](docs/experiments.md).

## 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 0. 기반 | 데이터 받기, 화자 기준 분리, 정규화 규칙, 측정 스크립트 | 완료 |
| 1. 기준선 | whisper-small, whisper-large-v3-turbo를 그대로 쓴 오류율·속도·GPU 메모리 | 진행 중 |
| 2. small 전체 파인튜닝 | 후보 설정 중 validation으로 선택, 다른 도메인·영어 확인 | 예정 |
| 3. turbo LoRA | 같은 절차 | 예정 |
| 4. 전화 음질 | 8kHz·μ-law 조건의 기준선과, 그 조건을 섞어 학습한 모델 | 예정 |
| 5. 정리 | 결과표, 한계, 추론 속도 | 예정 |

## 실행

준비: NVIDIA GPU(11GB에서 확인), Python 3.11, [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                       # torch는 CUDA 12.8 빌드를 받는다 (약 2.5GB)
uv run python -m whisper_ko_ft.download       # Zeroth-Korean, FLEURS 한국어·영어 (약 4GB)
uv run python -m whisper_ko_ft.prepare        # 16kHz 오디오 저장소와 학습·validation·test 목록 (약 6.5GB)
uv run python -m whisper_ko_ft.evaluate --model openai/whisper-small --set zeroth-val
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
