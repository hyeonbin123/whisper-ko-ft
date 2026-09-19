# whisper-ko-ft

공개 한국어 음성 데이터(Zeroth-Korean, 51.6시간)로 Whisper를 파인튜닝하고, **미리 정한 규칙으로 전후를 측정**하는 프로젝트. 같은 도메인에서 얼마나 좋아지는지뿐 아니라 다른 도메인(FLEURS)과 영어에서 무엇을 잃는지, 전화 음질(8kHz)에서는 어떻게 되는지를 함께 잰다.

진행 중이다. 계획은 [docs/plan.md](docs/plan.md), 측정 규칙과 결과는 [docs/experiments.md](docs/experiments.md).

## 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 0. 기반 | 데이터 받기, 화자 기준 분리, 정규화 규칙, 측정 스크립트 | 완료 |
| 1. 기준선 | whisper-small, whisper-large-v3-turbo를 그대로 쓴 오류율·속도·GPU 메모리 | 완료 |
| 2. small 전체 파인튜닝 | 후보 설정 중 validation으로 선택, 다른 도메인·영어 확인 | 완료 (도메인 전용) |
| 3. turbo LoRA | 같은 절차 | 완료 (도메인 전용) |
| 4. 전화 음질 | 8kHz·μ-law 조건의 기준선과, 그 조건을 섞어 학습한 모델 | 예정 |
| 5. 정리 | 결과표, 한계, 추론 속도 | 예정 |

## 지금까지의 결과

RTX 2080 Ti, 대괄호는 95% 부트스트랩 구간. 전체 표와 규칙은 [docs/experiments.md](docs/experiments.md).

**test (단계마다 한 번 측정)**

| 묶음 | whisper-small | whisper-large-v3-turbo | 파인튜닝한 small | turbo + LoRA |
|---|---|---|---|---|
| Zeroth test 457발화, CER | 10.31% [9.39, 11.29] | 4.48% [3.77, 5.32] | 3.78% [2.00, 6.61] | **1.96%** [1.36, 2.66] |
| 〃 숫자 발화 142개 제외 | 7.36% [6.30, 8.64] | 3.29% [2.34, 4.50] | 4.68% [2.01, 10.37] | **2.30%** [1.44, 3.45] |
| FLEURS 한국어 test, CER | 7.62% [6.56, 8.73] | **5.21%** [4.19, 6.16] | 11.58% [9.30, 14.66] | 6.94% [5.73, 8.08] |
| FLEURS 영어 test, WER | 7.08% [6.38, 7.76] | **4.95%** [4.51, 5.42] | 7.63% [6.89, 8.39] | 5.23% [4.78, 5.71] |
| 배치 1 속도 (음성 초 ÷ 걸린 초) | 12.1 | 21.3 | 12.0 | turbo와 같은 구조 |
| 배치 1 GPU 메모리 | 551MB | 1,599MB | 550MB | 〃 |

**3단계(turbo LoRA)의 판정: 도메인 전용.** 어텐션과 MLP에 LoRA(rank 16)를 붙여 1,500스텝 학습했다. validation에서 같은 도메인 CER은 4.15% → 1.24%(숫자 발화 제외 2.18% → 1.26%)로 줄었고, FLEURS 한국어는 4.75% → 7.08%로 허용 폭을 넘었다. 되풀이 오류는 없었다. test에서는 숫자가 없는 FLEURS 발화도 조금 나빠졌다(+0.53%p [+0.24, +0.85]).

**2단계(whisper-small 전체 파인튜닝)의 판정: 도메인 전용.** validation에서 같은 도메인 CER은 9.62% → 1.63%(숫자 발화 제외 6.03% → 1.67%)로 줄었지만, FLEURS 한국어는 8.07% → 10.77%로 허용 폭(1.0%p)을 넘게 나빠졌다.

- **표기가 결과를 크게 좌우했다.** Zeroth-Korean의 정답은 숫자를 한글로 풀어 쓰고("오 월") Whisper는 아라비아 숫자("5월")로 쓴다. 기준선에서는 이 차이가 오류로 잡혔고(발화의 32%), 파인튜닝한 모델은 반대로 FLEURS의 "1978년"을 "천 구백 칠십 팔 년"으로 적어 오류가 됐다. FLEURS 한국어의 악화는 모두 정답에 숫자가 있는 발화에서 나왔고, 나머지 발화에서는 차이가 없었다(validation +0.07%p, test +0.42%p, 둘 다 구간이 0을 포함)
- **파인튜닝한 모델에서만 되풀이 오류가 나왔다.** test에서 한 발화가 같은 구절을 끝없이 되풀이해 오류의 37%를 차지했다. 이 발화를 빼면 Zeroth test CER은 2.39%다. 기준선 모델에는 없던 현상이다
- **작은 validation 묶음으로 골랐다면 다른 후보를 골랐을 것이다.** 학습 중 440발화로는 학습률 3e-5가 나았지만, 2,214발화 전체에서는 같았고 되풀이 오류 하나가 그 후보 오류의 21%였다
- 한국어만 학습한 비용은 작았다. FLEURS 영어 WER은 small 전체 파인튜닝에서 +0.55%p, turbo LoRA에서 +0.28%p(test)
- turbo는 크기가 3배지만 한 발화씩 처리할 때는 small보다 1.8배 빠르다(디코더가 4층). 파인튜닝한 small이 turbo를 대신할 수 있는 경우는 GPU 메모리가 묶여 있고 도메인과 표기가 정해진 때뿐이다

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
