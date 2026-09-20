# whisper-ko-ft

공개 한국어 음성 데이터(Zeroth-Korean, 51.6시간)로 Whisper를 파인튜닝하고, **미리 정한 규칙으로 전후를 측정**하는 프로젝트. 같은 도메인에서 얼마나 좋아지는지뿐 아니라 다른 도메인(FLEURS)과 영어에서 무엇을 잃는지, 전화 음질(8kHz)에서는 어떻게 되는지를 함께 잰다.

계획은 [docs/plan.md](docs/plan.md), 측정 규칙과 결과는 [docs/experiments.md](docs/experiments.md).

## 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 0. 기반 | 데이터 받기, 화자 기준 분리, 정규화 규칙, 측정 스크립트 | 완료 |
| 1. 기준선 | whisper-small, whisper-large-v3-turbo를 그대로 쓴 오류율·속도·GPU 메모리 | 완료 |
| 2. small 전체 파인튜닝 | 후보 설정 중 validation으로 선택, 다른 도메인·영어 확인 | 완료 (도메인 전용) |
| 3. turbo LoRA | 같은 절차 | 완료 (도메인 전용) |
| 4. 전화 음질 | 8kHz·μ-law 조건의 기준선과, 그 조건을 섞어 학습한 모델 | 완료 (섞어 학습하는 것이 이득, 크기는 작음) |
| 5. 되풀이 오류와 추론 엔진 | faster-whisper의 재시도가 되풀이 오류를 막는지 | 완료 (막는다) |
| 6. 정답의 숫자 표기를 바꿔 학습 | 학습 정답의 "이천 십 팔 년"을 "2018년"으로 바꾸면 다른 도메인의 악화가 없어지는지 (처음 계획에 없던 단계) | 완료 (범용으로 쓸 수 있다) |
| 7. 더 오래 학습 | 6단계 모델에서 이어서 1,500스텝을 더 학습하면 나아지는지 (처음 계획에 없던 단계) | 완료 (1,500스텝으로 충분하다) |

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

**7단계(더 오래 학습)의 판정: 1,500스텝으로 충분하다.** 6단계 모델에서 이어서 학습하자 validation의 맞춘 CER은 1.25% → 1.41%였다(차이의 구간이 0을 포함). 더 학습한 모델이 한 발화에서 같은 구절을 되풀이해 오류의 17%를 냈고, 그 발화를 빼면 0.07%p 나았다. test에서는 거꾸로 6단계 모델이 되풀이해 순서가 뒤집혔다(2.68%와 2.01%). 추가 학습의 이득은 0.1%p 안쪽이고, 되풀이 오류 한 번이 그보다 크다. FLEURS 한국어는 더 학습할수록 조금씩 나빠졌다(test +0.18%p).

**6단계(숫자 표기)의 판정: 표기를 바꿔 학습하면 범용으로 쓸 수 있다.** 2·3단계의 "도메인 전용" 판정은 대부분 표기에서 나왔다. 학습 정답의 숫자를 규칙으로 아라비아 숫자로 바꾸고(20,049문장 중 4,703문장) 3단계와 같은 설정으로 다시 학습했다. Zeroth는 정답과 가설의 숫자 표기를 같은 규칙으로 맞춘 뒤 잰 "맞춘 CER"이다.

| test | whisper-large-v3-turbo | turbo + LoRA (3단계) | turbo + LoRA, 숫자를 바꿔 학습 |
|---|---|---|---|
| Zeroth test, 맞춘 CER | 4.22% [3.53, 5.00] | 2.04% [1.47, 2.76] | 2.68% [1.48, 4.62] |
| 〃 되풀이 오류가 난 한 발화 제외 (사후 분석) | | 1.80% | 1.82% |
| FLEURS 한국어 test, CER | 5.21% [4.19, 6.16] | 6.94% [5.73, 8.08] | 5.55% [4.41, 6.59] |
| 〃 정답에 숫자가 있는 86발화 | 9.15% | 14.96% | 9.12% |
| FLEURS 영어 test, WER | 4.95% [4.51, 5.42] | 5.23% [4.78, 5.71] | 5.31% [4.75, 5.84] |

validation에서 맞춘 CER은 3.60% → 1.25%(3단계 모델 1.28%)로 같은 도메인의 이득이 그대로였고, FLEURS 한국어의 악화는 +2.33%p에서 +0.53%p로 줄어 허용 폭 안에 들어왔다. test에서는 한 발화가 같은 음절을 끝까지 되풀이해 Zeroth의 구간이 넓어졌다(그 발화가 오류의 32%). 숫자가 없는 FLEURS 발화의 작은 악화(+0.45%p)는 표기를 바꿔도 그대로 남았다.

**5단계(되풀이 오류)의 판정: 추론 엔진의 재시도가 막는다.** 파인튜닝한 small을 faster-whisper로 돌리면 재시도를 껐을 때 2,214발화 중 1개가 같은 구절을 되풀이했고(오류 199~231개), 재시도를 켜면 그 한 발화만 다시 디코딩되어 0개가 됐다. 다른 발화와 속도는 그대로다. 되풀이에 빠지는 발화는 엔진(Transformers, faster-whisper)에 따라 달라서, 특정 발화가 아니라 모델의 성질로 봐야 한다.

**4단계(전화 음질)의 판정: 섞어 학습하는 것이 이득.** 음성을 300~3400Hz, 8kHz, μ-law로 바꾼 조건이다(실제 통화 녹음이 아니다).

| Zeroth test, CER | whisper-large-v3-turbo | turbo + LoRA | turbo + LoRA, 절반을 전화 음질로 학습 |
|---|---|---|---|
| 깨끗한 음성 | 4.48% [3.77, 5.32] | 1.96% [1.36, 2.66] | 1.71% [1.29, 2.25] |
| 전화 음질 | 6.38% [5.46, 7.35] | 2.52% [2.01, 3.13] | 2.08% [1.62, 2.64] |

이득의 대부분은 도메인 파인튜닝에서 나온다. 깨끗한 음성으로만 학습해도 전화 음질에서 기준선보다 훨씬 낮고, 전화 음질을 섞으면 그 위에 작은 몫이 더해진다(validation에서 악화 0.58%p 중 0.21%p를 되돌림). 깨끗한 음성의 오류율은 나빠지지 않았다.

**3단계(turbo LoRA)의 판정: 도메인 전용.** 어텐션과 MLP에 LoRA(rank 16)를 붙여 1,500스텝 학습했다. validation에서 같은 도메인 CER은 4.15% → 1.24%(숫자 발화 제외 2.18% → 1.26%)로 줄었고, FLEURS 한국어는 4.75% → 7.08%로 허용 폭을 넘었다. 되풀이 오류는 없었다. test에서는 숫자가 없는 FLEURS 발화도 조금 나빠졌다(+0.53%p [+0.24, +0.85]).

**2단계(whisper-small 전체 파인튜닝)의 판정: 도메인 전용.** validation에서 같은 도메인 CER은 9.62% → 1.63%(숫자 발화 제외 6.03% → 1.67%)로 줄었지만, FLEURS 한국어는 8.07% → 10.77%로 허용 폭(1.0%p)을 넘게 나빠졌다.

- **표기가 결과를 크게 좌우했다.** Zeroth-Korean의 정답은 숫자를 한글로 풀어 쓰고("오 월") Whisper는 아라비아 숫자("5월")로 쓴다. 기준선에서는 이 차이가 오류로 잡혔고(발화의 32%), 파인튜닝한 모델은 반대로 FLEURS의 "1978년"을 "천 구백 칠십 팔 년"으로 적어 오류가 됐다. FLEURS 한국어의 악화는 모두 정답에 숫자가 있는 발화에서 나왔고, 나머지 발화에서는 차이가 없었다(validation +0.07%p, test +0.42%p, 둘 다 구간이 0을 포함)
- **파인튜닝한 모델에서만 되풀이 오류가 나왔다.** test에서 한 발화가 같은 구절을 끝없이 되풀이해 오류의 37%를 차지했다. 이 발화를 빼면 Zeroth test CER은 2.39%다. 기준선 모델에는 없던 현상이다
- **작은 validation 묶음으로 골랐다면 다른 후보를 골랐을 것이다.** 학습 중 440발화로는 학습률 3e-5가 나았지만, 2,214발화 전체에서는 같았고 되풀이 오류 하나가 그 후보 오류의 21%였다
- 한국어만 학습한 비용은 작았다. FLEURS 영어 WER은 small 전체 파인튜닝에서 +0.55%p, turbo LoRA에서 +0.28%p(test)
- turbo는 크기가 3배지만 한 발화씩 처리할 때는 small보다 1.8배 빠르다(디코더가 4층). 파인튜닝한 small이 turbo를 대신할 수 있는 경우는 GPU 메모리가 묶여 있고 도메인과 표기가 정해진 때뿐이다

## 한계
- **데이터**: 읽는 말투 한 종류(Zeroth-Korean, 뉴스 문장 낭독)로만 학습했다. 대화체, 실제 통화, 잡음이 있는 환경에서는 재지 않았다. 전화 음질은 공개 데이터를 변환한 것이지 통화 녹음이 아니다
- **Whisper가 이미 본 데이터일 수 있다**: Whisper의 학습 데이터는 공개되지 않았다. Zeroth-Korean이나 FLEURS가 들어 있었는지 확인할 방법이 없다
- **작은 test 묶음**: Zeroth test는 457발화, 화자 10명이다. 발화 하나가 구간을 크게 흔들었다(2단계). 세 모델이 똑같이 틀린, 정답과 녹음이 맞지 않아 보이는 발화도 하나 있다(빼지 않았다)
- **규칙에 없던 분석**: FLEURS를 숫자 유무로 나눈 분석은 2단계 결과를 본 뒤에 했다. 사후 분석이라고 표시했고 판정은 바꾸지 않았다. 3단계부터는 재기 전에 규칙에 넣었다
- **학습 설정을 넓게 찾지 않았다**: 후보는 단계마다 한두 개다. 7단계에서 이어서 학습해 본 것은 학습률 일정이 두 번 도는 방식이라, 처음부터 길게 잡은 학습과 같지 않다. 더 나은 설정이 있을 수 있다
- **한 대의 PC**: RTX 2080 Ti(fp16만 지원) 한 장에서 학습하고 쟀다. 일부 측정은 같은 GPU를 다른 프로그램이 쓰는 동안 했고, 그런 측정의 속도 수치는 쓰지 않았다
- **표기**: 2~5단계의 파인튜닝한 모델은 숫자와 영문 약어를 한글로 풀어 쓴다. 6단계에서 학습 정답의 숫자만 규칙으로 바꿔 보았다. 규칙은 단위 앞의 숫자만 바꾸므로 빠뜨리는 것이 있고(학습 정답에 두 표기가 섞여 있다), 영문 약어와 인식 결과를 되돌리는 후처리는 다루지 않았다
- **되풀이 오류는 turbo LoRA에서도 나왔다**: 6단계와 7단계 모델이 2,671발화 중 한 번씩, 서로 다른 발화에서 냈다. 5단계의 재시도가 이 모델에서도 막는지는 재지 않았다

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

faster-whisper로 재기 (5단계):

```bash
uv sync --group ct2
uv run ct2-transformers-converter --model outputs/small-a/best --output_dir outputs/small-a/ct2 --quantization float16 --copy_files tokenizer.json
uv run python -m whisper_ko_ft.evaluate_ct2 --model outputs/small-a/ct2 --name small-a-ct2-fallback --set zeroth-val --fallback
```

LoRA 후보와 전화 음질 조건의 명령은 [docs/experiments.md](docs/experiments.md)의 각 단계 설정과 `train.py`, `evaluate.py`의 도움말(`--lora-r`, `--lora-targets`, `--telephone-prob`, `--adapter`, `--channel telephone`)을 따른다.

6단계: `uv run python -m whisper_ko_ft.prepare_itn`이 숫자를 바꾼 학습·validation 목록(`zeroth-train-itn`, `zeroth-val500-itn`)을 만들고, `train`에 `--train-set`, `--eval-set`으로 준다. 맞춘 CER은 `analyze table --harmonized`, 판정은 `analyze verdict6`.

테스트: `uv run pytest`, `uv run ruff check .`

## 데이터와 모델

| 쓰임 | 이름 | 라이선스 | 원본 |
|---|---|---|---|
| 학습·평가 | Zeroth-Korean | CC BY 4.0 | [openslr.org/40](https://openslr.org/40/), [kresnik/zeroth_korean](https://huggingface.co/datasets/kresnik/zeroth_korean) |
| 평가 | FLEURS (한국어, 영어) | CC BY 4.0 | [google/fleurs](https://huggingface.co/datasets/google/fleurs) |
| 모델 | Whisper small, large-v3-turbo | MIT | [openai/whisper-small](https://huggingface.co/openai/whisper-small), [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |

데이터와 학습 결과물은 저장소에 넣지 않는다.
