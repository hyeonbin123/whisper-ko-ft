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
| 8. 재시도 디코딩과 출력 숫자 변환 | 가중치는 그대로 두고 Transformers의 재시도 디코딩이 되풀이 오류를 막는지, 출력의 숫자 표기만 바꾸는 후처리가 다른 도메인의 격차를 줄이는지 (처음 계획에 없던 단계) | 완료 (재시도 디코딩을 쓴다, 출력 숫자 변환을 쓴다) |
| 9. 다른 기준 모델 | 계열이 다른 공개 모델 Qwen3-ASR-1.7B(Apache 2.0)를 학습 없이 같은 묶음·채점으로 재서, 다른 도메인(FLEURS 한국어)에서 기준선 turbo보다 나은지와 속도·GPU 메모리 (처음 계획에 없던 단계) | 완료 (다른 도메인에서 더 나은 기준 모델이다. 학습 데이터에 FLEURS가 들어 있었을 수 있음) |
| 10. 다른 기준 모델에 LoRA | 9단계의 Qwen3-ASR-1.7B에 6단계와 같은 데이터·같은 방식(fp16, 숫자를 바꾼 정답)으로 LoRA를 학습해, 같은 도메인에서는 6단계 모델만큼, 다른 도메인에서는 그보다 나은 범용 모델이 되는지 (처음 계획에 없던 단계) | 완료 (N을 유지한다. 같은 도메인은 6단계 모델보다 낮지만 다른 도메인 이득 없음) |

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

**10단계(다른 기준 모델에 LoRA, 2026-10-05)의 판정: N을 유지한다 (다른 도메인 이득 없음).** 9단계의 Qwen3-ASR-1.7B에 6단계 모델(N)과 같은 데이터(숫자를 바꾼 Zeroth 정답 20,049문장)·같은 종류의 LoRA 위치(오디오 인코더와 언어 모델의 어텐션·MLP, rank 16)·같은 스텝 수(1,500)로 LoRA를 학습했다(fp16 기본 가중치, fp32 어댑터, GradScaler. 1,500스텝에서 건너뜀 한 번, GPU 공유 메모리 넘침 없음, torch 최대 할당 5.1GB, 약 5.7시간). 같은 도메인에서는 N보다 낮았지만(조건 1 만족), 다른 도메인에서는 학습 전의 이점을 잃어 N과 비슷해져 조건 2(FLEURS 한국어에서 N보다 낮다)를 만족하지 못했다.

| | turbo + LoRA (6단계, N) | Qwen3-ASR-1.7B (학습 없이) | **Qwen3-ASR-1.7B + LoRA** |
|---|---|---|---|
| Zeroth validation 맞춘 CER (조건 1) | 1.25% | 3.45% | **0.92%** [0.83, 1.02] (N − 0.33%p [−0.42, −0.24]) |
| FLEURS 한국어 validation CER (조건 2) | 5.28% | **4.22%** | 5.40% [4.15, 6.73] (N + 0.12%p [−0.43, +0.70]) |
| FLEURS 영어 validation WER | 6.09% | **4.40%** | 4.91% [4.03, 5.77] |
| Zeroth test 맞춘 CER (참고) | 2.68% (재시도 디코딩 1.98%) | 4.24% | **1.52%** [1.01, 2.14] |
| FLEURS 한국어 test CER (참고) | 5.55% | **4.47%** | 5.68% [4.60, 6.74] |
| 전화 음질 Zeroth validation 맞춘 CER (판정 없음) | | 3.81% | **1.34%** (4단계 전화 음질 섞어 학습한 turbo 1.67%) |

FLEURS 한국어에서 학습 전 모델보다 +1.18%p [+0.59, +1.84] 나빠져 미리 정한 망각 허용 폭(1.0%p)을 넘었다. 숫자가 없는 발화와 표기 무관 채점에서도 나빠져 표기만의 문제는 아니다. 속도는 새로 재지 않았고, 병합한 모델은 학습 전 Qwen과 구조가 같아 9단계의 수치(한 발화씩 처리하면 turbo보다 약 6.5배 느림, CTranslate2 경로 없음)가 그대로 적용된다. FLEURS 수치에는 9단계와 같은 단서(Qwen의 학습 데이터에 FLEURS가 들어 있었을 수 있음)가 붙는다.

**9단계(다른 기준 모델, 2026-10-04)의 판정: 다른 도메인에서 더 나은 기준 모델이다.** 계열이 다른 공개 모델 Qwen3-ASR-1.7B(Apache 2.0, 오디오 인코더 + Qwen3 언어 모델, 약 20억 파라미터)를 학습 없이 같은 묶음·같은 채점으로 쟀다(fp16, greedy, 언어 고정, 앞 30초). FLEURS 한국어 validation CER이 기준선 turbo 4.75%에서 4.22%로 낮았고(−0.53%p [−0.99, −0.11]), 미리 정한 기준(차이의 구간 상한 < 0)을 만족했다.

| | whisper-large-v3-turbo | turbo + LoRA (6단계) | Qwen3-ASR-1.7B (학습 없이) |
|---|---|---|---|
| FLEURS 한국어 validation CER (판정) | 4.75% | 5.28% | **4.22%** [3.25, 5.31] |
| FLEURS 한국어 test CER (참고) | 5.21% [4.19, 6.16] | 5.55% | **4.47%** [3.48, 5.46] |
| FLEURS 영어 test WER (참고) | 4.95% [4.51, 5.42] | 5.31% | **3.91%** [3.50, 4.34] |
| Zeroth validation 맞춘 CER | 3.60% | **1.25%** | 3.45% [3.25, 3.65] |
| Zeroth test 맞춘 CER (참고) | 4.22% | 2.68% (재시도 디코딩 1.98%) | 4.24% [3.47, 5.14] |
| 배치 1 속도 / 한 발화 지연 중앙값 (Zeroth val-500, 번갈아 잼) | 20.7 / 0.39초 | turbo와 같은 구조 | 3.2 / 2.54초 |
| 배치 1 GPU 메모리 (torch 최대 할당) | 1,599MB | 〃 | 4,062MB |

같은 도메인(Zeroth)에서는 학습 없이 기준선 turbo와 비슷하고(차이의 구간이 0을 포함), 이 도메인으로 학습한 turbo LoRA에는 크게 못 미친다. fp16으로 돌려도 넘침은 없었고(fp32와 CER 차이 0.05%p), 되풀이 오류도 없었다(공식 반복 제거 후처리는 한 발화도 바꾸지 않았다). 대신 이 하네스(Transformers)에서는 느리고 메모리를 더 쓴다. 한 발화씩 처리하면 turbo보다 약 6.5배 느리고, 배치 16에서는 약 2.7배 느리며 11GB 카드를 거의 채운다(Zeroth val-500에서 두 모델을 번갈아 두 번씩 잰 평균이다. 1회차의 일부만 다른 프로그램의 GPU 부하와 겹쳤고, 회차별 비율은 배치 1 6.1~6.8배, 배치 16 2.6~2.8배다. 다른 추론 엔진은 재지 않았다). 다만 Qwen3-ASR의 학습 데이터는 공개되지 않아 FLEURS와 Zeroth-Korean이 학습에 들어 있었을 수 있다. 그래서 "학습하지 않은 도메인에서 낫다"고 단정하지 않는다(아래 한계). 이 모델에 LoRA로 학습하는 것은 이번 단계 밖이다. 작은 판 Qwen3-ASR-0.6B(참고, validation만)는 한국어에서 기준선 turbo보다 나빴다(FLEURS 한국어 5.31%, Zeroth 맞춘 CER 5.01%).

**8단계(재시도 디코딩과 출력 숫자 변환, 2026-10-04)의 판정: 둘 다 쓴다.** 가중치는 그대로 두고 디코딩과 후처리만 바꿨다. Transformers의 재시도(temperature 0~1.0, 토큰 압축비가 1.35를 넘으면 그 발화만 다시 디코딩)를 켜자, 6·7단계 두 모델의 validation 5,668발화와 6단계 모델의 test 839발화 중 다시 디코딩된 것은 기록된 되풀이 2개뿐이었고 둘 다 되풀이가 없어졌다(7단계 모델은 test를 재지 않았다). 다른 발화의 가설은 하나도 바뀌지 않았고, 배치 1 속도도 줄지 않았다. 7단계 모델의 validation 맞춘 CER은 1.41% → 1.20%, 6단계 모델의 Zeroth test는 2.68% → 1.98%(참고용, 구간 [1.48, 2.58])다. 다만 되풀이가 없어진 두 발화도 오류가 남았다(`152_003_1752`는 정답에 없는 말을 지어냈고, `105_003_0478`은 기준선 turbo도 받아 적은, 녹음에 섞인 것으로 보이는 다른 말이 붙었다). 로그확률 기준을 더한 후보도 결과가 같았고(배치 16 실행 시간은 약 20% 길었다. 한 번씩 잰 참고값), 규칙대로 더 단순한 압축비 기준만 쓴다. 출력의 숫자 낱말만 아라비아 숫자로 바꾸는 후처리는 FLEURS 한국어 validation에서 6단계 모델을 5.28% → 4.91%로 낮춰 기준선(4.75%)과의 격차 +0.53%p 중 숫자 몫을 없앴다. 이 효과는 규칙을 적기 전에 사후 계산으로 이미 알고 있었다. 그래서 이 판정은 눈을 가린 측정이 아니라, 미리 정한 규칙으로 커밋된 리포트에서 다시 계산한 확인이다. test(참고용)에서는 정답이 숫자를 한글로 적은 발화에서 생긴 오류와 상쇄되어 5.55% 그대로였다.

**추가 측정 (2026-10-02): 6단계 모델을 CTranslate2로.** 같은 가중치를 CTranslate2(fp16)로 바꾸면 한 발화씩 처리할 때 Transformers보다 1.46배 빨랐고(val-500에서 26.9 대 18.4배속. 배경화면 프로그램이 GPU를 쓰는 중에 번갈아 쟀으므로 비율만 결과로 본다) 오류율은 같았다. faster-whisper에서는 재시도를 꺼도 2,671발화 중 되풀이가 하나도 없어 재시도가 막는지는 판정하지 못했다(Transformers에서 되풀이한 test 발화 `105_003_0478`은 이 엔진에서 되풀이하지 않았지만, 맞춘 채점으로 정답 50글자에 오류가 20개라 여전히 많이 틀렸다). 새로 클론해 README대로 whisper-small 기준선을 다시 재자 2,214발화의 가설이 모두 커밋된 리포트와 같았다.

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
- **되풀이 오류는 주로 파인튜닝한 모델에서 나왔다.** test에서 한 발화가 같은 구절을 끝없이 되풀이해 오류의 37%를 차지했다. 이 발화를 빼면 Zeroth test CER은 2.39%다. 학습 전 small도 validation 한 발화(`217_003_1879`, "1. 1. 1. …")에서 같은 현상을 보였다 (2026-10-09 바로잡음: 처음에는 기준선 모델에는 없던 현상이라고 적었다, `reports/whisper-small/zeroth-val.json`)
- **작은 validation 묶음으로 골랐다면 다른 후보를 골랐을 것이다.** 학습 중 440발화로는 학습률 3e-5가 나았지만, 2,214발화 전체에서는 같았고 되풀이 오류 하나가 그 후보 오류의 21%였다
- 한국어만 학습한 비용은 작았다. FLEURS 영어 WER은 small 전체 파인튜닝에서 +0.55%p, turbo LoRA에서 +0.28%p(test)
- turbo는 크기가 3배지만 한 발화씩 처리할 때는 small보다 1.8배 빠르다(디코더가 4층). 파인튜닝한 small이 turbo를 대신할 수 있는 경우는 GPU 메모리가 묶여 있고 도메인과 표기가 정해진 때뿐이다

## 한계
- **데이터**: 읽는 말투 한 종류(Zeroth-Korean, 뉴스 문장 낭독)로만 학습했다. 대화체, 실제 통화, 잡음이 있는 환경에서는 재지 않았다. 전화 음질은 공개 데이터를 변환한 것이지 통화 녹음이 아니다
- **Whisper가 이미 본 데이터일 수 있다**: Whisper의 학습 데이터는 공개되지 않았다. Zeroth-Korean이나 FLEURS가 들어 있었는지 확인할 방법이 없다
- **9단계 모델(Qwen3-ASR)의 학습 데이터도 모른다**: 공개되지 않았고, FLEURS와 Zeroth-Korean은 공개 데이터라 학습에 들어 있었을 수 있다. 한 한국어 파인튜닝 모델 카드는 기본 Qwen3-ASR-1.7B의 FLEURS 한국어 CER을 1.42%로 적었다(이 저장소의 채점으로는 test 4.47%. 정규화나 묶음이 다를 수 있다). 9단계의 "다른 도메인에서 더 나은 기준 모델"은 이 오염 가능성을 배제하지 못한 결과다
- **작은 test 묶음**: Zeroth test는 457발화, 화자 10명이다. 발화 하나가 구간을 크게 흔들었다(2단계). 세 모델이 똑같이 틀린, 정답과 녹음이 맞지 않아 보이는 발화도 하나 있다(빼지 않았다)
- **규칙에 없던 분석**: FLEURS를 숫자 유무로 나눈 분석은 2단계 결과를 본 뒤에 했다. 사후 분석이라고 표시했고 판정은 바꾸지 않았다. 3단계부터는 재기 전에 규칙에 넣었다
- **학습 설정을 넓게 찾지 않았다**: 후보는 단계마다 한두 개다. 7단계에서 이어서 학습해 본 것은 학습률 일정이 두 번 도는 방식이라, 처음부터 길게 잡은 학습과 같지 않다. 더 나은 설정이 있을 수 있다
- **한 대의 PC**: RTX 2080 Ti(fp16만 지원) 한 장에서 학습하고 쟀다. 일부 측정은 같은 GPU를 다른 프로그램이 쓰는 동안 했다. 단계별 결과의 속도 수치는 그런 측정에서 가져오지 않았다. 예외는 둘이다. 2026-10-02의 CTranslate2 비교는 60분을 기다려도 GPU가 비지 않아 배경화면 프로그램이 GPU를 35% 안팎 쓰는 중에 두 엔진을 번갈아 쟀다. 그래서 두 엔진의 비율(1.46배)만 결과로 보고, 배속의 절대값은 낮게 나왔을 수 있다. 9단계의 Qwen3-ASR 속도도 60분을 기다려도 쉬는 상태가 되지 않아 그대로 쟀다. 1회차의 일부(turbo 배치 1과 Qwen 배치 1의 앞 약 4.5분)가 다른 프로그램의 GPU 부하(약 30%)와 겹쳤고, 나머지는 GPU가 거의 빈 상태에서 시작했지만 실행 중 CPU는 기록하지 않았다. 그래서 여기서도 번갈아 잰 비율을 주로 본다
- **표기**: 2~5단계의 파인튜닝한 모델은 숫자와 영문 약어를 한글로 풀어 쓴다. 6단계에서 학습 정답의 숫자만 규칙으로 바꿔 보았다. 규칙은 단위 앞의 숫자만 바꾸므로 빠뜨리는 것이 있고(학습 정답에 두 표기가 섞여 있다), 영문 약어와 인식 결과를 되돌리는 후처리는 다루지 않았다(8단계에서 출력의 숫자 표기를 바꾸는 후처리만 재 보았다. 영문 약어는 판정 없는 보조 지표로만)
- **되풀이 오류는 turbo LoRA에서도 나왔다**: 6단계와 7단계 모델이 2,671발화 중 한 번씩, 서로 다른 발화에서 냈다(Transformers로 잰 것). 6단계 모델을 faster-whisper로 재자(2026-10-02) 재시도를 꺼도 되풀이가 하나도 나오지 않아, 5단계의 재시도가 이 모델에서도 막는지는 판정하지 못했다(재시도를 켜도 결과는 같았다). Transformers에서는 8단계(2026-10-04)에서 재시도 디코딩이 두 되풀이를 모두 막았다

## 실행

준비: NVIDIA GPU(11GB에서 확인), Python 3.11, [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                       # torch는 CUDA 12.8 빌드를 받는다 (약 2.5GB)
uv run python -m whisper_ko_ft.download       # Zeroth-Korean, FLEURS 한국어·영어 (약 4GB)
uv run python -m whisper_ko_ft.prepare        # 16kHz 오디오 저장소와 학습·validation·test 목록 (약 6.5GB)
uv run python -m whisper_ko_ft.evaluate --model openai/whisper-small --set zeroth-val --overwrite   # 리포트가 커밋되어 있어 다시 재려면 --overwrite
uv run python -m whisper_ko_ft.analyze table --set zeroth-val --reports whisper-small whisper-large-v3-turbo
uv run python -m whisper_ko_ft.train --run-name small-a --learning-rate 1e-5   # 3,000스텝, 이 PC에서 몇 시간
```

- `prepare`는 Zeroth-Korean train에서 화자 10명을 validation으로 떼어 낸다 (화자 ID의 SHA-256 순, 규칙은 experiments.md)
- `evaluate`는 `reports/<이름>/<묶음>.json`에 요약과 발화별 정답·가설·오류 수를 남긴다. test 묶음은 `--allow-test`를 줘야 잰다 (단계마다 한 번만 재기 위한 장치)
- 리포트는 커밋된 기록이라 이미 있으면 `--overwrite`를 줘야 다시 잰다(`evaluate_ct2`도 같다). `--adapter`로 잴 때는 `--name`이 필수다 (기본 이름이 기준 모델의 리포트 폴더가 되므로). `--limit`으로 일부만 잰 리포트는 `analyze`가 받지 않는다

faster-whisper로 재기 (5단계):

```bash
uv sync --group ct2
uv run ct2-transformers-converter --model outputs/small-a/best --output_dir outputs/small-a/ct2 --quantization float16 --copy_files tokenizer.json
uv run python -m whisper_ko_ft.evaluate_ct2 --model outputs/small-a/ct2 --name small-a-ct2-fallback --set zeroth-val --fallback
```

LoRA 어댑터는 기본 모델에 합친 뒤 바꾼다. large-v3·turbo는 멜 필터가 128개라서 faster-whisper가 읽는 `preprocessor_config.json`을 함께 복사한다(`merge_adapter`가 만든다. transformers 5는 이 설정을 `processor_config.json` 안에 저장해서, 없으면 faster-whisper가 80개로 계산한다). `evaluate_ct2`는 변환된 모델의 멜 필터 수를 따른다.

```bash
uv run python -m whisper_ko_ft.merge_adapter --model openai/whisper-large-v3-turbo --adapter outputs/turbo-n/best --output outputs/turbo-n/merged
uv run ct2-transformers-converter --model outputs/turbo-n/merged --output_dir outputs/turbo-n/ct2 --quantization float16 --copy_files tokenizer.json preprocessor_config.json
```

LoRA 후보와 전화 음질 조건의 명령은 [docs/experiments.md](docs/experiments.md)의 각 단계 설정과 `train.py`, `evaluate.py`의 도움말(`--lora-r`, `--lora-targets`, `--telephone-prob`, `--adapter`, `--channel telephone`)을 따른다.

6단계: `uv run python -m whisper_ko_ft.prepare_itn`이 숫자를 바꾼 학습·validation 목록(`zeroth-train-itn`, `zeroth-val500-itn`)을 만들고, `train`에 `--train-set`, `--eval-set`으로 준다. 맞춘 CER은 `analyze table --harmonized`, 판정은 `analyze verdict6`.

8단계: `evaluate --fallback ratio`(압축비 기준) 또는 `--fallback ratio-logprob`(압축비와 로그확률)가 재시도 디코딩을 켜고, 리포트에 발화마다 마지막 temperature를 남긴다. `analyze changes`는 기록된 greedy 리포트와 비교해 다시 디코딩한 발화·바뀐 발화·되풀이를 세고, 판정은 `analyze verdict8 --models turbo-n turbo-n2 --arms d1a d1`(디코딩)과 `analyze itn8 --model turbo-n --baseline whisper-large-v3-turbo`(출력 숫자 변환). `table`·`loops`의 `--scoring itn|itn-latin|agnostic`은 저장된 가설을 바꾸지 않고 채점할 때만 후처리를 적용한다.

9단계: `uv run python -m whisper_ko_ft.evaluate_qwen --set zeroth-val`이 Qwen3-ASR-1.7B(고정한 revision, fp16, greedy, 언어 고정)로 디코딩해 한 번에 두 리포트를 남긴다. `reports/qwen3-asr-1.7b/`는 받아 적은 글 그대로(raw), `reports/qwen3-asr-1.7b-fixed/`는 공식 파서의 반복 제거를 거친 것이다. fp16 점검은 `--limit 100 --out <파일>`(fp32는 `--device cpu --precision fp32`)과 `analyze gate9`, 판정은 `analyze verdict9 --candidate qwen3-asr-1.7b --baseline whisper-large-v3-turbo`.

10단계: `uv run python -m whisper_ko_ft.train_qwen --run-name qwen-qn`이 Qwen3-ASR-1.7B에 LoRA를 학습한다(fp16 기본 가중치, fp32 어댑터, GradScaler, 장치당 2 × 누적 16, 1,500스텝). 스텝마다 `outputs/<run>/steps.jsonl`에 손실·기울기 크기·배율·건너뜀·GPU 메모리를 남기고, `outputs/<run>/PAUSE`가 있는 동안은 다음 스텝을 시작하지 않으며, `--resume`으로 이어 한다. 평가는 `evaluate_qwen --adapter outputs/qwen-qn/checkpoint-<스텝> --name qwen3-asr-1.7b-qn`(어댑터를 병합, `--channel telephone` 가능), 관문은 `analyze gate10`·`smoke10`, 체크포인트 고르기는 `analyze pick10`, 판정은 `analyze verdict10 --candidate qwen3-asr-1.7b-qn --n turbo-n --untrained qwen3-asr-1.7b`.

테스트: `uv run pytest`, `uv run ruff check .`

## 데이터와 모델

| 쓰임 | 이름 | 라이선스 | 원본 |
|---|---|---|---|
| 학습·평가 | Zeroth-Korean | CC BY 4.0 | [openslr.org/40](https://openslr.org/40/), [kresnik/zeroth_korean](https://huggingface.co/datasets/kresnik/zeroth_korean) |
| 평가 | FLEURS (한국어, 영어) | CC BY 4.0 | [google/fleurs](https://huggingface.co/datasets/google/fleurs) |
| 모델 | Whisper small, large-v3-turbo | MIT | [openai/whisper-small](https://huggingface.co/openai/whisper-small), [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| 모델 (9단계 학습 없이 평가, 10단계 LoRA) | Qwen3-ASR-1.7B, 0.6B (Transformers 판) | Apache 2.0 | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) |

데이터와 학습 결과물은 저장소에 넣지 않는다.
