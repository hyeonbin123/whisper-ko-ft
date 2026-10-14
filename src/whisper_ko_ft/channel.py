"""전화 채널 흉내: 300-3400 Hz 대역, 8 kHz 표본화, 8비트 mu-law(G.711 방식).

입력과 출력은 16 kHz float32라 결과가 같은 특징 추출기를 거친다. 잡음, 패킷 손실, 수화기 특성은 넣지 않았다.
docs/experiments.md는 이를 4단계의 한계로 적어 두었다.
"""

# [파일 역할] 4단계(전화 음질) 실험에서 쓰는 '전화 채널 흉내' 모듈이다. 깨끗한 16 kHz 음성을 받아 전화선을 한
# 번 거친 것처럼 바꾼 뒤 다시 16 kHz로 돌려준다.
# [왜 필요한가] 실제 통화 음성 데이터(AI Hub 등)는 신청·승인이 필요해 계획에서 뺐다(docs/plan.md). 그래서 공개
# 데이터(Zeroth-Korean)를 전화 음질로 바꿔 (1) 기본 모델이 전화 음질에서 얼마나 나빠지는지 재고 (2) 이렇게
# 바꾼 음성을 섞어 학습하면 얼마나 회복되는지 본다.
# [처리 순서] 300~3400 Hz 대역 통과 필터 → 8 kHz로 다운샘플 → 8비트 mu-law로 양자화했다가 복원(G.711 방식) →
# 16 kHz로 업샘플.
# [한계] 잡음·패킷 손실·수화기 특성은 넣지 않았다. 실제 통화와 다르다는 점은 docs/experiments.md 4단계 한계로
# 적혀 있다.

from __future__ import annotations

# numpy: 파형(float32 배열)에 mu-law 압축 공식을 벡터 연산으로 한 번에 적용한다.
# scipy.signal: 대역 통과 필터 설계(butter), 위상 왜곡 없는 필터링(sosfiltfilt), 정수비
# 리샘플링(resample_poly)을 직접 구현하지 않고 검증된 구현으로 쓰려고 가져온다.
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

# 프로젝트 전체가 쓰는 표본화율(16 kHz, Whisper 입력 규격)을 한곳에서 가져와 필터 설계 기준과 어긋나지 않게
# 한다.
from whisper_ko_ft.store import SAMPLE_RATE

# MU: mu-law의 압축 강도. 북미·일본 전화망 표준(G.711 mu-law)의 값 255를 그대로 쓴다. 작은 소리는 촘촘히, 큰
# 소리는 성기게 양자화해 8비트만으로도 말소리를 알아듣게 하는 방식이다.
MU = 255.0
# _BAND: 전화 음성 대역(300~3400 Hz)만 통과시키는 4차 Butterworth 대역 통과 필터. 모듈을 불러올 때 한 번만
# 설계해 두고 모든 발화에 재사용한다.
# output="sos"(2차 구간 직렬 형식)를 쓰는 이유: 같은 필터를 ba(분자·분모 계수) 형식으로 두면 차수가 높을 때
# 수치 오차로 불안정해질 수 있어서, scipy가 권장하는 sos 형식을 쓴다.
_BAND = butter(4, [300, 3400], btype="bandpass", fs=SAMPLE_RATE, output="sos")


# 8비트 mu-law로 '압축 → 256단계 양자화 → 복원'을 한 번 거친 파형을 돌려준다.
# 입력: -1~1 범위의 float 파형. 출력: 같은 길이의 float32 파형(양자화 잡음이 섞인 상태).
# 왜 압축만이 아니라 왕복(roundtrip)인가: 모델에 들어가는 것은 결국 복원된 파형이므로, 전화망에서 생기는
# '양자화 손실'만 남기고 나머지 형식은 원래와 같게 맞춘다.
def mu_law_roundtrip(samples: np.ndarray) -> np.ndarray:
    """8비트 mu-law 코드로 압축했다가 다시 펼친다."""
    # mu-law 공식은 -1~1 범위를 가정하므로 먼저 범위를 넘는 표본을 잘라 낸다.
    clipped = np.clip(samples, -1.0, 1.0)
    # mu-law 압축: sign(x) * ln(1 + MU*|x|) / ln(1 + MU). 작은 진폭을 크게 늘려 양자화 단계를 작은 소리 쪽에
    # 몰아 준다.
    compressed = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    # -1~1을 0~255 정수 코드(8비트, 256단계)로 반올림한다. 여기서 실제 정보 손실(양자화 잡음)이 생긴다.
    codes = np.round((compressed + 1.0) / 2.0 * 255.0)  # 256단계
    # 코드를 다시 -1~1 범위로 되돌린 뒤, 다음 줄에서 압축의 역함수(지수 확장)로 원래 진폭 척도로 복원한다.
    restored = codes / 255.0 * 2.0 - 1.0
    return (np.sign(restored) * np.expm1(np.abs(restored) * np.log1p(MU)) / MU).astype(np.float32)


# 16 kHz 음성을 '좁은 대역 전화 채널을 거친 소리'로 바꾸되, 출력은 다시 16 kHz로 돌려준다.
# 왜 16 kHz로 되돌리나: 학습·평가 코드가 같은 Whisper 특징 추출기(16 kHz 입력)를 그대로 쓰게 하려는 것이다.
# 이렇게 하면 깨끗한 조건과 전화 조건의 차이가 '음질'에서만 나온다.
# 호출 위치: 학습(train.py)에서는 전화 음질을 섞어 학습할 때 직접 부르고, 평가(evaluate.py,
# evaluate_qwen.py)에서는 --channel telephone 옵션으로 아래 CHANNELS를 거쳐 부른다.
def telephone(samples: np.ndarray) -> np.ndarray:
    """16 kHz 음성을 좁은 대역 전화 채널을 거친 소리로 바꾼다. 출력도 16 kHz다."""
    # sosfiltfilt는 양 끝을 덧대어(padding) 앞뒤로 두 번 거르는데, 표본이 너무 짧으면 덧댈 길이가 모자라
    # 오류가 난다. 그런 극단적으로 짧은 입력은 변환 없이 그대로 돌려준다.
    if len(samples) < 64:  # 영위상 필터의 덧대기 길이보다 짧다
        return samples.astype(np.float32)
    # 1) 대역 제한: sosfiltfilt는 앞으로 한 번, 뒤로 한 번 걸러 위상 지연이 0이 된다. 그래서 파형이 시간축으로
    #    밀리지 않고, 원래 길이·시점이 그대로인 채 주파수 대역만 좁아진다.
    band_limited = sosfiltfilt(_BAND, samples)
    # 2) 16 kHz → 8 kHz 다운샘플(1/2배). resample_poly는 내부에 앨리어싱 방지 필터가 있어 정수비 변환에
    #    알맞다.
    narrow = resample_poly(band_limited, 1, 2)  # 8 kHz로
    # 3) 8 kHz에서 mu-law 왕복으로 전화망 양자화 손실을 넣고, 4) 다시 16 kHz로 업샘플한다. 업샘플해도 4 kHz
    #    이상의 고역은 되살아나지 않으므로 '전화 음질'은 그대로 남는다.
    wide = resample_poly(mu_law_roundtrip(narrow), 2, 1)  # 다시 16 kHz로
    # 리샘플 과정에서 길이가 한두 표본 달라질 수 있어 원래 길이로 잘라 맞춘다(정렬·배치 처리 시 길이 불일치
    # 방지).
    return wide[: len(samples)].astype(np.float32)


# 채널 이름 → 변환 함수 표. evaluate.py와 evaluate_qwen.py의 --channel 옵션이 이 표의 키를 선택지로 쓴다. 다른
# 음질 조건을 추가하려면 함수를 만들고 여기에 한 줄만 더하면 된다.
CHANNELS = {"telephone": telephone}
