"""faster-whisper용 CTranslate2 Whisper 폴더의 로그 멜 특징 설정.

faster-whisper는 모델 폴더의 preprocessor_config.json에서 특징 설정을 읽고, 파일이 없으면 멜 필터 80개를
쓴다. whisper-large-v3와 large-v3-turbo는 128개를 받는다. transformers 5는 특징 추출기 설정을 그 파일 대신
processor_config.json 안에 저장한다. 그래서 합치거나 파인튜닝한 폴더에 preprocessor_config.json이 없었고,
ct2-transformers-converter가 복사할 파일이 없어 turbo 모델이 멜 80개짜리 특징을
받았다("expected (1, 128, 3000), but got (1, 80, 3000)", 2026-09-20).

`merge_adapter`는 합친 모델 옆에 preprocessor_config.json을 쓰고(변환할 때 --copy_files tokenizer.json
preprocessor_config.json을 준다), `evaluate_ct2`는 변환된 모델 자체에서 멜 필터 수를 가져온다. 그래서 이
파일 없이 변환된 폴더도 바르게 읽는다.
"""

# [파일 역할] faster-whisper(CTranslate2)로 large-v3·turbo 계열을 잴 때 멜 필터 수를 바로잡는 도우미.
# merge_adapter가 설정 파일을 쓰고(write_preprocessor_config), evaluate_ct2가 읽는다(feature_kwargs).
# [왜 필요한가] 위 docstring의 사고(2026-09-20 시도가 첫 발화의 인코더 입력에서 멈춤) 때문에 생겼다. 원인과
# 수정은 docs/experiments.md '추가 측정' 절과 커밋 61b4b2f에 있다. 5단계의 small은 멜 80개라 영향이 없었다.
# [설계] 두 겹으로 막는다. (1) 합친 모델 폴더에 preprocessor_config.json을 써 두고 변환할 때 복사한다. (2)
# 평가할 때는 변환된 모델 자체의 멜 수(n_mels)를 기준으로 삼아, 파일이 없는 폴더는 멜 수만 맞춰 읽고 멜 수가
# 틀린 파일은 거부한다.

from __future__ import annotations

import json
from pathlib import Path

# faster-whisper가 모델 폴더에서 읽는 특징 설정 파일 이름.
PREPROCESSOR_CONFIG = "preprocessor_config.json"
# COPY_FILES: 변환할 때 함께 복사할 파일 목록. README와 merge_adapter 사용법의 --copy_files 인자와
# 같다(tests/test_ct2_features.py가 설정 파일이 이 목록에 있는지 확인한다). merge_adapter가 쓴 폴더를
# ct2-transformers-converter로 변환할 때 --copy_files에 주는 값이다.
COPY_FILES = ("tokenizer.json", PREPROCESSOR_CONFIG)
# FEATURE_KEYS: 설정 파일에서 꺼내 faster-whisper FeatureExtractor에 넘길 키. transformers가 저장한 설정에는
# 다른 키도 들어 있어, 이 생성자(faster_whisper.feature_extractor.FeatureExtractor)가 받는 인자만 골라
# 넘긴다.
FEATURE_KEYS = ("feature_size", "sampling_rate", "hop_length", "chunk_length", "n_fft")


# 특징 추출기 설정을 faster-whisper가 읽는 평평한 preprocessor_config.json으로 저장한다. 입력: transformers의
# WhisperFeatureExtractor, 모델의 멜 수(config.num_mel_bins), 저장 폴더. 출력: 쓴 파일 경로.
# 호출 위치: merge_adapter.main.
def write_preprocessor_config(feature_extractor, num_mel_bins: int, folder: Path) -> Path:
    """WhisperFeatureExtractor를 faster-whisper가 읽는 평평한 preprocessor_config.json으로 저장한다.

    `num_mel_bins`는 모델의 값(config.num_mel_bins)이다. 이 값과 맞지 않는 특징 추출기는 거부한다.
    """
    # 특징 추출기와 모델의 멜 수가 다르면 쓰지 않고 멈춘다. 틀린 설정을 써 두면 변환 뒤 평가에서 같은 사고가
    # 다시 나기 때문이다.
    if feature_extractor.feature_size != num_mel_bins:
        raise ValueError(
            f"the feature extractor makes {feature_extractor.feature_size} mel bins, "
            f"but the model takes {num_mel_bins}"
        )
    target = Path(folder) / PREPROCESSOR_CONFIG
    # sort_keys·indent로 내용이 같으면 늘 같은 바이트가 나오게 하고, 줄바꿈을 LF로 고정한다.
    text = json.dumps(feature_extractor.to_dict(), indent=2, sort_keys=True) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


# 변환된 모델 폴더에서 faster-whisper FeatureExtractor에 넘길 인자를 만든다. n_mels는 변환된 모델 자체가 받는
# 멜 수다(evaluate_ct2가 model.model.n_mels로 넘긴다).
# 설정 파일이 있으면 그 값을 쓰되 멜 수가 모델과 다르면 오류로 멈춘다. 없으면 멜 수만 모델에 맞추고 나머지는
# faster-whisper 기본값을 둔다. 그래서 파일 없이 변환된 예전 폴더도 바르게 읽는다.
# 호출 위치: evaluate_ct2.main.
def feature_kwargs(folder: Path, n_mels: int) -> dict:
    """인코더가 멜 필터 `n_mels`개를 받는 변환된 모델에 넘길 FeatureExtractor 인자.

    폴더에 preprocessor_config.json이 있으면 그 설정을 쓰며, 그 안의 멜 필터 수가 모델과 같아야 한다. 파일이
    없으면 멜 필터 수만 모델에 맞추고 나머지는 faster-whisper 기본값을 둔다.
    """
    path = Path(folder) / PREPROCESSOR_CONFIG
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    # FeatureExtractor가 받는 키만 골라낸다.
    kwargs = {key: config[key] for key in FEATURE_KEYS if key in config}
    # 파일에 feature_size가 없으면 모델의 멜 수를 넣고, 있으면 모델과 맞는지 확인한다.
    size = kwargs.setdefault("feature_size", n_mels)
    if size != n_mels:
        raise ValueError(f"{path} has feature_size={size}, but the model takes {n_mels} mel bins")
    return kwargs
