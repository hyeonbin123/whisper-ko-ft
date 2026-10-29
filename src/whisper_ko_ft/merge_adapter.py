"""LoRA 어댑터를 기본 모델에 합쳐 저장한다. ct2-transformers-converter가 읽을 수 있게 하려는 것이다.

사용법:
    uv run python -m whisper_ko_ft.merge_adapter --model openai/whisper-large-v3-turbo \
        --adapter outputs/turbo-n/best --output outputs/turbo-n/merged
    uv run ct2-transformers-converter --model outputs/turbo-n/merged --output_dir outputs/turbo-n/ct2 \
        --quantization float16 --copy_files tokenizer.json preprocessor_config.json

처리기 외에, faster-whisper가 멜 필터 개수를 읽어 가는 평평한 preprocessor_config.json도 폴더에
쓴다(large-v3와 turbo는 128개. `ct2_features` 참고).
"""

# [파일 역할] 학습한 LoRA 어댑터(예: outputs/turbo-n/best)를 기본 Whisper 가중치에 합쳐 보통의 Transformers
# 모델 폴더로 저장한다. 이 폴더를 ct2-transformers-converter로 CTranslate2 형식으로 바꾸면
# evaluate_ct2(faster-whisper)로 잴 수 있다.
# [왜 필요한가] 변환기는 어댑터 폴더가 아니라 완성된 모델 폴더를 읽는다(위 docstring). 그래서 turbo LoRA
# 모델(6단계 N 등)을 faster-whisper로 재려면 먼저 합쳐야 한다(README '실행', docs/experiments.md '추가 측정').
# [주의] transformers 5는 특징 추출 설정을 processor_config.json 안에 저장해서, 처리기만 저장하면 변환기가
# 복사할 preprocessor_config.json이 생기지 않는다. 그 파일이 없으면 faster-whisper가 멜 80개로 계산해
# turbo(128개)가 실패했으므로 여기서 따로 써 둔다(ct2_features, 커밋 61b4b2f).
# [기본 모델 revision] 2026-10-29부터 --model이 등록한 Hub ID(whisper-small, large-v3-turbo)면 기록된 측정이
# 쓴 캐시 스냅숏과 같은 revision(provenance.WHISPER_REVISIONS)으로 기본 모델과 처리기를 받는다. 학습(train)과
# 평가(evaluate)가 쓰는 기본 가중치와 합치는 기본 가중치가 늘 같게 하려는 것이다.

from __future__ import annotations

import argparse
from pathlib import Path

# torch: 기본 가중치를 fp16으로 올린다. peft.PeftModel: 저장된 어댑터를 기본 모델에 붙이고 가중치에 더해
# 넣는다. transformers: Whisper 모델과 처리기(토크나이저 + 특징 추출기).
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# write_preprocessor_config: faster-whisper가 읽는 평평한 특징 설정 파일을 쓴다. revision_kwargs: 등록한 기본
# 모델의 고정 revision(로컬 폴더면 빈 dict).
from whisper_ko_ft.ct2_features import write_preprocessor_config
from whisper_ko_ft.provenance import revision_kwargs


# 명령줄 진입점. --model은 기본 모델 이름, --adapter는 어댑터 폴더, --output은 합친 모델을 쓸 폴더다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 기본 모델이 등록한 Hub ID면 고정한 revision으로 받는다(위 [기본 모델 revision]). 로컬 폴더면 빈 dict라
    # 지금까지와 같은 호출이다.
    pinned = revision_kwargs(args.model)
    # 기본 모델을 fp16으로 올린다. 학습 때 기본 가중치를 fp16으로 얼려 둔 것과 같고(docs/experiments.md 3단계
    # 공통 설정), 변환도 float16으로 한다.
    base = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16, **pinned)
    # 어댑터를 붙인 뒤 merge_and_unload로 LoRA의 저랭크 갱신을 원래 가중치에 더하고 PEFT 껍데기를 벗긴다.
    # 결과는 기본 모델과 구조가 같은 보통의 Whisper 모델이다(docs/experiments.md: 합친 모델의 torch 최대
    # 할당이 기본 turbo와 같았다).
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    output = Path(args.output)
    merged.save_pretrained(output)
    # 처리기는 기본 모델의 것을 그대로 쓴다(train.py도 기본 모델의 처리기로 학습한다). 변환기가
    # tokenizer.json을 복사할 수 있게 같은 폴더에 저장한다.
    processor = WhisperProcessor.from_pretrained(args.model, **pinned)
    processor.save_pretrained(output)
    # faster-whisper용 preprocessor_config.json을 따로 쓴다. 특징 추출기와 모델의 멜 수(turbo는 128)가 맞지
    # 않으면 여기서 멈춘다.
    write_preprocessor_config(processor.feature_extractor, merged.config.num_mel_bins, output)
    print(f"wrote {output} ({merged.config.num_mel_bins} mel bins)")


if __name__ == "__main__":
    main()
