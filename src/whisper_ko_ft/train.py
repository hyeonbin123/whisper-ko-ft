"""준비한 묶음으로 Whisper 모델을 파인튜닝한다: 전체 파인튜닝 또는 LoRA, fp16 혼합 정밀도.

사용법:
    uv run python -m whisper_ko_ft.train --run-name small-a --model openai/whisper-small --learning-rate 1e-5
    uv run python -m whisper_ko_ft.train --run-name turbo-l1 --learning-rate 1e-4 --lora-r 16 ...

후보들이 함께 쓰는 설정이 여기의 기본값이며, docs/experiments.md에 고정되어 있다.
outputs/<run-name>/에 쓴다: 체크포인트, `best/`(val-500 CER이 가장 낮은 것: 모델, 또는 --lora-r이나
--init-adapter를 주면 LoRA 어댑터, 그리고 처리기), `run.json`.
--init-adapter는 저장된 어댑터에서 rank, alpha, dropout, 대상 층을 가져오며 --lora-*는 무시한다.
"""

# [파일 역할] Whisper 학습 스크립트. 2단계(whisper-small 전체 파인튜닝), 3단계(turbo LoRA L1·L2), 4단계(전화
# 음질을 섞은 T), 6단계(숫자를 바꾼 정답으로 N), 7단계(N의 어댑터에서 이어 학습한 N2)를 모두 이 파일로
# 학습했다. 후보 사이의 차이는 명령줄 인자(학습률, LoRA 설정, --telephone-prob, --train-set, --init-adapter
# 등)뿐이다.
# [설계 이유] Hugging Face Seq2SeqTrainer를 그대로 쓰고, 이 프로젝트에 필요한 것만 더했다: 오디오 저장소에서
# 읽는 데이터셋, 모델마다 멜 수가 다른 것을 처리하는 콜레이터, 학습 중 val-500 CER로 체크포인트를 고르는
# compute_metrics, 실행 기록 run.json.
# [공통 설정] fp16 혼합 정밀도(RTX 2080 Ti에는 bf16이 없다), AdamW(Trainer 기본), warmup 뒤 선형 감소,
# gradient checkpointing, 실효 배치 32, 500스텝마다 val-500 평가(docs/experiments.md 2·3단계 공통 설정).
# [결과물] outputs/<run>/: 체크포인트, best/(val-500 CER이 가장 낮은 것. LoRA면 어댑터만), run.json(인자,
# 커밋, 데이터 규모, 평가 이력, 최대 GPU 메모리).
# [연결] train_qwen이 UtteranceDataset을 가져다 쓴다. 학습한 LoRA는 evaluate --adapter로 재고, merge_adapter로
# 합쳐 faster-whisper로 잰다.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

# numpy: 학습 중 평가에서 라벨 패딩(-100)을 바꾸고 오류율을 계산한다. torch: 데이터셋과 dtype. transformers:
# Seq2SeqTrainer(학습 루프·평가·체크포인트 관리), Whisper 모델과 처리기.
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# channel.telephone: 4단계에서 학습 음성 일부를 전화 음질로 바꾼다. evaluate: 평가와 같은 최대 토큰 수, 커밋
# 기록. metrics: val-500 채점. store: 묶음과 오디오 저장소.
from whisper_ko_ft.channel import telephone
from whisper_ko_ft.evaluate import MAX_NEW_TOKENS, git_commit
from whisper_ko_ft.metrics import error_rate, score
from whisper_ko_ft.paths import OUTPUTS
from whisper_ko_ft.store import SAMPLE_RATE, AudioStore, Utterance, load_set

# 묶음의 언어 코드를 Whisper 토크나이저·generate가 받는 언어 이름으로 바꾼다.
LANGUAGE_NAMES = {"ko": "korean", "en": "english"}


# 묶음의 발화를 (원본 오디오, 정답) 한 쌍씩 돌려주는 데이터셋. 특징(로그 멜)은 여기서 만들지 않고 Collator가
# 만든다(위 docstring: small은 멜 80개, large-v3·turbo는 128개라 모델의 처리기에 맡긴다).
# telephone_prob를 주면 그 비율만큼 발화를 무작위로 전화 음질로 바꿔 돌려준다(4단계 T는 0.5,
# docs/experiments.md 4단계). 매번 새로 뽑으므로 같은 발화도 에폭마다 깨끗한 판과 전화 음질 판이 섞인다.
# 호출 위치: main(학습·평가 데이터셋), train_qwen.main.
class UtteranceDataset(Dataset):
    """원본 오디오와 글. 특징은 콜레이터에서 계산해 모델마다 자기 멜 수를 쓸 수 있게 한다."""

    def __init__(self, utterances: list[Utterance], telephone_prob: float = 0.0) -> None:
        self.utterances = utterances
        self.telephone_prob = telephone_prob
        # 저장소 memmap은 처음 읽을 때 연다. DataLoader 워커로 데이터셋을 넘길 때 피클로 보내는데, 열린
        # memmap은 넘길 수 없어 빈 dict 상태로 넘긴다.
        self._stores: dict[str, AudioStore] = {}  # 처음 읽을 때 연다: memmap은 워커로 피클해 보낼 수 없다

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> dict:
        utterance = self.utterances[index]
        # 각 워커 안에서 처음 필요할 때 그 저장소를 연다.
        if utterance.store not in self._stores:
            self._stores[utterance.store] = AudioStore(utterance.store)
        audio = self._stores[utterance.store].read(utterance)
        # 전화 음질 적용 여부를 torch 난수로 정한다. torch가 실행의 seed(Trainer의 0)에서 DataLoader 워커마다
        # 시드를 정하므로, 같은 실행 설정이면 같은 선택이 나온다.
        # getattr로 읽는 이유: Windows의 DataLoader 워커는 이 모듈을 다시 import하고, 예전 코드로 만든
        # 데이터셋 객체를 풀 수 있다. 학습 도중 코드가 바뀌어도 속성이 없다고 죽지 않게 기본값을 둔다.
        telephone_prob = getattr(self, "telephone_prob", 0.0)
        if telephone_prob and torch.rand(1).item() < telephone_prob:
            audio = telephone(audio)
        return {"audio": audio, "text": utterance.text}


# 배치(행 목록)를 모델 입력으로 바꾸는 콜레이터. 로그 멜 특징, 특징의 attention mask, 정답 토큰(labels)을
# 만든다.
# 왜 여기서 특징을 만드나: 처리기가 모델에 맞는 멜 수를 알고 있어 같은 데이터셋을 small·turbo에 모두 쓸 수
# 있다.
# 호출 위치: main의 Seq2SeqTrainer(data_collator), tests/test_collator.py.
@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int
    # features_dtype: LoRA 실행은 fp16, 전체 파인튜닝은 fp32(main이 정한다). LoRA 평가에서 dtype이 맞지 않아
    # 실패한 것을 고친 설정이다(커밋 6905b6e). LoRA 실행은 기본 모델을 fp16으로 둔다. 학습은 autocast 아래에서
    # 돌지만 평가 중의 생성(generate)은 그렇지 않으므로, 특징이 처음부터 모델의 dtype이어야 한다.
    features_dtype: torch.dtype = torch.float32

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        # 로그 멜 특징. 특징 추출기가 30초로 채우거나 자른다(학습 데이터는 가장 긴 것이 20.7초라 잘리는 것이
        # 없다, docs/experiments.md 10단계 후보).
        inputs = self.processor.feature_extractor(
            [row["audio"] for row in rows],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        # 정답을 토큰으로 바꾼다. main이 set_prefix_tokens로 언어·task 접두 토큰을 정해 두었으므로
        # '<시작><언어><transcribe><타임스탬프 없음> 정답 <끝>' 꼴이 된다. 패딩 위치는 -100으로 바꿔 손실에서
        # 뺀다.
        labels = self.processor.tokenizer([row["text"] for row in rows], return_tensors="pt", padding=True)
        ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        # 맨 앞이 디코더 시작 토큰이면 떼어 낸다. 모델이 labels를 오른쪽으로 밀어 디코더 입력을 만들 때 시작
        # 토큰을 스스로 붙이므로, 남겨 두면 두 번 들어간다.
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        return {
            "input_features": inputs.input_features.to(self.features_dtype),
            "attention_mask": inputs.attention_mask,
            "labels": ids,
        }


# 실제로 쓸 LoRA 설정을 args에 채워, 모델 dtype·콜레이터 dtype·run.json이 실제 학습과 맞게 한다.
# --init-adapter(7단계 N2)면 저장된 어댑터의 rank·alpha·dropout·대상 층을 쓰고 --lora-*는 무시한다. 새 LoRA면
# alpha 기본값을 2 × rank로 둔다(rank 16 → alpha 32, docs/experiments.md 3단계).
# 왜 생겼나: 예전에는 --lora-r만 보고 LoRA 여부를 정해서, --lora-r 없이 --init-adapter만 주면 어댑터를 버리고
# 기본 모델 전체를 fp32로 학습했다(커밋 8ad5a18). 기록된 7단계 실행은 --lora-r 16을 함께 줘서 영향이 없었다.
# 호출 위치: main, tests/test_train_args.py.
def resolve_lora_args(args: argparse.Namespace) -> None:
    """실제로 쓸 LoRA 설정을 채운다. run.json에 그 값이 남고, --init-adapter만 줘도 LoRA로 학습한다."""
    if args.init_adapter:
        from peft import PeftConfig

        saved = PeftConfig.from_pretrained(args.init_adapter)
        targets = saved.target_modules
        args.lora_r, args.lora_alpha, args.lora_dropout = saved.r, saved.lora_alpha, saved.lora_dropout
        # PEFT는 대상 층을 set으로 저장하기도 해서, run.json에 순서가 고정된 목록으로 남도록 정렬한다(문자열
        # 정규식이면 그대로).
        args.lora_targets = sorted(targets) if isinstance(targets, (set, list, tuple)) else targets
    elif args.lora_r and args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r


# 명령줄 진입점. 기본값은 후보들이 함께 쓰는 설정이다(위 docstring): small, 3,000스텝, warmup 500, 500스텝마다
# 평가, 장치당 16 × 누적 2.
# 후보별로 주는 것: 2단계 A/B는 --learning-rate 1e-5/3e-5. 3단계는 --model turbo --lora-r 16 --learning-rate
# 1e-4 --max-steps 1500(L2는 --lora-targets q_proj k_proj v_proj out_proj fc1 fc2). 4단계 T는 --telephone-prob
# 0.5. 6단계 N은 --train-set zeroth-train-itn --eval-set zeroth-val500-itn. 7단계 N2는 --init-adapter.
# --spec-augment는 과적합이 보일 때만 돌리기로 한 후보 C용이다(docs/experiments.md 2단계).
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--spec-augment", action="store_true", help="mask_time_prob=0.05(후보 C)")
    parser.add_argument("--lora-r", type=int, default=0, help="LoRA rank. 0이면 모든 가중치를 학습한다")
    parser.add_argument("--lora-alpha", type=int, help="기본값은 2 x rank")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument(
        "--init-adapter",
        help="이 저장된 LoRA 어댑터에서 이어서 학습한다(LoRA 학습이 된다. 어댑터 자체의 rank, alpha, "
        "dropout, 대상 층을 쓰고 --lora-*는 무시한다)",
    )
    parser.add_argument(
        "--telephone-prob",
        type=float,
        default=0.0,
        help="학습 음성 중 전화 채널을 통과시킬 비율",
    )
    parser.add_argument("--train-set", default="zeroth-train")
    parser.add_argument("--eval-set", default="zeroth-val500")
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-train", type=int, help="스모크 실행 전용: 학습 발화를 앞의 N개로 줄인다")
    parser.add_argument("--limit-eval", type=int, help="스모크 실행 전용: 평가 발화를 앞의 N개로 줄인다")
    args = parser.parse_args()
    resolve_lora_args(args)

    # 학습 묶음과 평가 묶음(val-500)을 읽는다. --limit-*는 스모크 실행에서만 쓴다.
    language, train_utterances = load_set(args.train_set)
    _, eval_utterances = load_set(args.eval_set)
    if args.limit_train:
        train_utterances = train_utterances[: args.limit_train]
    if args.limit_eval:
        eval_utterances = eval_utterances[: args.limit_eval]

    # 처리기는 기본 모델의 것을 쓴다. 토크나이저에 언어와 task 접두 토큰을 정해 두어 정답 토큰이 추론 때와
    # 같은 접두로 시작하게 한다.
    processor = WhisperProcessor.from_pretrained(args.model)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE_NAMES[language], task="transcribe")
    # LoRA는 얼린 기본 가중치를 fp16으로 올려 11GB GPU에 맞추고(PEFT가 만드는 어댑터 가중치는 fp32다), 전체
    # 파인튜닝은 fp32 마스터 가중치로 올린 뒤 Trainer의 fp16 혼합 정밀도로 학습한다.
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16 if args.lora_r else torch.float32
    )
    # 학습 중 평가(predict_with_generate)의 디코딩을 evaluate.py와 맞춘다: 언어 고정, transcribe, 최대
    # 256토큰. 예전 방식의 forced_decoder_ids와 max_length는 지워 language·task·max_new_tokens 설정과 겹치지
    # 않게 한다.
    model.generation_config.language = LANGUAGE_NAMES[language]
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS
    model.generation_config.max_length = None
    model.config.use_cache = False  # gradient checkpointing과 함께 쓸 수 없다
    # 후보 C: SpecAugment(시간 축 마스킹 비율 0.05)를 켠다.
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
    decoder_start_token_id = model.config.decoder_start_token_id
    # LoRA 붙이기. --init-adapter면 저장된 어댑터를 학습 가능 상태로 불러와 이어서 학습하고, 아니면 새
    # LoRA(bias 없음)를 만든다. peft는 LoRA 실행에서만 불러온다. 학습하는 파라미터 수를 찍어 확인한다.
    if args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        model.print_trainable_parameters()
    elif args.lora_r:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets,
            bias="none",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    # 학습 중 val-500 평가 지표. generate로 만든 가설과 정답 토큰을 글로 되돌려 evaluate.py와 같은
    # 정규화·채점으로 묶음 CER을 낸다. 이 값으로 가장 좋은 체크포인트를 고른다(metric_for_best_model).
    # 정답의 -100(손실에서 뺀 패딩)은 디코딩할 수 없으므로 패딩 토큰으로 바꾼다. 정답이 빈 발화는 뺀다.
    # 주의: 6·7단계는 숫자를 바꾼 val-500으로 원래 CER을 쟀다. 10단계(train_qwen)는 이와 달리 학습이 끝난 뒤
    # 맞춘 CER로 고른다(docs/experiments.md 10단계).
    def compute_metrics(prediction) -> dict[str, float]:
        label_ids = np.where(
            prediction.label_ids == -100, processor.tokenizer.pad_token_id, prediction.label_ids
        )
        hypotheses = processor.batch_decode(prediction.predictions, skip_special_tokens=True)
        references = processor.batch_decode(label_ids, skip_special_tokens=True)
        scored = [score(r, h, language) for r, h in zip(references, hypotheses, strict=True)]
        kept = [s for s in scored if s is not None]
        edits, lengths = np.array([s.edits for s in kept]), np.array([s.length for s in kept])
        return {"cer": error_rate(edits, lengths)}

    # Trainer 설정. 이 프로젝트에서 뜻이 있는 것:
    # - fp16=True: Turing GPU라 bf16 대신 fp16 혼합 정밀도(손실 배율 자동 조절).
    # - gradient_checkpointing: 활성값을 저장하지 않고 역전파 때 다시 계산해 메모리를 아낀다.
    #   use_reentrant=False는 PyTorch가 권장하는 방식이다.
    # - 실효 배치 = 장치당 배치 × 누적 횟수 = 32. 6·7단계는 GPU를 다른 프로그램과 나눠 써서 8 × 4로
    #   바꿨다(docs/experiments.md 6단계 결과).
    # - eval_steps마다 평가하고 저장하며, 끝나면 CER이 가장 낮은 체크포인트를 다시
    #   불러온다(load_best_model_at_end). 디스크를 아끼려고 체크포인트는 2개만 남긴다.
    # - predict_with_generate: 평가 때 teacher forcing이 아니라 실제 디코딩으로 CER을 잰다.
    # - remove_unused_columns=False: 데이터셋이 모델 인자가 아닌 audio·text를 돌려주므로 Trainer가 열을 지우지
    #   않게 한다.
    # - report_to=[]: 외부 기록 서비스를 쓰지 않는다. seed=0.
    # 주의(docs/experiments.md 10단계 '재기 전에 아는 것'): transformers 5.17의 Trainer는 Whisper 손실을 누적
    # 횟수로 나누지 않아, 한 스텝의 기울기가 누적한 묶음 평균의 합이었고 기울기 자르기(기본 1.0)가 거의 매
    # 스텝 걸렸다. N의 기록은 그대로 두고 10단계에서만 바꿨다.
    output_dir = OUTPUTS / args.run_name
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        logging_steps=25,
        dataloader_num_workers=args.workers,
        # 7단계 첫 실행이 500스텝 평가의 DataLoader 워커를 띄우다가 멈춘 뒤 넣은 설정이다(커밋 30e739b,
        # docs/experiments.md 7단계 결과). 학습 결과에는 영향이 없다. Windows는 워커를 spawn으로 띄우므로,
        # 워커를 유지해 평가 때마다가 아니라 실행마다 한 번만 띄우게 한다.
        dataloader_persistent_workers=args.workers > 0,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=0,
    )
    # 학습 데이터셋에만 전화 음질 비율을 준다. 평가 데이터셋(val-500)은 깨끗한 음성 그대로다(4단계도
    # 체크포인트는 깨끗한 val-500으로 골랐다). processing_class는 transformers 5의 이름이다(4.x의 tokenizer=
    # 대신).
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=UtteranceDataset(train_utterances, args.telephone_prob),
        eval_dataset=UtteranceDataset(eval_utterances),
        data_collator=Collator(
            processor, decoder_start_token_id, torch.float16 if args.lora_r else torch.float32
        ),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    # 학습을 시작하기 전에 run.json을 먼저 쓴다(인자, 커밋, 학습 발화 수와 시간, 평가 발화 수, torch, GPU).
    # 학습이 도중에 죽어도 어떤 설정으로 돌렸는지 남는다.
    output_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "args": vars(args),
        "commit": git_commit(),
        "train_utterances": len(train_utterances),
        "train_hours": round(sum(u.seconds for u in train_utterances) / 3600, 2),
        "eval_utterances": len(eval_utterances),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8", newline="\n")

    trainer.train()

    # 학습 중에는 gradient checkpointing과 맞지 않아 껐던 KV 캐시를 추론용으로 다시 켜고 저장한다. best/에는
    # 가장 좋은 체크포인트(LoRA면 어댑터만)와 처리기가 들어간다.
    model.config.use_cache = True
    trainer.save_model(str(output_dir / "best"))
    processor.save_pretrained(str(output_dir / "best"))
    # run.json에 평가 이력(스텝별 val-500 CER), 고른 체크포인트, 그 CER, 학습 중 최대 GPU 메모리를 더해 다시
    # 쓴다. docs/experiments.md의 단계별 val-500 수치가 여기서 나온다.
    history = [entry for entry in trainer.state.log_history if "eval_cer" in entry]
    run["eval_history"] = [{"step": e["step"], "cer": round(e["eval_cer"], 5)} for e in history]
    run["best_checkpoint"] = trainer.state.best_model_checkpoint
    run["best_cer"] = trainer.state.best_metric
    run["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    (output_dir / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(run["eval_history"]))


if __name__ == "__main__":
    main()
