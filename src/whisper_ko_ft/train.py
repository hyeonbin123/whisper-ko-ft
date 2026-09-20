"""Fine-tune a Whisper model on a prepared set: full fine-tuning or LoRA, fp16 mixed precision.

Usage:
    uv run python -m whisper_ko_ft.train --run-name small-a --model openai/whisper-small --learning-rate 1e-5
    uv run python -m whisper_ko_ft.train --run-name turbo-l1 --learning-rate 1e-4 --lora-r 16 ...

Settings that candidates share are the defaults here and are fixed in docs/experiments.md.
Writes outputs/<run-name>/: checkpoints, `best/` (lowest val-500 CER: the model, or the LoRA adapter when
--lora-r is set, plus the processor) and `run.json`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from whisper_ko_ft.channel import telephone
from whisper_ko_ft.evaluate import MAX_NEW_TOKENS, git_commit
from whisper_ko_ft.metrics import error_rate, score
from whisper_ko_ft.paths import OUTPUTS
from whisper_ko_ft.store import SAMPLE_RATE, AudioStore, Utterance, load_set

LANGUAGE_NAMES = {"ko": "korean", "en": "english"}


class UtteranceDataset(Dataset):
    """Raw audio and text; features are computed in the collator so each model can use its own mel size."""

    def __init__(self, utterances: list[Utterance], telephone_prob: float = 0.0) -> None:
        self.utterances = utterances
        self.telephone_prob = telephone_prob
        self._stores: dict[str, AudioStore] = {}  # opened lazily: a memmap can't be pickled to workers

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> dict:
        utterance = self.utterances[index]
        if utterance.store not in self._stores:
            self._stores[utterance.store] = AudioStore(utterance.store)
        audio = self._stores[utterance.store].read(utterance)
        # torch seeds every loader worker from the run's seed, so the draw is repeatable per run.
        # getattr: loader workers re-import this module, and may unpickle a dataset made by older code.
        telephone_prob = getattr(self, "telephone_prob", 0.0)
        if telephone_prob and torch.rand(1).item() < telephone_prob:
            audio = telephone(audio)
        return {"audio": audio, "text": utterance.text}


@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int
    # A LoRA run keeps the base model in fp16. Training runs under autocast, but generation during
    # evaluation does not, so the features must already have the model's dtype.
    features_dtype: torch.dtype = torch.float32

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        inputs = self.processor.feature_extractor(
            [row["audio"] for row in rows],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        labels = self.processor.tokenizer([row["text"] for row in rows], return_tensors="pt", padding=True)
        ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        # The model prepends the start token itself when it shifts the labels right.
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        return {
            "input_features": inputs.input_features.to(self.features_dtype),
            "attention_mask": inputs.attention_mask,
            "labels": ids,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--spec-augment", action="store_true", help="mask_time_prob=0.05 (candidate C)")
    parser.add_argument("--lora-r", type=int, default=0, help="LoRA rank; 0 trains every weight")
    parser.add_argument("--lora-alpha", type=int, help="defaults to 2 x rank")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument(
        "--init-adapter", help="continue from this saved LoRA adapter (its own rank and targets are used)"
    )
    parser.add_argument(
        "--telephone-prob",
        type=float,
        default=0.0,
        help="share of training audio sent through the telephone channel",
    )
    parser.add_argument("--train-set", default="zeroth-train")
    parser.add_argument("--eval-set", default="zeroth-val500")
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-train", type=int, help="smoke runs only")
    parser.add_argument("--limit-eval", type=int, help="smoke runs only")
    args = parser.parse_args()

    language, train_utterances = load_set(args.train_set)
    _, eval_utterances = load_set(args.eval_set)
    if args.limit_train:
        train_utterances = train_utterances[: args.limit_train]
    if args.limit_eval:
        eval_utterances = eval_utterances[: args.limit_eval]

    processor = WhisperProcessor.from_pretrained(args.model)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE_NAMES[language], task="transcribe")
    # LoRA keeps the frozen base in fp16 to fit 11 GB; PEFT creates the adapter weights in fp32.
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16 if args.lora_r else torch.float32
    )
    model.generation_config.language = LANGUAGE_NAMES[language]
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS
    model.generation_config.max_length = None
    model.config.use_cache = False  # not compatible with gradient checkpointing
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
    decoder_start_token_id = model.config.decoder_start_token_id
    if args.lora_r and args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        model.print_trainable_parameters()
    elif args.lora_r:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha or 2 * args.lora_r,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets,
            bias="none",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

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
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=0,
    )
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

    model.config.use_cache = True
    trainer.save_model(str(output_dir / "best"))
    processor.save_pretrained(str(output_dir / "best"))
    history = [entry for entry in trainer.state.log_history if "eval_cer" in entry]
    run["eval_history"] = [{"step": e["step"], "cer": round(e["eval_cer"], 5)} for e in history]
    run["best_checkpoint"] = trainer.state.best_model_checkpoint
    run["best_cer"] = trainer.state.best_metric
    run["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    (output_dir / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(run["eval_history"]))


if __name__ == "__main__":
    main()
