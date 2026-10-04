"""Train a LoRA adapter on Qwen3-ASR (stage 10): fp16 base, fp32 adapter, fp16 autocast and GradScaler.

Usage:
    uv run --no-sync python -m whisper_ko_ft.train_qwen --run-name qwen-qn > outputs/qwen-qn/train.log 2>&1
    smoke (a gate of 10단계): --run-name qwen-qn-smoke --max-steps 50 --longest-at 25 --save-steps 0
    the initial adapter only, no training (CPU is enough): --run-name qwen-qn-init --max-steps 0 --device cpu
    a stopped run continues from its last saved state with the same command and --resume

Settings are fixed in docs/experiments.md (10단계). This is a plain loop, not the Trainer of train.py, so that
every optimizer step is logged with its loss, gradient norm, loss scale, whether GradScaler skipped it and its
GPU memory, and so that a run can pause: no step starts while outputs/<run>/PAUSE exists (a watcher creates it
when Windows moves GPU memory to shared system memory), and paused time is not part of any step time. The
settings the Trainer gave the stage 6 model N are kept: AdamW (fused on the GPU, betas 0.9/0.999, eps 1e-8, no
weight decay), linear warmup then linear decay to zero, gradient clipping at 1.0, GradScaler's defaults,
and no scheduler step on a step that GradScaler skipped.

The loss is the model's causal LM loss on the transcript tokens and the end token <|im_end|> only. The prompt
is the one inference uses (`apply_transcription_request`, ending in "language Korean<asr_text>"), so the
template, the audio placeholders and the forced language are masked, as is the padding. Logits are computed at
those positions only (the same value as the model's own loss without vocabulary-sized logits for every
position). A step sums the loss over the tokens of all its micro-batches and divides by their number.

Writes outputs/<run>/: run.json, steps.jsonl (one line per step and per event; a resumed run appends),
checkpoint-<step>/ (the adapter, every --save-steps) and resume/ (the adapter and the optimizer, scaler,
scheduler and random states, every --resume-every steps and at the end).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from whisper_ko_ft.evaluate import git_commit
from whisper_ko_ft.evaluate_qwen import DEFAULT_MODEL, LANGUAGES, REGISTERED, first_window
from whisper_ko_ft.paths import OUTPUTS
from whisper_ko_ft.store import SAMPLE_RATE, load_set
from whisper_ko_ft.train import UtteranceDataset

# The kinds of layers N's adapter is on (stage 6: attention and MLP of the Whisper encoder and decoder): the
# audio encoder's attention and fc1/fc2 and the language model's attention and MLP. Not the projector, the
# convolutions or the output layer. A regex, because the encoder's attention has q_proj/k_proj/v_proj too.
LORA_TARGETS = (
    r"model\.audio_tower\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)"
    r"|model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)
ENCODER_TARGETS_PER_LAYER = 6
LANGUAGE_MODEL_TARGETS_PER_LAYER = 7
END_TOKEN = "<|im_end|>"
IGNORE = -100
MODEL_INPUTS = ("input_ids", "attention_mask", "input_features", "input_features_mask")
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


# --- data ---


class StepBatches:
    """The index lists of the micro-batches, step by step: a fixed schedule, so a resumed run sees the same
    data.

    Epoch e is a permutation seeded with seed + e, cut into micro-batches (the remainder is dropped). Step s
    takes micro-batches (s-1)*accum ... s*accum-1 of that endless sequence. `longest_at` (smoke runs) gives
    that one step the `longest` utterances instead; its usual micro-batches are skipped.
    """

    def __init__(
        self,
        count: int,
        batch_size: int,
        accum: int,
        seed: int = 0,
        first_step: int = 1,
        last_step: int = 1,
        longest: list[int] | None = None,
        longest_at: int | None = None,
    ) -> None:
        if count < batch_size:
            raise ValueError("fewer utterances than one micro-batch")
        if longest_at is not None and (longest is None or len(longest) != batch_size * accum):
            raise ValueError("the longest step needs batch_size * accum utterances")
        self.count, self.batch_size, self.accum, self.seed = count, batch_size, accum, seed
        self.first_step, self.last_step = first_step, last_step
        self.longest, self.longest_at = longest, longest_at
        self.per_epoch = count // batch_size
        self._epoch: tuple[int, list[int]] | None = None

    def __len__(self) -> int:
        return max(0, self.last_step - self.first_step + 1) * self.accum

    def _permutation(self, epoch: int) -> list[int]:
        if self._epoch is None or self._epoch[0] != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._epoch = (epoch, torch.randperm(self.count, generator=generator).tolist())
        return self._epoch[1]

    def micro(self, index: int) -> list[int]:
        epoch, position = divmod(index, self.per_epoch)
        start = position * self.batch_size
        return self._permutation(epoch)[start : start + self.batch_size]

    def step(self, step: int) -> list[list[int]]:
        if step == self.longest_at:
            size = self.batch_size
            return [self.longest[m * size : (m + 1) * size] for m in range(self.accum)]
        return [self.micro((step - 1) * self.accum + m) for m in range(self.accum)]

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.first_step, self.last_step + 1):
            yield from self.step(step)


def longest_indices(utterances, count: int) -> list[int]:
    """The `count` longest utterances (audio length, then text length), longest first."""
    order = sorted(range(len(utterances)), key=lambda i: (utterances[i].length, len(utterances[i].text)))
    return order[::-1][:count]


def answer_ids(tokenizer, text: str) -> list[int]:
    """What the model should generate after the prompt: the transcript, then the end token."""
    return tokenizer(text, add_special_tokens=False)["input_ids"] + [
        tokenizer.convert_tokens_to_ids(END_TOKEN)
    ]


def assemble(
    prompt_ids: torch.Tensor, prompt_mask: torch.Tensor, answers: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-padded input ids, attention mask and labels: each row is its prompt (left padding removed) and
    its answer, and only the answer is labelled (aligned with the inputs; the model shifts the labels)."""
    prompts = [ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, prompt_mask, strict=True)]
    width = max(len(p) + len(a) for p, a in zip(prompts, answers, strict=True))
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(prompts), width), dtype=torch.long)
    labels = torch.full((len(prompts), width), IGNORE, dtype=torch.long)
    for row, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
        end = len(prompt) + len(answer)
        input_ids[row, :end] = torch.tensor(prompt + answer)
        attention[row, :end] = 1
        labels[row, len(prompt) : end] = torch.tensor(answer)
    return input_ids, attention, labels


@dataclass
class QwenCollator:
    """Audio and transcripts to model inputs, with the prompt and audio window of `evaluate_qwen`."""

    processor: object
    language: str = "Korean"

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        audio = [first_window(row["audio"]) for row in rows]
        prompt = self.processor.apply_transcription_request(audio, language=self.language)
        tokenizer = self.processor.tokenizer
        answers = [answer_ids(tokenizer, row["text"]) for row in rows]
        input_ids, attention, labels = assemble(
            prompt["input_ids"], prompt["attention_mask"], answers, tokenizer.pad_token_id
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "input_features": prompt["input_features"],
            "input_features_mask": prompt["input_features_mask"],
            "seconds": torch.tensor([len(samples) / SAMPLE_RATE for samples in audio]),
        }


# --- model ---


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Gradient checkpointing on both towers, then a fresh LoRA adapter (fp32 even on an fp16 base)."""
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, bias="none")
    return get_peft_model(model, config)


def resume_model(model, folder: Path):
    """The base model with a saved adapter, trainable, as `add_lora` would have left it."""
    from peft import PeftModel

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return PeftModel.from_pretrained(model, str(folder), is_trainable=True)


def expected_lora_modules(config) -> int:
    audio, text = config.audio_config.encoder_layers, config.text_config.num_hidden_layers
    return ENCODER_TARGETS_PER_LAYER * audio + LANGUAGE_MODEL_TARGETS_PER_LAYER * text


def lora_summary(model) -> dict:
    """Where the adapter is and what trains (gate (b) of stage 10)."""
    modules = {name.split(".lora_A.")[0] for name, _ in model.named_parameters() if ".lora_A." in name}
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "modules": len(modules),
        "encoder_modules": sum(".audio_tower." in name for name in modules),
        "language_model_modules": sum(".language_model." in name for name in modules),
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "trainable_dtypes": sorted({str(p.dtype) for _, p in trainable}),
        "other_trainable": [name for name, _ in trainable if ".lora_" not in name],
    }


def check_lora(summary: dict, config) -> None:
    expected = expected_lora_modules(config)
    if summary["modules"] != expected:
        raise ValueError(f"LoRA is on {summary['modules']} modules, expected {expected}")
    if summary["trainable_dtypes"] != ["torch.float32"]:
        raise ValueError(f"the adapter must train in torch.float32, not {summary['trainable_dtypes']}")
    if summary["other_trainable"]:
        raise ValueError(f"other weights would train: {summary['other_trainable'][:3]}")


def answer_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Summed cross-entropy of the labelled tokens and their number = the model's loss x that number."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = inner.model(**{key: batch[key] for key in MODEL_INPUTS}, use_cache=False).last_hidden_state
    targets = batch["labels"][:, 1:]
    keep = targets != IGNORE
    logits = inner.lm_head(hidden[:, :-1][keep])
    return F.cross_entropy(logits.float(), targets[keep], reduction="sum"), int(keep.sum())


# --- one optimizer step, the loop, saved state ---


def linear_schedule(optimizer, warmup: int, total: int):
    return transformers.get_linear_schedule_with_warmup(optimizer, warmup, total)


def optimizer_step(scaler, optimizer, params, max_grad_norm: float) -> tuple[float, float, bool]:
    """Unscale, clip, step: (gradient norm before clipping, loss scale used, whether GradScaler skipped)."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return float(norm), float(scale), scaler.get_scale() < scale


def wait_while_paused(flag: Path, sleep: Callable[[float], None] = time.sleep, poll: float = 5.0) -> float:
    """Seconds spent waiting for `flag` to disappear (0 when it was not there)."""
    if not flag.exists():
        return 0.0
    started = time.perf_counter()
    while flag.exists():
        sleep(poll)
    return time.perf_counter() - started


@dataclass
class LoopSettings:
    first_step: int
    last_step: int
    accum: int
    device: str = "cuda"
    amp: bool = True
    max_grad_norm: float = 1.0
    pause_flag: Path | None = None
    longest_at: int | None = None


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_steps(
    model,
    batches: Iterator[dict],
    optimizer,
    scheduler,
    scaler,
    settings: LoopSettings,
    log: Callable[[dict], None],
    after_step: Callable[[int], None] = lambda step: None,
) -> dict:
    """Optimizer steps first_step..last_step, `accum` micro-batches each; one log entry per step."""
    params = [p for p in model.parameters() if p.requires_grad]
    cuda = settings.device == "cuda"
    summary = {"steps": 0, "skipped_steps": [], "nonfinite_steps": [], "peak_allocated_mb": 0}
    summary.update({"seconds": 0.0, "paused_seconds": 0.0})
    for step in range(settings.first_step, settings.last_step + 1):
        if settings.pause_flag is not None:
            paused = wait_while_paused(settings.pause_flag)
            if paused:
                log({"event": "pause", "before_step": step, "seconds": round(paused, 1), "time": now()})
                summary["paused_seconds"] += paused
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        micro = [next(batches) for _ in range(settings.accum)]
        data_seconds = time.perf_counter() - started
        tokens = sum(int((batch["labels"][:, 1:] != IGNORE).sum()) for batch in micro)
        loss_total = 0.0
        for batch in micro:
            inputs = {key: batch[key].to(settings.device) for key in (*MODEL_INPUTS, "labels")}
            with torch.autocast(device_type=settings.device, dtype=torch.float16, enabled=settings.amp):
                loss_sum, _ = answer_loss(model, inputs)
            scaler.scale(loss_sum / tokens).backward()
            loss_total += loss_sum.item()
        lr = optimizer.param_groups[0]["lr"]
        norm, scale, skipped = optimizer_step(scaler, optimizer, params, settings.max_grad_norm)
        if not skipped:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        loss = loss_total / tokens
        peak = round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None
        seconds_per_utterance = torch.cat([batch["seconds"] for batch in micro])
        log(
            {
                "step": step,
                "time": now(),
                "loss": loss,
                "tokens": tokens,
                "utterances": len(seconds_per_utterance),
                "audio_seconds": round(float(seconds_per_utterance.sum()), 2),
                "max_seconds": round(float(seconds_per_utterance.max()), 2),
                "grad_norm": norm,
                "scale": scale,
                "skipped": skipped,
                "lr": lr,
                "seconds": round(seconds, 3),
                "data_seconds": round(data_seconds, 3),
                "peak_allocated_mb": peak,
                "reserved_mb": round(torch.cuda.memory_reserved() / 2**20) if cuda else None,
                "longest": step == settings.longest_at,
            }
        )
        summary["steps"] += 1
        summary["seconds"] += seconds
        if skipped:
            summary["skipped_steps"].append(step)
        if not math.isfinite(loss):
            summary["nonfinite_steps"].append(step)
        if peak is not None:
            summary["peak_allocated_mb"] = max(summary["peak_allocated_mb"], peak)
        after_step(step)
    return summary


def _random_states() -> dict:
    """Plain types and tensors only, so that the state loads with torch.load(weights_only=True)."""
    name, keys, position, has_gauss, gauss = np.random.get_state()
    states = {
        "python": random.getstate(),
        "numpy": [name, keys.tolist(), int(position), int(has_gauss), float(gauss)],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def _set_random_states(states: dict) -> None:
    version, internal, gauss = states["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, position, has_gauss, cached = states["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def save_state(folder: Path, model, optimizer, scaler, scheduler, step: int) -> None:
    """The adapter and everything a resumed run needs, written next to `folder` and then moved into place."""
    partial = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(partial, ignore_errors=True)
    model.save_pretrained(str(partial))
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "random": _random_states(),
    }
    torch.save(state, partial / "state.pt")
    shutil.rmtree(folder, ignore_errors=True)
    partial.rename(folder)


def resume_folder(run_dir: Path) -> Path | None:
    """The last complete saved state of a run (a crash while moving it into place leaves it as resume.tmp)."""
    for name in ("resume", "resume.tmp"):
        folder = run_dir / name
        if (folder / "state.pt").exists() and (folder / "adapter_model.safetensors").exists():
            return folder
    return None


def saved_step(folder: Path) -> int:
    """The last finished step of a saved state."""
    return int(torch.load(folder / "state.pt", weights_only=True)["step"])


def restore_state(folder: Path, optimizer, scaler, scheduler) -> int:
    """Load the optimizer, scaler, scheduler and random states; returns the last finished step."""
    state = torch.load(folder / "state.pt", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    scheduler.load_state_dict(state["scheduler"])
    _set_random_states(state["random"])
    return int(state["step"])


# --- the run ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="defaults to the pinned revision of a registered model")
    parser.add_argument("--train-set", default="zeroth-train-itn")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1_500, help="0 saves the initial adapter and stops")
    parser.add_argument("--save-steps", type=int, default=500, help="adapter checkpoints; 0 for none")
    parser.add_argument("--resume-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--longest-at", type=int, help="smoke: this step gets the longest utterances")
    parser.add_argument("--limit-train", type=int, help="CPU checks only")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=list(DTYPES), default="fp16", help="of the frozen base")
    parser.add_argument("--resume", action="store_true", help="continue from outputs/<run>/resume")
    args = parser.parse_args(argv)
    if args.revision is None:
        if args.model not in REGISTERED:
            parser.error("a model that is not registered needs --revision")
        args.revision = REGISTERED[args.model][0]
    if args.device == "cpu" and args.precision == "fp16" and args.max_steps > 0:
        parser.error("training on the CPU needs --precision fp32 (and is never a measured run)")
    run_dir = OUTPUTS / args.run_name
    if args.resume and resume_folder(run_dir) is None:
        parser.error(f"{run_dir} has no saved state to resume from")
    if not args.resume and (run_dir / "run.json").exists():
        parser.error(f"{run_dir} already has a run; pass --resume or pick another --run-name")
    if args.longest_at is not None and not 1 <= args.longest_at <= args.max_steps:
        parser.error("--longest-at must be one of the steps")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from peft import __version__ as peft_version
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    run_dir = OUTPUTS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    amp = args.device == "cuda" and args.precision == "fp16"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    commit = git_commit()

    language, utterances = load_set(args.train_set)
    if args.limit_train:
        utterances = utterances[: args.limit_train]
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=DTYPES[args.precision], attn_implementation="sdpa"
    )
    state_folder = resume_folder(run_dir) if args.resume else None
    if state_folder is not None:
        model = resume_model(model, state_folder)
    else:
        model = add_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)  # initialised on the CPU
    lora = lora_summary(model)
    check_lora(lora, model.config)
    model.to(args.device)

    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if args.resume else {}
    run.setdefault("starts", []).append({"time": now(), "commit": commit, "resume_from": str(state_folder)})
    run.update(
        {
            "args": vars(args),
            "commit": commit,
            "lora_targets": LORA_TARGETS,
            "lora": lora,
            "train_utterances": len(utterances),
            "train_hours": round(sum(u.seconds for u in utterances) / 3600, 2),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft_version,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        }
    )
    run_path.write_text(json.dumps(run, indent=1, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(lora), flush=True)
    if args.max_steps == 0:
        model.save_pretrained(str(run_dir / "checkpoint-0"))
        print(f"wrote {run_dir / 'checkpoint-0'} (the initial adapter)", flush=True)
        return

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fused=args.device == "cuda",
    )
    scheduler = linear_schedule(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.amp.GradScaler(args.device, enabled=amp)
    done = saved_step(state_folder) if state_folder is not None else 0
    model.train()

    longest = longest_indices(utterances, args.batch_size * args.grad_accum) if args.longest_at else None
    sampler = StepBatches(
        len(utterances),
        args.batch_size,
        args.grad_accum,
        args.seed,
        done + 1,
        args.max_steps,
        longest,
        args.longest_at,
    )
    loader = torch.utils.data.DataLoader(
        UtteranceDataset(utterances),
        batch_sampler=sampler,
        collate_fn=QwenCollator(processor, LANGUAGES[language]),
        num_workers=args.workers,
        # Windows spawns workers; one iterator for the whole run starts them once.
        persistent_workers=args.workers > 0,
    )
    batches = iter(loader)  # draws the workers' base seed, so it comes before the saved random state
    if state_folder is not None:
        restore_state(state_folder, optimizer, scaler, scheduler)
    steps_path = run_dir / "steps.jsonl"

    def log(entry: dict) -> None:
        with steps_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(entry) + "\n")
        if "event" in entry or entry["step"] % 10 == 0 or entry["step"] <= 3 or entry["skipped"]:
            print(json.dumps(entry), flush=True)

    def after_step(step: int) -> None:
        if args.save_steps and step % args.save_steps == 0:
            model.save_pretrained(str(run_dir / f"checkpoint-{step}"))
            log({"event": "checkpoint", "step": step, "time": now()})
        if step % args.resume_every == 0 or step == args.max_steps:
            save_state(run_dir / "resume", model, optimizer, scaler, scheduler, step)

    log({"event": "start", "first_step": done + 1, "last_step": args.max_steps, "time": now()})
    settings = LoopSettings(
        first_step=done + 1,
        last_step=args.max_steps,
        accum=args.grad_accum,
        device=args.device,
        amp=amp,
        max_grad_norm=args.max_grad_norm,
        pause_flag=run_dir / "PAUSE",
        longest_at=args.longest_at,
    )
    summary = run_steps(model, batches, optimizer, scheduler, scaler, settings, log, after_step)
    log({"event": "end", **summary, "time": now()})

    entries = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()]
    steps = {e["step"]: e for e in entries if "event" not in e}  # a resumed run logs some steps twice
    times = [e["seconds"] for e in steps.values() if not e["longest"]]
    run["finished"] = {
        "time": now(),
        "steps": len(steps),
        "skipped_steps": sorted(s for s, e in steps.items() if e["skipped"]),
        "nonfinite_steps": sorted(s for s, e in steps.items() if not math.isfinite(e["loss"])),
        "median_step_seconds": round(statistics.median(times), 3) if times else None,
        "peak_allocated_mb": max((e["peak_allocated_mb"] or 0) for e in steps.values()),
        "paused_seconds": round(sum(e["seconds"] for e in entries if e.get("event") == "pause"), 1),
    }
    run_path.write_text(json.dumps(run, indent=1, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(run["finished"]), flush=True)


if __name__ == "__main__":
    main()
