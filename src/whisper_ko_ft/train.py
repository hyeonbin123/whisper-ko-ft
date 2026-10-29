"""준비한 묶음으로 Whisper 모델을 파인튜닝한다: 전체 파인튜닝 또는 LoRA, fp16 혼합 정밀도.

사용법:
    uv run python -m whisper_ko_ft.train --run-name small-a --model openai/whisper-small --learning-rate 1e-5
    uv run python -m whisper_ko_ft.train --run-name turbo-l1 --learning-rate 1e-4 --lora-r 16 ...
    멈춘 실행은 같은 명령에 --resume을 더하면 끝까지 저장된 마지막 체크포인트에서 이어 간다

후보들이 함께 쓰는 설정이 여기의 기본값이며, docs/experiments.md에 고정되어 있다.
outputs/<run-name>/에 쓴다: 체크포인트, `best/`(val-500 CER이 가장 낮은 것: 모델, 또는 --lora-r이나
--init-adapter를 주면 LoRA 어댑터, 그리고 처리기), `run.json`(시작할 때마다 starts에 한 줄), `hang.log`(학습이
15분 넘게 나아가지 않으면 주 프로세스의 스레드 스택을 덧붙인다).
그 폴더에 이미 학습 결과(run.json, best/, 체크포인트)가 있으면 --resume 없이는 시작하지 않는다.
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
# [결과물] outputs/<run>/: 체크포인트, best/(val-500 CER이 가장 낮은 것. LoRA면 어댑터만), run.json(시작 기록
# starts, 인자, 커밋, 기본 모델의 revision, 데이터 규모, torch·transformers·peft 판, 평가 이력, 최대 GPU
# 메모리), hang.log(멈춤 진단).
# [기본 모델 revision] 2026-10-29부터 --model이 등록한 Hub ID(whisper-small, large-v3-turbo)면 기록된 학습이
# 쓴 캐시 스냅숏과 같은 revision(provenance.WHISPER_REVISIONS)으로 처리기와 모델을 받는다. Hub의 모델이
# 바뀌어도 같은 가중치에서 시작하게 하려는 것이고, 가중치가 같아 학습 경로는 그대로다.
# [실행 폴더 보호] outputs/는 커밋하지 않으므로, 기록한 리포트를 낸 가중치는 그 폴더가 유일한 사본이다.
# 예전에는 같은 --run-name으로 다시 돌리면 run.json을 곧바로 덮어쓰고, 새 체크포인트를 저장하면서 예전
# 체크포인트를 지우고(save_total_limit 2), 끝에 best/를 덮어썼다. 그래서 학습 결과가 있는 폴더에서는 --resume
# 없이 시작하지 않는다(train_qwen과 같은 규칙과 문구). --resume은 Trainer가 체크포인트에 남긴
# 옵티마이저·학습률 일정·GradScaler·난수 상태로 이어 가고, run.json은 지우지 않고 starts에 시작을 덧붙인다.
# [연결] train_qwen이 UtteranceDataset을 가져다 쓴다. 학습한 LoRA는 evaluate --adapter로 재고, merge_adapter로
# 합쳐 faster-whisper로 잰다.

from __future__ import annotations

import argparse
import faulthandler
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# numpy: 학습 중 평가에서 라벨 패딩(-100)을 바꾸고 오류율을 계산한다. torch: 데이터셋과 dtype. transformers:
# Seq2SeqTrainer(학습 루프·평가·체크포인트 관리), Whisper 모델과 처리기, 멈춤 진단을 거는 TrainerCallback,
# 그리고 체크포인트 폴더 이름(checkpoint-<스텝>)과 Trainer가 체크포인트에 마지막으로 쓰는
# 파일(trainer_state.json)의 이름. 이름을 Trainer에서 가져와 판이 바뀌어도 Trainer가 쓰는 이름과 어긋나지 않게
# 한다.
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

# channel.telephone: 4단계에서 학습 음성 일부를 전화 음질로 바꾼다. evaluate: 평가와 같은 최대 토큰 수.
# metrics: val-500 채점. provenance: 학습 코드의 커밋 기록, 기본 모델의 고정 revision, run.json에 적는 peft
# 판. store: 묶음과 오디오 저장소.
from whisper_ko_ft.channel import telephone
from whisper_ko_ft.evaluate import MAX_NEW_TOKENS
from whisper_ko_ft.metrics import error_rate, score
from whisper_ko_ft.paths import OUTPUTS
from whisper_ko_ft.provenance import git_commit, peft_version, revision_kwargs, whisper_revision
from whisper_ko_ft.store import SAMPLE_RATE, AudioStore, Utterance, load_set

# 묶음의 언어 코드를 Whisper 토크나이저·generate가 받는 언어 이름으로 바꾼다.
LANGUAGE_NAMES = {"ko": "korean", "en": "english"}

# 멈춤 진단의 기다림(초): 학습 스텝도, 평가 배치도 끝나지 않은 채 이만큼 지나면 hang.log에 스택을 남긴다.
# 6단계 N 학습의 스텝당 약 12.7~27초(docs/experiments.md 6단계 결과)보다 훨씬 길고, 평가는 배치마다 다시
# 세므로 val-500 평가 전체의 길이와는 관계없다. 10단계 학습의 멈춤 감지 기준(steps.jsonl이 15분 넘게 늘지
# 않으면)과도 같다.
HANG_SECONDS = 900

# 체크포인트 폴더 이름(checkpoint-<스텝>)에서 스텝 수를 읽는다. 글자 순으로 고르면 checkpoint-500이
# checkpoint-1000보다 뒤가 되므로 스텝 수로 비교한다.
CHECKPOINT_NAME = re.compile(rf"^{PREFIX_CHECKPOINT_DIR}-(\d+)$")


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


# run.json의 시작 기록과 hang.log 머리 줄에 남길 지금 시각(초 단위, 이 PC의 시간대). train_qwen의 now와 같은
# 꼴이다(train_qwen이 이 모듈을 가져다 쓰므로 거꾸로 가져오면 순환 import가 된다).
def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# 이어 할 체크포인트: run_dir의 checkpoint-<스텝> 폴더 중 trainer_state.json이 있는 것 가운데 스텝이 가장 큰
# 것. 없으면(폴더가 없을 때도) None.
# 왜 trainer_state.json을 보나: Trainer는 체크포인트를 모델 → 옵티마이저·일정·GradScaler·난수 상태 →
# trainer_state.json 순서로 쓰고(transformers 5.17 Trainer._save_checkpoint), 이어 하기에는 그 파일이 꼭
# 있어야 한다. 저장 도중에 PID로 끝내면 그 파일이 없는 폴더가 남는데, transformers의 get_last_checkpoint는
# 폴더 이름만 보고 그런 폴더도 골라 이어 하기가 실패한다. 예전 체크포인트는 새 체크포인트를 끝까지 쓴 뒤에
# 지우므로, 끝까지 쓴 체크포인트가 하나는 남는다.
# 호출 위치: check_run_dir, tests/test_train_args.py.
def last_complete_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    found = []
    for folder in run_dir.iterdir():
        match = CHECKPOINT_NAME.match(folder.name)
        if match and folder.is_dir() and (folder / TRAINER_STATE_NAME).is_file():
            found.append((int(match.group(1)), folder))
    return max(found)[1] if found else None


# 실행 폴더 보호와 이어 할 체크포인트 고르기. 데이터·처리기·모델을 불러오기 전에(main이 parse_args 바로 뒤에)
# 부르므로, 거부할 때는 아무것도 읽거나 쓰지 않는다.
# - --resume이 없으면: run.json, best/, checkpoint-* 폴더(끝까지 저장됐는지와 관계없이) 중 하나라도 있으면
#   멈춘다. 셸 리디렉션으로 로그를 받으려고 미리 만든 빈 폴더나 train.log만 있는 폴더는 학습 결과가 아니라
#   통과한다.
# - --resume이면: 끝까지 저장된 체크포인트가 없으면 멈추고, 있으면 그 경로를 돌려준다. 더 늦은 체크포인트가
#   저장 도중에 끝나 건너뛸 때는 그 사실을 찍는다. 끝난 실행에 --resume을 주는 것(스모크에서 더 긴
#   --max-steps로 잇기 등)은 막지 않는다. 같은 이름을 실수로 다시 쓰는 것이 아니라 일부러 이어 가는 것이기
#   때문이다.
# 오류 문구는 train_qwen.parse_args처럼 짧은 영어 한 줄이다. 호출 위치: main.
def check_run_dir(parser: argparse.ArgumentParser, run_dir: Path, resume: bool) -> Path | None:
    latest = get_last_checkpoint(str(run_dir)) if run_dir.is_dir() else None
    if not resume:
        if (run_dir / "run.json").exists() or (run_dir / "best").exists() or latest is not None:
            parser.error(f"{run_dir} already has a run; pass --resume or pick another --run-name")
        return None
    checkpoint = last_complete_checkpoint(run_dir)
    if checkpoint is None:
        parser.error(f"{run_dir} has no complete checkpoint to resume from")
    if Path(latest) != checkpoint:
        print(f"skipping {Path(latest).name}: no {TRAINER_STATE_NAME} (its save did not finish)", flush=True)
    print(f"resuming from {checkpoint}", flush=True)
    return checkpoint


# 학습이 멈췄을 때 원인을 볼 수 있게, 정한 시간(HANG_SECONDS) 동안 진행(학습 스텝 끝, 평가 배치, 평가 끝)이
# 없으면 faulthandler가 주 프로세스의 모든 스레드 스택을 run 폴더의 hang.log에 덧붙이는 Trainer 콜백.
# 왜 넣었나: 7단계 첫 학습이 500스텝 평가의 DataLoader 워커를 띄우다가 원인을 모른 채
# 멈췄고(docs/experiments.md 7단계 결과), 그때는 사람이 py-spy를 붙이는 것 말고는 스택을 볼 방법이 없었다.
# 동작: 진행이 있을 때마다 dump_traceback_later를 다시 걸어 시간을 처음부터 센다(이전 감시는 faulthandler가
# 스스로 취소한다). repeat=True라 멈춘 동안 HANG_SECONDS마다 다시 쓰므로, 스택이 바뀌는지(느리지만 살아 있음)
# 그대로인지(멈춤)를 볼 수 있다. 프로세스는 끝내지 않는다(exit=False). 끝내는 것은 사람이 PID로 하고
# --resume으로 잇는다.
# 주의: 주 프로세스의 스택만 남는다. DataLoader 워커(Windows는 spawn으로 띄운 별도 프로세스) 안에서 멈추면 주
# 프로세스가 워커의 배치를 기다리는 자리만 보이고 워커 안은 보이지 않는다.
# 주의: faulthandler의 감시는 프로세스 전체에 하나뿐이고 파일 번호에 직접 쓴다. 그래서 닫을 때는 감시를 먼저
# 풀고 파일을 닫는다(닫힌 번호가 다른 파일에 다시 쓰이면 엉뚱한 파일에 쓰게 된다). 학습이 예외로 끝나도 main의
# finally가 close를 부른다. 파일을 닫아야 Windows에서 실행 폴더를 지울 수 있다.
# hang.log는 덧붙여 쓴다(이어 한 실행이 멈춘 앞 실행의 덤프를 지우지 않게). 열 때 시작 시각을 머리 줄로
# 남긴다.
# 호출 위치: main(Seq2SeqTrainer의 callbacks), tests/test_train_args.py.
class HangWatchdog(TrainerCallback):
    def __init__(self, path: Path, seconds: float = HANG_SECONDS) -> None:
        self.path = path
        self.seconds = seconds
        self.file = None

    # 감시를 (다시) 건다. 처음이면 hang.log를 덧붙이기로 열고 머리 줄을 쓴 뒤, faulthandler가 파일 번호에 직접
    # 쓰는 덤프와 섞이지 않게 바로 flush한다.
    def arm(self) -> None:
        if self.file is None:
            self.file = self.path.open("a", encoding="utf-8", newline="\n")
            self.file.write(
                f"--- {now()} watching; thread stacks follow if nothing moves for {self.seconds:g} s\n"
            )
            self.file.flush()
        faulthandler.dump_traceback_later(self.seconds, repeat=True, file=self.file)

    # 감시를 풀고 파일을 닫는다. 두 번 불러도 된다(on_train_end와 main의 finally가 모두 부른다).
    def close(self) -> None:
        if self.file is not None:
            faulthandler.cancel_dump_traceback_later()
            self.file.close()
            self.file = None

    # Trainer 사건: 학습 시작(첫 스텝 전 DataLoader 워커를 띄우는 동안도 본다), 옵티마이저 스텝 끝, 평가 배치
    # 하나 끝, 평가 끝에서 다시 걸고, 학습 끝에서 푼다. 반환값이 None이면 Trainer의 control은 그대로다.
    def on_train_begin(self, args, state, control, **kwargs):
        self.arm()

    def on_step_end(self, args, state, control, **kwargs):
        self.arm()

    def on_prediction_step(self, args, state, control, **kwargs):
        self.arm()

    def on_evaluate(self, args, state, control, **kwargs):
        self.arm()

    def on_train_end(self, args, state, control, **kwargs):
        self.close()


# 명령줄 진입점. 기본값은 후보들이 함께 쓰는 설정이다(위 docstring): small, 3,000스텝, warmup 500, 500스텝마다
# 평가, 장치당 16 × 누적 2.
# 후보별로 주는 것: 2단계 A/B는 --learning-rate 1e-5/3e-5. 3단계는 --model turbo --lora-r 16 --learning-rate
# 1e-4 --max-steps 1500(L2는 --lora-targets q_proj k_proj v_proj out_proj fc1 fc2). 4단계 T는 --telephone-prob
# 0.5. 6단계 N은 --train-set zeroth-train-itn --eval-set zeroth-val500-itn. 7단계 N2는 --init-adapter.
# --spec-augment는 과적합이 보일 때만 돌리기로 한 후보 C용이다(docs/experiments.md 2단계).
# --resume은 멈춘 실행을 같은 명령으로 잇는다(check_run_dir, HangWatchdog).
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        required=True,
        help="outputs/<이 이름>/에 쓴다. 그 폴더에 학습 결과(run.json, best/, checkpoint-*)가 있으면 "
        "--resume 없이는 시작하지 않는다(기록한 학습 결과를 덮어쓰지 않게)",
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="멈춘 실행을 outputs/<run-name>/의 마지막 체크포인트(trainer_state.json까지 끝까지 "
        "저장된 것)에서 잇는다. 처음과 같은 명령에 이것만 더한다. 옵티마이저·학습률 일정·GradScaler·난수 "
        "상태는 체크포인트에서 이어지고, run.json의 starts에 이번 시작이 덧붙는다",
    )
    args = parser.parse_args()
    # 실행 폴더 보호: 데이터·처리기·모델을 불러오기 전에 확인해, 거부할 때는 아무것도 읽거나 쓰지 않는다.
    run_dir = OUTPUTS / args.run_name
    resume_from = check_run_dir(parser, run_dir, args.resume)
    resolve_lora_args(args)

    # 학습 묶음과 평가 묶음(val-500)을 읽는다. --limit-*는 스모크 실행에서만 쓴다.
    language, train_utterances = load_set(args.train_set)
    _, eval_utterances = load_set(args.eval_set)
    if args.limit_train:
        train_utterances = train_utterances[: args.limit_train]
    if args.limit_eval:
        eval_utterances = eval_utterances[: args.limit_eval]

    # 기본 모델이 등록한 Hub ID면 고정한 revision으로 받는다(위 [기본 모델 revision]). outputs/ 아래 로컬
    # 폴더에는 revision이 없어 빈 dict가 되고 지금까지와 같은 호출이다.
    pinned = revision_kwargs(args.model)
    # 처리기는 기본 모델의 것을 쓴다. 토크나이저에 언어와 task 접두 토큰을 정해 두어 정답 토큰이 추론 때와
    # 같은 접두로 시작하게 한다.
    processor = WhisperProcessor.from_pretrained(args.model, **pinned)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE_NAMES[language], task="transcribe")
    # LoRA는 얼린 기본 가중치를 fp16으로 올려 11GB GPU에 맞추고(PEFT가 만드는 어댑터 가중치는 fp32다), 전체
    # 파인튜닝은 fp32 마스터 가중치로 올린 뒤 Trainer의 fp16 혼합 정밀도로 학습한다.
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16 if args.lora_r else torch.float32, **pinned
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
    # 실행 폴더 보호·이어 하기·멈춤 진단을 더할 때 이 설정 값은 하나도 바꾸지 않았다
    # (tests/test_train_args.py가 기본 인자의 설정 전체를 고정한다).
    output_dir = run_dir
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
    # 대신). callbacks의 HangWatchdog은 학습 계산에 관여하지 않고, 멈췄을 때 스택만 hang.log에 남긴다.
    watchdog = HangWatchdog(output_dir / "hang.log")
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
        callbacks=[watchdog],
    )

    # 학습을 시작하기 전에 run.json을 먼저 쓴다(시작 기록, 인자, 커밋, 기본 모델의 revision, 학습 발화 수와
    # 시간, 평가 발화 수, torch·transformers·peft 판, GPU). 학습이 도중에 죽어도 어떤 설정으로 돌렸는지
    # 남는다.
    # starts: 시작할 때마다 시각·커밋·이어 한 체크포인트(새로 시작하면 null)를 한 줄 덧붙인다(train_qwen과
    # 같은 꼴). 이어 하기면 기존 run.json을 읽어 그 뒤에 덧붙이고, 맨 위의 인자·커밋·데이터 규모·판·GPU는
    # 이번 시작의 값으로 바꾼다(train_qwen과 같다. 시작마다의 커밋은 starts에 남는다). starts가 생기기 전에
    # 만든 run.json을 이어 하면 그 실행의 커밋을 starts 첫 줄로 옮겨 잃지 않는다(그때의 시작 시각은 기록이
    # 없어 null).
    # revision은 등록한 Hub ID면 고정한 sha, 로컬 폴더면 null. peft는 LoRA 실행(--lora-r 또는 --init-adapter,
    # resolve_lora_args가 lora_r을 채운다)에서만 적고 전체 파인튜닝은 null이다. 2026-10-29에 더한 세 키라 그
    # 전의 run.json에는 없다.
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if resume_from and run_path.exists() else {}
    if "starts" not in run and "commit" in run:
        run["starts"] = [{"time": None, "commit": run["commit"], "resume_from": None}]
    commit = git_commit()
    run.setdefault("starts", []).append(
        {"time": now(), "commit": commit, "resume_from": str(resume_from) if resume_from else None}
    )
    run.update(
        {
            "args": vars(args),
            "commit": commit,
            "revision": whisper_revision(args.model),
            "train_utterances": len(train_utterances),
            "train_hours": round(sum(u.seconds for u in train_utterances) / 3600, 2),
            "eval_utterances": len(eval_utterances),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft_version() if args.lora_r else None,
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    run_path.write_text(json.dumps(run, indent=1), encoding="utf-8", newline="\n")

    # 이어 하기면 Trainer가 그 체크포인트에서 모델(LoRA면 어댑터)·옵티마이저·학습률 일정·GradScaler·난수
    # 상태와 trainer_state.json(스텝, 평가 이력, 가장 좋은 체크포인트)을 읽고, 데이터를 그 스텝까지 건너뛴 뒤
    # 잇는다. 학습이 예외로 끝나도 멈춤 감시를 풀고 hang.log를 닫는다.
    try:
        trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    finally:
        watchdog.close()

    # 학습 중에는 gradient checkpointing과 맞지 않아 껐던 KV 캐시를 추론용으로 다시 켜고 저장한다. best/에는
    # 가장 좋은 체크포인트(LoRA면 어댑터만)와 처리기가 들어간다.
    model.config.use_cache = True
    trainer.save_model(str(output_dir / "best"))
    processor.save_pretrained(str(output_dir / "best"))
    # run.json에 평가 이력(스텝별 val-500 CER), 고른 체크포인트, 그 CER, 학습 중 최대 GPU 메모리를 더해 다시
    # 쓴다. docs/experiments.md의 단계별 val-500 수치가 여기서 나온다. 이어 한 실행의 평가 이력에는 이어 하기
    # 전의 평가도 들어간다(trainer_state.json에서 이어짐). 최대 GPU 메모리는 이 프로세스(마지막 시작)의
    # 값이다.
    history = [entry for entry in trainer.state.log_history if "eval_cer" in entry]
    run["eval_history"] = [{"step": e["step"], "cer": round(e["eval_cer"], 5)} for e in history]
    run["best_checkpoint"] = trainer.state.best_model_checkpoint
    run["best_cer"] = trainer.state.best_metric
    run["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    run_path.write_text(json.dumps(run, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(run["eval_history"]))


if __name__ == "__main__":
    main()
