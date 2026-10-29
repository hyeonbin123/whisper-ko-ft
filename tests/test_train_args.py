# [테스트 묶음] train.py의 LoRA 인자 처리(resolve_lora_args)와, --init-adapter만 준 실행이 실제로 어댑터를
# 불러오는지 확인한다. 그리고 실행 폴더 보호(같은 --run-name이면 시작하지 않음), --resume(끝까지 저장된 마지막
# 체크포인트에서 잇기, run.json의 starts 덧붙이기, 이미 --max-steps에 닿은 실행은 거부), 멈춤 진단(진행이
# 없으면 스레드 스택을 hang.log에 남기는 콜백), 기본 모델 revision 고정(등록한 Hub ID에만)과 run.json의 환경
# 기록(revision, transformers·peft 판)을 확인한다.
# [왜 중요한가] 예전 코드는 --lora-r만 보고 LoRA 여부를 정해서, --lora-r 없이 --init-adapter만 주면 어댑터를
# 버리고 기본 모델 전체를 fp32로 학습했다(커밋 8ad5a18에서 고침). 오류 없이 다른 실험이 되어 버리는 버그라
# 테스트로 막는다. 또 outputs/는 커밋하지 않아 기록한 리포트를 낸 가중치의 유일한 사본인데, 예전 코드는 같은
# --run-name으로 다시 돌리면 run.json·체크포인트·best/를 경고 없이 덮어썼다. 몇 시간짜리 학습이 멈추면 이어 할
# 방법도 없었다.
# [방법] 모델·처리기·Trainer·faulthandler를 monkeypatch로 가짜로 바꿔, 내려받기나 GPU 없이 main이 어디까지
# 가는지와 무엇을 남기는지만 본다. 진짜 faulthandler는 짧은 시간으로 한 번만 확인한다.

import argparse
import json
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import peft
import pytest
import torch
import transformers
from peft import LoraConfig

from whisper_ko_ft import train
from whisper_ko_ft.provenance import WHISPER_REVISIONS
from whisper_ko_ft.train import resolve_lora_args


# train.main이 parse_args로 만드는 것과 같은 LoRA 관련 인자 묶음. 기본값은 전체 파인튜닝(rank 0)이고 바꿀 값만
# 넘긴다.
def lora_args(**changes) -> argparse.Namespace:
    values = {
        "lora_r": 0,
        "lora_alpha": None,
        "lora_dropout": 0.05,
        "lora_targets": ["q_proj", "v_proj"],
        "init_adapter": None,
    }
    return argparse.Namespace(**{**values, **changes})


# rank 16, alpha 32, dropout 0.05, 대상 q_proj·fc1인 어댑터 설정을 폴더에 저장한다(저장된 어댑터 흉내).
def save_adapter(folder) -> None:
    LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "fc1"], bias="none"
    ).save_pretrained(folder)


# --init-adapter를 주면 명령줄의 --lora-* 값(dropout 0.1, q/v 대상)이 아니라 저장된 어댑터의 설정을 쓴다. 대상
# 층은 정렬된 목록으로 run.json에 남는다.
def test_init_adapter_implies_lora_with_the_saved_settings(tmp_path):
    save_adapter(tmp_path)
    args = lora_args(lora_dropout=0.1, init_adapter=str(tmp_path))

    resolve_lora_args(args)

    used = (args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_targets)
    assert used == (16, 32, 0.05, ["fc1", "q_proj"])  # --lora-* 값이 아니다


# 새 LoRA에서 alpha를 주지 않으면 2 × rank가 된다(3단계 rank 16 → alpha 32).
def test_default_alpha_is_twice_the_rank():
    args = lora_args(lora_r=8)
    resolve_lora_args(args)
    assert args.lora_alpha == 16


# LoRA를 쓰지 않는 전체 파인튜닝(2단계)은 인자가 바뀌지 않는다.
def test_full_fine_tuning_is_untouched():
    args = lora_args()
    resolve_lora_args(args)
    assert (args.lora_r, args.lora_alpha) == (0, None)


# main이 확인하려는 지점까지 왔음을 알리려고 일부러 던지는 예외.
class Reached(Exception):
    pass


# 명령줄에 --init-adapter만 주고 main을 돌리면 기본 모델을 fp16으로 올리고(LoRA 경로) 저장된 어댑터를 불러와야
# 한다. 어댑터를 건너뛰면 Trainer 설정 단계에서 멈추게 해 두어, 고치기 전 동작(fp32로 올리고 어댑터를
# 건너뜀)이면 seen 값이 달라져 실패한다.
def test_init_adapter_without_lora_r_loads_the_base_in_fp16_and_the_adapter(tmp_path, monkeypatch):
    save_adapter(tmp_path)
    seen = {}

    def fake_model(*args, **kwargs):
        seen["dtype"] = kwargs["dtype"]
        return SimpleNamespace(
            generation_config=SimpleNamespace(), config=SimpleNamespace(decoder_start_token_id=1)
        )

    def fake_adapter(model, path, **kwargs):
        seen["adapter"] = path
        raise Reached

    def training_starts(**kwargs):
        raise Reached

    processor = SimpleNamespace(tokenizer=SimpleNamespace(set_prefix_tokens=lambda **kwargs: None))
    # 데이터, 처리기, 모델, PEFT 불러오기, Trainer 설정을 모두 가짜로 바꾼다. monkeypatch는 테스트가 끝나면
    # 원래대로 되돌린다. 실행 폴더 검사가 이 PC의 진짜 outputs/를 보지 않게 빈 임시 폴더를 준다.
    monkeypatch.setattr(train, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(train, "load_set", lambda name: ("ko", []))
    monkeypatch.setattr(train.WhisperProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(train.WhisperForConditionalGeneration, "from_pretrained", fake_model)
    monkeypatch.setattr("peft.PeftModel.from_pretrained", fake_adapter)
    monkeypatch.setattr(train, "Seq2SeqTrainingArguments", training_starts)  # 어댑터를 건너뛰면 여기서 멈춘다
    argv = ["train", "--run-name", "x", "--learning-rate", "5e-5", "--init-adapter", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(Reached):
        train.main()

    assert seen == {"dtype": torch.float16, "adapter": str(tmp_path)}


# --- 실행 폴더 보호, --resume, 멈춤 진단 ---


# faulthandler 대신 넣는 가짜. 감시를 거는(arm) 호출과 푸는(cancel) 호출을 순서대로 남기고, 넘겨받은 파일
# 객체도 모아 둔다. 진짜 faulthandler의 감시는 프로세스 전체에 하나뿐이라, 가짜 Trainer 테스트에서 900초짜리
# 감시를 진짜로 걸어 두면 한참 뒤 다른 테스트 도중에 스택을 쓸 수 있다.
class FakeFaulthandler:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.files: list = []

    def dump_traceback_later(self, timeout, repeat=False, file=None, exit=False):
        self.calls.append(("arm", timeout, repeat, file.name, file.closed))
        self.files.append(file)

    def cancel_dump_traceback_later(self):
        self.calls.append(("cancel",))


# main이 내려받기·GPU 없이 끝까지 돌도록 바꾼다: 묶음, 처리기, 모델, Trainer 설정과 Trainer, 커밋, GPU 이름과
# 최대 메모리, faulthandler. 데이터·처리기·모델을 불러오면 seen["loaded"]에 남고(보호 검사가 그보다 먼저인지
# 보려고), from_pretrained가 받은 키워드 인자(dtype, revision)는 seen["kwargs"]에 종류별로 남는다. 가짜
# Trainer의 train은 받은 resume_from_checkpoint와 그 순간의 run.json을 남기고, 진짜 Trainer처럼 콜백의 학습
# 시작·스텝 끝·평가 배치·평가 끝·학습 끝을 차례로 부른다. fail이면 스텝 끝 뒤에 예외를 낸다(학습이 죽는 경우).
def fake_training(monkeypatch, outputs, fail: bool = False) -> dict:
    seen: dict = {"loaded": [], "saved": [], "kwargs": {}, "faulthandler": FakeFaulthandler()}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks") or []
            seen["callbacks"] = self.callbacks
            self.state = SimpleNamespace(
                log_history=[{"step": 500, "loss": 0.31}, {"step": 500, "eval_cer": 0.02512}],
                best_model_checkpoint=str(outputs / "x" / "checkpoint-500"),
                best_metric=0.02512,
            )

        def train(self, resume_from_checkpoint=None):
            seen["resume_from_checkpoint"] = resume_from_checkpoint
            seen["run_at_start"] = json.loads((outputs / "x" / "run.json").read_text(encoding="utf-8"))
            for event in ("on_train_begin", "on_step_end", "on_prediction_step", "on_evaluate"):
                for callback in self.callbacks:
                    getattr(callback, event)(None, self.state, None)
                if fail and event == "on_step_end":
                    raise RuntimeError("worker died")
            for callback in self.callbacks:
                callback.on_train_end(None, self.state, None)

        def save_model(self, path):
            seen["saved"].append(path)

    def loader(kind: str, value):
        def load(*args, **kwargs):
            seen["loaded"].append(kind)
            seen["kwargs"][kind] = kwargs
            return value

        return load

    def training_args(**kwargs):
        seen["training_args"] = kwargs
        return SimpleNamespace(**kwargs)

    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(set_prefix_tokens=lambda **kwargs: None),
        save_pretrained=lambda path: None,
    )
    model = SimpleNamespace(
        generation_config=SimpleNamespace(), config=SimpleNamespace(decoder_start_token_id=1)
    )
    monkeypatch.setattr(train, "OUTPUTS", outputs)
    monkeypatch.setattr(train, "load_set", loader("data", ("ko", [])))
    monkeypatch.setattr(train.WhisperProcessor, "from_pretrained", loader("processor", processor))
    monkeypatch.setattr(train.WhisperForConditionalGeneration, "from_pretrained", loader("model", model))
    monkeypatch.setattr(train, "Seq2SeqTrainingArguments", training_args)
    monkeypatch.setattr(train, "Seq2SeqTrainer", FakeTrainer)
    monkeypatch.setattr(train, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(train.torch.cuda, "get_device_name", lambda index: "Fake GPU")
    monkeypatch.setattr(train.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(train, "faulthandler", seen["faulthandler"])
    return seen


# 실행 이름 x, 학습률 1e-5(README 예시와 같은 꼴)에 extra 인자를 더해 main을 부른다.
def run_main(monkeypatch, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--run-name", "x", "--learning-rate", "1e-5", *extra])
    train.main()


# 학습 결과(run.json, best/, 체크포인트 중 하나라도)가 있는 폴더에 --resume 없이 같은 이름으로 시작하면,
# 데이터·처리기·모델을 불러오기 전에 멈추고 폴더에 아무것도 쓰지 않는다. 고치기 전에는 run.json을 곧바로
# 덮어쓰고, 새 체크포인트를 저장할 때 예전 체크포인트를 지웠으며(save_total_limit 2), 끝에 best/를 덮어썼다.
@pytest.mark.parametrize("artifact", ["run.json", "best", "checkpoint-2500"])
def test_an_existing_run_is_never_overwritten(tmp_path, monkeypatch, capsys, artifact):
    seen = fake_training(monkeypatch, tmp_path)
    run_dir = tmp_path / "x"
    run_dir.mkdir()
    if artifact == "run.json":
        (run_dir / "run.json").write_text('{"commit": "35087f3"}', encoding="utf-8")
    else:
        (run_dir / artifact).mkdir()
    before = sorted(path.name for path in run_dir.iterdir())

    with pytest.raises(SystemExit):
        run_main(monkeypatch)

    assert "already has a run; pass --resume or pick another --run-name" in capsys.readouterr().err
    assert seen["loaded"] == []  # 불러오기 전에 멈춘다
    assert sorted(path.name for path in run_dir.iterdir()) == before  # run.json·hang.log도 새로 생기지 않는다
    if artifact == "run.json":
        assert (run_dir / "run.json").read_text(encoding="utf-8") == '{"commit": "35087f3"}'


# 학습 결과가 없는 폴더(셸 리디렉션으로 로그를 받으려고 미리 만든 폴더 같은 것)에서는 새로 시작한다.
# run.json은 학습을 시작하기 전에 쓰고, starts에는 이번 시작(시각, 커밋, 이어 한 체크포인트 없음) 한 줄이
# 남는다. Trainer 설정은 보호·이어 하기·진단을 더하기 전과 같다(측정한 학습과 같은 계산이어야 한다).
def test_a_fresh_run_records_one_start_before_training_with_the_same_settings(tmp_path, monkeypatch):
    seen = fake_training(monkeypatch, tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "train.log").write_text("", encoding="utf-8")

    run_main(monkeypatch)

    assert seen["resume_from_checkpoint"] is None
    (start,) = seen["run_at_start"]["starts"]
    assert (start["commit"], start["resume_from"]) == ("abc1234", None)
    assert start["time"] == datetime.fromisoformat(start["time"]).isoformat(timespec="seconds")
    assert seen["run_at_start"]["args"]["resume"] is False
    run = json.loads((tmp_path / "x" / "run.json").read_text(encoding="utf-8"))
    assert run["starts"] == [start]
    assert run["eval_history"] == [{"step": 500, "cer": 0.02512}]
    assert (run["best_checkpoint"], run["best_cer"]) == (str(tmp_path / "x" / "checkpoint-500"), 0.02512)
    assert seen["saved"] == [str(tmp_path / "x" / "best")]
    assert seen["training_args"] == {
        "output_dir": str(tmp_path / "x"),
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "lr_scheduler_type": "linear",
        "warmup_steps": 500,
        "max_steps": 3000,
        "fp16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_strategy": "steps",
        "eval_steps": 500,
        "save_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "predict_with_generate": True,
        "logging_steps": 25,
        "dataloader_num_workers": 4,
        "dataloader_persistent_workers": True,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "report_to": [],
        "seed": 0,
    }


# --resume인데 이어 할 체크포인트가 없으면(폴더가 없거나, run.json만 있거나, 체크포인트가 저장 도중에 끝나
# trainer_state.json이 없으면) 데이터·처리기·모델을 불러오기 전에 멈춘다.
@pytest.mark.parametrize("layout", ["no folder", "run.json only", "incomplete checkpoint"])
def test_resume_needs_a_complete_checkpoint(tmp_path, monkeypatch, capsys, layout):
    seen = fake_training(monkeypatch, tmp_path)
    run_dir = tmp_path / "x"
    if layout != "no folder":
        run_dir.mkdir()
        (run_dir / "run.json").write_text("{}", encoding="utf-8")
    if layout == "incomplete checkpoint":
        (run_dir / "checkpoint-500").mkdir()
        (run_dir / "checkpoint-500" / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--resume")

    assert "has no complete checkpoint to resume from" in capsys.readouterr().err
    assert seen["loaded"] == []


# --resume은 끝까지 저장된(trainer_state.json이 있는) 체크포인트 중 스텝이 가장 큰 것에서 잇는다(이름의 글자
# 순이 아니라 스텝 수 순). 저장 도중 끝난 더 늦은 체크포인트는 건너뛰고 그 사실을 찍는다. run.json은 지우지
# 않고 starts에 이번 시작을 덧붙이며 앞 시작의 기록은 그대로 둔다. hang.log도 덧붙여 써서, 멈춘 앞 실행의 스택
# 덤프가 남는다.
def test_resume_picks_the_last_complete_checkpoint_and_keeps_the_records(tmp_path, monkeypatch, capsys):
    seen = fake_training(monkeypatch, tmp_path)
    run_dir = tmp_path / "x"
    run_dir.mkdir()
    first = {"time": "2026-10-24T01:00:00", "commit": "0db1c24", "resume_from": None}
    (run_dir / "run.json").write_text(json.dumps({"starts": [first], "commit": "0db1c24"}), encoding="utf-8")
    (run_dir / "hang.log").write_text("Timeout (0:15:00)!\n", encoding="utf-8")
    for step, complete in ((500, True), (1000, True), (1500, False)):
        (run_dir / f"checkpoint-{step}").mkdir()
        if complete:
            (run_dir / f"checkpoint-{step}" / "trainer_state.json").write_text("{}", encoding="utf-8")

    run_main(monkeypatch, "--resume")

    last = str(run_dir / "checkpoint-1000")
    assert seen["resume_from_checkpoint"] == last
    assert "checkpoint-1500" in capsys.readouterr().out
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["starts"][0] == first
    assert [start["resume_from"] for start in run["starts"]] == [None, last]
    assert run["starts"][1]["commit"] == run["commit"] == "abc1234"
    assert run["args"]["resume"] is True
    assert (run_dir / "hang.log").read_text(encoding="utf-8").startswith("Timeout (0:15:00)!\n--- ")


# 이 기능 전에 만든 run.json(starts가 없고 커밋만 있음)을 이어 하면, 그 실행의 커밋을 starts 첫 줄로 옮겨 잃지
# 않는다(그때의 시작 시각은 기록이 없어 null).
def test_resuming_a_run_recorded_before_starts_keeps_its_first_commit(tmp_path, monkeypatch):
    fake_training(monkeypatch, tmp_path)
    run_dir = tmp_path / "x"
    (run_dir / "checkpoint-500").mkdir(parents=True)
    (run_dir / "checkpoint-500" / "trainer_state.json").write_text("{}", encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({"commit": "35087f3"}), encoding="utf-8")

    run_main(monkeypatch, "--resume")

    starts = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["starts"]
    assert starts[0] == {"time": None, "commit": "35087f3", "resume_from": None}
    assert starts[1]["resume_from"] == str(run_dir / "checkpoint-500")


# Trainer가 체크포인트에 쓰는 것과 같은 trainer_state.json(TrainerState.save_to_json)을 단 체크포인트 폴더를
# 만든다. 이어 하기는 이 파일의 global_step에서 잇는다.
def saved_checkpoint(run_dir, step: int, max_steps: int) -> None:
    folder = run_dir / f"checkpoint-{step}"
    folder.mkdir(parents=True)
    transformers.TrainerState(global_step=step, max_steps=max_steps).save_to_json(
        str(folder / "trainer_state.json")
    )


# 마지막 체크포인트가 이미 --max-steps에 닿은(끝난) 실행에 --resume을 주면, 데이터·처리기·모델을 불러오기
# 전에 멈추고 폴더에 아무것도 쓰지 않는다. transformers 5.17의 Trainer는 이어 하기 전에 이를 확인하지 않아,
# 고치기 전에는 같은 명령으로 다시 돌리면(1500) 옵티마이저 스텝을 하나 더 돌고 평가한 뒤
# checkpoint-<max+1>을 저장했다(save_total_limit 2 회전으로 앞 체크포인트가 지워지고, 그 체크포인트가 가장
# 좋으면 best/도 바뀐다). 더 작은 --max-steps(1000)도 더 학습할 것이 없으므로 같이 거부한다. 그때 Trainer가
# 스텝을 더 도는지는 에폭 계산에 달렸지만, 어느 쪽이든 새 프로세스가 best/와 run.json을 다시 쓴다.
@pytest.mark.parametrize("max_steps", ["1500", "1000"])
def test_a_finished_run_is_not_resumed(tmp_path, monkeypatch, capsys, max_steps):
    seen = fake_training(monkeypatch, tmp_path)
    run_dir = tmp_path / "x"
    for step in (1000, 1500):
        saved_checkpoint(run_dir, step, 1500)
    (run_dir / "run.json").write_text('{"commit": "0db1c24"}', encoding="utf-8")
    before = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*"))

    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--resume", "--max-steps", max_steps)

    err = capsys.readouterr().err
    assert f"checkpoint-1500 is already at step 1500 of --max-steps {max_steps}" in err
    assert "nothing left to train; pass a larger --max-steps to extend" in err
    assert seen["loaded"] == []  # 불러오기 전에 멈춘다
    assert sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*")) == before  # hang.log도 없다
    assert (run_dir / "run.json").read_text(encoding="utf-8") == '{"commit": "0db1c24"}'


# 더 긴 --max-steps로 잇는 것은 그대로 된다(13단계 스모크의 checkpoint-2 → --resume --max-steps 4).
def test_a_run_can_be_extended_with_a_larger_max_steps(tmp_path, monkeypatch, capsys):
    seen = fake_training(monkeypatch, tmp_path)
    run_dir = tmp_path / "x"
    saved_checkpoint(run_dir, 2, 2)
    (run_dir / "run.json").write_text("{}", encoding="utf-8")

    run_main(monkeypatch, "--resume", "--max-steps", "4")

    assert seen["resume_from_checkpoint"] == str(run_dir / "checkpoint-2")
    assert seen["training_args"]["max_steps"] == 4
    assert "(step 2 of --max-steps 4)" in capsys.readouterr().out


# 고른 체크포인트의 trainer_state.json을 읽을 수 없으면(저장 도중 끝나 반만 쓰인 파일 등) 불러오기 전에 그
# 파일을 짚어 멈춘다. 그대로 두면 데이터와 모델을 다 불러온 뒤 Trainer가 JSON 오류로 멈춘다.
def test_an_unreadable_trainer_state_stops_before_loading(tmp_path, monkeypatch, capsys):
    seen = fake_training(monkeypatch, tmp_path)
    run_dir = tmp_path / "x"
    (run_dir / "checkpoint-500").mkdir(parents=True)
    (run_dir / "checkpoint-500" / "trainer_state.json").write_text('{"global_step": 5', encoding="utf-8")

    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--resume")

    assert "cannot read the step of checkpoint-500 from trainer_state.json" in capsys.readouterr().err
    assert seen["loaded"] == []


# 멈춤 진단: 학습을 시작할 때 hang.log를 열어 감시(900초, 반복)를 걸고, 스텝 끝·평가 배치·평가 끝마다 다시
# 걸어 시간을 처음부터 센다. 학습이 끝나면 감시를 먼저 풀고 파일을 닫는다.
def test_the_hang_watchdog_is_rearmed_on_progress_and_released_at_the_end(tmp_path, monkeypatch):
    seen = fake_training(monkeypatch, tmp_path)

    run_main(monkeypatch)

    arm = ("arm", 900, True, str(tmp_path / "x" / "hang.log"), False)
    assert seen["faulthandler"].calls == [arm, arm, arm, arm, ("cancel",)]
    assert all(file.closed for file in seen["faulthandler"].files)
    assert (tmp_path / "x" / "hang.log").read_text(encoding="utf-8").startswith("--- ")


# 학습이 예외로 끝나도 감시를 풀고 hang.log를 닫는다. 그래야 감시 스레드가 닫힌 파일 번호에 쓰지 않고,
# Windows에서 실행 폴더를 지울 수 있다.
def test_the_hang_watchdog_is_released_when_training_fails(tmp_path, monkeypatch):
    seen = fake_training(monkeypatch, tmp_path, fail=True)

    with pytest.raises(RuntimeError, match="worker died"):
        run_main(monkeypatch)

    assert seen["faulthandler"].calls[-1] == ("cancel",)
    assert seen["faulthandler"].files and all(file.closed for file in seen["faulthandler"].files)


# 진짜 faulthandler로 확인: 진행 없이 정한 시간이 지나면 hang.log에 모든 스레드의 스택이 쓰이고(기다리는 이
# 테스트 함수가 주 스레드 스택에 보인다), 닫은 뒤에는 Windows에서도 파일을 지울 수 있다. 두 번 닫아도 괜찮다.
# 시간은 0.2초로 줄였다.
def test_the_hang_watchdog_writes_every_thread_stack_after_the_timeout(tmp_path):
    path = tmp_path / "hang.log"
    watchdog = train.HangWatchdog(path, seconds=0.2)
    watchdog.on_train_begin(None, None, None)
    try:
        deadline = time.monotonic() + 10
        while "Timeout" not in path.read_text(encoding="utf-8") and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        watchdog.close()
    watchdog.close()

    text = path.read_text(encoding="utf-8")
    assert text.startswith("--- ")
    assert "Timeout (0:00:00.200000)!" in text
    assert "test_the_hang_watchdog_writes_every_thread_stack_after_the_timeout" in text
    path.unlink()


# --- 기본 모델 revision 고정과 run.json의 환경 기록 ---


# 등록한 Hub ID(기본값 openai/whisper-small)면 처리기·모델 from_pretrained에 고정한 revision을 넘기고,
# run.json에 그 revision과 transformers 판을 적는다. 전체 파인튜닝은 peft를 쓰지 않으므로 peft는 null이다.
def test_the_run_record_pins_the_revision_and_names_the_library_versions(tmp_path, monkeypatch):
    seen = fake_training(monkeypatch, tmp_path)

    run_main(monkeypatch)

    small = WHISPER_REVISIONS["openai/whisper-small"]
    assert seen["kwargs"]["processor"] == {"revision": small}
    assert seen["kwargs"]["model"] == {"dtype": torch.float32, "revision": small}
    run = json.loads((tmp_path / "x" / "run.json").read_text(encoding="utf-8"))
    assert (run["revision"], run["transformers"], run["peft"]) == (small, transformers.__version__, None)


# outputs/ 아래 로컬 폴더를 기본 모델로 주면 revision을 넘기지 않고 run.json의 revision은 null이다. LoRA
# 실행(--lora-r)은 어댑터를 만든 peft 판을 적는다(get_peft_model은 가짜로 바꿔 가중치 없이 돈다).
def test_a_local_base_model_gets_no_revision_and_a_lora_run_names_peft(tmp_path, monkeypatch):
    seen = fake_training(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "peft.get_peft_model",
        lambda model, config: SimpleNamespace(**vars(model), print_trainable_parameters=lambda: None),
    )

    run_main(monkeypatch, "--model", "outputs/turbo-n/merged", "--lora-r", "16")

    assert seen["kwargs"]["processor"] == {}
    assert seen["kwargs"]["model"] == {"dtype": torch.float16}
    run = json.loads((tmp_path / "x" / "run.json").read_text(encoding="utf-8"))
    assert (run["revision"], run["transformers"], run["peft"]) == (
        None,
        transformers.__version__,
        peft.__version__,
    )
