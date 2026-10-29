# [테스트 묶음] scripts/speed.py(속도·GPU 메모리 측정 도우미)의 프리셋이 기록된 속도 수치를 만든 명령과
# 같은지, GPU 없이 확인할 수 있는 순수 함수(nvidia-smi 줄 읽기, 쉬는 상태 판단, 요약 꺼내기), 그리고
# speed.json에 측정 코드의 커밋(scripts/ 자신까지 본 +dirty 표시)이 남는지 확인한다. 2026-10-29부터는 측정
# 하네스의 견고성도 본다: 실행 하나가 예외로 끝나도 표본 프로세스(nvidia-smi, PowerShell)를 끝내는지, 1분보다
# 짧은 확인 간격을 거부하는지, 중간에 멈춘 측정이 앞 실행의 기록을 speed.json에 complete: false로 남기는지,
# 캐시 데우기가 저장소마다 memmap을 한 번만 여는지.
# [왜 중요한가] README의 속도 비율(CTranslate2 1.46배, Qwen 약 6.5배)은 이 절차로 잰 것이다. 세 측정
# 스크립트를 하나로 합치면서 명령·순서·환경 변수가 달라지면, 다시 잰 수치가 같은 절차의 수치가 아니게 된다.
# 그래서 그때 쓴 명령 목록을 여기에 그대로 적어 두고 프리셋과 비교한다. 하네스 리포트에는 커밋이 남지만
# speed.json에는 없어서, 어느 절차(스크립트 판)로 잰 속도인지 파일만으로는 알 수 없었다. 끝나지 않고 남은
# 표본 프로세스는 다음 측정의 쉬는 상태 판정(CPU 1분 평균)과 사용률 표본을 흐리고, 끝에 한 번만 쓰던
# speed.json은 마지막 실행이 실패하면 앞 실행의 기록까지 모두 잃게 했다.
# [범위] 측정을 돌리지 않는다(GPU, nvidia-smi, PowerShell 없이 돈다). 커밋 기록·부분 기록 테스트는 대기·캐시
# 데우기·실행을 가짜로 바꿔 main을 돌리고, 표본 프로세스 테스트는 subprocess를 가짜로 바꿔 run()을 돌린다.

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# scripts/는 패키지가 아니라서 파일 경로로 불러온다.
@pytest.fixture(scope="module")
def speed():
    spec = importlib.util.spec_from_file_location("speed", ROOT / "scripts" / "speed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PY = ["uv", "run", "--no-sync", "python", "-m"]
SET = "zeroth-val500"
# 2026-10-02 측정에 쓴 명령(그때 스크립트의 RUNS를 펼친 것).
HF_1002 = [*PY, "whisper_ko_ft.evaluate", "--model", "openai/whisper-large-v3-turbo", "--adapter",
           "outputs/turbo-n/best", "--set", SET]  # fmt: skip
CT2_1002 = [*PY, "whisper_ko_ft.evaluate_ct2", "--model", "outputs/turbo-n/ct2", "--set", SET]
RUNS_1002 = [
    ("hf-b1-1", [*HF_1002, "--name", "turbo-n-b1", "--batch-size", "1"]),
    ("ct2-1", [*CT2_1002, "--name", "turbo-n-ct2"]),
    ("hf-b1-2", [*HF_1002, "--name", "turbo-n-b1", "--batch-size", "1", "--no-report"]),
    ("ct2-2", [*CT2_1002, "--name", "turbo-n-ct2", "--no-report"]),
    ("hf-b16", [*HF_1002, "--name", "turbo-n-b1", "--batch-size", "16", "--no-report"]),
]
# 8단계 측정에 쓴 명령.
HF_STAGE8 = [*PY, "whisper_ko_ft.evaluate", "--model", "openai/whisper-large-v3-turbo", "--adapter",
             "outputs/turbo-n/best", "--name", "speed", "--set", SET, "--batch-size", "1",
             "--no-report"]  # fmt: skip
RUNS_STAGE8 = [
    ("d0-1", HF_STAGE8),
    ("d1a-1", [*HF_STAGE8, "--fallback", "ratio"]),
    ("d1-1", [*HF_STAGE8, "--fallback", "ratio-logprob"]),
    ("d0-2", HF_STAGE8),
    ("d1a-2", [*HF_STAGE8, "--fallback", "ratio"]),
    ("d1-2", [*HF_STAGE8, "--fallback", "ratio-logprob"]),
]
# 9단계 측정에 쓴 명령(Qwen에 넘긴 추가 인자는 없었다).
TURBO_STAGE9 = [*PY, "whisper_ko_ft.evaluate", "--model", "openai/whisper-large-v3-turbo", "--name", "speed",
                "--set", SET, "--no-report"]  # fmt: skip
QWEN_STAGE9 = [*PY, "whisper_ko_ft.evaluate_qwen", "--set", SET, "--no-report"]
RUNS_STAGE9 = [
    (f"{label}-{n}", [*command, "--batch-size", batch])
    for n in (1, 2)
    for label, command, batch in (
        ("turbo-b1", TURBO_STAGE9, "1"),
        ("qwen-b1", QWEN_STAGE9, "1"),
        ("turbo-b16", TURBO_STAGE9, "16"),
        ("qwen-b16", QWEN_STAGE9, "16"),
    )
]


# 세 프리셋의 실행 목록(라벨, 명령, 순서)과 환경 변수가 그때와 같다.
@pytest.mark.parametrize(
    ("preset", "expected", "env"),
    [
        ("ct2-n", RUNS_1002, {"PYTHONUTF8": "1"}),
        ("fallback-n", RUNS_STAGE8, {"PYTHONUTF8": "1"}),
        ("qwen-turbo", RUNS_STAGE9, {"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}),
    ],
)
def test_presets_repeat_the_recorded_commands(speed, preset, expected, env):
    runs, env_extra = speed.build_runs(preset, [])
    assert runs == expected and env_extra == env


# Qwen에 넘기는 추가 인자와 fp32일 때의 큰 배치는 Qwen 실행에만 들어간다(그때의 sys.argv와 QWEN_LARGE_BATCH).
def test_qwen_arguments_and_large_batch_go_to_the_qwen_runs_only(speed):
    runs, _ = speed.build_runs("qwen-turbo", ["--precision", "fp32"], "4")
    commands = dict(runs)
    assert commands["qwen-b16-1"][-4:] == ["--precision", "fp32", "--batch-size", "4"]
    assert commands["turbo-b16-1"][-2:] == ["--batch-size", "16"]
    assert "--precision" not in commands["turbo-b1-2"]


# 12단계 프리셋: Transformers 배치 1 → CTranslate2 float16 → int8_float16을 두 번, 모두 리포트 없이.
def test_the_int8_preset_alternates_three_runs_twice_without_reports(speed):
    runs, _ = speed.build_runs("ct2-int8", [])
    assert [label for label, _ in runs] == [
        "hf-b1-1", "ct2-fp16-1", "ct2-int8-1", "hf-b1-2", "ct2-fp16-2", "ct2-int8-2",
    ]  # fmt: skip
    assert all("--no-report" in command for _, command in runs)
    assert dict(runs)["ct2-int8-2"][-2:] == ["--compute-type", "int8_float16"]


# --speed-only는 리포트를 남기던 두 실행에만 --no-report를 붙이고, 나머지 명령은 그대로 둔다.
def test_speed_only_adds_no_report_where_it_was_missing(speed):
    runs = speed.speed_only(RUNS_1002)
    assert all(command.count("--no-report") == 1 for _, command in runs)
    assert runs[2:] == RUNS_1002[2:]


# nvidia-smi --format=csv,noheader,nounits의 한 줄과 프로세스 표 읽기.
def test_nvidia_smi_lines_are_read(speed):
    assert speed.parse_gpu_line("1599, 37") == (1599, 37)
    table = "\n".join(
        [
            "| Processes:                                                     |",
            "|  GPU   GI   CI        PID   Type   Process name     GPU Memory |",
            "|        ID   ID                                     Usage      |",
            "|================================================================|",
            "|    0   N/A  N/A      4242    C+G   ...\\game.exe      N/A      |",
            "+----------------------------------------------------------------+",
        ]
    )
    assert speed.parse_gpu_processes(table) == ["0   N/A  N/A      4242    C+G   ...\\game.exe      N/A"]
    assert speed.parse_gpu_processes("No devices were found") == []


# 쉬는 상태 기준: GPU 사용률 1분 평균 10% 이하이고 CPU 1분 평균 15% 미만(둘 다).
def test_idle_needs_both_a_quiet_gpu_and_a_quiet_cpu(speed):
    assert speed.is_idle({"gpu_util_avg": 10.0, "cpu_percent_avg": 14.9})
    assert not speed.is_idle({"gpu_util_avg": 10.1, "cpu_percent_avg": 5.0})
    assert not speed.is_idle({"gpu_util_avg": 3.0, "cpu_percent_avg": 15.0})


# evaluate_qwen은 raw와 fixed 요약 두 개를 찍는다. 첫 요약(raw)을 꺼낸다.
def test_the_first_printed_summary_is_read(speed):
    text = 'loading\n{\n  "output": "raw",\n  "error_rate": 0.1\n}\n{\n  "output": "fixed"\n}\nwrote x\n'
    assert speed.first_summary(text) == {"output": "raw", "error_rate": 0.1}


# speed.json에 측정 코드의 커밋이 남는다. 하네스와 같은 git_commit을 쓰되 scripts/(이 스크립트 자신)까지 보아,
# 스크립트를 고친 채 재면 +dirty가 붙는다. 대기·캐시 데우기·실행·끝 조건은 가짜로 바꿔 GPU 없이 main을 돌린다.
def test_speed_json_records_the_commit_including_the_scripts_folder(speed, tmp_path, monkeypatch):
    seen = {}

    def fake_commit(root, paths):
        seen["root"], seen["paths"] = root, tuple(paths)
        return "abc1234+dirty"

    monkeypatch.setattr(speed, "git_commit", fake_commit)
    monkeypatch.setattr(speed, "wait_for_idle", lambda max_wait, poll: (True, []))
    monkeypatch.setattr(speed, "warm_cache", lambda: None)
    monkeypatch.setattr(speed, "conditions", lambda: {"time": "t"})
    monkeypatch.setattr(speed, "run", lambda label, command, env_extra, out_dir: {"label": label})
    monkeypatch.setattr(sys, "argv", ["speed.py", "--preset", "ct2-int8", "--out", str(tmp_path)])

    speed.main()

    payload = json.loads((tmp_path / "speed.json").read_text(encoding="utf-8"))
    assert payload["commit"] == "abc1234+dirty"
    assert seen == {"root": speed.ROOT, "paths": ("src", "pyproject.toml", "uv.lock", "scripts")}
    assert [record["label"] for record in payload["runs"]] == [
        "hf-b1-1", "ct2-fp16-1", "ct2-int8-1", "hf-b1-2", "ct2-fp16-2", "ct2-int8-2",
    ]  # fmt: skip
    # 끝까지 돈 측정은 complete: true이고 끝난 뒤의 확인(after)까지 남는다.
    assert payload["complete"] is True and payload["after"] == {"time": "t"}


# --- 측정 하네스 견고성 (2026-10-29) ---


# speed.json은 실행마다 그때까지의 기록으로 다시 쓴다(complete: false, 끝까지 돌면 true). 둘째 실행에서
# 하네스가 실패하거나(run이 SystemExit(1)) 사람이 멈추면(KeyboardInterrupt) 첫 실행의 기록과 쉬는 상태 확인이
# 남는다. 고치기 전에는 모든 실행이 끝난 뒤에만 써서, 최대 대기까지 기다린 측정이 마지막에 실패하면 앞 실행의
# 기록까지 모두 잃었다(analyze verdict12는 speed.json이 있어야 한다).
@pytest.mark.parametrize("failure", [SystemExit(1), KeyboardInterrupt()])
def test_a_failed_run_keeps_the_earlier_runs_as_incomplete(speed, tmp_path, monkeypatch, failure):
    def run(label, command, env_extra, out_dir):
        if label != "hf-b1-1":
            raise failure
        return {"label": label}

    def after():
        raise AssertionError("the check after the runs ran")

    monkeypatch.setattr(speed, "git_commit", lambda root, paths: "abc1234")
    monkeypatch.setattr(speed, "wait_for_idle", lambda max_wait, poll: (False, [{"time": "check"}]))
    monkeypatch.setattr(speed, "warm_cache", lambda: None)
    monkeypatch.setattr(speed, "conditions", after)
    monkeypatch.setattr(speed, "run", run)
    monkeypatch.setattr(sys, "argv", ["speed.py", "--preset", "ct2-int8", "--out", str(tmp_path)])

    with pytest.raises(type(failure)):
        speed.main()

    payload = json.loads((tmp_path / "speed.json").read_text(encoding="utf-8"))
    assert payload["complete"] is False and "after" not in payload
    assert payload["runs"] == [{"label": "hf-b1-1"}]
    assert payload["commit"] == "abc1234"
    assert (payload["idle_reached"], payload["checks"]) == (False, [{"time": "check"}])


# --poll-minutes는 1 이상만 받는다. 한 번의 확인이 1분(CHECK_SECONDS)이라 그보다 짧으면 확인 사이의 sleep
# 길이가 음수가 되어 첫 확인 뒤에 ValueError로 죽었다(nan은 round에서 죽었다). 쉬는 상태를 기다리기 전에
# 멈춘다.
@pytest.mark.parametrize("minutes", ["0.5", "nan"])
def test_a_poll_interval_below_one_minute_is_refused(speed, tmp_path, monkeypatch, capsys, minutes):
    def wait(max_wait, poll):
        raise AssertionError("waited for the idle state")

    monkeypatch.setattr(speed, "git_commit", lambda root, paths: "abc1234")
    monkeypatch.setattr(speed, "wait_for_idle", wait)
    argv = ["speed.py", "--preset", "ct2-int8", "--out", str(tmp_path), "--poll-minutes", minutes]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        speed.main()
    assert "--poll-minutes" in capsys.readouterr().err
    assert not (tmp_path / "speed.json").exists()


# run()의 바깥 프로세스를 가짜로 바꾸는 fixture. subprocess.Popen 대역은 표본 프로세스를 띄우지 않고 받은
# 호출(terminate, wait)만 적으며, stdout으로 받은 파일에 표본 한 줄을 써 두어(nvidia-smi는 "메모리, 사용률",
# PowerShell은 "시각,사용률") run()이 실행 뒤 CSV를 읽는 길도 GPU 없이 돈다. subprocess.run 대역은 하네스
# 명령에 state.harness(결과 또는 던질 예외)를 돌려주고 taskkill은 적기만 한다. state.hang이면 terminate 뒤 첫
# wait가 시간 초과를 낸다(끝나지 않는 프로세스). state.fail_popen에 실행 파일 이름을 주면 그 Popen이
# FileNotFoundError를 낸다(PATH에 없음).
@pytest.fixture
def processes(speed, monkeypatch):
    state = SimpleNamespace(samplers=[], commands=[], harness=None, hang=False, fail_popen=None)

    class Sampler:
        def __init__(self, args, stdout=None, **kwargs):
            if args[0] == state.fail_popen:
                raise FileNotFoundError(args[0])
            self.args, self.calls, self.pid = args, [], 4000 + len(state.samplers)
            state.samplers.append(self)
            stdout.write("1500, 40\n" if args[0] == "nvidia-smi" else "12:00:00,25\n")

        def terminate(self):
            self.calls.append("terminate")

        def wait(self, timeout=None):
            self.calls.append("wait")
            if state.hang and self.calls.count("wait") == 1:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return 1

    def run(command, **kwargs):
        state.commands.append(command)
        if command[0] == "taskkill":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if isinstance(state.harness, BaseException):
            raise state.harness
        return state.harness

    fake = SimpleNamespace(
        Popen=Sampler, run=run, TimeoutExpired=subprocess.TimeoutExpired, PIPE=subprocess.PIPE
    )
    monkeypatch.setattr(speed, "subprocess", fake)
    monkeypatch.setattr(speed, "gpu_sample", lambda: (1000, 3))
    return state


# 하네스 요약 대역(evaluate_ct2가 찍는 JSON 요약의 일부).
FINISHED = SimpleNamespace(
    returncode=0, stdout='{\n "audio_seconds_per_second": 20.0,\n "error_rate": 0.1\n}\nwrote\n', stderr=""
)
HARNESS = [*PY, "whisper_ko_ft.evaluate_ct2", "--set", SET, "--no-report"]


# 하네스를 띄우지 못하거나(uv가 PATH에 없음: FileNotFoundError) 실행 중에 멈추면(KeyboardInterrupt) 예외는
# 그대로 나가되, 두 표본 프로세스는 terminate한 뒤 끝나기를 기다린다. 고치기 전에는 subprocess.run과 terminate
# 사이에 try/finally가 없어 두 프로세스가 남았다.
@pytest.mark.parametrize("error", [FileNotFoundError("uv"), KeyboardInterrupt()])
def test_the_samplers_are_stopped_when_the_harness_does_not_finish(speed, processes, tmp_path, error):
    processes.harness = error
    with pytest.raises(type(error)):
        speed.run("ct2-1", HARNESS, {}, tmp_path)
    assert [sampler.args[0] for sampler in processes.samplers] == ["nvidia-smi", "powershell.exe"]
    assert all(sampler.calls == ["terminate", "wait"] for sampler in processes.samplers)


# PowerShell 표본을 띄우지 못하면(FileNotFoundError) 먼저 띄운 nvidia-smi 표본을 끝내고, 하네스는 돌리지
# 않는다.
def test_the_gpu_sampler_is_stopped_when_the_cpu_sampler_cannot_start(speed, processes, tmp_path):
    processes.fail_popen = "powershell.exe"
    with pytest.raises(FileNotFoundError):
        speed.run("ct2-1", HARNESS, {}, tmp_path)
    assert [(sampler.args[0], sampler.calls) for sampler in processes.samplers] == [
        ("nvidia-smi", ["terminate", "wait"])
    ]
    assert processes.commands == []


# terminate 뒤 정한 시간 안에 끝나지 않는 표본 프로세스는 taskkill /T /F로 자식까지 트리째 끝내고 다시
# 기다린다(Windows의 Popen.kill은 terminate와 같은 TerminateProcess라 다시 불러도 다를 것이 없다). 고치기
# 전에는 시간 제한 없이 기다렸다.
def test_a_sampler_that_does_not_exit_is_ended_with_its_tree(speed, processes, tmp_path):
    processes.harness, processes.hang = FINISHED, True
    speed.run("ct2-1", HARNESS, {}, tmp_path)
    assert all(sampler.calls == ["terminate", "wait", "wait"] for sampler in processes.samplers)
    killed = [["taskkill", "/T", "/F", "/PID", str(sampler.pid)] for sampler in processes.samplers]
    assert processes.commands == [HARNESS, *killed]


# 정상 실행: 표본 프로세스를 끝내고, 하네스 요약과 두 CSV의 표본으로 기록을 만든다(고치기 전과 같은 기록).
def test_a_finished_run_stops_the_samplers_and_reads_their_samples(speed, processes, tmp_path):
    processes.harness = FINISHED
    record = speed.run("ct2-1", HARNESS, {}, tmp_path)
    assert all(sampler.calls == ["terminate", "wait"] for sampler in processes.samplers)
    assert processes.commands == [HARNESS]
    assert record["summary"] == {"audio_seconds_per_second": 20.0, "error_rate": 0.1}
    memory = [record[f"gpu_memory_{key}_mb"] for key in ("before", "max", "added")]
    assert memory == [1000, 1500, 500]
    assert (record["gpu_util_avg_during"], record["cpu_percent_avg_during"]) == (40.0, 25.0)


# 캐시 데우기도 저장소마다 memmap을 한 번만 연다(하네스와 같다). 고치기 전에는 setdefault의 기본값 인자
# AudioStore(...)가 키가 있어도 먼저 계산되어 발화마다 새 memmap을 열었다.
def test_warming_the_cache_opens_each_store_once(speed, monkeypatch):
    from whisper_ko_ft import store

    opened = []

    class CountingStore:
        def __init__(self, name):
            opened.append(name)

        def read(self, utterance):
            return np.zeros(utterance.length, dtype=np.float32)

    names = zip("abcd", "xyxy", strict=True)
    utterances = [store.Utterance(i, "s", "", name, 0, 16_000) for i, name in names]
    monkeypatch.setattr(store, "AudioStore", CountingStore)
    monkeypatch.setattr(store, "load_set", lambda name: ("ko", utterances))

    speed.warm_cache()

    assert opened == ["x", "y"]
