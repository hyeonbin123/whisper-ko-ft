# [테스트 묶음] scripts/speed.py(속도·GPU 메모리 측정 도우미)의 프리셋이 기록된 속도 수치를 만든 명령과
# 같은지, 그리고 GPU 없이 확인할 수 있는 순수 함수(nvidia-smi 줄 읽기, 쉬는 상태 판단, 요약 꺼내기)를
# 확인한다.
# [왜 중요한가] README의 속도 비율(CTranslate2 1.46배, Qwen 약 6.5배)은 이 절차로 잰 것이다. 세 측정
# 스크립트를 하나로 합치면서 명령·순서·환경 변수가 달라지면, 다시 잰 수치가 같은 절차의 수치가 아니게 된다.
# 그래서 그때 쓴 명령 목록을 여기에 그대로 적어 두고 프리셋과 비교한다.
# [범위] 측정을 돌리지 않는다(GPU, nvidia-smi, PowerShell 없이 돈다).

import importlib.util
from pathlib import Path

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
