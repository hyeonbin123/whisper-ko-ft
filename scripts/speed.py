"""속도·GPU 메모리 측정 도우미(Windows 전용). README와 docs/experiments.md의 배속·지연 수치를 만든 절차.

사용법(저장소 폴더에서, GPU 한 장, 다른 GPU 작업이 없을 때):
    uv run --no-sync python scripts/speed.py --preset ct2-n --speed-only
        (2026-10-02 N: Transformers 대 CTranslate2)
    uv run --no-sync python scripts/speed.py --preset fallback-n
        (8단계: 재시도 디코딩의 배치 1 속도)
    uv run --no-sync python scripts/speed.py --preset qwen-turbo
        (9단계: Qwen3-ASR-1.7B 대 turbo)
    uv run --no-sync python scripts/speed.py --preset ct2-int8 --max-wait-minutes 10 --poll-minutes 3
        (12단계: N의 CTranslate2 float16 대 int8_float16)
    uv run --no-sync python scripts/speed.py --preset ct2-n --dry-run
        (돌릴 명령만 찍는다)

절차(프리셋마다 같다): 쉬는 상태(다른 프로그램의 GPU 사용률 10% 이하, CPU 1분 평균 15% 미만)를 기다린다.
1분짜리 확인을 --poll-minutes마다 하고, --max-wait-minutes 안에 쉬는 상태가 오지 않으면 그대로 잰다. 그다음
묶음의 오디오를 한 번 읽어 파일 캐시를 데우고, 프리셋의 실행을 정해진 순서로 하나씩 돌린다. 실행마다
nvidia-smi를 0.5초 간격으로 기록해 GPU 메모리 증가(최댓값 − 실행 직전 값)와 사용률을 내고, 하네스가 찍은
요약(배속, 배치 시간, torch 최대 할당, 오류율)을 모은다. 결과는 --out 폴더(기본
outputs/speed/<프리셋>-<시각>/)의 speed.json과 실행별 CSV에 남는다. speed.json은 쉬는 상태 확인 뒤와 실행이
하나 끝날 때마다 그때까지의 기록으로 다시 쓰고(complete: false), 끝난 뒤의 확인까지 마치면 complete: true로
바꾼다. 중간에 멈춘 측정도 앞 실행의 기록이 남는다(analyze verdict12는 complete: false인 파일을 거부한다).
"""

# [파일 역할] 공개 저장소만으로 속도 수치를 다시 잴 수 있게 하는 측정 스크립트. README의 "CTranslate2로
# 1.46배"(2026-10-02), 8단계의 "재시도 디코딩을 켜도 배치 1 속도가 그대로", 9단계의 "한 발화씩 처리하면
# Qwen이 turbo보다 약 6.5배 느림"을 만든 절차다. 그때 쓴 스크립트 세 개는 공통 부분이 같아(쉬는 상태 대기,
# 캐시 데우기, nvidia-smi 표본, 요약 읽기) 하나로 합치고 실행 목록만 프리셋으로 남겼다. 프리셋의 명령
# 목록·순서·환경 변수·대기 기준은 그때와 같다 (tests/test_speed_script.py가 그때의 명령 목록과 같음을
# 고정한다).
# [그때와 다른 점] (1) 결과 파일 위치(--out). (2) 기록 항목을 더했다: 쉬는 상태 확인마다 CPU를 많이 쓴
# 프로세스 이름과 nvidia-smi의 프로세스 목록, 실행마다 CPU 사용률 표본(1~2초 간격). 9단계 기록의 정정("실행
# 중 CPU는 기록하지 않았다")을 다음 측정에서 메우려는 것이다. 측정 명령과 순서는 바꾸지 않았다. (3)
# --max-wait-minutes와 --poll-minutes (그때는 60분·5분으로 고정). 2026-10-04부터는 속도 측정도 최대 10분만
# 기다리고, 쉬는 상태가 아니면 그대로 재서 사용률과 함께 돈 프로그램을 수치 옆에 적는다(docs/experiments.md
# 12단계). (4) ct2-int8 프리셋(12단계)을 더했다.
# (5) --speed-only: 그때 첫 두 실행이 남긴 리포트가 이미 커밋되어 있어, 다시 재면 그 실행이 덮어쓰기 보호로
# 멈춘다. 이 옵션은 모든 실행에 --no-report를 붙여 속도만 잰다. (6) 2026-10-29부터 speed.json에 측정 코드의
# 커밋(commit)을 적는다. 하네스 리포트의 commit과 같은 git_commit이되 scripts/(이 파일)까지 보아, 측정 절차를
# 고친 채 잰 것은 +dirty로 드러난다. (7) 2026-10-29부터 측정 하네스가 실패에 견딘다: 실행 하나가 예외로
# 끝나도(uv가 PATH에 없음, Ctrl+C 등) 두 표본 프로세스를 끝내고(run의 try/finally와 stop), --poll-minutes가 1
# 미만이면 시작하기 전에 거부하며, speed.json을 실행마다 다시 써서(complete 키) 중간에 멈춰도 앞 실행의 기록이
# 남는다. 그전에는 남은 표본 프로세스(1초마다 Get-CimInstance를 부르는 PowerShell, nvidia-smi)가 다음 측정의
# 쉬는 상태 판정과 사용률 표본을 흐릴 수 있었고, 확인 간격이 1분보다 짧으면 첫 확인 뒤에 음수 sleep으로
# 죽었으며, 마지막 실행이 실패하면 앞 실행의 기록까지 모두 잃었다. 하네스와 함께 캐시 데우기도 저장소마다
# memmap을 한 번만 연다(warm_cache). 측정 명령·순서·대기 기준은 그대로다.
# [주의] Windows 전용이다: CPU 사용률과 프로세스 이름을 PowerShell(Get-CimInstance, Get-Process)로 읽는다.
# nvidia-smi와 uv가 PATH에 있어야 한다. 쉬는 상태 확인의 Ollama 항목은 같은 PC에서 GPU를 나눠 쓰는 로컬
# Ollama 서버에 올라간 모델을 적는 것이고, 서버가 없으면 "unreachable"로 남는다(측정에는 영향이 없다).
# 이 프로세스만 PID로 끝내면(Stop-Process, 작업 관리자의 '작업 끝내기') finally가 돌지 않아 표본 프로세스를
# 정리하지 못한다. 측정을 멈출 때는 콘솔에서 Ctrl+C를 누르거나 `taskkill /T /F /PID <이 프로세스 PID>`로
# 자식까지 트리째 끝낸다(Job Object로 묶으면 부모가 죽을 때 자식도 끝나게 할 수 있지만 Windows API 코드가
# 늘어나 쓰지 않았다).

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ROOT: 저장소 폴더(이 파일은 scripts/ 안에 있다). 모든 실행은 이 폴더에서 돈다.
ROOT = Path(__file__).resolve().parents[1]
# 저장소의 src/를 import 경로에 넣어, 이 스크립트를 어디서 돌려도 whisper_ko_ft를 찾게 한다(아래 provenance와
# warm_cache의 store). provenance는 torch·transformers를 불러오지 않아, 측정 명령을 자식 프로세스로 돌리는 이
# 부모 프로세스가 가벼운 채로 남는다. import가 모듈 중간에 오므로 E402 검사를 끈다.
sys.path.insert(0, str(ROOT / "src"))
from whisper_ko_ft.provenance import CODE_PATHS, git_commit  # noqa: E402

# COMMIT_PATHS: speed.json에 적는 커밋의 +dirty 판단에 보는 경로. 하네스가 보는 코드·잠금(CODE_PATHS:
# src·pyproject.toml·uv.lock)에 이 스크립트가 있는 scripts/를 더한다.
COMMIT_PATHS = (*CODE_PATHS, "scripts")
# 속도 묶음: Zeroth val-500(440발화, 1.07시간). 모든 프리셋이 같은 묶음을 같은 순서로 처리한다.
SET = "zeroth-val500"
PY = ["uv", "run", "--no-sync", "python", "-m"]
# 쉬는 상태 기준(2026-10-02 규칙부터 같다): 다른 프로그램을 포함한 GPU 사용률 1분 평균 10% 이하, CPU 1분
# 평균 15% 미만.
GPU_IDLE_UTIL = 10
CPU_IDLE = 15
# 한 번의 확인에 걸리는 시간(초). 5초 간격 표본 12개다. --poll-minutes는 이보다 짧을 수 없다(확인 사이의
# sleep이 poll − CHECK_SECONDS라 음수가 된다).
CHECK_SECONDS = 60
# 표본 프로세스를 끝낼 때 한 단계마다 기다리는 시간(초). Windows의 terminate(TerminateProcess)는 보통 바로
# 끝나므로 넉넉한 값이고, 정상 실행에서는 기다리는 일이 없다.
STOP_SECONDS = 10

# 1분 동안 CPU 전체 사용률을 5초마다 12번 읽어 평균·최댓값을 첫 줄에 찍고, 그 1분 동안 CPU 시간을 가장 많이 쓴
# 프로세스 다섯 개(이름:초)를 둘째 줄에 찍는다. Win32_PerfFormattedData는 Windows 언어와 관계없이 이름이 같다.
CPU_PS = (
    "$b = @{}; Get-Process | ForEach-Object { if ($_.CPU -ne $null) { $b[$_.Id] = $_.CPU } }; "
    "$s = 1..12 | ForEach-Object { (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor "
    "-Filter \"Name='_Total'\").PercentProcessorTime; Start-Sleep -Seconds 5 }; "
    '$m = $s | Measure-Object -Average -Maximum; "$($m.Average) $($m.Maximum)"; '
    "$t = Get-Process | Where-Object { $_.CPU -ne $null -and $b.ContainsKey($_.Id) } | "
    "ForEach-Object { [pscustomobject]@{ N = $_.ProcessName; S = $_.CPU - $b[$_.Id] } } | "
    "Sort-Object S -Descending | Select-Object -First 5 | "
    "ForEach-Object { \"$($_.N):$([math]::Round($_.S, 1))\" }; $t -join ' '"
)
# 실행 중 CPU 표본: 끝날 때까지 1초 남짓 간격으로 "시각,사용률"을 찍는다(실행이 끝나면 프로세스를 끝낸다).
CPU_SAMPLER_PS = (
    "while ($true) { $c = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor "
    "-Filter \"Name='_Total'\").PercentProcessorTime; "
    "\"$(Get-Date -Format 'HH:mm:ss'),$c\"; Start-Sleep -Seconds 1 }"
)


# --- 프리셋: 실행 목록 (라벨, 명령, 환경 변수) ---


def preset_ct2_n() -> tuple[list[tuple[str, list[str]]], dict]:
    """2026-10-02 'CTranslate2와 Transformers의 추론 속도': N(turbo + LoRA)을 두 엔진으로 번갈아.

    Transformers 배치 1 → CTranslate2 → Transformers 배치 1 → CTranslate2 → Transformers 배치 16. 처음 두
    실행은 리포트(reports/turbo-n-b1/, reports/turbo-n-ct2/의 zeroth-val500.json)를 남겼다. 결과:
    1.46배(26.9 대 18.4배속).
    """
    hf = [
        *PY,
        "whisper_ko_ft.evaluate",
        "--model",
        "openai/whisper-large-v3-turbo",
        "--adapter",
        "outputs/turbo-n/best",
        "--set",
        SET,
    ]
    ct2 = [*PY, "whisper_ko_ft.evaluate_ct2", "--model", "outputs/turbo-n/ct2", "--set", SET]
    runs = [
        ("hf-b1-1", [*hf, "--name", "turbo-n-b1", "--batch-size", "1"]),
        ("ct2-1", [*ct2, "--name", "turbo-n-ct2"]),
        ("hf-b1-2", [*hf, "--name", "turbo-n-b1", "--batch-size", "1", "--no-report"]),
        ("ct2-2", [*ct2, "--name", "turbo-n-ct2", "--no-report"]),
        ("hf-b16", [*hf, "--name", "turbo-n-b1", "--batch-size", "16", "--no-report"]),
    ]
    return runs, {"PYTHONUTF8": "1"}


def preset_fallback_n() -> tuple[list[tuple[str, list[str]]], dict]:
    """8단계 '속도': N의 배치 1에서 greedy(D0), 압축비 재시도(D1a), 압축비 + 로그확률 재시도(D1)를 번갈아.

    D0 → D1a → D1을 두 번 돈다. 결과: 재시도를 켜도 배치 1 속도가 줄지 않았다.
    """
    hf = [
        *PY,
        "whisper_ko_ft.evaluate",
        "--model",
        "openai/whisper-large-v3-turbo",
        "--adapter",
        "outputs/turbo-n/best",
        "--name",
        "speed",
        "--set",
        SET,
        "--batch-size",
        "1",
        "--no-report",
    ]
    arms = {"d0": [], "d1a": ["--fallback", "ratio"], "d1": ["--fallback", "ratio-logprob"]}
    runs = [(f"{arm}-{n}", [*hf, *extra]) for n in (1, 2) for arm, extra in arms.items()]
    return runs, {"PYTHONUTF8": "1"}


def preset_qwen_turbo(qwen_args: list[str], qwen_large_batch: str = "16") -> tuple[list, dict]:
    """9단계 '속도와 GPU 메모리': 기준선 turbo(Transformers)와 Qwen3-ASR-1.7B, 배치 1과 16, 두 번.

    turbo 배치 1 → Qwen 배치 1 → turbo 배치 16 → Qwen 배치 16을 두 번. qwen_args는 evaluate_qwen에 그대로
    넘긴다 (예: fp16 점검에 실패했을 때 --precision fp32). qwen_large_batch는 그때 Qwen의 '배치 16' 실행에
    쓸 배치이고 라벨은 b16 그대로 둔다(요약에 실제 배치가 남는다). 결과: 배치 1에서 약 6.5배 느림.
    """
    turbo = [
        *PY,
        "whisper_ko_ft.evaluate",
        "--model",
        "openai/whisper-large-v3-turbo",
        "--name",
        "speed",
        "--set",
        SET,
        "--no-report",
    ]
    qwen = [*PY, "whisper_ko_ft.evaluate_qwen", "--set", SET, "--no-report", *qwen_args]

    def batch_size(model: str, batch: int) -> str:
        return qwen_large_batch if model == "qwen" and batch == 16 else str(batch)

    runs = [
        (f"{model}-b{batch}-{n}", [*command, "--batch-size", batch_size(model, batch)])
        for n in (1, 2)
        for batch in (1, 16)
        for model, command in (("turbo", turbo), ("qwen", qwen))
    ]
    return runs, {"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}


def preset_ct2_int8() -> tuple[list[tuple[str, list[str]]], dict]:
    """12단계(2026-10-15 규칙): N을 Transformers 배치 1, CTranslate2 float16, int8_float16으로 두 번 번갈아.

    세 실행 모두 리포트를 남기지 않는다(속도와 메모리만). CTranslate2는 불러올 때 양자화하므로 같은 변환
    폴더를 쓴다.
    """
    hf = [
        *PY,
        "whisper_ko_ft.evaluate",
        "--model",
        "openai/whisper-large-v3-turbo",
        "--adapter",
        "outputs/turbo-n/best",
        "--name",
        "speed",
        "--set",
        SET,
        "--batch-size",
        "1",
        "--no-report",
    ]
    ct2 = [
        *PY,
        "whisper_ko_ft.evaluate_ct2",
        "--model",
        "outputs/turbo-n/ct2",
        "--name",
        "speed",
        "--set",
        SET,
        "--no-report",
    ]
    runs = []
    for n in (1, 2):
        runs.append((f"hf-b1-{n}", hf))
        runs.append((f"ct2-fp16-{n}", [*ct2, "--compute-type", "float16"]))
        runs.append((f"ct2-int8-{n}", [*ct2, "--compute-type", "int8_float16"]))
    return runs, {"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}


PRESETS = ("ct2-n", "fallback-n", "qwen-turbo", "ct2-int8")


def build_runs(preset: str, qwen_args: list[str], qwen_large_batch: str = "16") -> tuple[list, dict]:
    """프리셋 이름 → (실행 목록, 더할 환경 변수)."""
    if preset == "ct2-n":
        return preset_ct2_n()
    if preset == "fallback-n":
        return preset_fallback_n()
    if preset == "qwen-turbo":
        return preset_qwen_turbo(qwen_args, qwen_large_batch)
    if preset == "ct2-int8":
        return preset_ct2_int8()
    raise ValueError(f"unknown preset {preset}")


def speed_only(runs: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """--no-report가 없는 실행에 붙인다. 커밋된 리포트를 건드리지 않고 속도만 다시 잴 때 쓴다."""
    return [
        (label, command if "--no-report" in command else [*command, "--no-report"]) for label, command in runs
    ]


# --- 측정 도우미 ---


def log(message: str) -> None:
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {message}", flush=True)


def parse_gpu_line(line: str) -> tuple[int, int]:
    """nvidia-smi --format=csv,noheader,nounits의 한 줄 "메모리MiB, 사용률%" → (메모리, 사용률)."""
    used, util = (int(x) for x in line.split(","))
    return used, util


def gpu_sample() -> tuple[int, int]:
    out = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    return parse_gpu_line(out)


def parse_gpu_processes(text: str) -> list[str]:
    """nvidia-smi 표의 'Processes:' 아래에서 프로세스 줄만(GPU 번호로 시작하는 줄) 골라 돌려준다.

    Windows(WDDM)에서는 그래픽 프로그램(C+G, G)도 나와, 게임이나 배경화면 프로그램이 GPU를 쓰는지 볼 수 있다.
    프로세스가 없거나 표가 없으면 빈 목록이다.
    """
    after = text.split("Processes:", 1)[1] if "Processes:" in text else ""
    return [line.strip(" |") for line in after.splitlines() if line.strip(" |")[:1].isdigit()]


def gpu_processes() -> list[str]:
    return parse_gpu_processes(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)


def ollama_models() -> str:
    try:
        return subprocess.run(
            ["curl", "-s", "localhost:11434/api/ps"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unreachable"


def is_idle(check: dict) -> bool:
    """쉬는 상태 기준: GPU 사용률 1분 평균 10% 이하이고 CPU 1분 평균 15% 미만."""
    return check["gpu_util_avg"] <= GPU_IDLE_UTIL and check["cpu_percent_avg"] < CPU_IDLE


def conditions() -> dict:
    """1분: CPU(PowerShell, 5초마다 12번)와 그동안 CPU를 많이 쓴 프로세스, GPU 사용률·메모리(5초마다)."""
    cpu = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-Command", CPU_PS], stdout=subprocess.PIPE, text=True
    )
    gpu = []
    for _ in range(CHECK_SECONDS // 5):
        gpu.append(gpu_sample())
        time.sleep(5)
    lines = cpu.communicate()[0].strip().splitlines()
    cpu_avg, cpu_max = (float(x) for x in lines[0].split())
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "cpu_percent_avg": round(cpu_avg, 1),
        "cpu_percent_max": round(cpu_max, 1),
        "cpu_top_processes": lines[1] if len(lines) > 1 else "",
        "gpu_util_avg": round(sum(u for _, u in gpu) / len(gpu), 1),
        "gpu_util_max": max(u for _, u in gpu),
        "gpu_memory_used_mb": max(m for m, _ in gpu),
        "gpu_processes": gpu_processes(),
        "ollama": ollama_models(),
    }


def wait_for_idle(max_wait_seconds: int, poll_seconds: int) -> tuple[bool, list[dict]]:
    """쉬는 상태가 될 때까지 확인한다. 다음 확인이 끝날 시각이 max_wait를 넘으면 그만 기다린다.

    그때의 스크립트와 같은 식이다: 60분·5분이면 확인 12번, 10분·3분이면 4번(0, 3, 6, 9분에 시작).
    """
    checks = []
    started = time.monotonic()
    while True:
        check = conditions()
        checks.append(check)
        idle = is_idle(check)
        log(f"conditions {json.dumps(check, ensure_ascii=False)} idle={idle}")
        if idle:
            return True, checks
        if time.monotonic() - started + poll_seconds > max_wait_seconds:
            return False, checks
        time.sleep(poll_seconds - CHECK_SECONDS)


def warm_cache() -> None:
    """묶음의 오디오를 한 번 읽어 운영체제 파일 캐시에 올린다(첫 실행만 디스크를 읽어 느려지지 않게)."""
    # store는 여기서만 쓰므로 이 안에서 불러온다(src/는 모듈 머리에서 import 경로에 넣었다).
    from whisper_ko_ft.store import AudioStore, load_set

    _, utterances = load_set(SET)
    stores: dict[str, AudioStore] = {}
    total = 0
    for u in utterances:
        # 저장소마다 memmap을 한 번만 연다(하네스와 같다). setdefault(u.store, AudioStore(u.store))는 기본값
        # 인자를 키가 있어도 먼저 계산해 발화마다 새로 열었다(2026-10-29까지).
        if u.store not in stores:
            stores[u.store] = AudioStore(u.store)
        total += stores[u.store].read(u).size
    log(f"warmed {len(utterances)} utterances, {total / 16000 / 3600:.2f} h")


def first_summary(text: str) -> dict:
    """하네스가 찍은 첫 JSON 요약. evaluate·evaluate_ct2는 하나, evaluate_qwen은 raw·fixed 두 개를 찍는다."""
    start = text.index("{\n")
    return json.loads(text[start : text.index("\n}", start) + 2])


def stop(process: subprocess.Popen) -> None:
    """표본 프로세스 하나를 끝낸다: terminate → wait(STOP_SECONDS) → 안 끝나면 taskkill /T /F → wait.

    두 번째 단계를 Popen.kill이 아니라 taskkill /T /F(자식까지 트리째 강제 종료)로 한 것은, Windows의 kill이
    terminate와 같은 TerminateProcess라 다시 불러도 다를 것이 없어서다. 그래도 끝나지 않으면(또는 taskkill을
    띄우지 못하면) PID를 적고 넘어간다: run의 finally에서 불리므로 여기서 예외를 내면 원래 예외를 가리고 다른
    표본 프로세스를 정리하지 못한다. 남은 프로세스는 그 로그 줄의 명령으로 손으로 끝낸다.
    """
    process.terminate()
    try:
        process.wait(timeout=STOP_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, text=True)
        process.wait(timeout=STOP_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as error:
        log(
            f"warning: sampler pid {process.pid} may still run ({error!r});"
            f" end it with taskkill /T /F /PID {process.pid}"
        )


def run(label: str, command: list[str], env_extra: dict, out_dir: Path) -> dict:
    """실행 하나: nvidia-smi(0.5초)와 CPU(약 1초) 표본을 남기며 명령을 돌리고, 요약과 표본 통계를 돌려준다."""
    before_mem, before_util = gpu_sample()
    gpu_path, cpu_path = out_dir / f"gpu_{label}.csv", out_dir / f"cpu_{label}.csv"
    # 띄운 표본 프로세스는 이 목록에 넣고, 하네스가 어떻게 끝나든(정상, 실패, uv가 PATH에 없는
    # FileNotFoundError, Ctrl+C의 KeyboardInterrupt) finally에서 모두 끝낸다. 둘째 표본(PowerShell)을 띄우지
    # 못해도 먼저 띄운 nvidia-smi를 끝낸다. 예외는 그대로 위로 나간다(main이 그때까지의 speed.json을 남긴다).
    # 하네스를 띄운 바로 아래 자식(uv)은 subprocess.run이 예외(KeyboardInterrupt 포함) 때 스스로 kill한다.
    samplers: list[subprocess.Popen] = []
    with gpu_path.open("w") as gpu_file, cpu_path.open("w") as cpu_file:
        try:
            samplers.append(
                subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                        "-lms",
                        "500",
                    ],
                    stdout=gpu_file,
                )
            )
            samplers.append(
                subprocess.Popen(
                    ["powershell.exe", "-NoProfile", "-Command", CPU_SAMPLER_PS], stdout=cpu_file, text=True
                )
            )
            log(f"{label} start: gpu {before_mem} MiB, {before_util}%")
            env = {**os.environ, **env_extra}
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env
            )
        finally:
            for process in samplers:
                stop(process)
    if result.returncode:
        log(f"{label} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
        raise SystemExit(1)
    summary = first_summary(result.stdout)
    rows = [parse_gpu_line(line) for line in gpu_path.read_text().splitlines() if line.strip()]
    memory, util = [m for m, _ in rows], [u for _, u in rows]
    cpu = [float(line.split(",")[1]) for line in cpu_path.read_text().splitlines() if "," in line]
    record = {
        "label": label,
        "command": " ".join(command),
        "gpu_memory_before_mb": before_mem,
        "gpu_memory_max_mb": max(memory),
        "gpu_memory_added_mb": max(memory) - before_mem,
        "gpu_util_avg_during": round(sum(util) / len(util), 1),
        "cpu_percent_avg_during": round(sum(cpu) / len(cpu), 1) if cpu else None,
        "cpu_percent_max_during": max(cpu) if cpu else None,
        "summary": summary,
    }
    log(
        f"{label}: {summary['audio_seconds_per_second']}x, wall {summary.get('wall_seconds')} s, "
        f"batch median {summary.get('batch_seconds_median')} s, p90 {summary.get('batch_seconds_p90')} s, "
        f"error {summary['error_rate']}, nvidia-smi +{record['gpu_memory_added_mb']} MiB, "
        f"torch peak {summary.get('peak_gpu_memory_mb')}, cpu {record['cpu_percent_avg_during']}%"
    )
    return record


def write_speed_json(target: Path, payload: dict) -> None:
    """speed.json을 통째로 다시 쓴다(UTF-8, LF). main이 실행마다 부른다(부분 기록)."""
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="속도·GPU 메모리 측정 절차를 프리셋대로 돈다(Windows 전용, docs/experiments.md 속도 절).",
        epilog="qwen-turbo 프리셋에서 알 수 없는 인자는 evaluate_qwen에 그대로 넘긴다(예: --precision fp32).",
    )
    parser.add_argument("--preset", required=True, choices=PRESETS, help="실행 목록(위 docstring)")
    parser.add_argument(
        "--max-wait-minutes",
        type=float,
        default=60,
        help="쉬는 상태를 기다리는 최대 시간(기본 60, 12단계는 10)",
    )
    parser.add_argument(
        "--poll-minutes",
        type=float,
        default=5,
        help="쉬는 상태 확인 간격(기본 5, 12단계는 3). 한 번의 확인이 1분 걸리므로 1 이상이어야 한다",
    )
    parser.add_argument("--out", type=Path, help="결과 폴더(기본 outputs/speed/<프리셋>-<시각>)")
    parser.add_argument(
        "--speed-only",
        action="store_true",
        help="모든 실행에 --no-report를 붙인다(커밋된 리포트를 건드리지 않음)",
    )
    parser.add_argument(
        "--qwen-large-batch", default="16", help="qwen-turbo의 Qwen '배치 16' 실행에 쓸 배치(fp32면 4)"
    )
    parser.add_argument("--dry-run", action="store_true", help="돌릴 명령만 찍고 끝낸다")
    args, extra = parser.parse_known_args()
    if extra and args.preset != "qwen-turbo":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    # 확인 사이의 sleep은 poll − CHECK_SECONDS다. 1분보다 짧으면 음수가 되어 첫 확인 뒤에 ValueError로
    # 죽으므로, 쉬는 상태를 기다리기(최대 --max-wait-minutes) 전에 거부한다. nan·inf도 여기서 거부한다
    # (round에서 죽는다). --dry-run에도 같은 검사를 한다.
    if not (math.isfinite(args.poll_minutes) and args.poll_minutes * 60 >= CHECK_SECONDS):
        parser.error(
            f"--poll-minutes must be at least {CHECK_SECONDS // 60} (one check takes {CHECK_SECONDS} s)"
        )
    runs, env_extra = build_runs(args.preset, extra, args.qwen_large_batch)
    if args.speed_only:
        runs = speed_only(runs)
    if args.dry_run:
        for label, command in runs:
            print(label, " ".join(command))
        return

    # 측정 코드의 커밋(scripts/ 자신까지 보아 +dirty). 하네스 리포트의 commit과 같은 방식이고, 하네스처럼
    # 측정을 시작하기 전에 읽는다.
    commit = git_commit(ROOT, COMMIT_PATHS)
    log(f"commit {commit}")
    out_dir = args.out or ROOT / "outputs" / "speed" / f"{args.preset}-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_wait, poll = round(args.max_wait_minutes * 60), round(args.poll_minutes * 60)
    idle, checks = wait_for_idle(max_wait, poll)
    log(f"idle reached: {idle}")
    # speed.json 부분 기록: 쉬는 상태 확인 뒤 한 번, 실행이 하나 끝날 때마다 그때까지의 기록으로 다시 쓴다
    # (complete: false). 끝난 뒤의 확인(after)까지 마치면 complete: true로 쓴다. 그래서 실행 하나가 실패하거나
    # (run의 SystemExit) 사람이 멈춰도(Ctrl+C) 앞 실행의 기록과 확인이 남고, analyze verdict12는 false인
    # 파일을 판정에 쓰지 않는다. 키가 없는 이전 speed.json은 모든 실행이 끝난 뒤에만 쓴 것이라 끝까지 돈
    # 것으로 본다.
    target = out_dir / "speed.json"
    payload = {
        "preset": args.preset,
        "complete": False,
        "commit": commit,
        "max_wait_seconds": max_wait,
        "poll_seconds": poll,
        "idle_reached": idle,
        "checks": checks,
        "runs": [],
    }
    write_speed_json(target, payload)
    try:
        warm_cache()
        for label, command in runs:
            payload["runs"].append(run(label, command, env_extra, out_dir))
            write_speed_json(target, payload)
        payload["after"] = conditions()
    except BaseException:
        # 어디서 멈췄는지와 남은 기록의 위치만 알리고 예외는 그대로 다시 낸다(종료 코드·추적 정보 유지).
        done = len(payload["runs"])
        log(f"stopped after {done} of {len(runs)} runs; {target} keeps them (complete: false)")
        raise
    log(f"conditions after {json.dumps(payload['after'], ensure_ascii=False)}")
    payload["complete"] = True
    write_speed_json(target, payload)
    log(f"wrote {target}")


if __name__ == "__main__":
    main()
