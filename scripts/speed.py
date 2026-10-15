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
outputs/speed/<프리셋>-<시각>/)의 speed.json과 실행별 CSV에 남는다.
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
# 멈춘다. 이 옵션은 모든 실행에 --no-report를 붙여 속도만 잰다.
# [주의] Windows 전용이다: CPU 사용률과 프로세스 이름을 PowerShell(Get-CimInstance, Get-Process)로 읽는다.
# nvidia-smi와 uv가 PATH에 있어야 한다. 쉬는 상태 확인의 Ollama 항목은 같은 PC에서 GPU를 나눠 쓰는 로컬
# Ollama 서버에 올라간 모델을 적는 것이고, 서버가 없으면 "unreachable"로 남는다(측정에는 영향이 없다).

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ROOT: 저장소 폴더(이 파일은 scripts/ 안에 있다). 모든 실행은 이 폴더에서 돈다.
ROOT = Path(__file__).resolve().parents[1]
# 속도 묶음: Zeroth val-500(440발화, 1.07시간). 모든 프리셋이 같은 묶음을 같은 순서로 처리한다.
SET = "zeroth-val500"
PY = ["uv", "run", "--no-sync", "python", "-m"]
# 쉬는 상태 기준(2026-10-02 규칙부터 같다): 다른 프로그램을 포함한 GPU 사용률 1분 평균 10% 이하, CPU 1분
# 평균 15% 미만.
GPU_IDLE_UTIL = 10
CPU_IDLE = 15
# 한 번의 확인에 걸리는 시간(초). 5초 간격 표본 12개다.
CHECK_SECONDS = 60

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
    sys.path.insert(0, str(ROOT / "src"))
    from whisper_ko_ft.store import AudioStore, load_set

    _, utterances = load_set(SET)
    stores: dict[str, AudioStore] = {}
    total = sum(stores.setdefault(u.store, AudioStore(u.store)).read(u).size for u in utterances)
    log(f"warmed {len(utterances)} utterances, {total / 16000 / 3600:.2f} h")


def first_summary(text: str) -> dict:
    """하네스가 찍은 첫 JSON 요약. evaluate·evaluate_ct2는 하나, evaluate_qwen은 raw·fixed 두 개를 찍는다."""
    start = text.index("{\n")
    return json.loads(text[start : text.index("\n}", start) + 2])


def run(label: str, command: list[str], env_extra: dict, out_dir: Path) -> dict:
    """실행 하나: nvidia-smi(0.5초)와 CPU(약 1초) 표본을 남기며 명령을 돌리고, 요약과 표본 통계를 돌려준다."""
    before_mem, before_util = gpu_sample()
    gpu_path, cpu_path = out_dir / f"gpu_{label}.csv", out_dir / f"cpu_{label}.csv"
    with gpu_path.open("w") as gpu_file, cpu_path.open("w") as cpu_file:
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-lms",
                "500",
            ],
            stdout=gpu_file,
        )
        cpu_sampler = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", CPU_SAMPLER_PS], stdout=cpu_file, text=True
        )
        log(f"{label} start: gpu {before_mem} MiB, {before_util}%")
        env = {**os.environ, **env_extra}
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)
        for process in (sampler, cpu_sampler):
            process.terminate()
            process.wait()
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
        "--poll-minutes", type=float, default=5, help="쉬는 상태 확인 간격(기본 5, 12단계는 3)"
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
    runs, env_extra = build_runs(args.preset, extra, args.qwen_large_batch)
    if args.speed_only:
        runs = speed_only(runs)
    if args.dry_run:
        for label, command in runs:
            print(label, " ".join(command))
        return

    out_dir = args.out or ROOT / "outputs" / "speed" / f"{args.preset}-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_wait, poll = round(args.max_wait_minutes * 60), round(args.poll_minutes * 60)
    idle, checks = wait_for_idle(max_wait, poll)
    log(f"idle reached: {idle}")
    warm_cache()
    records = [run(label, command, env_extra, out_dir) for label, command in runs]
    after = conditions()
    log(f"conditions after {json.dumps(after, ensure_ascii=False)}")
    payload = {
        "preset": args.preset,
        "max_wait_seconds": max_wait,
        "poll_seconds": poll,
        "idle_reached": idle,
        "checks": checks,
        "runs": records,
        "after": after,
    }
    target = out_dir / "speed.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    log(f"wrote {target}")


if __name__ == "__main__":
    main()
