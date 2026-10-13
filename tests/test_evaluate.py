# [테스트 묶음] evaluate.py의 규칙 보호 장치(인자 검사)와 리포트 요약 도우미를 확인한다: --adapter에는 --name
# 필수, 있는 리포트는 --overwrite 없이 덮어쓰지 않음, 측정 코드가 바뀌면 커밋에 +dirty, 스모크용 기준값은
# --no-report에서만, 행 채점과 배치 시간 요약.
# [왜 중요한가] 리포트는 커밋된 측정 기록이고 모든 판정이 이것을 다시 읽는다. 어댑터 평가가 기준선 리포트를
# 덮어쓰거나, 일부만 잰 리포트가 전체 리포트를 대신하는 사고(커밋 a29a975가 막은 것)가 다시 생기지 않게 한다.
# [방법] 인자 검사를 통과하면 모델을 불러오기 직전에 Reached를 던지게 해, GPU 없이 '멈췄는지/통과했는지'만
# 본다.

import subprocess
import sys

import pytest

from whisper_ko_ft import evaluate


# 인자 검사를 통과해 묶음을 읽으려던 순간에 던지는 예외(아래 fixture가 load_set·load_model을 이것으로 바꾼다).
class Reached(Exception):
    """The argument checks passed and the set was about to be loaded."""


# 리포트 폴더와 저장소 루트를 임시 폴더로 바꾸고, 묶음·모델 불러오기를 Reached로 바꾸는 fixture. 실제
# reports/를 건드리지 않는다.
@pytest.fixture
def reports(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(evaluate, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "load_set", stop)
    monkeypatch.setattr(evaluate, "load_model", stop)
    return tmp_path


# 명령줄 인자를 바꿔 evaluate.main을 한 번 부른다.
def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["evaluate", *args])
    evaluate.main()


# --adapter인데 --name이 없으면 parser.error로 멈추고(오류 메시지에 --name), --name을 주면 측정 단계까지 간다.
def test_adapter_needs_a_report_name(reports, monkeypatch, capsys):
    adapter = ["--model", "openai/whisper-large-v3-turbo", "--adapter", "x", "--set", "zeroth-val"]
    # Without --name the report would land in the base model's folder.
    with pytest.raises(SystemExit):
        run(monkeypatch, *adapter)
    assert "--name" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, *adapter, "--name", "l2")


# 이미 있는 리포트(채널 판 포함)는 --overwrite 없이는 멈추고, --overwrite나 --no-report(리포트를 쓰지 않는
# 속도·스모크 실행)면 진행한다.
@pytest.mark.parametrize("channel", [[], ["--channel", "telephone"]])
def test_an_existing_report_is_not_replaced_without_overwrite(reports, monkeypatch, capsys, channel):
    suffix = "@telephone" if channel else ""
    (reports / "m").mkdir()
    (reports / "m" / f"zeroth-val{suffix}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val", *channel)
    assert "--overwrite" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val", *channel, "--overwrite")
    with pytest.raises(Reached):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val", *channel, "--no-report", "--limit", "5")


# 임시 저장소에서 git을 부르는 도우미. 사용자 이름·메일을 명령에서 주어 전역 설정 없이도 커밋된다.
def git(root, *args):
    command = ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args]
    subprocess.run(command, cwd=root, check=True, capture_output=True)


# 임시 git 저장소를 만들어, 리포트만 바뀌면 커밋 표시가 그대로이고 src 코드가 바뀌면 '+dirty'가 붙는지
# 본다(커밋 6ba3c27의 규칙).
def test_commit_is_marked_dirty_only_for_uncommitted_code(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "r.json").write_text("{}\n", encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "init")

    clean = evaluate.git_commit(tmp_path)
    assert clean and not clean.endswith("+dirty")
    (tmp_path / "reports" / "r.json").write_text('{"a": 1}\n', encoding="utf-8")  # reports do not count
    assert evaluate.git_commit(tmp_path) == clean
    (tmp_path / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert evaluate.git_commit(tmp_path) == f"{clean}+dirty"


# git 실행 파일이 없으면 예외로 죽지 않고 'unknown'을 남긴다.
def test_commit_without_git(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(evaluate.subprocess, "run", missing)
    assert evaluate.git_commit(tmp_path) == "unknown"


# 8단계 재시도 후보 D1a(ratio)로 재는 기본 인자.
SMOKE = ["--model", "m", "--set", "zeroth-val", "--fallback", "ratio"]


# 스모크용으로 기준값을 낮추면 --no-report가 있어야 한다. 등록하지 않은 기준으로 만든 리포트가 남지 않게
# 하려는 것이다.
def test_threshold_overrides_are_for_smoke_runs_only(reports, monkeypatch, capsys):
    # A lowered threshold is not a registered candidate, so it must not leave a report behind.
    with pytest.raises(SystemExit):
        run(monkeypatch, *SMOKE, "--compression-ratio-threshold", "1.0")
    assert "--no-report" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, *SMOKE, "--compression-ratio-threshold", "1.0", "--no-report", "--limit", "16")


# 기준값은 맞는 후보가 있을 때만 받는다: 재시도 없이 압축비 기준만 주거나, 로그확률 기준이 없는 D1a에 로그확률
# 기준을 주면 멈춘다.
@pytest.mark.parametrize(
    "args",
    [
        ["--model", "m", "--set", "zeroth-val", "--compression-ratio-threshold", "1.0"],  # no fallback
        [*SMOKE, "--logprob-threshold", "-0.5"],  # D1a has no log-probability threshold
    ],
)
def test_threshold_overrides_need_the_matching_fallback(reports, monkeypatch, args):
    with pytest.raises(SystemExit):
        run(monkeypatch, *args, "--no-report")


# 등록한 두 후보(ratio, ratio-logprob)와 --seed는 검사를 통과해 측정 단계까지 간다.
def test_fallback_runs_reach_the_measurement(reports, monkeypatch):
    with pytest.raises(Reached):
        run(monkeypatch, *SMOKE)
    with pytest.raises(Reached):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val", "--fallback", "ratio-logprob", "--seed", "3")


# score_rows가 기록된 리포트와 같은 필드를 채우는지: 편집 수·길이·가설의 숫자 여부, 정규화하면 비는 정답은 0과
# 0으로 두고 그 수를 돌려준다. 9단계에서 main 밖으로 옮긴 함수라 동작이 그대로인지 고정한다(옆 주석).
def test_rows_are_scored_as_before():
    # Moved out of main for the Qwen3-ASR harness (stage 9); rows must come out as recorded reports have them.
    rows = [
        {"id": "a", "reference": "오는 오 월", "hypothesis": "오는 5월"},
        {"id": "b", "reference": "...", "hypothesis": "아무 말"},  # empty after normalization: skipped
        {"id": "c", "reference": "문장", "hypothesis": "문장"},
    ]
    assert evaluate.score_rows(rows, "ko") == 1
    assert (rows[0]["edits"], rows[0]["length"], rows[0]["has_digit"]) == (1, 4, True)
    assert (rows[1]["edits"], rows[1]["length"]) == (0, 0) and "has_digit" not in rows[1]
    assert (rows[2]["edits"], rows[2]["length"], rows[2]["has_digit"]) == (0, 2, False)


# 배치 시간의 중앙값과 90번째 백분위(numpy 선형 보간: 0.4와 1.0 사이 → 0.76), 빈 목록이면 None.
def test_batch_times_in_the_summary():
    # Per-utterance latency at batch size 1 (stage 9 speed check).
    assert evaluate.batch_seconds_summary([0.1, 0.2, 0.3, 0.4, 1.0]) == {
        "batch_seconds_median": 0.3,
        "batch_seconds_p90": 0.76,
    }
    assert evaluate.batch_seconds_summary([]) == {"batch_seconds_median": None, "batch_seconds_p90": None}
