import subprocess
import sys

import pytest

from whisper_ko_ft import evaluate


class Reached(Exception):
    """The argument checks passed and the set was about to be loaded."""


@pytest.fixture
def reports(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(evaluate, "REPORTS", tmp_path)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "load_set", stop)
    monkeypatch.setattr(evaluate, "load_model", stop)
    return tmp_path


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["evaluate", *args])
    evaluate.main()


def test_adapter_needs_a_report_name(reports, monkeypatch, capsys):
    adapter = ["--model", "openai/whisper-large-v3-turbo", "--adapter", "x", "--set", "zeroth-val"]
    # Without --name the report would land in the base model's folder.
    with pytest.raises(SystemExit):
        run(monkeypatch, *adapter)
    assert "--name" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, *adapter, "--name", "l2")


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


def git(root, *args):
    command = ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args]
    subprocess.run(command, cwd=root, check=True, capture_output=True)


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


def test_commit_without_git(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(evaluate.subprocess, "run", missing)
    assert evaluate.git_commit(tmp_path) == "unknown"


SMOKE = ["--model", "m", "--set", "zeroth-val", "--fallback", "ratio"]


def test_threshold_overrides_are_for_smoke_runs_only(reports, monkeypatch, capsys):
    # A lowered threshold is not a registered candidate, so it must not leave a report behind.
    with pytest.raises(SystemExit):
        run(monkeypatch, *SMOKE, "--compression-ratio-threshold", "1.0")
    assert "--no-report" in capsys.readouterr().err
    with pytest.raises(Reached):
        run(monkeypatch, *SMOKE, "--compression-ratio-threshold", "1.0", "--no-report", "--limit", "16")


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


def test_fallback_runs_reach_the_measurement(reports, monkeypatch):
    with pytest.raises(Reached):
        run(monkeypatch, *SMOKE)
    with pytest.raises(Reached):
        run(monkeypatch, "--model", "m", "--set", "zeroth-val", "--fallback", "ratio-logprob", "--seed", "3")


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


def test_batch_times_in_the_summary():
    # Per-utterance latency at batch size 1 (stage 9 speed check).
    assert evaluate.batch_seconds_summary([0.1, 0.2, 0.3, 0.4, 1.0]) == {
        "batch_seconds_median": 0.3,
        "batch_seconds_p90": 0.76,
    }
    assert evaluate.batch_seconds_summary([]) == {"batch_seconds_median": None, "batch_seconds_p90": None}
