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
