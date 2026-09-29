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
