# [테스트 묶음] provenance.py(측정·학습 기록에 남길 지문)를 확인한다: 등록한 Whisper 두 모델의 고정 revision과
# 그것을 from_pretrained 인자로 바꾸는 규칙(Hub ID만, 로컬 폴더는 없음), 큰 파일을 조각으로 읽는 SHA-256,
# 어댑터 폴더·CTranslate2 변환 폴더의 지문, peft 판, 그리고 코드의 커밋 표시(+dirty 규칙과 보는 경로).
# [왜 중요한가] 리포트와 run.json은 커밋되는(또는 유일한 사본인) 기록이다. 기본 모델의 revision이 고정되지
# 않으면 Hub의 모델이 바뀐 뒤 새 클론의 측정이 경고 없이 다른 가중치로 돌고, 어댑터·변환 폴더의 지문이 없으면
# 같은 이름의 폴더를 다시 만들었을 때 리포트만으로는 같은 가중치인지 알 수 없다. 여기의 sha 값이 바뀌면 기록된
# 측정과 다른 가중치를 받는 것이므로 값 자체를 고정한다.
# [방법] 네트워크·GPU 없이 돈다. 커밋 표시는 임시 git 저장소로 확인한다.

import hashlib
import subprocess

import peft
import pytest

from whisper_ko_ft import provenance
from whisper_ko_ft.provenance import (
    CODE_PATHS,
    WHISPER_REVISIONS,
    adapter_hashes,
    adapter_info,
    converted_model_hashes,
    git_commit,
    peft_version,
    revision_kwargs,
    sha256_file,
    whisper_revision,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# 등록한 revision은 기록된 측정이 쓴 Hugging Face 캐시 스냅숏이다(2026-10-24·29에 Hub main과 같음을 확인). Hub
# ID면 그 sha를, outputs/ 아래 로컬 폴더나 등록하지 않은 Hub ID면 None을 돌려주고, from_pretrained 인자는
# 등록된 모델에만 revision을 넣는다.
def test_the_registered_whisper_revisions_are_the_recorded_snapshots():
    assert WHISPER_REVISIONS == {
        "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
        "openai/whisper-large-v3-turbo": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    }
    small = WHISPER_REVISIONS["openai/whisper-small"]
    assert whisper_revision("openai/whisper-small") == small
    assert revision_kwargs("openai/whisper-small") == {"revision": small}
    assert whisper_revision("outputs/small-a/best") is None  # 로컬 폴더: revision이 없다
    assert revision_kwargs("outputs/small-a/best") == {}
    assert whisper_revision("openai/whisper-medium") is None  # 등록하지 않은 Hub ID는 고정하지 않는다


# 파일 SHA-256은 조각으로 읽어도(1.6 GB짜리 model.bin을 통째로 올리지 않게) 한 번에 읽은 값과 같다.
def test_sha256_file_reads_in_chunks(tmp_path):
    data = bytes(range(256)) * 300
    path = tmp_path / "model.bin"
    path.write_bytes(data)
    assert sha256_file(path, chunk_bytes=7) == sha(data) == sha256_file(path)


# 어댑터 지문: adapter_model.safetensors와 adapter_config.json의 SHA-256. adapter_info는 거기에 어댑터 폴더의
# posix 경로를 더한 것(evaluate_qwen 리포트의 기존 꼴)이고, 어댑터가 없으면 모두 None이다.
def test_adapter_hashes_and_info(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_bytes(b'{"r": 16}')
    hashes = adapter_hashes(str(tmp_path))
    assert hashes == {"adapter_sha256": sha(b"weights"), "adapter_config_sha256": sha(b'{"r": 16}')}
    assert adapter_info(str(tmp_path)) == {"adapter": tmp_path.as_posix(), **hashes}
    assert adapter_hashes(None) == {"adapter_sha256": None, "adapter_config_sha256": None}
    assert adapter_info(None) == {"adapter": None, "adapter_sha256": None, "adapter_config_sha256": None}


# CTranslate2 변환 폴더 지문: model.bin(가중치)과 config.json의 SHA-256. 파일이 없으면 조용히 None을 적지 않고
# 멈춘다(변환 폴더가 아니다).
def test_converted_model_hashes(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"ct2")
    (tmp_path / "config.json").write_bytes(b"{}")
    assert converted_model_hashes(tmp_path) == {"model_sha256": sha(b"ct2"), "config_sha256": sha(b"{}")}
    with pytest.raises(FileNotFoundError):
        converted_model_hashes(tmp_path / "missing")


# 기록에 남기는 peft 판은 불러온 peft 모듈의 것이다.
def test_the_peft_version_is_the_imported_one():
    assert peft_version() == peft.__version__


# 임시 저장소에서 git을 부르는 도우미. 사용자 이름·메일을 명령에서 주어 전역 설정 없이도 커밋된다.
def git(root, *args):
    command = ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args]
    subprocess.run(command, cwd=root, check=True, capture_output=True)


# 임시 git 저장소를 만들어, 리포트만 바뀌면 커밋 표시가 그대로이고 src 코드가 바뀌면 '+dirty'가 붙는지
# 본다(커밋 6ba3c27의 규칙). 기본으로 보는 경로는 src·pyproject.toml·uv.lock이고, scripts/는 보는 경로를 준
# 호출(scripts/speed.py 자신)에서만 +dirty가 된다.
def test_commit_is_marked_dirty_only_for_uncommitted_code(tmp_path):
    for folder in ("src", "reports", "scripts"):
        (tmp_path / folder).mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "reports" / "r.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "scripts" / "speed.py").write_text("y = 1\n", encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "init")

    clean = git_commit(tmp_path)
    assert clean and not clean.endswith("+dirty")
    assert CODE_PATHS == ("src", "pyproject.toml", "uv.lock")
    (tmp_path / "reports" / "r.json").write_text('{"a": 1}\n', encoding="utf-8")  # 리포트는 세지 않는다
    assert git_commit(tmp_path) == clean
    (tmp_path / "scripts" / "speed.py").write_text("y = 2\n", encoding="utf-8")  # 측정 스크립트 자신
    assert git_commit(tmp_path) == clean  # 하네스의 기본 경로는 scripts/를 보지 않는다(지금까지와 같다)
    assert git_commit(tmp_path, (*CODE_PATHS, "scripts")) == f"{clean}+dirty"
    (tmp_path / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert git_commit(tmp_path) == f"{clean}+dirty"


# git 실행 파일이 없으면 예외로 죽지 않고 'unknown'을 남긴다.
def test_commit_without_git(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", missing)
    assert git_commit(tmp_path) == "unknown"
