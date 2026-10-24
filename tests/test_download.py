# [테스트 묶음] download.py의 원본 manifest와 확인 기능: 파일 SHA-256, manifest와 data/raw 견주기(check_raw:
# 없는 파일·manifest에 없는 파일·크기·해시), `--verify`(받은 파일을 manifest와 SHA-256으로 견줌, 네트워크
# 없음), 내려받기(.part에 받은 뒤 이름 바꾸기, 있으면 건너뛰기, --force)와 `--write-manifest`(로컬 해시를
# 적고, Hugging Face API의 refs/convert/parquet tree에 적힌 LFS oid가 모두 같을 때만 그 revision을 적음).
# [왜 중요한가] prepare는 이 manifest로 data/raw가 기록된 원본과 같은지 가린다. 내려받기가 끊기거나 원본이
# 다시 변환된 것을 알아채지 못하면 다른 데이터로 측정하게 된다. 검증을 거친 revision만 적어야 manifest가
# '이 파일들이 그 커밋의 변환본'이라는 근거가 된다.
# [방법] httpx를 가짜 모듈로 바꿔(목록·tree·revision API와 파일 스트림을 흉내) 네트워크 없이 돈다. 파일 내용은
# 짧은 바이트열이고 parquet으로 읽지 않는다.

import hashlib
import json
import re
import sys

import httpx
import pytest

from whisper_ko_ft import download


class FakeResponse:
    """httpx 응답 대역: JSON 본문(json), 바이트 본문(iter_bytes, content-length 헤더), with 문 지원."""

    def __init__(self, payload=None, content=b""):
        self.payload, self.content = payload, content
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

    def iter_bytes(self, size):
        for start in range(0, len(self.content), size):
            yield self.content[start : start + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHttpx:
    """download가 쓰는 httpx 함수(get·head·stream)를 흉내 낸다.

    contents는 (저장소, config, split) → 파일 바이트 목록, revisions는 저장소 → refs/convert/parquet 커밋
    sha다. oids로 tree에 적을 LFS oid를 바꿔 로컬과 다른 경우를, fail_tree로 API 실패를 흉내 낸다.
    stream으로 받은 URL은 downloads에 남아 다시 받았는지 셀 수 있다.
    """

    HTTPError = httpx.HTTPError
    LIST = re.compile(r"/api/datasets/(?P<repo>[^/]+/[^/]+)/parquet/(?P<config>[^/]+)/(?P<split>[^/]+)$")
    TREE = re.compile(
        r"/api/datasets/(?P<repo>[^/]+/[^/]+)/tree/refs%2Fconvert%2Fparquet/(?P<config>[^/]+)/(?P<split>[^/]+)$"
    )
    REVISION = re.compile(r"/api/datasets/(?P<repo>[^/]+/[^/]+)/revision/refs%2Fconvert%2Fparquet$")
    FILE = re.compile(
        r"https://fake/(?P<repo>[^/]+/[^/]+)/(?P<config>[^/]+)/(?P<split>[^/]+)/(?P<n>\d+)\.parquet$"
    )

    def __init__(self, contents, revisions, oids=None, fail_tree=False):
        self.contents, self.revisions = contents, revisions
        self.oids, self.fail_tree = oids or {}, fail_tree
        self.downloads = []

    def get(self, url, **kwargs):
        if match := self.LIST.search(url):
            key = (match["repo"], match["config"], match["split"])
            urls = [f"https://fake/{'/'.join(key)}/{n}.parquet" for n in range(len(self.contents[key]))]
            return FakeResponse(urls)
        if match := self.TREE.search(url):
            if self.fail_tree:
                raise httpx.HTTPError("tree listing failed")
            key = (match["repo"], match["config"], match["split"])
            files = self.contents[key]
            oids = self.oids.get(key) or [hashlib.sha256(f).hexdigest() for f in files]
            entries = [
                {
                    "type": "file",
                    "path": f"{match['config']}/{match['split']}/{n:04d}.parquet",
                    "size": len(f),
                    "lfs": {"oid": oid, "size": len(f), "pointerSize": 134},
                }
                for n, (f, oid) in enumerate(zip(files, oids, strict=True))
            ]
            return FakeResponse(list(reversed(entries)))  # 순서는 경로로 맞춰야 한다
        if match := self.REVISION.search(url):
            return FakeResponse({"sha": self.revisions[match["repo"]]})
        raise AssertionError(f"unexpected GET {url}")

    def head(self, url, **kwargs):
        return FakeResponse(content=self._file(url))

    def stream(self, method, url, **kwargs):
        self.downloads.append(url)
        return FakeResponse(content=self._file(url))

    def _file(self, url):
        match = self.FILE.search(url)
        return self.contents[(match["repo"], match["config"], match["split"])][int(match["n"])]


class NoNetwork:
    """httpx 자리에 꽂아, 네트워크를 쓰면 안 되는 모드가 httpx를 건드리는 순간 실패하게 한다."""

    def __getattr__(self, name):
        raise AssertionError(f"this mode must not use the network (httpx.{name})")


# download.SOURCES의 모든 (저장소, config, split)에 가짜 파일을 만든다: train은 두 파일, 나머지는 한 파일.
def fake_contents():
    contents = {}
    for name, (repo, configs) in download.SOURCES.items():
        for config, splits in configs.items():
            for split in splits:
                count = 2 if split == "train" else 1
                contents[(repo, config, split)] = [
                    f"{name}/{config}/{split}/{n}".encode() * (n + 3) for n in range(count)
                ]
    return contents


REVISIONS = {"kresnik/zeroth_korean": "z" * 40, "google/fleurs": "f" * 40}


# 가짜 httpx와 임시 RAW·MANIFEST를 꽂는다. 테스트가 fake를 바꿀 수 있게 fake를 돌려준다.
@pytest.fixture
def fake(tmp_path, monkeypatch):
    fake = FakeHttpx(fake_contents(), REVISIONS)
    monkeypatch.setattr(download, "httpx", fake)
    monkeypatch.setattr(download, "RAW", tmp_path / "raw")
    monkeypatch.setattr(download, "MANIFEST", tmp_path / "data_manifest.json")
    return fake


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["download", *args])
    download.main()


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# 파일 SHA-256은 한 번에 읽은 해시와 같다(여러 조각으로 읽어도 결과가 같은지, 3 MiB로 조각 경계를 넘긴다).
def test_sha256_file_hashes_the_whole_file(tmp_path):
    content = bytes(range(256)) * (3 * 4096)
    (tmp_path / "f.parquet").write_bytes(content)
    assert download.sha256_file(tmp_path / "f.parquet") == sha(content)


# check_raw: 요청한 폴더에서 manifest에만 있는 파일(missing), 디스크에만 있는 파일(not in the manifest),
# 크기가 다른 파일을 적고, hashes=True일 때만 크기가 같고 내용이 다른 파일을 SHA-256으로 잡는다. 요청하지 않은
# 폴더는 보지 않는다.
def test_check_raw_reports_missing_extra_size_and_hash(tmp_path):
    (tmp_path / "d/c/s").mkdir(parents=True)
    (tmp_path / "d/c/s/0.parquet").write_bytes(b"ok")
    (tmp_path / "d/c/s/1.parquet").write_bytes(b"xx")
    manifest = {
        "files": {
            "d/c/s/0.parquet": {"bytes": 2, "sha256": sha(b"ok")},
            "d/c/s/2.parquet": {"bytes": 3, "sha256": "0" * 64},
            "d/c/t/0.parquet": {"bytes": 1, "sha256": "0" * 64},
        }
    }

    problems = download.check_raw(manifest, tmp_path, ["d/c/s"], hashes=True)
    assert problems == ["d/c/s/1.parquet: not in the manifest", "d/c/s/2.parquet: missing"]

    (tmp_path / "d/c/s/0.parquet").write_bytes(b"ko")
    assert "d/c/s/0.parquet: SHA-256 differs from the manifest" not in download.check_raw(
        manifest, tmp_path, ["d/c/s"]
    )
    assert "d/c/s/0.parquet: SHA-256 differs from the manifest" in download.check_raw(
        manifest, tmp_path, ["d/c/s"], hashes=True
    )
    (tmp_path / "d/c/s/0.parquet").write_bytes(b"k")
    assert "d/c/s/0.parquet: 1 bytes, manifest 2" in download.check_raw(manifest, tmp_path, ["d/c/s"])
    assert download.check_raw(manifest, tmp_path, ["d/c/t"]) == ["d/c/t/0.parquet: missing"]


# 내려받기: 파일마다 통째로 받아 .part를 남기지 않고, 이미 있는 파일은 건너뛰며, --force면 다시 받는다.
def test_download_writes_each_file_whole_and_skips_existing_files(fake, tmp_path, monkeypatch):
    run(monkeypatch, "--datasets", "zeroth")

    raw = tmp_path / "raw"
    train = fake.contents[("kresnik/zeroth_korean", "default", "train")]
    assert (raw / "zeroth/default/train/1.parquet").read_bytes() == train[1]
    assert (raw / "zeroth/default/test/0.parquet").read_bytes() == fake.contents[
        ("kresnik/zeroth_korean", "default", "test")
    ][0]
    assert not list(raw.rglob("*.part")) and len(fake.downloads) == 3
    assert not (raw / "fleurs").exists()

    run(monkeypatch, "--datasets", "zeroth")
    assert len(fake.downloads) == 3  # 있는 파일은 다시 받지 않는다
    run(monkeypatch, "--datasets", "zeroth", "--force")
    assert len(fake.downloads) == 6


# --write-manifest: 받은 모든 파일의 상대 경로·크기·SHA-256을 적고, 데이터셋마다 tree의 LFS oid가 로컬 해시와
# 파일 수까지 모두 같을 때 refs/convert/parquet 커밋 sha와 확인 날짜를 적는다.
def test_write_manifest_records_hashes_and_the_verified_parquet_revision(fake, tmp_path, monkeypatch):
    run(monkeypatch)
    run(monkeypatch, "--write-manifest")

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["files"]) == [
        "fleurs/en_us/test/0.parquet",
        "fleurs/en_us/validation/0.parquet",
        "fleurs/ko_kr/test/0.parquet",
        "fleurs/ko_kr/validation/0.parquet",
        "zeroth/default/test/0.parquet",
        "zeroth/default/train/0.parquet",
        "zeroth/default/train/1.parquet",
    ]
    content = fake.contents[("kresnik/zeroth_korean", "default", "train")][1]
    assert manifest["files"]["zeroth/default/train/1.parquet"] == {
        "bytes": len(content),
        "sha256": sha(content),
    }
    assert manifest["upstream"] == {
        "zeroth": {
            "repo": "kresnik/zeroth_korean",
            "parquet_revision": "z" * 40,
            "checked": manifest["created"],
        },
        "fleurs": {"repo": "google/fleurs", "parquet_revision": "f" * 40, "checked": manifest["created"]},
    }
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", manifest["created"])


# 원격 oid가 하나라도 다르면, 또는 tree API가 실패하면 그 데이터셋의 revision은 null이고 날짜만 남는다. 로컬
# 해시는 그대로 적는다.
def test_write_manifest_leaves_the_revision_null_when_hashes_differ_or_the_api_fails(
    fake, tmp_path, monkeypatch
):
    run(monkeypatch)
    fake.oids[("kresnik/zeroth_korean", "default", "test")] = ["0" * 64]
    run(monkeypatch, "--write-manifest")
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["upstream"]["zeroth"]["parquet_revision"] is None
    assert manifest["upstream"]["zeroth"]["checked"] == manifest["created"]
    assert manifest["upstream"]["fleurs"]["parquet_revision"] == "f" * 40
    assert len(manifest["files"]) == 7

    fake.fail_tree = True
    run(monkeypatch, "--write-manifest")
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert [manifest["upstream"][name]["parquet_revision"] for name in ("zeroth", "fleurs")] == [None, None]


# --write-manifest는 모든 데이터셋을 다루므로 --datasets와 함께 쓸 수 없고, 받지 않은 파일이 있으면 멈춘다.
def test_write_manifest_needs_every_file_and_refuses_a_dataset_filter(fake, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, "--write-manifest", "--datasets", "zeroth")
    assert "--datasets" in capsys.readouterr().err
    run(monkeypatch, "--datasets", "zeroth")
    with pytest.raises(SystemExit, match="no parquet"):
        run(monkeypatch, "--write-manifest")
    assert not (tmp_path / "data_manifest.json").exists()


# --verify: 네트워크 없이 받은 파일을 manifest와 SHA-256으로 견준다. 크기가 같고 내용만 다른 파일도 잡고,
# --datasets로 좁히면 그 데이터셋만 본다. --dry-run과는 함께 쓸 수 없다.
def test_verify_compares_the_files_with_the_manifest_without_the_network(fake, tmp_path, monkeypatch, capsys):
    run(monkeypatch)
    run(monkeypatch, "--write-manifest")
    monkeypatch.setattr(download, "httpx", NoNetwork())

    run(monkeypatch, "--verify")
    assert "7 files match" in capsys.readouterr().out

    path = tmp_path / "raw/fleurs/ko_kr/test/0.parquet"
    original = path.read_bytes()
    path.write_bytes(b"\0" + original[1:])
    with pytest.raises(SystemExit) as info:
        run(monkeypatch, "--verify")
    assert "fleurs/ko_kr/test/0.parquet" in str(info.value) and "SHA-256" in str(info.value)
    run(monkeypatch, "--verify", "--datasets", "zeroth")

    path.unlink()
    with pytest.raises(SystemExit, match="missing"):
        run(monkeypatch, "--verify")
    with pytest.raises(SystemExit):
        run(monkeypatch, "--verify", "--dry-run")
    assert "--dry-run" in capsys.readouterr().err
