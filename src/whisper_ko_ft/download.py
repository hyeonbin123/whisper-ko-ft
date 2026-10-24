"""Zeroth-Korean과 FLEURS(한국어, 영어)의 parquet 파일을 Hugging Face에서 내려받고, 원본 manifest와 견준다.

사용법:
    uv run python -m whisper_ko_ft.download                   # 전부 받는다. 약 4 GB
    uv run python -m whisper_ko_ft.download --dry-run         # 파일 목록과 크기만 보여 준다
    uv run python -m whisper_ko_ft.download --verify          # 받은 파일을 manifest와 SHA-256으로 견준다
    uv run python -m whisper_ko_ft.download --write-manifest  # data/raw 전체로 manifest를 다시 쓴다

data/raw/<데이터셋>/<config>/<split>/<n>.parquet에 쓴다(git 제외). Hugging Face가 kresnik/zeroth_korean과
google/fleurs(둘 다 CC BY 4.0)에 대해 제공하는 parquet 변환본이다.

원본 manifest(src/whisper_ko_ft/data_manifest.json, 커밋)는 기록된 측정이 쓴 parquet 파일마다 상대 경로·바이트
크기·SHA-256을, 데이터셋마다 Hugging Face parquet 변환본(refs/convert/parquet)의 커밋을 적는다. 커밋은 그
커밋의 파일 목록(LFS oid)이 받은 파일의 SHA-256과 모두 같을 때만 적고, 아니면 null과 확인한 날짜만 남는다.
prepare가 시작할 때 data/raw를 이 manifest와 견준다.
"""

# [파일 역할] 0단계(기반)의 첫 단계. Hugging Face가 제공하는 parquet 변환본을 data/raw/에 받고, 받은 파일이
# 기록된 원본과 같은지 확인하는 수단(원본 manifest, --verify)을 둔다. 다음 단계 prepare.py가 이 파일들을 읽어
# 오디오 저장소를 만든다.
# [설계 이유] datasets 라이브러리로 받아 디코딩하지 않고 parquet 파일을 직접 받는다. datasets의 오디오
# 디코딩은 Windows에서 torchcodec이 필요해서, parquet 안의 오디오 바이트를 prepare가 soundfile로 직접 읽게
# 했다.
# [흐름] 데이터셋·config·split마다 API로 parquet URL 목록을 받는다 → 파일마다 이미 있으면 건너뛰고(--force면
# 다시 받음) → 없으면 .part 임시 파일에 받은 뒤 이름을 바꾼다. --dry-run은 받지 않고 원격 크기와 로컬 크기만
# 찍는다(새 클론 재현 때 원격과 크기가 같은지 이것으로 확인했다, docs/experiments.md).
# [원본 manifest] 2026-10-24 전에는 받은 파일의 크기·해시와 변환본의 커밋을 어디에도 남기지 않았다. 내려받기가
# 목록 중간에 끊기거나(파일마다 .part를 바꿔 넣으므로 앞 파일만 남는다) Hugging Face가 parquet을 다시
# 변환하면, 새 클론은 기록된 측정과 다른 데이터로 돌면서도 그 사실을 알 길이 없었다. 그래서 지금 data/raw의
# 파일로 만든 manifest를 코드와 함께 커밋하고, --verify와 prepare가 그것과 견준다(크기가 같고 내용만 다른
# 파일은 SHA-256만 잡는다). --write-manifest는 원본을 일부러 바꿨을 때만 쓴다.

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

# httpx: 파일 목록 API 호출(GET), 원격 크기 확인(HEAD), 큰 파일의 스트리밍 내려받기, 변환본의 tree·revision
# API 호출에 쓴다. 모든 요청에 follow_redirects=True를 줘 리다이렉트된 실제 파일 위치까지 따라간다.
import httpx

# RAW: 내려받은 parquet을 두는 data/raw(git 제외). MANIFEST: 원본 manifest(커밋).
from whisper_ko_ft.paths import MANIFEST, RAW

# API: 데이터셋 저장소의 parquet 변환본 URL 목록을 돌려주는 Hugging Face 엔드포인트. 뒤에 /<config>/<split>을
# 붙여 쓴다.
API = "https://huggingface.co/api/datasets/{repo}/parquet"
# PARQUET_REF: Hugging Face가 parquet 변환본을 두는 브랜치 refs/convert/parquet. 아래 두 URL의 경로 안에
# 들어가므로 '/'를 %2F로 적는다.
# TREE: 그 브랜치의 <config>/<split> 폴더 목록. 항목마다 path와 lfs.oid(파일 내용의 SHA-256)가 있어 받은
# 파일의 해시와 견줄 수 있다(2026-10-24에 kresnik/zeroth_korean·google/fleurs로 확인한 응답 모양). 폴더에
# 파일이 1,000개 넘으면 목록이 나눠 오지만, 여기서 보는 폴더는 모두 파일 몇 개뿐이다.
# REVISION: 그 브랜치의 지금 커밋(응답의 "sha").
PARQUET_REF = "refs%2Fconvert%2Fparquet"
TREE = "https://huggingface.co/api/datasets/{repo}/tree/" + PARQUET_REF + "/{config}/{split}"
REVISION = "https://huggingface.co/api/datasets/{repo}/revision/" + PARQUET_REF
# SOURCES: 받을 데이터. Zeroth-Korean은 train(학습과 validation의 원천)과 test, FLEURS는 한국어(ko_kr)와
# 영어(en_us)의 validation·test만 받는다. FLEURS는 다른 도메인과 영어를 확인하는 평가에만 쓰므로 train이 필요
# 없다(docs/plan.md 데이터 표). 라이선스는 둘 다 CC BY 4.0이다.
# 형식: 데이터셋 폴더 이름 -> (Hugging Face 저장소, {config: split 목록})
SOURCES = {
    "zeroth": ("kresnik/zeroth_korean", {"default": ("train", "test")}),
    "fleurs": ("google/fleurs", {"ko_kr": ("validation", "test"), "en_us": ("validation", "test")}),
}


# config·split 하나의 parquet 파일 URL 목록을 받는다. main이 목록 순서대로 0, 1, 2 … 번호를 붙여 저장하고,
# prepare는 그 번호순으로 읽는다.
def list_files(repo: str, config: str, split: str) -> list[str]:
    response = httpx.get(f"{API.format(repo=repo)}/{config}/{split}", follow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.json()


# HEAD 요청으로 원격 파일 크기(content-length)만 확인한다. --dry-run에서 로컬 파일 크기와 나란히 찍는 데 쓴다.
# 헤더가 없으면 0을 돌려준다.
def remote_size(url: str) -> int:
    response = httpx.head(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    return int(response.headers.get("content-length", 0))


# url을 target에 받는다. 1 MiB씩 스트리밍해 큰 parquet도 메모리에 한꺼번에 올리지 않는다.
# 먼저 .part 파일에 쓰고 끝까지 받은 뒤에만 이름을 바꾼다. 중간에 끊겨도 반쯤 받은 파일이 target 이름으로 남지
# 않으므로, 다음 실행의 '있으면 건너뛰기'가 깨진 파일을 건너뛰는 일이 없다.
def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            for chunk in response.iter_bytes(1 << 20):
                out.write(chunk)
    partial.replace(target)


# 데이터셋마다 받는 폴더를 data/raw 아래 상대 경로('zeroth/default/train')로 늘어놓는다. manifest의 파일
# 경로가 이 폴더로 시작하고, prepare.STORES의 폴더도 같은 구조다.
def source_folders(datasets: Iterable[str]) -> dict[str, tuple[str, str, str]]:
    """data/raw 아래 상대 폴더 → (Hugging Face 저장소, config, split)."""
    folders: dict[str, tuple[str, str, str]] = {}
    for name in datasets:
        repo, configs = SOURCES[name]
        for config, splits in configs.items():
            for split in splits:
                folders[f"{name}/{config}/{split}"] = (repo, config, split)
    return folders


# 폴더의 parquet 파일을 main이 붙인 번호(0, 1, 2 …) 순서로 돌려준다. 이름을 문자열로 정렬하면 10이 2보다 앞에
# 오므로 정수로 정렬한다. prepare가 저장소를 만들 때 이 순서로 읽고, manifest의 변환본 확인도 이 순서로 tree
# 목록과 짝짓는다. 폴더가 없으면 빈 목록이다.
def parquet_files(folder: Path) -> list[Path]:
    """folder의 *.parquet을 파일 번호순으로."""
    return sorted(folder.glob("*.parquet"), key=lambda p: int(p.stem))


# 파일 내용의 SHA-256. hashlib.file_digest가 조각 단위로 읽으므로 수백 MB짜리 parquet도 메모리에 올리지
# 않는다. manifest의 sha256 칸과 --verify, prepare --verify-hash가 이 값을 쓴다.
def sha256_file(path: Path) -> str:
    """파일 내용의 SHA-256(16진수 64자)."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


# 원본 manifest를 읽는다. 커밋된 파일이라 늘 있어야 하고, 없으면 무엇으로 다시 만드는지 알리며 멈춘다.
def load_manifest(path: Path) -> dict:
    """manifest JSON(files: 상대 경로 → {bytes, sha256}, upstream: 데이터셋 → 변환본 커밋, created: 날짜)."""
    if not path.exists():
        raise SystemExit(
            f"{path} is missing; it is committed with the code"
            " (`python -m whisper_ko_ft.download --write-manifest` rewrites it from data/raw)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# manifest와 디스크를 견준다. folders는 raw 아래 상대 폴더('zeroth/default/train') 목록이고, 폴더마다
# manifest에 적힌 파일과 디스크의 *.parquet을 모아 경로순으로 다른 점을 적는다. 요청하지 않은 폴더는 보지
# 않고, 받다가 남은 .part 파일은 prepare가 읽지 않으므로 세지 않는다.
# 기본은 크기만 견준다(HDD에서 약 4 GB를 매번 해시하지 않으려고). hashes=True면 크기가 같은 파일의 SHA-256도
# 견준다. 호출 위치: main(--verify, 늘 hashes=True), prepare.main(기본 크기, --verify-hash면 해시).
def check_raw(manifest: dict, raw: Path, folders: Iterable[str], hashes: bool = False) -> list[str]:
    """요청한 폴더에서 manifest와 디스크가 다른 점의 목록. 모두 같으면 빈 목록.

    항목 꼴: 'missing'(manifest에만 있음), 'not in the manifest'(디스크에만 있음), 'N bytes, manifest M'(크기
    차이), 'SHA-256 differs from the manifest'(hashes=True, 크기는 같고 내용이 다름).
    """
    recorded = manifest["files"]
    problems: list[str] = []
    for folder in folders:
        on_disk = {f"{folder}/{path.name}": path for path in (raw / folder).glob("*.parquet")}
        in_manifest = {path for path in recorded if PurePosixPath(path).parent.as_posix() == folder}
        for path in sorted(set(on_disk) | in_manifest):
            if path not in in_manifest:
                problems.append(f"{path}: not in the manifest")
            elif path not in on_disk:
                problems.append(f"{path}: missing")
            elif (size := on_disk[path].stat().st_size) != recorded[path]["bytes"]:
                problems.append(f"{path}: {size:,} bytes, manifest {recorded[path]['bytes']:,}")
            elif hashes and sha256_file(on_disk[path]) != recorded[path]["sha256"]:
                problems.append(f"{path}: SHA-256 differs from the manifest")
    return problems


# 데이터셋 하나의 refs/convert/parquet 커밋을 확인한다. local은 (config, split) → 받은 파일의 SHA-256
# 목록(파일 번호순)이다. 폴더마다 tree 목록의 parquet 항목을 경로순으로 늘어놓고 LFS oid(내용의 SHA-256)가
# 로컬 해시와 수·순서까지 같아야 한다. 하나라도 다르거나 API가 실패하면 None을 돌려준다(그 데이터셋의
# 커밋은 적지 않는다).
# 왜 이렇게 하나: 커밋 sha만 적으면 '지금 Hugging Face에 있는 것'일 뿐 '이 파일들이 그 커밋의 것'이라는
# 보장이 없다. 파일 내용의 해시가 같을 때만 적어야 manifest가 근거가 된다.
def parquet_revision(repo: str, local: dict[tuple[str, str], list[str]]) -> str | None:
    """refs/convert/parquet 커밋 sha. tree의 LFS oid가 받은 파일의 SHA-256과 모두 같을 때만, 아니면 None."""
    try:
        for (config, split), digests in local.items():
            response = httpx.get(
                TREE.format(repo=repo, config=config, split=split), follow_redirects=True, timeout=60
            )
            response.raise_for_status()
            entries = [
                e for e in response.json() if e.get("type") == "file" and e["path"].endswith(".parquet")
            ]
            oids = [entry.get("lfs", {}).get("oid") for entry in sorted(entries, key=lambda e: e["path"])]
            if oids != digests:
                print(
                    f"{repo} {config}/{split}: the parquet files on Hugging Face differ from the local files"
                    f" ({len(oids)} remote, {len(digests)} local); the revision is not recorded"
                )
                return None
        response = httpx.get(REVISION.format(repo=repo), follow_redirects=True, timeout=60)
        response.raise_for_status()
        return response.json()["sha"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"{repo}: could not check refs/convert/parquet ({error!r}); the revision is not recorded")
        return None


# data/raw 전체로 manifest 내용을 만든다. 먼저 모든 폴더에 parquet이 있는지 보고(하나라도 비면 해시를 시작하기
# 전에 멈춘다), 파일마다 크기·SHA-256을 계산한 뒤 데이터셋마다 변환본 커밋을 확인한다. 날짜는 UTC 기준이다.
def build_manifest() -> dict:
    """manifest 내용: created(날짜), files(경로 → bytes·sha256), upstream(데이터셋 → repo·커밋·날짜)."""
    folders = source_folders(SOURCES)
    paths = {folder: parquet_files(RAW / folder) for folder in folders}
    for folder, files in paths.items():
        if not files:
            raise SystemExit(f"{folder}: no parquet files in {RAW / folder}; download the files first")
    files_entry: dict[str, dict] = {}
    digests: dict[str, dict[tuple[str, str], list[str]]] = {name: {} for name in SOURCES}
    for folder, (_, config, split) in folders.items():
        hashes = []
        for path in paths[folder]:
            print(f"{folder}/{path.name}: hashing ...", flush=True)
            hashes.append(sha256_file(path))
            files_entry[f"{folder}/{path.name}"] = {"bytes": path.stat().st_size, "sha256": hashes[-1]}
        digests[folder.split("/", 1)[0]][(config, split)] = hashes
    today = datetime.now(UTC).date().isoformat()
    upstream = {
        name: {"repo": repo, "parquet_revision": parquet_revision(repo, digests[name]), "checked": today}
        for name, (repo, _) in SOURCES.items()
    }
    return {"created": today, "files": files_entry, "upstream": upstream}


# --verify: 받은 파일을 manifest와 SHA-256으로 견준다. 네트워크를 쓰지 않는다. 다른 점이 있으면 목록을 적고
# 멈춘다(다시 받으려면 그 파일을 지우고 download를 돌린다).
def verify(datasets: Iterable[str]) -> None:
    """요청한 데이터셋의 파일이 manifest와 모두 같은지 확인하고, 다르면 SystemExit."""
    manifest = load_manifest(MANIFEST)
    folders = source_folders(datasets)
    problems = check_raw(manifest, RAW, folders, hashes=True)
    if problems:
        raise SystemExit("data/raw differs from the manifest:\n  " + "\n  ".join(problems))
    count = sum(1 for path in manifest["files"] if PurePosixPath(path).parent.as_posix() in folders)
    print(f"{count} files match the manifest (SHA-256)")


# 명령줄 진입점. --datasets로 일부만 받을 수 있다. --dry-run은 받지 않고 목록과 크기만, --force는 이미 있는
# 파일도 다시 받는다. --verify는 받은 파일을 manifest와 견주고, --write-manifest는 data/raw 전체(--datasets와
# 함께 쓸 수 없다)로 manifest를 다시 쓴다. 세 모드는 서로 배타적이다.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hugging Face의 parquet 변환본을 data/raw에 받고, 받은 파일을 원본 manifest와 견준다",
        epilog="manifest는 src/whisper_ko_ft/data_manifest.json(커밋). prepare도 시작할 때 이것과 견준다.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(SOURCES),
        help="이 데이터셋만(기본은 전부). --write-manifest에는 못 쓴다",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="파일 목록과 원격·로컬 크기만 보여 주고 받지 않는다"
    )
    mode.add_argument(
        "--verify", action="store_true", help="받은 파일을 manifest와 SHA-256으로 견준다(네트워크 없음)"
    )
    mode.add_argument(
        "--write-manifest",
        action="store_true",
        help="data/raw의 모든 parquet으로 manifest를 다시 쓴다(원본을 바꾼 뒤에만)",
    )
    parser.add_argument("--force", action="store_true", help="파일이 이미 있어도 다시 받는다")
    args = parser.parse_args()
    if args.force and (args.dry_run or args.verify or args.write_manifest):
        parser.error(
            "--force downloads the files again; it does nothing with --dry-run, --verify or --write-manifest"
        )
    if args.write_manifest and args.datasets:
        parser.error("--write-manifest records every dataset; drop --datasets")
    datasets = args.datasets or list(SOURCES)

    if args.verify:
        verify(datasets)
        return
    if args.write_manifest:
        manifest = build_manifest()
        # 들여쓰기 2칸, 줄바꿈 LF. 커밋되는 파일이라 diff가 읽기 쉽게 한 항목을 한 줄씩 둔다.
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        revisions = {name: entry["parquet_revision"] for name, entry in manifest["upstream"].items()}
        print(f"wrote {MANIFEST} ({len(manifest['files'])} files, revisions {revisions})")
        return

    for name in datasets:
        repo, configs = SOURCES[name]
        for config, splits in configs.items():
            for split in splits:
                for n, url in enumerate(list_files(repo, config, split)):
                    # 저장 위치: data/raw/<데이터셋>/<config>/<split>/<n>.parquet. prepare.STORES가 이 폴더
                    # 구조를 그대로 읽는다.
                    target = RAW / name / config / split / f"{n}.parquet"
                    label = f"{name}/{config}/{split}/{n}"
                    if args.dry_run:
                        # --dry-run: 원격 크기와 로컬 크기(없으면 None)를 나란히 찍어 받은 파일이 온전한지
                        # 눈으로 확인한다. 내용까지 확인하려면 --verify.
                        local = target.stat().st_size if target.exists() else None
                        print(f"{label}: remote {remote_size(url):,} bytes, local {local}")
                    elif target.exists() and not args.force:
                        # 이미 받은 파일은 건너뛴다. 약 4GB를 다시 받지 않게 하려는 것이다(다시 받으려면
                        # --force).
                        print(f"{label}: exists, skipped (--force downloads it again)")
                    else:
                        print(f"{label}: downloading ...", flush=True)
                        download(url, target)
                        print(f"{label}: {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
