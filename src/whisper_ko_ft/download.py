"""Download the parquet files of Zeroth-Korean and FLEURS (Korean, English) from Hugging Face.

Usage:
    uv run python -m whisper_ko_ft.download            # everything, about 4 GB
    uv run python -m whisper_ko_ft.download --dry-run  # list the files and sizes only

Writes data/raw/<dataset>/<config>/<split>/<n>.parquet (git-ignored). These are the parquet
conversions Hugging Face serves for kresnik/zeroth_korean and google/fleurs (both CC BY 4.0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from whisper_ko_ft.paths import RAW

API = "https://huggingface.co/api/datasets/{repo}/parquet"
# dataset folder -> (repo, {config: splits})
SOURCES = {
    "zeroth": ("kresnik/zeroth_korean", {"default": ("train", "test")}),
    "fleurs": ("google/fleurs", {"ko_kr": ("validation", "test"), "en_us": ("validation", "test")}),
}


def list_files(repo: str, config: str, split: str) -> list[str]:
    response = httpx.get(f"{API.format(repo=repo)}/{config}/{split}", follow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.json()


def remote_size(url: str) -> int:
    response = httpx.head(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    return int(response.headers.get("content-length", 0))


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            for chunk in response.iter_bytes(1 << 20):
                out.write(chunk)
    partial.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes, download nothing")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()

    for name in args.datasets:
        repo, configs = SOURCES[name]
        for config, splits in configs.items():
            for split in splits:
                for n, url in enumerate(list_files(repo, config, split)):
                    target = RAW / name / config / split / f"{n}.parquet"
                    label = f"{name}/{config}/{split}/{n}"
                    if args.dry_run:
                        local = target.stat().st_size if target.exists() else None
                        print(f"{label}: remote {remote_size(url):,} bytes, local {local}")
                    elif target.exists() and not args.force:
                        print(f"{label}: exists, skipped (--force downloads it again)")
                    else:
                        print(f"{label}: downloading ...", flush=True)
                        download(url, target)
                        print(f"{label}: {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
