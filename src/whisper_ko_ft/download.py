"""Zeroth-Korean과 FLEURS(한국어, 영어)의 parquet 파일을 Hugging Face에서 내려받는다.

사용법:
    uv run python -m whisper_ko_ft.download            # 전부 받는다. 약 4 GB
    uv run python -m whisper_ko_ft.download --dry-run  # 파일 목록과 크기만 보여 준다

data/raw/<데이터셋>/<config>/<split>/<n>.parquet에 쓴다(git 제외). Hugging Face가 kresnik/zeroth_korean과
google/fleurs(둘 다 CC BY 4.0)에 대해 제공하는 parquet 변환본이다.
"""

# [파일 역할] 0단계(기반)의 첫 단계. Hugging Face가 제공하는 parquet 변환본을 data/raw/에 받는다. 다음 단계
# prepare.py가 이 파일들을 읽어 오디오 저장소를 만든다.
# [설계 이유] datasets 라이브러리로 받아 디코딩하지 않고 parquet 파일을 직접 받는다. datasets의 오디오
# 디코딩은 Windows에서 torchcodec이 필요해서, parquet 안의 오디오 바이트를 prepare가 soundfile로 직접 읽게
# 했다.
# [흐름] 데이터셋·config·split마다 API로 parquet URL 목록을 받는다 → 파일마다 이미 있으면 건너뛰고(--force면
# 다시 받음) → 없으면 .part 임시 파일에 받은 뒤 이름을 바꾼다. --dry-run은 받지 않고 원격 크기와 로컬 크기만
# 찍는다(새 클론 재현 때 원격과 크기가 같은지 이것으로 확인했다, docs/experiments.md).

from __future__ import annotations

import argparse
from pathlib import Path

# httpx: 파일 목록 API 호출(GET), 원격 크기 확인(HEAD), 큰 파일의 스트리밍 내려받기에 쓴다. 모든 요청에
# follow_redirects=True를 줘 리다이렉트된 실제 파일 위치까지 따라간다.
import httpx

# RAW: 내려받은 parquet을 두는 data/raw(git 제외).
from whisper_ko_ft.paths import RAW

# API: 데이터셋 저장소의 parquet 변환본 URL 목록을 돌려주는 Hugging Face 엔드포인트. 뒤에 /<config>/<split>을
# 붙여 쓴다.
API = "https://huggingface.co/api/datasets/{repo}/parquet"
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


# 명령줄 진입점. --datasets로 일부만 받을 수 있다. --dry-run은 받지 않고 목록과 크기만, --force는 이미 있는
# 파일도 다시 받는다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--dry-run", action="store_true", help="파일 목록과 크기만 보여 주고 받지 않는다")
    parser.add_argument("--force", action="store_true", help="파일이 이미 있어도 다시 받는다")
    args = parser.parse_args()

    for name in args.datasets:
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
                        # 눈으로 확인한다.
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
