"""측정·학습 기록에 남길 지문: 고정한 Whisper revision, 가중치·변환 파일의 SHA-256, peft 판, 코드의 커밋.

리포트(reports/<이름>/<묶음>.json)와 run.json은 커밋되는 기록이거나 outputs/의 유일한 사본이다. 수치만
남으면 나중에 "어떤 가중치와 어떤 코드·라이브러리로 쟀는지"를 되짚을 수 없다. 이 모듈은 그 질문에 답할
값들을 한곳에서 만든다. evaluate·evaluate_ct2·evaluate_qwen·train·merge_adapter와 scripts/speed.py가
가져다 쓴다.

torch·transformers를 불러오지 않아 가볍다. scripts/speed.py(측정 명령을 자식 프로세스로 돌리는 도우미)가
커밋을 적을 때 부모 프로세스에 torch를 올리지 않게 하려는 것이다.
"""

# [파일 역할] 재현에 필요한 '무엇으로 쟀는가'를 기록하는 도우미 모음. (1) 기본 Whisper 모델의 고정
# revision(WHISPER_REVISIONS, revision_kwargs, whisper_revision), (2) 파일 지문(sha256_file,
# adapter_hashes, adapter_info, converted_model_hashes), (3) 라이브러리 판(peft_version), (4) 측정·학습
# 코드의 커밋(git_commit).
# [왜 생겼나] Qwen 경로(evaluate_qwen)는 처음부터 REGISTERED로 revision을 고정하고 리포트에 transformers
# 판과 어댑터 SHA-256을 남겼지만, Whisper 경로(evaluate·train·merge_adapter)는 from_pretrained에 revision을
# 넘기지 않아 Hub의 main을 받았고 요약에는 torch 판과 GPU 이름뿐이었다. Hub의 모델이 바뀌면 새 클론의 측정이
# 경고 없이 다른 입력으로 돌고, outputs/의 어댑터나 변환 폴더를 다시 만들어도 리포트만으로는 같은 가중치인지
# 알 수 없었다. 2026-10-29에 Whisper 경로도 같은 수준으로 맞추면서 공용 함수를 여기로 모았다(adapter_info와
# git_commit은 각각 evaluate_qwen·evaluate에서 옮겨 왔고 출력은 그대로다).
# [기록 수치] 등록한 revision은 기록된 모든 측정·학습이 쓴 Hugging Face 캐시 스냅숏이라 수치는 바뀌지 않는다.
# 리포트에는 새 키만 더해지고 기존 키의 값·꼴은 그대로다.

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path

# ROOT: 저장소 루트. git_commit이 기본으로 git을 돌리는 폴더다.
from whisper_ko_ft.paths import ROOT

# WHISPER_REVISIONS: 기본 모델(Hub ID) → 고정한 커밋 sha. 기록된 측정·학습(1~12단계)이 쓴 로컬 Hugging Face
# 캐시 스냅숏이며(캐시에 모델마다 스냅숏이 이것 하나뿐이다), 2026-10-24와 10-29에 Hub의 main과 같음을
# 확인했다(main의 마지막 변경은 whisper-small 2024-02-29, large-v3-turbo 2024-10-04로 이 저장소의 첫 측정보다
# 앞선다). 그래서 고정해도 받는 가중치는 지금까지와 같다.
# 새 클론은 이 sha를 받으므로, Hub의 모델(가중치나 generation_config)이 뒤에 바뀌어도 측정이 조용히 달라지지
# 않는다. evaluate·train·merge_adapter가 from_pretrained에 revision_kwargs로 넘기고(모델과 처리기 모두),
# 요약·run.json에 whisper_revision을 적는다.
# outputs/ 아래 로컬 폴더(체크포인트, 합친 모델)에는 revision이 없어 넘기지 않는다. 등록하지 않은 Hub ID는
# 고정하지 않고(지금까지처럼 main을 받는다) 요약에 null이 남는다. 새 기본 모델을 쓰기 시작하면 여기에 그때의
# sha를 더한다.
WHISPER_REVISIONS = {
    "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
    "openai/whisper-large-v3-turbo": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
}

# CODE_PATHS: git_commit이 '+dirty'를 판단할 때 보는 경로. 측정·학습 코드와 의존성 잠금이다. 리포트·문서만
# 바뀐 것은 측정 코드가 아니므로 보지 않는다(커밋 6ba3c27의 규칙). scripts/speed.py는 여기에 "scripts"를 더해
# 자신도 함께 본다(측정 절차 스크립트를 고친 채 잰 것도 드러나게).
CODE_PATHS = ("src", "pyproject.toml", "uv.lock")


# 등록한 Hub ID면 고정한 revision, 로컬 폴더나 등록하지 않은 Hub ID면 None. 요약·run.json의 revision 항목에
# 그대로 들어간다. 호출 위치: evaluate.main, train.main, revision_kwargs.
def whisper_revision(model: str) -> str | None:
    """등록한 Hub ID면 고정한 revision, 로컬 폴더나 등록하지 않은 ID면 None."""
    return WHISPER_REVISIONS.get(model)


# from_pretrained에 펼쳐 넣을(**) 키워드 인자. 등록한 Hub ID면 {"revision": sha}, 아니면 빈 dict라 인자를
# 아예 넘기지 않는다(로컬 폴더에 revision=None을 넘기는 대신 지금까지와 같은 호출을 유지한다). 호출 위치:
# evaluate.load_model, train.main, merge_adapter.main.
def revision_kwargs(model: str) -> dict[str, str]:
    """from_pretrained 인자: 등록한 Hub ID면 {"revision": sha}, 아니면 빈 dict(인자를 넘기지 않음)."""
    revision = whisper_revision(model)
    return {"revision": revision} if revision else {}


# 파일 하나의 SHA-256. 조각(기본 16 MiB)으로 읽어 CTranslate2 model.bin(약 1.6 GB)도 메모리에 다 올리지
# 않는다. 파일이 없으면 FileNotFoundError로 멈춘다(지문을 조용히 null로 남기지 않는다). 호출 위치:
# adapter_hashes, converted_model_hashes.
def sha256_file(path: Path, chunk_bytes: int = 1 << 24) -> str:
    """파일의 SHA-256(16진수). 조각으로 읽어 큰 파일도 메모리에 다 올리지 않는다."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


# LoRA 어댑터 폴더의 가중치(adapter_model.safetensors)와 설정(adapter_config.json: rank, alpha, 대상 층)의
# SHA-256. 체크포인트 폴더 이름만으로는 같은 가중치인지 알 수 없어서, 어떤 가중치로 쟀는지 내용으로 확인할 수
# 있게 한다. 어댑터가 없으면 두 값 모두 None. evaluate의 요약은 기존 adapter 키(준 경로 그대로)에 이 두 키만
# 더한다. 호출 위치: evaluate.main, adapter_info.
def adapter_hashes(adapter: str | None) -> dict[str, str | None]:
    """어댑터 폴더의 가중치·설정 파일의 SHA-256(어댑터가 없으면 None)."""
    if adapter is None:
        return {"adapter_sha256": None, "adapter_config_sha256": None}
    folder = Path(adapter)
    return {
        "adapter_sha256": sha256_file(folder / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(folder / "adapter_config.json"),
    }


# evaluate_qwen 리포트 요약의 어댑터 항목(10단계부터): 어댑터 폴더의 posix 경로와 위 두 SHA-256. 11단계처럼
# 같은 가중치를 다른 스케일로 합친 실행은 요약의 스케일 항목(adapter_scale)으로 구분한다. 2026-10-29에
# evaluate_qwen에서 옮겨 왔고 키와 값의 꼴은 그대로다. 호출 위치: evaluate_qwen.main.
def adapter_info(adapter: str | None) -> dict:
    """리포트 요약에 넣을 어댑터 폴더(posix 경로)와 그 가중치·설정의 SHA-256."""
    return {"adapter": None if adapter is None else Path(adapter).as_posix(), **adapter_hashes(adapter)}


# CTranslate2 변환 폴더(ct2-transformers-converter 출력)의 가중치(model.bin)와 설정(config.json)의 SHA-256.
# 같은 경로에 다시 변환한 이력(outputs/turbo-n/ct2)이 있어, 리포트만으로 어느 변환으로 쟀는지 알 수 있게
# 한다. 변환기는 두 파일을 늘 쓴다. model.bin이 없으면 faster_whisper.WhisperModel이 불러오다 멈추지만,
# config.json이 없는 폴더는 띄우기는 하고 첫 디코딩에서야 알기 어려운 JSON 오류(type_error.305)로 멈춘다
# (2026-10-29 CPU에서 작은 무작위 모델로 확인). 그래서 디코딩 전에 여기서 FileNotFoundError로 멈춘다(지문을
# null로 남기지 않는다). 호출하는 쪽이 측정 시계 밖에서 한 번만 부른다. 호출 위치: evaluate_ct2.main.
def converted_model_hashes(folder: Path) -> dict[str, str]:
    """CTranslate2 변환 폴더의 model.bin·config.json SHA-256."""
    folder = Path(folder)
    return {
        "model_sha256": sha256_file(folder / "model.bin"),
        "config_sha256": sha256_file(folder / "config.json"),
    }


# 불러온 peft의 판. 어댑터를 합치거나 LoRA를 학습하는 실행에서만 부른다(그때는 peft가 이미 올라와 있어
# import가 공짜다). peft를 쓰지 않는 실행(기본 모델 측정, 전체 파인튜닝)은 부르지 않고 null을 적는다. 그
# 실행의 계산에는 peft가 들지 않고, evaluate가 어댑터 없이는 peft를 불러오지 않던 동작도 그대로 둔다. 호출
# 위치: evaluate.main, train.main.
def peft_version() -> str:
    """불러온 peft 모듈의 판."""
    import peft

    return peft.__version__


# 측정·학습한 코드의 커밋을 리포트·run.json·speed.json에 남긴다. 짧은 HEAD 해시이고, paths(기본 CODE_PATHS:
# src·pyproject.toml·uv.lock)가 HEAD와 다르면 '+dirty'를 붙인다. 리포트나 문서만 바뀐 것은 측정 코드가
# 아니므로 보지 않는다. git이 없으면 'unknown', 커밋이 없으면 'uncommitted'.
# 2026-10-29에 evaluate에서 옮겨 오며 paths 인자를 더했다(기본값은 그대로). 호출 위치:
# evaluate·evaluate_ct2·evaluate_qwen·train·train_qwen(evaluate를 거쳐)·scripts/speed.py.
def git_commit(root: Path = ROOT, paths: Sequence[str] = CODE_PATHS) -> str:
    """짧은 HEAD 해시. `paths`가 HEAD와 다르면 '+dirty'를 붙인다(리포트·문서는 보지 않는다)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=root,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        return "unknown"
    if not head:
        return "uncommitted"
    return f"{head}+dirty" if dirty else head
