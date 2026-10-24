"""스크립트들이 읽고 쓰는 폴더. `data/`와 `outputs/`는 git에서 제외하고, `reports/`는 커밋한다."""

# [파일 역할] 스크립트들이 읽고 쓰는 폴더 경로를 한곳에 모은 모듈이다.
# download·prepare·prepare_itn·evaluate·evaluate_ct2·evaluate_qwen·train·train_qwen·analyze가 모두 여기서
# 경로를 가져간다.
# [왜 필요한가] 경로를 스크립트마다 따로 적으면 폴더 구조를 바꿀 때 여러 곳을 고쳐야 하고, 실행한 현재
# 폴더(cwd)에 따라 결과가 다른 곳에 생길 수 있다. 이 파일의 위치를 기준으로 절대 경로를 만들어 어디서 실행하든
# 같은 폴더를 가리키게 한다.
# [커밋 여부] data/·outputs/는 크기가 커서 git에서 뺀다. reports/는 발화별 정답·가설·오류 수를 남긴 측정
# 기록이라 커밋한다. 수치를 나중에 리포트에서 다시 계산할 수 있게 하려는 것이다.

from pathlib import Path

# ROOT: 저장소 루트. 이 파일(src/whisper_ko_ft/paths.py)에서 두 단계 위 폴더다. 소스 폴더에 있는 이 파일을
# 그대로 import한다는 전제의 계산이다. evaluate의 git_commit이 커밋 해시를 읽을 때와, 리포트 경로를 루트 기준
# 상대 경로로 찍을 때 쓴다.
ROOT = Path(__file__).resolve().parents[2]
# DATA: 데이터 전체(git 제외). RAW는 download가 받은 parquet
# 원본(data/raw/<데이터셋>/<config>/<split>/<n>.parquet), CACHE는 prepare가 만든 16 kHz int16 오디오 저장소와
# 묶음 목록(data/cache/sets/<묶음>.json)이다.
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
# OUTPUTS: 학습 결과(체크포인트, LoRA 어댑터, run.json, 로그)를 두는 곳(git 제외). train과 train_qwen이
# outputs/<run 이름>/ 아래에 쓴다.
# REPORTS: 평가 리포트 reports/<모델 이름>/<묶음>.json. 커밋되는 측정 기록이며 analyze와
# tests/test_recorded_reports.py가 다시 읽어 채점한다.
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
# MANIFEST: 원본 manifest(커밋). data/raw의 parquet마다 상대 경로·바이트 크기·SHA-256과, 데이터셋마다 Hugging
# Face parquet 변환본(refs/convert/parquet)의 커밋을 적은 JSON이다. `download --write-manifest`가 쓰고,
# `download --verify`와 prepare가 읽어 받은 파일이 기록된 측정의 원본과 같은지 가린다. data/는 git 제외라
# 커밋할 수 없어 패키지 폴더에 둔다(en_norm의 english.json과 같은 방식). 패키지와 함께 설치되므로 어디서
# 실행하든 같은 기록을 읽는다.
MANIFEST = Path(__file__).with_name("data_manifest.json")
