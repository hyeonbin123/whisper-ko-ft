"""커밋된 평가 리포트 파일 목록: reports/<이름>/<묶음>.json 중 파생 파일 폴더를 뺀 것."""

# [파일 역할] 커밋된 리포트를 모두 다시 계산하는 테스트(test_recorded_reports.py의 발화별 채점,
# test_metrics.py의 요약 구간)가 같은 목록을 쓰게 하는 도우미다. 테스트 파일이 아니라 pytest가 모으지 않는다.
# [왜 필요한가] 예전에는 두 파일이 파생 폴더 제외 목록을 따로 두었다. 새 파생 폴더(13단계 문장 hold-out
# 정의)를 더할 때 한쪽만 고쳐, 다른 쪽이 그 폴더의 파일을 리포트로 읽다가 KeyError('summary')로 깨졌다.
# 목록을 여기 하나로 두어 다음 파생 폴더도 한 곳만 고치게 한다.
# [연결] tests 폴더에 __init__.py가 없어 pytest가 이 폴더를 import 경로에 넣으므로, 두 테스트 파일이
# 'from report_files import …'로 가져온다(tiny_qwen.py와 같은 방식).

from pathlib import Path

from whisper_ko_ft.paths import REPORTS

# DERIVED: reports/ 아래에서 평가 리포트가 아닌 파생 파일 폴더. 셋 다 요약(summary)과 발화별 정답·가설이 없다.
# - digit_utterances: 묶음의 '숫자 발화' 목록(analyze digits가 쓴다)
# - sentence_overlap: 학습·평가 묶음의 문장 겹침(analyze overlap이 쓰고 tests/test_analyze.py가 수를 고정한다)
# - sentence_holdout: 13단계 문장 hold-out 정의(prepare_heldout이 쓰고 tests/test_prepare_heldout.py가 수를
#   고정한다)
# 리포트가 아닌 폴더를 reports/에 새로 커밋하면 여기에만 더한다.
DERIVED = frozenset({"digit_utterances", "sentence_overlap", "sentence_holdout"})


def report_files() -> list[Path]:
    """reports/<이름>/<묶음>.json 중 DERIVED 폴더를 뺀 평가 리포트의 경로(경로 순)."""
    return sorted(p for p in REPORTS.glob("*/*.json") if p.parent.name not in DERIVED)
