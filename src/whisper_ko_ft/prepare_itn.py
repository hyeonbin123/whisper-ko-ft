"""Write copies of the Zeroth training sets whose references have numbers as digits (stage 6).

Usage:
    uv run python -m whisper_ko_ft.prepare_itn

Reads data/cache/sets/zeroth-train.json and zeroth-val500.json (from `prepare`) and writes
zeroth-train-itn.json and zeroth-val500-itn.json next to them. The audio is shared.
"""

# [파일 역할] 6단계(정답의 숫자 표기를 바꿔 학습)용 묶음 목록을 만든다. prepare가 만든 zeroth-train과
# zeroth-val500의 정답에 itn.to_digits를 적용해 '이천 십 팔 년' 같은 숫자 낱말을 '2018년'처럼 아라비아 숫자로
# 바꾼 사본(zeroth-train-itn, zeroth-val500-itn)을 쓴다.
# [왜 필요한가] 2·3단계의 '도메인 전용' 판정은 대부분 표기에서 나왔다. Zeroth 정답은 숫자를 한글로 풀어
# 쓰는데, 이를 배운 모델이 아라비아 숫자로 적는 FLEURS에서 오류를 냈다. 학습 정답의 표기만 바꿔, 같은 도메인의
# 이득을 지키면서 이 악화를 없앨 수 있는지 보려는 것이다(docs/experiments.md 6단계).
# [설계 이유] 오디오 저장소는 그대로 두고 정답만 바꾼 목록을 따로 쓴다. 원래 목록을 덮어쓰지 않으므로 앞
# 단계의 묶음이 그대로 남는다. validation 전체와 test 묶음은 바꾸지 않고, 채점할 때 정답과 가설을 같은
# 규칙으로 맞춘 '맞춘 CER'로 비교한다.
# [연결] train --train-set zeroth-train-itn --eval-set zeroth-val500-itn(6·7단계 N, N2, README 6단계 명령)과
# train_qwen의 기본 학습 묶음(10단계 QN)이 이 목록을 쓴다.

from __future__ import annotations

import json

# itn.to_digits: 숫자 낱말 묶음 바로 뒤에 단위가 올 때만 아라비아 숫자로 바꾸는 규칙 기반 변환. CACHE: 묶음
# 목록이 있는 data/cache.
from whisper_ko_ft.itn import to_digits
from whisper_ko_ft.paths import CACHE


# 두 묶음 각각에 대해 정답을 바꾼 사본을 <이름>-itn.json으로 쓰고 바뀐 문장 수를 찍는다. 기록된 수는 학습
# 20,049문장 중 4,703문장, val-500 440문장 중 109문장이다(docs/experiments.md 6단계).
def main() -> None:
    for name in ("zeroth-train", "zeroth-val500"):
        data = json.loads((CACHE / "sets" / f"{name}.json").read_text(encoding="utf-8"))
        changed = 0
        # 발화마다 정답만 to_digits로 바꾸고 원문과 달라진 문장 수를 센다(불리언을 더하면 True가 1). 발화 ID와
        # 저장소 위치 등 나머지 키는 그대로라 같은 오디오를 가리킨다.
        for utterance in data["utterances"]:
            rewritten = to_digits(utterance["text"])
            changed += rewritten != utterance["text"]
            utterance["text"] = rewritten
        target = CACHE / "sets" / f"{name}-itn.json"
        target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8", newline="\n")
        print(f"{name}-itn: {changed:,} of {len(data['utterances']):,} references rewritten")


if __name__ == "__main__":
    main()
