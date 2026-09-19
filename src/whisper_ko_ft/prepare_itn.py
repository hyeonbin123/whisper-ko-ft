"""Write copies of the Zeroth training sets whose references have numbers as digits (stage 6).

Usage:
    uv run python -m whisper_ko_ft.prepare_itn

Reads data/cache/sets/zeroth-train.json and zeroth-val500.json (from `prepare`) and writes
zeroth-train-itn.json and zeroth-val500-itn.json next to them. The audio is shared.
"""

from __future__ import annotations

import json

from whisper_ko_ft.itn import to_digits
from whisper_ko_ft.paths import CACHE


def main() -> None:
    for name in ("zeroth-train", "zeroth-val500"):
        data = json.loads((CACHE / "sets" / f"{name}.json").read_text(encoding="utf-8"))
        changed = 0
        for utterance in data["utterances"]:
            rewritten = to_digits(utterance["text"])
            changed += rewritten != utterance["text"]
            utterance["text"] = rewritten
        target = CACHE / "sets" / f"{name}-itn.json"
        target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8", newline="\n")
        print(f"{name}-itn: {changed:,} of {len(data['utterances']):,} references rewritten")


if __name__ == "__main__":
    main()
