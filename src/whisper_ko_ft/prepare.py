"""Turn the downloaded parquet files into audio stores and the evaluation/training sets.

Usage:
    uv run python -m whisper_ko_ft.prepare

Writes data/cache/<store>.int16 + <store>.json, and data/cache/sets/<set>.json:
    zeroth-train    Zeroth-Korean train without the validation speakers
    zeroth-val      the 10 validation speakers (docs/experiments.md, "데이터 분리")
    zeroth-val500   50 utterances per validation speaker, for picking checkpoints during training
    zeroth-test, fleurs-ko-val, fleurs-ko-test, fleurs-en-val, fleurs-en-test
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

from whisper_ko_ft.paths import CACHE, RAW
from whisper_ko_ft.splits import val500_ids, validation_speakers
from whisper_ko_ft.store import SAMPLE_RATE

# store -> (parquet folder, text column, speaker column or None)
STORES = {
    "zeroth_train": (RAW / "zeroth" / "default" / "train", "text", "speaker_id"),
    "zeroth_test": (RAW / "zeroth" / "default" / "test", "text", "speaker_id"),
    "fleurs_ko_validation": (RAW / "fleurs" / "ko_kr" / "validation", "transcription", None),
    "fleurs_ko_test": (RAW / "fleurs" / "ko_kr" / "test", "transcription", None),
    "fleurs_en_validation": (RAW / "fleurs" / "en_us" / "validation", "transcription", None),
    "fleurs_en_test": (RAW / "fleurs" / "en_us" / "test", "transcription", None),
}


def decode(audio_bytes: bytes) -> np.ndarray:
    """16 kHz mono int16 samples."""
    samples, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        mono = resample_poly(mono, SAMPLE_RATE, rate).astype(np.float32)
    return np.clip(np.round(mono * 32768.0), -32768, 32767).astype(np.int16)


def build_store(name: str, folder: Path, text_column: str, speaker_column: str | None) -> list[dict]:
    """Write <name>.int16 and return its index entries. Skips the work if the index already exists."""
    index_path = CACHE / f"{name}.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))

    CACHE.mkdir(parents=True, exist_ok=True)
    columns = ["id", "audio", text_column] + ([speaker_column] if speaker_column else [])
    entries: list[dict] = []
    offset = 0
    partial = CACHE / f"{name}.int16.part"
    with partial.open("wb") as out:
        for parquet in sorted(folder.glob("*.parquet"), key=lambda p: int(p.stem)):
            reader = pq.ParquetFile(parquet)
            for batch in reader.iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    samples = decode(row["audio"]["bytes"])
                    out.write(samples.tobytes())
                    entries.append(
                        {
                            "id": str(row["id"]),
                            "speaker": str(row[speaker_column]) if speaker_column else "",
                            "text": row[text_column].strip(),
                            "store": name,
                            "offset": offset,
                            "length": len(samples),
                        }
                    )
                    offset += len(samples)
            print(f"{name}: {parquet.name} done, {len(entries):,} utterances", flush=True)
    partial.replace(CACHE / f"{name}.int16")
    index_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8", newline="\n")
    return entries


def write_set(name: str, language: str, utterances: list[dict]) -> None:
    target = CACHE / "sets" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"language": language, "utterances": utterances}
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    hours = sum(u["length"] for u in utterances) / SAMPLE_RATE / 3600
    speakers = len({u["speaker"] for u in utterances if u["speaker"]})
    print(f"set {name}: {len(utterances):,} utterances, {hours:.2f} h, {speakers} speakers")


def main() -> None:
    stores = {name: build_store(name, *spec) for name, spec in STORES.items()}

    train = stores["zeroth_train"]
    held_out = set(validation_speakers(u["speaker"] for u in train))
    validation = [u for u in train if u["speaker"] in held_out]
    picked = set(val500_ids(validation))
    print(f"validation speakers: {sorted(held_out)}")

    write_set("zeroth-train", "ko", [u for u in train if u["speaker"] not in held_out])
    write_set("zeroth-val", "ko", validation)
    write_set("zeroth-val500", "ko", [u for u in validation if u["id"] in picked])
    write_set("zeroth-test", "ko", stores["zeroth_test"])
    write_set("fleurs-ko-val", "ko", stores["fleurs_ko_validation"])
    write_set("fleurs-ko-test", "ko", stores["fleurs_ko_test"])
    write_set("fleurs-en-val", "en", stores["fleurs_en_validation"])
    write_set("fleurs-en-test", "en", stores["fleurs_en_test"])


if __name__ == "__main__":
    main()
