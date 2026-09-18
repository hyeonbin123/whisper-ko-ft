"""Folders the scripts read and write. `data/` and `outputs/` are git-ignored; `reports/` is committed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
