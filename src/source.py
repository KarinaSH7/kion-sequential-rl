"""Pinned public data download and complete-user sampling. No credentials needed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
BASE = f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{REVISION}"
SOURCES = {"data_en.zip": f"{BASE}/data_en.zip", "items.csv": f"{BASE}/data_original/items.csv"}


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    expected_file = ROOT / "checksums.json"
    expected = json.loads(expected_file.read_text()) if expected_file.exists() else {}
    for name, url in SOURCES.items():
        path = raw / name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
            tmp.replace(path)
        if name in expected and sha256(path) != expected[name]:
            raise ValueError(f"Checksum mismatch: {path}")

