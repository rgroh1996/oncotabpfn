"""Acquire all three published HANCOCK partitions; never substitute synthetic data."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from src.common import DATA, file_hash, write_json

URL = "https://data.fau.de/public/24/87/322108724/DataSplits_DataDictionaries.zip"
NAMES = [f"dataset_split_{split}.json" for split in ("in", "out", "Oropharynx")]


def main():
    destination = DATA / "paper"
    destination.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=(15, 90))
    response.raise_for_status()
    payloads = {}
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            if name in NAMES:
                if name in payloads or member.file_size > 1_000_000:
                    raise ValueError("Duplicate or oversized split file")
                payloads[name] = archive.read(member)
    if set(payloads) != set(NAMES):
        raise ValueError("Official archive is missing a paper split")
    for name, payload in payloads.items():
        path = destination / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace changed split: {name}")
        path.write_bytes(payload)
    write_json(destination / "manifest.json", {
        "source": "official", "url": URL,
        "sha256": {name: file_hash(destination / name) for name in NAMES},
    })
    print(f"Downloaded {len(payloads)} official splits to {destination}")


if __name__ == "__main__":
    main()
