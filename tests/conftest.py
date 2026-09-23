from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

from data.download_hancock import download
from data.dataset_builder import build_dataset


@pytest.fixture(scope="session")
def synthetic_cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    download(root / "raw", source="synthetic")
    frame, report = build_dataset(root / "raw", root / "processed.parquet")
    return frame, report, root
