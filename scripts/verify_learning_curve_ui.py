"""Verify the completed research view without consuming inference quota."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest

from src.common import ROOT, RESULTS, read_json, write_json


def main():
    root = RESULTS / "learning_curves"
    status = read_json(root / "status.json")
    assert status["complete"]
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=45).run()
    assert not app.exception, app.exception
    assert any("600 final evaluations completed" in item.value for item in app.success)
    assert any("Paper-matched sample efficiency" in item.value for item in app.subheader)
    primary = [table.value for table in app.dataframe if "reference" in table.value.columns and "size" in table.value.columns]
    assert any(len(table) == 10 and table["size"].str.startswith("Low-N").all() for table in primary)
    downloads = [item.proto.label for item in app.get("download_button")]
    assert "Download learning-curve metrics" in downloads
    assert "Download learning-curve report" in downloads
    assert "assessment" not in app.session_state
    write_json(root / "ui_verification.json", dict(protocol_id=status["protocol_id"], complete=True,
               rendered_without_exception=True, primary_comparison_rows=10, downloads_verified=True,
               live_inference_requested=False, native_browser_visual_inspection=False))
    print("Completed learning-curve dashboard view passed Streamlit AppTest; no inference requested.")


if __name__ == "__main__":
    main()
