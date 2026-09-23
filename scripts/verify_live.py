"""Opt-in, authenticated Streamlit integration check; consumes TabPFN API quota."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

from src.common import DATA, RESULTS, ROOT, TARGETS, file_hash, load_token, safe_error, write_json


def check_prediction(app: AppTest) -> dict:
    next(button for button in app.button if button.label == "Generate risk assessment").click()
    app.run(timeout=240)
    if app.exception:
        raise RuntimeError("Streamlit raised an exception during live prediction")
    assessment = app.session_state["assessment"]
    for target in TARGETS:
        result = assessment["results"][target]
        if "error" in result:
            raise RuntimeError(result["error"])
        if not 0 <= result["probability"] <= 1 or len(result["outcome_set"]) != 2:
            raise RuntimeError(f"Invalid prediction shape for {target}")
    return assessment


def main() -> None:
    if not load_token():
        raise RuntimeError("Set TABPFN_TOKEN before running this opt-in live check")
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=240).run()
    if app.exception:
        raise RuntimeError("Streamlit failed to render")
    print("Checking held-out cohort patient predictions…", flush=True)
    cohort = check_prediction(app)
    print("Both cohort endpoints returned calibrated predictions and conformal sets.", flush=True)
    app.radio[0].set_value("New patient").run()
    next(widget for widget in app.slider if widget.label == "Age at diagnosis").set_value(65)
    next(widget for widget in app.selectbox if widget.label == "Pathological T stage").set_value("T3")
    next(widget for widget in app.selectbox if widget.label == "Pathological N stage").set_value("N1")
    print("Checking manual patient predictions with cached model contexts…", flush=True)
    manual = check_prediction(app)
    result = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verification": "Authenticated Streamlit AppTest, not native-browser visual inspection",
        "processed_sha256": file_hash(DATA / "hancock_processed.parquet"),
        "cohort_assessment": cohort,
        "manual_assessment": manual,
    }
    write_json(RESULTS / "live_verification.json", result)
    print(f"Live integration verified for both modes and endpoints; report: {RESULTS / 'live_verification.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        sys.exit(1)
