import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.benchmark import run_benchmark
from src.common import ROOT, TARGETS, read_json


def test_resumable_benchmark_and_unavailable_api(synthetic_cohort, tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.load_token", lambda: "")
    df = synthetic_cohort[0]
    output = tmp_path / "benchmark"
    args = dict(output=output, models=["tabpfn", "lightgbm", "xgboost"], sizes=["50", "5000"], seeds=[42], targets=[TARGETS[0]])
    metrics = run_benchmark(df, **args)
    assert len(metrics.loc[metrics.status.eq("complete")]) == 2
    assert len(metrics.loc[metrics.status.eq("unavailable")]) == 1
    assert len(metrics.loc[metrics.status.eq("skipped")]) == 3
    assert (output / "benchmark_performance.png").stat().st_size > 1000
    assert (output / "roc_calibration.png").exists()
    completed = metrics.loc[metrics.status.eq("complete"), "run_id"]
    timestamps = {rid: (output / "runs" / f"{rid}.json").stat().st_mtime_ns for rid in completed}
    run_benchmark(df, **args)
    assert timestamps == {rid: (output / "runs" / f"{rid}.json").stat().st_mtime_ns for rid in completed}
    assert not read_json(output / "manifest.json")["comparison_complete"]
    pairs = pd.read_csv(output / "paired_comparisons.csv")
    assert len(pairs) == 4


def test_app_missing_token_and_manual_form(monkeypatch):
    monkeypatch.setattr("src.common.load_token", lambda: "")
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Patient assessment", "Paper replication", "Optimization & explanations", "Cohort & methodology"]
    assert any("763" in m.value for m in app.metric)
    submit = next(b for b in app.button if b.label == "Generate risk assessment")
    assert submit.disabled
    app.radio[0].set_value("New patient").run()
    assert not app.exception
    assert any(n.label == "Neutrophils · 10⁹/L" and n.value is None for n in app.number_input)


def test_app_prediction_render_with_mock_api(monkeypatch):
    # Exercise rendering/error paths without claiming authenticated model verification.
    import streamlit as st
    st.cache_resource.clear()
    monkeypatch.setattr("src.common.load_token", lambda: "test-token")
    class FakeDashboard:
        evaluation = {"conformal_coverage": .91, "conformal_mean_set_size": 1.4, "counts": {"test": 79}}
        def predict(self, frame):
            return np.array([.3]), np.array([[True, True]])
    monkeypatch.setattr("src.pipeline.fit_dashboard_model", lambda *a, **kw: FakeDashboard())
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    next(b for b in app.button if b.label == "Generate risk assessment").click().run()
    assert not app.exception
    assert app.session_state["assessment"]["results"][TARGETS[0]]["probability"] == .3
    assert any("Both outcomes remain plausible" in i.value for i in app.info)
    st.cache_resource.clear()


def test_app_api_failure_is_visible(monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    monkeypatch.setattr("src.common.load_token", lambda: "test-token-failure")
    def fail(*args, **kwargs):
        raise RuntimeError("API quota exhausted (simulated)")
    monkeypatch.setattr("src.pipeline.fit_dashboard_model", fail)
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    next(b for b in app.button if b.label == "Generate risk assessment").click().run()
    assert not app.exception
    assert any("quota exhausted" in w.value for w in app.warning)
    st.cache_resource.clear()
