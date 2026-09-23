import numpy as np
import pandas as pd
import pytest

from src.pipeline import (ConformalClassifier, FeaturePreprocessor, ModelUnavailable, dashboard_partition,
                          evaluation_splits, fit_dashboard_model, make_estimator, metric_scores, stratified_order)
from src.common import TARGETS


def test_preprocessing_excludes_leakage_and_unseen_categories(synthetic_cohort):
    frame = synthetic_cohort[0]
    train = frame.iloc[:100].copy()
    prep = FeaturePreprocessor().fit(train)
    before = prep.transform(frame.iloc[100:110])
    altered = frame.iloc[100:110].copy()
    for name in ["target_survival_5y", "target_recurrence", "patient_id", "days_to_last_information", "recurrence"]:
        altered[name] = "leakage"
    np.testing.assert_array_equal(before, prep.transform(altered))
    altered["primary_tumor_site"] = "unseen site"
    assert np.isfinite(prep.transform(altered)).all()
    assert "neutrophils" not in prep.numeric and "nlr" not in prep.numeric
    assert "smoking_pack_years" not in prep.numeric


def test_imputation_and_pca_fit_only_on_training():
    train = pd.DataFrame({"age": [40., 50., 60., np.nan], "embedding_0000": [1., 2., 3., np.nan],
                          "embedding_0001": [3., 1., 2., np.nan]})
    prep = FeaturePreprocessor().fit(train)
    assert prep.tabular.named_transformers_["numeric"].statistics_[0] == 50
    pca_mean = prep.embedding_scaler.mean_.copy()
    test = pd.DataFrame({"age": [500.], "embedding_0000": [10000.], "embedding_0001": [20000.]})
    assert np.isfinite(prep.transform(test)).all()
    np.testing.assert_array_equal(prep.embedding_scaler.mean_, pca_mean)
    assert prep.pca.n_components_ <= 2
    missing = prep.transform(pd.DataFrame({"age": [50.]}))
    assert missing[0, -1] == 0
    np.testing.assert_array_equal(missing[0, -(prep.pca.n_components_+1):-1], 0)


def test_nested_samples_and_split_isolation(synthetic_cohort):
    df = synthetic_cohort[0]
    for target in TARGETS:
        _, train, test = next(evaluation_splits(df, target))
        assert not set(train.patient_id) & set(test.patient_id)
        order = stratified_order(train, target, 42)
        assert len(set(order)) == len(train)
        assert set(order[:50]) <= set(order[:100]) <= set(order[:200])
        assert abs(train.loc[order[:50], target].mean()-train[target].mean()) <= .02
        parts = [*dashboard_partition(train), test]
        for i, a in enumerate(parts):
            for b in parts[i+1:]:
                assert not set(a.patient_id) & set(b.patient_id)
        folds = list(evaluation_splits(df, target, "cv"))
        assert len(folds) == 5
        assert sum(len(test) for _, _, test in folds) == df[target].notna().sum()


def test_metrics_and_one_class():
    scores = metric_scores([0, 1], [.1, .9])
    assert scores["roc_auc"] == 1
    assert scores["average_precision"] == 1
    assert scores["brier"] == pytest.approx(.01)
    assert scores["ece"] == pytest.approx(.1)
    assert metric_scores([0, 0], [.2, .3])["roc_auc"] is None
    with pytest.raises(ValueError):
        metric_scores([0, 1], [.2, 2.])


def test_conformal_finite_sample_quantile_and_empty_set():
    # Nine scores; corrected 90% quantile must be the maximum, not interpolated.
    conformal = ConformalClassifier(.1).fit(np.arange(1, 10) / 10, np.zeros(9))
    assert conformal.threshold == pytest.approx(.9)
    assert conformal.predict([.5]).tolist() == [[True, True]]
    small = ConformalClassifier(.1).fit([.1], [0])
    assert small.predict([.5]).tolist() == [[True, True]]
    confident = ConformalClassifier(.1).fit([.01]*10, [0]*10)
    assert confident.predict([.5]).tolist() == [[False, False]]


def test_missing_token_never_substitutes_a_baseline(monkeypatch):
    monkeypatch.setattr("src.pipeline.load_token", lambda: "")
    with pytest.raises(ModelUnavailable, match="TABPFN_TOKEN"):
        make_estimator("tabpfn")


def test_tabpfn_adapter_selects_explicit_version(monkeypatch):
    from tabpfn_client import TabPFNClassifier
    from tabpfn_client.constants import ModelVersion
    captured = {}
    monkeypatch.setattr("src.pipeline.load_token", lambda: "test-token-not-real")
    monkeypatch.setattr("tabpfn_client.set_access_token", lambda token: None)
    def factory(version, **kwargs):
        captured.update(version=version, **kwargs)
        return "adapter-tested-without-network"
    monkeypatch.setattr(TabPFNClassifier, "create_default_for_version", factory)
    assert make_estimator("tabpfn", 44) == "adapter-tested-without-network"
    assert captured == {"version": ModelVersion.V3_5, "random_state": 44}


def test_dashboard_calibration_integration_with_baseline(synthetic_cohort):
    df = synthetic_cohort[0]
    fitted = fit_dashboard_model(df, TARGETS[0], model_name="lightgbm", artifact_dir=None)
    p, sets = fitted.predict(df.loc[df.patient_id.isin(fitted.test_ids)].iloc[:5])
    assert p.shape == (5,) and sets.shape == (5, 2)
    assert ((p >= 0) & (p <= 1)).all()
    assert 0 <= fitted.evaluation["conformal_coverage"] <= 1
    assert sum(fitted.evaluation["counts"].values()) == df[TARGETS[0]].notna().sum()
