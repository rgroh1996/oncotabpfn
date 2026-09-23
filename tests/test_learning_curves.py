import numpy as np
import pytest

from src.learning_curve_protocol import (calibrate, folds_for, nested_order,
                                         resample_training, score_probabilities)


def test_nested_samples_preserve_budget_prevalence_and_identity():
    y = np.array([0] * 429 + [1] * 108)
    order = nested_order(y, 42)
    assert sorted(order.tolist()) == list(range(len(y)))
    assert np.array_equal(order, nested_order(y, 42))
    assert not np.array_equal(order, nested_order(y, 43))
    previous = set()
    for n in (50, 100, 200, 500, len(y)):
        chosen = order[:n]
        assert previous <= set(chosen)
        assert abs(y[chosen].sum()-n*y.mean()) <= .5 + 1e-10
        seen = []
        for fit, validation in folds_for(y[chosen], 42):
            assert not set(fit) & set(validation)
            assert set(fit) | set(validation) == set(range(n))
            assert np.bincount(y[chosen][fit]).min() >= 2
            seen.extend(validation.tolist())
        assert sorted(seen) == list(range(n))
        previous = set(chosen)


def test_sigmoid_uses_only_oof_labels_and_handles_extreme_probabilities():
    oof = np.array([0, .1, .2, .3, .6, .7, .8, 1])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    p, params = calibrate(oof, y, np.array([0, .5, 1]))
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    _, other = calibrate(oof, y, np.array([.97, .02]))
    assert params == other  # Test predictions cannot change calibration parameters.
    assert params["slope"] > 0
    assert np.isfinite(list(score_probabilities(y, oof).values())).all()


def test_small_minority_smote_does_not_touch_validation():
    pytest.importorskip("imblearn")
    x = np.arange(24).reshape(12, 2)
    y = np.array([0] * 10 + [1] * 2)
    before = x.copy()
    xx, yy = resample_training(x, y, 42)
    np.testing.assert_array_equal(x, before)
    assert len(xx) == 20 and np.bincount(yy).tolist() == [10, 10]


def test_quota_saving_drops_only_calibration_calls(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from src.common import read_json
    from src.learning_curve_protocol import run_case
    from src.learning_curve_budget import run_raw_case
    import src.learning_curve_protocol as protocol
    import src.learning_curve_budget as budget

    counts = []
    def fake_api(name, seed):
        counts.append(seed)
        return LogisticRegression(random_state=seed)
    monkeypatch.setattr(protocol, "make_estimator", fake_api)
    monkeypatch.setattr(budget, "make_estimator", fake_api)
    rng = np.random.default_rng(123)
    y = np.array([0, 1] * 9)
    x = rng.normal(size=(len(y), 3))
    data = dict(train=x, y=y, test=rng.normal(size=(6, 3)), test_y=np.array([0, 1] * 3))
    for fold, (fit, val) in enumerate(folds_for(y, 47)):
        data.update({f"fit_{fold}": x[fit], f"val_{fold}": x[val],
                     f"fit_indices_{fold}": fit, f"val_indices_{fold}": val})
    case = dict(name="sample", target="recurrence", size="50", n=len(y), seed=47,
                training_ids=[str(i) for i in range(len(y))], test_ids=[str(i) for i in range(30, 36)])
    for name in ("full", "raw"):
        (tmp_path / name / "inputs").mkdir(parents=True)
        np.savez_compressed(tmp_path / name / "inputs/sample.npz", **data)
    run_case(tmp_path / "full", case, "tabpfn", "test")
    assert len(counts) == 4
    run_raw_case(tmp_path / "raw", case, "test", "quota-test")
    assert len(counts) == 5
    full = read_json(tmp_path / "full/runs/sample_tabpfn.json")
    raw = read_json(tmp_path / "raw/runs/sample_tabpfn.json")
    np.testing.assert_allclose(full["probabilities"], raw["probabilities"])
    assert full["raw_metrics"] == raw["raw_metrics"]
    assert raw["calibration_available"] is False
    assert "calibrated_probabilities" not in raw


def test_token_cap_blocks_a_case_before_any_prediction(tmp_path, monkeypatch):
    import src.learning_curve_budget as budget
    from src.common import write_json
    write_json(tmp_path / "quota_usage_start.json", {"monthly_tokens_used": 0})
    case = dict(name="budget_check", seed=42)
    monkeypatch.setattr(budget, "prepare_amendment", lambda output: ({"cases": [case]}, {}))
    current = dict(monthly_tokens_used=1_080_000, monthly_token_limit=20_000_000,
                   daily_tokens_used=1_080_000, daily_token_limit=5_000_000)
    monkeypatch.setattr(budget, "usage", lambda: current)
    monkeypatch.setattr("src.learning_curves.report", lambda output: None)
    def forbidden(*args, **kwargs):
        pytest.fail("Prediction started despite insufficient remaining budget")
    monkeypatch.setattr(budget, "run_case", forbidden)
    with pytest.raises(RuntimeError, match="1.1-million"):
        budget.run_budget(tmp_path)


def test_nonnegative_sigmoid_matches_sklearn_and_never_inverts():
    from sklearn.linear_model import LogisticRegression

    from src.learning_curve_protocol import logit
    from src.learning_curve_recalibration import fit_sigmoid
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    informative = np.clip(.3 + .4 * y + rng.normal(0, .15, 80), .01, .99)
    reference = LogisticRegression(C=1., tol=1e-10, max_iter=10000).fit(logit(informative), y)
    np.testing.assert_allclose(fit_sigmoid(informative, y, nonnegative=True),
                               [reference.coef_[0, 0], reference.intercept_[0]], atol=1e-6)
    anti = 1 - informative
    assert fit_sigmoid(anti, y, nonnegative=False)[0] < 0
    slope, intercept = fit_sigmoid(anti, y, nonnegative=True)
    assert slope == 0 and np.isclose(intercept, np.log(y.mean() / (1 - y.mean())))
