import numpy as np
import pandas as pd
import pytest

from src.explain_tabpfn import CachedPredictor, aggregate_importance


def test_prediction_cache_keeps_nan_rows_and_feature_order(tmp_path):
    class Model:
        columns = ["age", "stage"]
        calls = 0
        def predict(self, values, batch_size):
            self.calls += 1
            return np.nan_to_num(values[:, 0], nan=.2)
    model = Model()
    predictor = CachedPredictor(model, tmp_path, batch_size=2)
    x = np.array([[.3, 1], [np.nan, 2], [.8, 0]])
    np.testing.assert_allclose(predictor(x), [.3, .2, .8])
    assert model.calls == 2
    np.testing.assert_allclose(predictor(x), [.3, .2, .8])
    assert model.calls == 2
    np.testing.assert_allclose(predictor(x[::-1]), [.8, .2, .3])
    with pytest.raises(ValueError):
        predictor(x[:, :1])


def test_shap_global_ranking_uses_absolute_not_signed_mean():
    values = np.array([[.4, .1], [-.4, .1]])
    ranking = aggregate_importance(values, ["varying", "constant"])
    assert ranking.iloc[0].feature == "varying"
    assert ranking.iloc[0].mean_signed_shap == 0


def test_shap_pipeline_preserves_categories_and_probability_additivity(tmp_path):
    shap = pytest.importorskip("shap")
    class Model:
        columns = ["age", "site"]
        def predict(self, values, batch_size):
            # Nominal codes are masked as a single feature; no invalid one-hot states.
            assert set(values[:, 1]) <= {0., 1., 2.}
            return .1 + .2 * values[:, 0] + .1 * (values[:, 1] == 2)
    model = Model()
    # SHAP's masker inherits background dtype; floats preserve fractional inputs.
    background = pd.DataFrame([[0, 0], [1, 1]], columns=model.columns, dtype=float)
    x = pd.DataFrame([[.5, 2]], columns=model.columns, dtype=float)
    predictor = CachedPredictor(model, tmp_path)
    explanation = shap.PermutationExplainer(predictor, background, seed=42)(x, max_evals=20)
    np.testing.assert_allclose(explanation.values.sum(axis=1)+explanation.base_values, predictor(x), atol=1e-8)
    np.testing.assert_allclose(explanation.values[0], [0, .1], atol=1e-8)
