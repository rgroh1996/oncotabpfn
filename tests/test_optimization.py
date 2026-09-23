import numpy as np
import pytest

from src.optimization import CONFIGS, OptimizationModel, native_categorical_indices, rank_configurations, selected_columns
from src.paper_protocol import load_features, load_partition
from sklearn.model_selection import StratifiedKFold


def test_candidates_have_no_outcomes_or_ids_and_preserve_categories():
    for config in CONFIGS:
        columns = selected_columns(config)
        assert len(columns) == len(set(columns))
        assert not {"patient_id", "target", "recurrence", "survival_status", "days_to_last_information"} & set(columns)
        categorical = [columns[i] for i in native_categorical_indices(columns)]
        assert {"sex", "primary_tumor_site", "hpv_association_p16"} <= set(categorical)
        assert "age_at_initial_diagnosis" not in categorical
    assert len(selected_columns(CONFIGS[4])) < len(selected_columns(CONFIGS[5])) < len(selected_columns(CONFIGS[1]))


def test_training_folds_exclude_official_test_and_cover_training_once():
    features = load_features()
    for target in ("survival_status", "recurrence"):
        train, test = load_partition(features, target, "in")
        seen = []
        for a, b in StratifiedKFold(3, shuffle=True, random_state=20260921).split(train, train.target):
            fit_ids, validation_ids = set(train.iloc[a].patient_id), set(train.iloc[b].patient_id)
            assert not fit_ids & validation_ids
            assert not (fit_ids | validation_ids) & set(test.patient_id)
            seen.extend(validation_ids)
        assert len(seen) == len(set(seen)) == len(train)


def test_selection_ignores_incomplete_candidates_and_test_scores():
    rows = []
    for config, auc in (("paper_default", .75), ("native_default", .77)):
        rows += [dict(config=config, fold=i, status="complete", roc_auc=auc, brier=.2, test_auc=1-auc) for i in range(3)]
    rows += [dict(config="paper_thinking", fold=0, status="complete", roc_auc=.99, brier=.01)]
    assert rank_configurations(rows)[0]["config"] == "native_default"
    with pytest.raises(ValueError):
        rank_configurations([])


def test_predict_masks_before_onehot_and_batches_original_columns():
    model = OptimizationModel(CONFIGS[4])
    class FakeEstimator:
        def predict_proba(self, x):
            assert x.shape[1] == len(model.columns)
            p = np.clip(np.asarray(x)[:, 0], 0, 1)
            return np.column_stack((1-p, p))
    model.estimator = FakeEstimator()
    rows = np.ones((11, len(model.columns))) * .3
    np.testing.assert_allclose(model.predict(rows, batch_size=4), .3)
