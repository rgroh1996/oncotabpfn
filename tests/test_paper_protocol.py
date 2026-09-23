"""Guard endpoint quirks and parity with the frozen published implementation."""
import ast

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from src.paper_protocol import (VENDOR, SPLITS, SPLIT_LABELS, TARGETS, eligibility,
                                literal_lists, load_features, load_partition,
                                paired_auc_bootstrap, published_forest, setup_preprocessor)


def test_recurrence_released_code_boundaries():
    frame = pd.DataFrame({
        "recurrence": ["yes", "yes", "no", "no", "no", "no"],
        "days_to_recurrence": [1095, 1096, np.nan, np.nan, np.nan, np.nan],
        "days_to_last_information": [1095, 1096, 1095, 1096, 10, 10],
        "survival_status": ["deceased", "deceased", "deceased", "deceased", "living", "deceased"],
    })
    assert eligibility(frame, "recurrence").tolist() == [True, False, False, True, True, False]


def test_death_status_has_no_fixed_horizon_and_retains_unknown_cause():
    frame = pd.DataFrame({"survival_status_with_cause": ["deceased not tumor specific", "deceased tumor specific", "unknown", "living", np.nan]})
    assert eligibility(frame, "survival_status").tolist() == [False, True, True, True, True]


def published_function(relative, name, namespace):
    if not VENDOR.exists():
        pytest.skip("Frozen upstream checkout not available")
    tree = ast.parse((VENDOR / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(relative), "exec"), namespace)
    return namespace[name]


def test_frozen_rf_parameter_parity_all_six_configurations():
    original = published_function("multimodal_machine_learning/execution/outcome_prediction.py", "return_optimal_random_forest",
                                  {"np": np, "RandomForestClassifier": RandomForestClassifier})
    for target in TARGETS:
        for split in SPLITS:
            actual = published_forest(target, split, 42).get_params()
            expected = original(target=target, data_split=SPLIT_LABELS[split], random_state=42).get_params()
            assert actual == expected


def test_preprocessor_matches_original_and_split_patients_are_disjoint():
    if not VENDOR.exists():
        pytest.skip("Frozen upstream checkout not available")
    namespace = {"ColumnTransformer": ColumnTransformer, "Pipeline": Pipeline, "SimpleImputer": SimpleImputer,
                 "OneHotEncoder": OneHotEncoder, "StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler}
    namespace.update(literal_lists(VENDOR / "feature_extraction/extract_tabular_features.py"))
    namespace.update(literal_lists(VENDOR / "feature_extraction/extract_tma_features.py"))
    original = published_function("data_exploration/umap_embedding.py", "setup_preprocessing_pipeline", namespace)
    features = load_features()
    assert len(features) == 763
    for target in TARGETS:
        for split in SPLITS:
            train, test = load_partition(features, target, split)
            assert not set(train.patient_id) & set(test.patient_id)
            x_train, x_test = train.iloc[:, 2:], test.iloc[:, 2:]
            actual, expected = setup_preprocessor(x_train.columns), original(x_train.columns)
            np.testing.assert_array_equal(actual.fit_transform(x_train), expected.fit_transform(x_train))
            np.testing.assert_array_equal(actual.transform(x_test), expected.transform(x_test))


def test_bootstrap_pairs_patients_and_preserves_ties():
    y = [0, 0, 1, 1]
    tied = np.full((5, 4), .5)
    perfect = np.array([[.1, .2, .8, .9]] * 5)
    result = paired_auc_bootstrap(y, perfect, tied, samples=100)
    assert result["auc_difference"] == .5
    assert result["ci_low"] == result["ci_high"] == .5
    identical = paired_auc_bootstrap(y, perfect, perfect, samples=100)
    assert identical["auc_difference"] == identical["ci_low"] == identical["ci_high"] == 0
    with pytest.raises(ValueError):
        paired_auc_bootstrap(y, perfect[:, :3], tied)
