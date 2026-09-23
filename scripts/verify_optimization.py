"""Verify selection isolation, recorded metrics, and SHAP probability reconstruction."""
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from src.common import ROOT, RESULTS, file_hash, read_json, write_json
from src.optimization import rank_configurations
from src.pipeline import metric_scores


def main():
    root = RESULTS / "optimization"
    protocol = read_json(root / "protocol.json")
    status = read_json(root / "status.json")
    assert status["tuning_complete"] and status["completed_cv_runs"] == 36
    assert protocol["implementation_sha256"] == file_hash(ROOT / "src/optimization.py")
    records = [read_json(p) for p in (root / "runs").glob("*.json")]
    assert len(records) == 36 and all(r["status"] == "complete" for r in records)
    targets = ("survival_status", "recurrence")
    selections = {t: read_json(root / f"selection_{t}.json") for t in targets}
    latest_selection = max(datetime.fromisoformat(s["selected_utc"]) for s in selections.values())
    explanation_checks = []
    for target in targets:
        cohort = protocol["cohorts"][target]
        training, test = set(cohort["training_ids"]), set(cohort["reserved_test_ids"])
        assert not training & test
        target_records = [r for r in records if r["target"] == target]
        for record in target_records:
            assert record["protocol_id"] == protocol["protocol_id"]
            fit, validation = set(record["training_ids"]), set(record["validation_ids"])
            assert not fit & validation
            assert fit | validation == training
            assert not (fit | validation) & test
            assert record["training_ids"] == cohort["folds"][record["fold"]]["train_ids"]
            assert record["validation_ids"] == cohort["folds"][record["fold"]]["validation_ids"]
            for metric, expected in metric_scores(record["y"], record["probabilities"]).items():
                assert np.isclose(record[metric], expected, atol=1e-12)
        selection = selections[target]
        assert selection["selected"] == rank_configurations(target_records)[0]["config"]
        evaluation = read_json(root / f"test_{target}.json")
        assert evaluation["selected"] == selection["selected"]
        saved_model = joblib.load(root / "models" / target / "preprocessing.joblib")
        assert saved_model["config"]["name"] == selection["selected"] and saved_model["seed"] == evaluation["seed"]
        assert datetime.fromisoformat(evaluation["evaluated_utc"]) >= latest_selection
        assert set(evaluation["patient_ids"]) == test
        for metric, expected in metric_scores(evaluation["y"], evaluation["probabilities"]).items():
            assert np.isclose(evaluation[metric], expected, atol=1e-12)
        directory = root / "explanations" / target
        explanation_protocol = read_json(directory / "protocol.json")
        summary = read_json(directory / "summary.json")
        assert summary["complete"] and explanation_protocol["protocol_id"] == protocol["protocol_id"]
        assert explanation_protocol["server_model_sha256"] == file_hash(root / "models" / target / "server_model.json")
        assert explanation_protocol["preprocessing_sha256"] == file_hash(root / "models" / target / "preprocessing.joblib")
        assert explanation_protocol["implementation_sha256"] == file_hash(ROOT / "src/explain_tabpfn.py")
        assert set(explanation_protocol["background_ids"]) <= training
        assert set(explanation_protocol["explained_ids"]) <= test
        with np.load(directory / "shap_arrays.npz", allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays["values"].sum(axis=1)+arrays["baseline"], arrays["probability"], atol=1e-4)
            np.testing.assert_allclose(arrays["values"], (arrays["first"]+arrays["second"])/2, atol=1e-12)
            assert arrays["patient_ids"].tolist() == explanation_protocol["explained_ids"]
            assert arrays["features"].tolist() == explanation_protocol["original_features"]
            shap_features = pd.read_csv(directory / "feature_importance.csv").set_index("feature")
            np.testing.assert_allclose(shap_features.loc[arrays["features"], "mean_absolute_shap"], np.abs(arrays["values"]).mean(axis=0), atol=1e-12)
        for filename in ("shap_beeswarm.png", "modality_importance.png", *[f"waterfall_{i}.png" for i in summary["patient_ids"]]):
            assert (directory / filename).stat().st_size > 1000
        explanation_checks.append(dict(target=target, patients=summary["explained_patients"],
                                       max_additivity_error=summary["max_additivity_error"],
                                       ordering_rank_spearman=summary["ordering_rank_spearman"]))
    result = dict(protocol_id=protocol["protocol_id"], all_36_metrics_recomputed=True,
                  folds_exclude_official_test=True, both_selections_frozen_before_tests=True,
                  selected_models_match_validation_ranking=True, saved_models_match_selection=True, training_only_shap_background=True,
                  held_out_explained_patients=True, explanations=explanation_checks)
    write_json(root / "verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
