"""Audit the completed paper-comparison artifacts without calling the API."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.common import RESULTS, file_hash, read_json, write_json
from src.pipeline import metric_scores


def main():
    output = RESULTS / "paper"
    status = read_json(output / "status.json")
    assert status["complete"] and status["completed_runs"] == 90
    protocol_id = status["protocol_id"]
    runs = [read_json(path) for path in (output / "runs").glob("*.json")]
    assert len(runs) == 90 and all(r["protocol_id"] == protocol_id for r in runs)
    summary = pd.read_csv(output / "summary.csv")
    comparisons = pd.read_csv(output / "paired_comparisons.csv")
    assert len(summary) == 18 and len(comparisons) == 12
    for record in runs:
        matrix_file = output / "inputs" / f"{record['target']}_{record['split']}_{record['iteration']}.npz"
        assert file_hash(matrix_file) == record["input_hash"]
        with np.load(matrix_file, allow_pickle=False) as matrices:
            assert not set(matrices["train_ids"]) & set(matrices["test_ids"])
            assert matrices["test_ids"].tolist() == record["patient_ids"]
            assert matrices["y_test"].tolist() == record["y_test"]
            assert len(matrices["y_train"]) == record["n_train"]
            context_y = matrices["y_train" if record["model"] == "tabpfn_native" else "y_smote"]
            assert len(context_y) == record["n_context"]
            assert np.isfinite(matrices["x_smote"]).all() and np.isfinite(matrices["x_test"]).all()
        for metric, value in metric_scores(record["y_test"], record["probabilities"]).items():
            assert np.isclose(value, record[metric], atol=1e-12)
    forest = summary.loc[summary.model.eq("random_forest")]
    assert len(forest) == 6
    assert all(round(r.roc_auc_mean, 2) == r.published_rf_auc for r in forest.itertuples())
    for comparison in comparisons.itertuples():
        group = summary.loc[summary.target.eq(comparison.target) & summary.split.eq(comparison.split)].set_index("model")
        expected = group.loc[comparison.model, "roc_auc_mean"] - group.loc["random_forest", "roc_auc_mean"]
        assert np.isclose(expected, comparison.auc_difference, atol=1e-12)
        assert comparison.ci_low <= comparison.ci_high
    for filename in ("paper_comparison.png", "roc_comparison.png", "REPORT.md"):
        assert (output / filename).stat().st_size > 1000
    checks = {"protocol_id": protocol_id, "completed_runs": 90, "metrics_recomputed": True,
              "patient_alignment_and_disjointness": True, "matrix_hashes_match": True,
              "published_auc_rounding_matches": 6, "paired_differences_verified": 12}
    write_json(output / "verification.json", checks)
    print(checks)


if __name__ == "__main__":
    main()
