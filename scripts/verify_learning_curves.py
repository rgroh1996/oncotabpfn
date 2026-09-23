"""Audit the completed matched-label-budget study without making API requests."""
from datetime import datetime
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import expit

from src.common import ROOT, file_hash, read_json, write_json
from src.learning_curve_protocol import (GRIDS, MODELS, OUTPUT, SEEDS, SIZES, calibrate,
                                         nested_order, prepare, score_probabilities)
from src.learning_curve_budget import prepare_amendment
from src.paper_protocol import load_features, load_partition, paired_auc_bootstrap, setup_preprocessor


def main():
    protocol = prepare(OUTPUT)  # Validates frozen sources, packages, data and all input hashes.
    _, amendment = prepare_amendment(OUTPUT)
    calibration_seeds = amendment["calibration_seeds"]
    status = read_json(OUTPUT / "status.json")
    assert status["complete"] and status["completed_runs"] == 600
    metrics = pd.read_csv(OUTPUT / "metrics.csv", dtype={"size": str})
    predictions = pd.read_csv(OUTPUT / "predictions.csv", dtype={"patient_id": str, "size": str})
    assert len(metrics) == 900
    assert len(metrics.loc[metrics.calibration.eq("raw")]) == 600
    assert len(metrics.loc[metrics.calibration.eq("calibrated")]) == 300
    assert set(metrics.loc[metrics.calibration.eq("calibrated"), "seed"]) == set(calibration_seeds)
    calibration_summary = pd.read_csv(OUTPUT / "calibration_summary.csv")
    assert calibration_summary.roc_auc_count.eq(5).all()
    assert len(list((OUTPUT / "runs").glob("*.json"))) == 600
    features = load_features()
    partitions = {target: load_partition(features, target, "in") for target in protocol["targets"]}
    matrices_checked, folds_checked, cache_checked = 0, 0, 0
    indexed = {}
    for case in protocol["cases"]:
        target, seed, size, n = case["target"], case["seed"], case["size"], case["n"]
        train, test = partitions[target]
        assert case["test_ids"] == test.patient_id.tolist()
        chosen = train.iloc[nested_order(train.target.to_numpy(), seed)[:n]].reset_index(drop=True)
        assert case["training_ids"] == chosen.patient_id.tolist()
        assert not set(case["training_ids"]) & set(case["test_ids"])
        assert len(set(case["training_ids"])) == n
        columns = protocol["cohorts"][target]["features"]
        with np.load(OUTPUT / "inputs" / f"{case['name']}.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["y"], chosen.target)
            np.testing.assert_array_equal(arrays["test_y"], test.target)
            prep = setup_preprocessor(columns)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Skipping features without any observed values")
                np.testing.assert_allclose(arrays["train"], prep.fit_transform(chosen[columns]))
                np.testing.assert_allclose(arrays["test"], prep.transform(test[columns]))
                matrices_checked += 2
                validation_union = []
                for fold, split in enumerate(case["inner_folds"]):
                    fit, val = split["fit_indices"], split["val_indices"]
                    assert not set(fit) & set(val) and set(fit) | set(val) == set(range(n))
                    np.testing.assert_array_equal(arrays[f"fit_indices_{fold}"], fit)
                    np.testing.assert_array_equal(arrays[f"val_indices_{fold}"], val)
                    prep = setup_preprocessor(columns)
                    np.testing.assert_allclose(arrays[f"fit_{fold}"], prep.fit_transform(chosen.iloc[fit][columns]))
                    np.testing.assert_allclose(arrays[f"val_{fold}"], prep.transform(chosen.iloc[val][columns]))
                    matrices_checked += 2
                    folds_checked += 1
                    validation_union.extend(val)
                assert sorted(validation_union) == list(range(n))
        for model in MODELS:
            path = OUTPUT / "runs" / f"{case['name']}_{model}.json"
            record = read_json(path)
            indexed[target, size, seed, model] = record
            assert record["status"] == "complete" and record["protocol_id"] == protocol["protocol_id"]
            assert record["training_ids"] == case["training_ids"] and record["patient_ids"] == case["test_ids"]
            assert record["training_y"] == chosen.target.tolist() and record["y"] == test.target.tolist()
            assert datetime.fromisoformat(protocol["prepared_utc"]) < datetime.fromisoformat(record["started_utc"])
            assert record["started_utc"] <= record["selection_utc"] <= record["completed_utc"]
            candidate_scores = []
            available = record.get("calibration_available", True)
            if not available:
                assert model == "tabpfn" and seed not in calibration_seeds
                assert record["amendment_id"] == amendment["amendment_id"]
                assert record["candidates"] == [] and record["selected"] == 0
                assert record["started_utc"] >= amendment["amended_utc"]
            for candidate, params in enumerate(GRIDS[model] if available else []):
                assert record["candidates"][candidate]["parameters"] == params
                aucs, oof = [], np.full(n, np.nan)
                for fold, split in enumerate(case["inner_folds"]):
                    cached = read_json(OUTPUT / "inner_cache" / f"{case['name']}_{model}_c{candidate}_f{fold}.json")
                    val = split["val_indices"]
                    assert cached["validation_indices"] == val and cached["protocol_id"] == protocol["protocol_id"]
                    assert cached["completed_utc"] <= record["selection_utc"]
                    oof[val] = cached["probabilities"]
                    aucs.append(score_probabilities(chosen.target.iloc[val], cached["probabilities"])["roc_auc"])
                    cache_checked += 1
                np.testing.assert_allclose(aucs, record["candidates"][candidate]["fold_auc"])
                assert np.isclose(np.mean(aucs), record["candidates"][candidate]["mean_auc"])
                candidate_scores.append(np.mean(aucs))
                if candidate == record["selected"]:
                    np.testing.assert_allclose(oof, record["oof_probabilities"])
            if available:
                assert record["selected"] == int(np.argmax(candidate_scores))
                calibrated, params = calibrate(record["oof_probabilities"], record["training_y"], record["probabilities"])
                for key, value in params.items():
                    assert np.isclose(value, record["calibration"][key], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(calibrated, record["calibrated_probabilities"])
                p = np.clip(record["probabilities"], 1e-6, 1-1e-6)
                np.testing.assert_allclose(expit(params["slope"]*np.log(p/(1-p))+params["intercept"]), calibrated)
            table = metrics.loc[(metrics.target == target) & (metrics["size"] == size) & (metrics.seed == seed) & (metrics.model == model)]
            pred = predictions.loc[(predictions.target == target) & (predictions["size"] == size) & (predictions.seed == seed) & (predictions.model == model)]
            included_calibration = seed in calibration_seeds
            assert len(table) == (2 if included_calibration else 1) and pred.patient_id.tolist() == case["test_ids"]
            if not included_calibration:
                assert pred.calibrated.isna().all()
            for calibration, key in (("raw", "probabilities"), ("calibrated", "calibrated_probabilities")):
                if calibration == "calibrated" and not available:
                    continue
                included = calibration == "raw" or included_calibration
                if included:
                    np.testing.assert_allclose(pred[calibration], record[key], atol=1e-14)
                for metric, expected in score_probabilities(record["y"], record[key]).items():
                    assert np.isclose(expected, record[f"{calibration}_metrics"][metric])
                    if included:
                        assert np.isclose(table.loc[table.calibration.eq(calibration), metric].item(), expected)
            if model == "tabpfn":
                assert "3.5" in str(record["estimator"]["requested_config"])
    pairs = pd.read_csv(OUTPUT / "paired_comparisons.csv", dtype={"size": str})
    assert len(pairs) == 60
    for row in pairs.to_dict("records"):
        target, size, baseline = row["target"], row["size"], row["reference"]
        sizes = ("50", "100", "200") if size.startswith("Low-N") else (size,)
        a = np.array([indexed[target, s, seed, "tabpfn"]["probabilities"] for s in sizes for seed in SEEDS])
        b = np.array([indexed[target, s, seed, baseline]["probabilities"] for s in sizes for seed in SEEDS])
        comparison = paired_auc_bootstrap(protocol["cohorts"][target]["test_y"], a, b, seed=20260922)
        for key, value in comparison.items():
            assert np.isclose(row[key], value)
    for name in ("benchmark_performance.png", "calibration_comparison.png"):
        assert (OUTPUT / name).stat().st_size > 1000
    usage_start = read_json(OUTPUT / "quota_usage_start.json")
    usage_end = read_json(OUTPUT / "quota_usage_latest.json")
    additional_tokens = usage_end["monthly_tokens_used"] - usage_start["monthly_tokens_used"]
    assert 0 <= additional_tokens <= amendment["max_additional_tokens"]
    result = dict(complete=True, protocol_id=protocol["protocol_id"], amendment_id=amendment["amendment_id"], final_runs=600,
                  metric_rows=900, preprocessing_matrices_reconstructed=matrices_checked,
                  disjoint_inner_folds=folds_checked, candidate_fold_scores_recomputed=cache_checked,
                  paired_intervals_recomputed=60, nested_sizes=list(SIZES), matched_seeds=list(SEEDS),
                  calibration_reconstructed=True, no_extra_training_labels=True,
                  calibration_seeds=calibration_seeds, additional_tokens_after_amendment=additional_tokens,
                  verifier_sha256=file_hash(ROOT / "scripts/verify_learning_curves.py"))
    write_json(OUTPUT / "verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
