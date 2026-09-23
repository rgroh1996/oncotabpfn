"""Reproduce HANCOCK Figure 2 RF and evaluate prespecified TabPFN replacements."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results/paper/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve

from src.common import RESULTS, ROOT, file_hash, fingerprint, read_json, safe_error, write_json
from src.paper_protocol import (PUBLISHED_AUC, SPLITS, SPLIT_LABELS, TARGETS, load_features,
                                load_partition, paired_auc_bootstrap, provenance,
                                published_forest, setup_preprocessor)
from src.pipeline import make_estimator, metric_scores, model_metadata

OUT = RESULTS / "paper"
DISPLAY = {"random_forest": "Published RF + SMOTE", "tabpfn_smote": "TabPFN-3.5 + SMOTE (primary)",
           "tabpfn_native": "TabPFN-3.5, no SMOTE (secondary)"}
COLORS = {"random_forest": "#66758c", "tabpfn_smote": "#008c82", "tabpfn_native": "#c78031"}
ENDPOINTS = {"survival_status": "Death status (paper exclusions)", "recurrence": "Recurrence (released-code eligibility)"}


def make_protocol():
    return {
        "study": "HANCOCK Figure 2 released-code replication; no test-based tuning",
        "source": provenance(), "targets": list(TARGETS), "splits": list(SPLITS), "repetitions": 5,
        "primary": "Untuned TabPFN-3.5 replaces frozen Random Forest on identical SMOTE matrices",
        "secondary": "Untuned TabPFN-3.5 on the same original training patients without SMOTE",
        "tabpfn_seeds": list(range(42, 47)), "tabpfn_model": "v3.5_default",
        "rf_rng": "Separate SMOTE and RF RandomState(42) streams advance across repetitions and splits; reset per endpoint",
        "primary_metric": "Mean ROC AUC across five repetitions",
        "uncertainty": "2000 stratified paired patient bootstraps of mean repetition AUC differences; unadjusted exploratory 95% intervals",
        "implementation": {name: file_hash(ROOT / name) for name in ("src/paper_protocol.py", "src/paper_benchmark.py")},
    }


def run_record(target, split, iteration, model, probabilities, y_test, ids, protocol_id, **extra):
    return dict(protocol_id=protocol_id, target=target, split=split, iteration=iteration, model=model,
                seed=42 + iteration if model != "random_forest" else None,
                n_test=len(y_test), test_events=int(y_test.sum()),
                patient_ids=list(ids), y_test=y_test.tolist(), probabilities=np.asarray(probabilities).tolist(),
                **metric_scores(y_test, probabilities), **extra)


def prepare_and_run_rf(protocol_id):
    features = load_features()
    cohort_audit = []
    for target in TARGETS:
        smote_rng, rf_rng = np.random.RandomState(42), np.random.RandomState(42)
        for split in SPLITS:
            train, test = load_partition(features, target, split)
            for iteration in range(5):
                preprocessor = setup_preprocessor(train.columns[2:])
                x_train = preprocessor.fit_transform(train.drop(columns=["patient_id", "target"]))
                x_test = preprocessor.transform(test.drop(columns=["patient_id", "target"]))
                y_train, y_test = train.target.to_numpy(), test.target.to_numpy()
                x_smote, y_smote = SMOTE(random_state=smote_rng).fit_resample(x_train, y_train)
                stem = f"{target}_{split}_{iteration}"
                matrix_path = OUT / "inputs" / f"{stem}.npz"
                np.savez_compressed(matrix_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                    x_smote=x_smote, y_smote=y_smote,
                                    train_ids=train.patient_id.to_numpy(dtype=str), test_ids=test.patient_id.to_numpy(dtype=str))
                model = published_forest(target, split, rf_rng)
                started = time.perf_counter()
                model.fit(x_smote, y_smote)
                probability = model.predict_proba(x_test)[:, 1]
                metadata = model.get_params(deep=False)
                metadata["random_state"] = "Shared RandomState(42) stream; see protocol"
                record = run_record(target, split, iteration, "random_forest", probability, y_test,
                                    test.patient_id, protocol_id, n_train=len(train), n_context=len(y_smote),
                                    n_features=x_train.shape[1], seconds=time.perf_counter()-started,
                                    input_hash=file_hash(matrix_path), parameters=metadata)
                write_json(OUT / "runs" / f"{stem}_random_forest.json", record)
                print(f"RF {target}/{split} repeat {iteration+1}/5: AUC={record['roc_auc']:.4f}", flush=True)
                if iteration == 0:
                    cohort_audit.append(dict(target=target, split=split, n_train=len(train), n_test=len(test),
                                             train_events=int(y_train.sum()), test_events=int(y_test.sum()),
                                             n_context_smote=len(y_smote), n_features=x_train.shape[1],
                                             train_patient_ids=train.patient_id.tolist(), test_patient_ids=test.patient_id.tolist(),
                                             feature_names=preprocessor.get_feature_names_out().tolist()))
    write_json(OUT / "cohort_audit.json", cohort_audit)


def run_tabpfn(protocol_id):
    for target in TARGETS:
        for split in SPLITS:
            for iteration in range(5):
                stem = f"{target}_{split}_{iteration}"
                matrix_path = OUT / "inputs" / f"{stem}.npz"
                reference = read_json(OUT / "runs" / f"{stem}_random_forest.json")
                input_hash = file_hash(matrix_path)
                if reference["protocol_id"] != protocol_id or reference["input_hash"] != input_hash:
                    raise ValueError("Stale RF/matrix artifact; rerun RF phase")
                with np.load(matrix_path, allow_pickle=False) as matrices:
                    for model_name in ("tabpfn_smote", "tabpfn_native"):
                        destination = OUT / "runs" / f"{stem}_{model_name}.json"
                        if destination.exists():
                            cached = read_json(destination)
                            if cached.get("protocol_id") == protocol_id and cached.get("input_hash") == input_hash:
                                print(f"Cached {stem}/{model_name}", flush=True)
                                continue
                        augmented = model_name == "tabpfn_smote"
                        x_train = matrices["x_smote" if augmented else "x_train"]
                        y_train = matrices["y_smote" if augmented else "y_train"]
                        started = time.perf_counter()
                        try:
                            model = make_estimator("tabpfn", 42 + iteration)
                            model.fit(x_train, y_train)
                            probabilities = model.predict_proba(matrices["x_test"])[:, 1]
                            metadata = model_metadata(model)
                            # SDK exposes requested config; some defaults are resolved only by its server.
                            if "resolved_config" in metadata:
                                metadata["requested_config"] = metadata.pop("resolved_config")
                            record = run_record(target, split, iteration, model_name, probabilities,
                                                matrices["y_test"], matrices["test_ids"], protocol_id,
                                                n_train=len(matrices["y_train"]), n_context=len(y_train),
                                                n_features=x_train.shape[1], seconds=time.perf_counter()-started,
                                                input_hash=input_hash, model_metadata=metadata)
                            write_json(destination, record)
                        except Exception as exc:
                            message = safe_error(exc)
                            write_json(OUT / "last_error.json", {"run": stem, "model": model_name, "error": message})
                            raise RuntimeError(message) from None
                        print(f"{model_name} {target}/{split} repeat {iteration+1}/5: AUC={record['roc_auc']:.4f} ({record['seconds']:.1f}s)", flush=True)


def export_results(protocol_id):
    records = [read_json(path) for path in sorted((OUT / "runs").glob("*.json"))]
    records = [r for r in records if r["protocol_id"] == protocol_id]
    if not records:
        raise ValueError("No matching results")
    columns = ["target", "split", "model", "iteration", "n_train", "n_context", "n_test", "test_events", "n_features",
               "roc_auc", "average_precision", "brier", "ece", "seconds"]
    metrics = pd.DataFrame([{key: r[key] for key in columns} for r in records])
    metrics.to_csv(OUT / "metrics.csv", index=False)
    prediction_rows = []
    for r in records:
        for patient, y, p in zip(r["patient_ids"], r["y_test"], r["probabilities"]):
            prediction_rows.append(dict(target=r["target"], split=r["split"], model=r["model"], iteration=r["iteration"], patient_id=patient, y=y, probability=p))
    pd.DataFrame(prediction_rows).to_csv(OUT / "predictions.csv", index=False)
    summary, comparisons = [], []
    for target in TARGETS:
        for split_index, split in enumerate(SPLITS):
            selected = [r for r in records if r["target"] == target and r["split"] == split]
            reference = sorted([r for r in selected if r["model"] == "random_forest"], key=lambda r: r["iteration"])
            for model in DISPLAY:
                group = sorted([r for r in selected if r["model"] == model], key=lambda r: r["iteration"])
                if not group:
                    continue
                row = dict(target=target, split=split, model=model, repeats=len(group), n_train=group[0]["n_train"],
                           n_test=group[0]["n_test"], published_rf_auc=PUBLISHED_AUC[target][split_index])
                for metric in ("roc_auc", "average_precision", "brier", "ece"):
                    row[f"{metric}_mean"] = float(np.mean([r[metric] for r in group]))
                    row[f"{metric}_std"] = float(np.std([r[metric] for r in group], ddof=0))
                summary.append(row)
                if model != "random_forest" and len(group) == len(reference) == 5:
                    for a, b in zip(group, reference):
                        if a["iteration"] != b["iteration"] or a["patient_ids"] != b["patient_ids"] or a["y_test"] != b["y_test"]:
                            raise ValueError("Unpaired patient predictions")
                    comparisons.append(dict(target=target, split=split, model=model,
                                            **paired_auc_bootstrap(group[0]["y_test"], [r["probabilities"] for r in group], [r["probabilities"] for r in reference])))
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)
    complete = len(records) == 90 and all(row["repeats"] == 5 for row in summary)
    write_json(OUT / "status.json", {"protocol_id": protocol_id, "complete": complete, "completed_runs": len(records), "expected_runs": 90})
    report = ["# HANCOCK paper comparison", "", f"Completed runs: {len(records)}/90. Protocol: `{protocol_id}`.", "",
              "This reproduces the released Figure 2 Random Forest experiment, then replaces RF with untuned TabPFN-3.5. "
              "The primary comparison retains identical SMOTE-augmented matrices. A prespecified secondary comparison omits SMOTE for TabPFN.", "",
              "| Endpoint | Split | Model | ROC AUC, mean ± SD |", "|---|---|---|---:|"]
    for row in summary:
        report.append(f"| {ENDPOINTS[row['target']]} | {SPLIT_LABELS[row['split']]} | {DISPLAY[row['model']]} | {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f} |")
    report += ["", "## Paired differences versus reproduced RF", "", "95% intervals resample test patients, retaining all five repetitions together. "
               "They describe test-sample uncertainty conditional on these training sets; they are exploratory, unadjusted for multiple comparisons. "
               "Repetitions and the three overlapping test partitions are not independent cohorts.", "",
               "| Endpoint | Split | TabPFN variant | ΔAUC | 95% paired interval |", "|---|---|---|---:|---:|"]
    for row in comparisons:
        report.append(f"| {ENDPOINTS[row['target']]} | {SPLIT_LABELS[row['split']]} | {DISPLAY[row['model']]} | {row['auc_difference']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |")
    report += ["", "## Reproduction checks and limits", "",
               "Published RF AUCs (rounded): death status 0.79 / 0.78 / 0.71; recurrence 0.79 / 0.71 / 0.69 (in / out / Oropharynx). "
               "Our reproduced RF is the paired comparator; published rounded numbers are reference values only.", "",
               "- Official, real 763-patient cohort; endpoint eligibility reduces the train/test counts. No synthetic fallback.",
               "- Frozen clinical, pathology, hematology, ICD and CD3/CD8 TMA feature tables; train-fitted one-hot encoding, imputation and scaling. No UMAP model input.",
               "- Reused feature tables inherit the authors' upstream cohort-wide encoding/blood imputation; this is a replication, not a fully re-engineered leakage-free preprocessing study.",
               "- Death status excludes known non-tumor-specific deaths, retains unknown causes, and has no five-year horizon.",
               "- Recurrence uses released-code eligibility: positive within 1095 days, or nonrecurrent with >1095 days follow-up OR living status. The living exception differs from the prose. This is not a strict censoring-aware three-year endpoint.",
               "- Frozen RF hyperparameters are used without rerunning search. Original core dependency versions and both advancing RNG streams are preserved. Ancillary dependencies/platform can still affect exact numerical reproduction.",
               "- TabPFN uses five seeds with v3.5_default through tabpfn-client 0.6.0, without tuning, feature selection or probability recalibration. Server-resolved defaults/weights are not fully pinned by the client.",
               "- Brier/ECE are descriptive raw-probability metrics; SMOTE changes class prevalence. No claim of calibrated clinical probabilities follows from this comparison.",
               "- This compares Figure 2 RF, not the paper's separate foundation-model/RFS experiments. No external validation or treatment-effect inference.", "",
               "Sources: [paper](https://doi.org/10.1038/s41467-025-62386-6), "
               "[frozen authors' code](https://github.com/ankilab/HANCOCK_MultimodalDataset/tree/521b99b03a94008b28df5c3df4aa5f82aa14b25a). "
               "Full provenance, versions, feature names, patient memberships, per-run predictions and configuration are saved alongside this report."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    plot(records, summary)
    print(f"Exported {len(records)}/90 runs to {OUT}", flush=True)


def plot(records, summary):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for i, target in enumerate(TARGETS):
        for j, split in enumerate(SPLITS):
            ax = axes[i, j]
            for model in DISPLAY:
                runs = [r for r in records if (r["target"], r["split"], r["model"]) == (target, split, model)]
                if not runs:
                    continue
                grid = np.linspace(0, 1, 101)
                curves = []
                for r in runs:
                    fpr, tpr, _ = roc_curve(r["y_test"], r["probabilities"])
                    curve = np.interp(grid, fpr, tpr)
                    curve[0], curve[-1] = 0, 1
                    curves.append(curve)
                auc = np.mean([r["roc_auc"] for r in runs])
                ax.plot(grid, np.mean(curves, axis=0), color=COLORS[model], label=f"{DISPLAY[model]}\nAUC {auc:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="#cccccc")
            ax.set(title=f"{ENDPOINTS[target]}\n{SPLIT_LABELS[split]}", xlabel="False-positive rate", ylabel="True-positive rate")
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("HANCOCK Figure 2 protocol · five repetitions · held-out test patients")
    fig.savefig(OUT / "roc_comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        for k, model in enumerate(DISPLAY):
            rows = [next((r for r in summary if (r["target"], r["split"], r["model"]) == (target, split, model)), None) for split in SPLITS]
            if any(r is None for r in rows):
                continue
            ax.errorbar(np.arange(3) + (k-1)*.16, [r["roc_auc_mean"] for r in rows],
                        yerr=[r["roc_auc_std"] for r in rows], marker="o", linestyle="none", capsize=4,
                        color=COLORS[model], label=DISPLAY[model])
        ax.set(title=ENDPOINTS[target], ylabel="ROC AUC (mean ± repetition SD)", ylim=(.45, .95), xticks=range(3), xticklabels=list(SPLIT_LABELS.values()))
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "paper_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "rf", "tabpfn", "report"], default="all")
    args = parser.parse_args()
    for name in ("runs", "inputs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    protocol = make_protocol()
    protocol_id = fingerprint(protocol)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path)["protocol_id"] != protocol_id:
        raise ValueError("Protocol/provenance changed. Archive results/paper before starting a new study.")
    if not protocol_path.exists():
        write_json(protocol_path, {"protocol_id": protocol_id, "created_utc": datetime.now(timezone.utc).isoformat(), **protocol})
    if args.phase in ("all", "rf"):
        prepare_and_run_rf(protocol_id)
    if args.phase in ("all", "tabpfn"):
        run_tabpfn(protocol_id)
    export_results(protocol_id)


if __name__ == "__main__":
    main()
