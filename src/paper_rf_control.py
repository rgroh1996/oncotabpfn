"""Supplementary control: the published RF on the frozen paper matrices without SMOTE.

The frozen 90-run study compares no-SMOTE TabPFN against RF + SMOTE, which mixes a model change
with a pipeline change. This control isolates the model: it refits the published per-split RF
hyperparameters on the identical original training matrices, and pairs it with the saved TabPFN
predictions. It reads results/paper without modifying it and writes results/paper_rf_control.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.common import RESULTS, ROOT, file_hash, fingerprint, read_json, write_json
from src.paper_protocol import (PUBLISHED_AUC, SPLITS, SPLIT_LABELS, TARGETS, paired_auc_bootstrap,
                                provenance, published_forest)
from src.pipeline import metric_scores

PAPER = RESULTS / "paper"
OUT = RESULTS / "paper_rf_control"
ENDPOINTS = {"survival_status": "Death status", "recurrence": "Recurrence"}
MODELS = {"random_forest": "Published RF + SMOTE", "random_forest_native": "Published RF, no SMOTE",
          "tabpfn_native": "TabPFN-3.5, no SMOTE"}
# (candidate, reference) pairs; the first isolates the model, the second the SMOTE step.
COMPARISONS = (("tabpfn_native", "random_forest_native"), ("random_forest_native", "random_forest"))


def make_protocol(paper_protocol_id):
    return {
        "study": "Supplementary no-SMOTE RF control for the frozen HANCOCK paper comparison; added after results were seen",
        "paper_protocol_id": paper_protocol_id, "source": provenance(), "targets": list(TARGETS),
        "splits": list(SPLITS), "repetitions": 5,
        "model": "Published per-split RF hyperparameters fitted on the frozen original (non-SMOTE) training matrices",
        "rf_rng": "RandomState(42) per endpoint, advancing across splits and repetitions, as in the frozen RF + SMOTE runs",
        "limitation": "Hyperparameters were searched by the authors for the SMOTE pipeline; they are not retuned here",
        "uncertainty": "2000 stratified paired patient bootstraps, all five repetitions together; unadjusted exploratory 95% intervals",
        "implementation": {name: file_hash(ROOT / name) for name in ("src/paper_rf_control.py", "src/paper_protocol.py")},
    }


def load_run(target, split, iteration, model, paper_protocol_id):
    record = read_json(PAPER / "runs" / f"{target}_{split}_{iteration}_{model}.json")
    if record["protocol_id"] != paper_protocol_id:
        raise ValueError(f"Stale paper run: {target}/{split}/{iteration}/{model}")
    return record


def run_forests(paper_protocol_id, protocol_id):
    records = []
    for target in TARGETS:
        rf_rng = np.random.RandomState(42)
        for split in SPLITS:
            for iteration in range(5):
                stem = f"{target}_{split}_{iteration}"
                matrix_path = PAPER / "inputs" / f"{stem}.npz"
                reference = load_run(target, split, iteration, "random_forest", paper_protocol_id)
                if reference["input_hash"] != file_hash(matrix_path):
                    raise ValueError(f"Frozen matrix changed: {stem}")
                with np.load(matrix_path, allow_pickle=False) as matrices:
                    model = published_forest(target, split, rf_rng)
                    started = time.perf_counter()
                    model.fit(matrices["x_train"], matrices["y_train"])
                    probability = model.predict_proba(matrices["x_test"])[:, 1]
                    if matrices["test_ids"].tolist() != reference["patient_ids"] or matrices["y_test"].tolist() != reference["y_test"]:
                        raise ValueError(f"Unpaired test patients: {stem}")
                    record = dict(protocol_id=protocol_id, target=target, split=split, iteration=iteration,
                                  model="random_forest_native", n_train=len(matrices["y_train"]),
                                  n_test=len(reference["y_test"]), patient_ids=reference["patient_ids"],
                                  y_test=reference["y_test"], probabilities=probability.tolist(),
                                  input_hash=reference["input_hash"], seconds=time.perf_counter() - started,
                                  **metric_scores(reference["y_test"], probability))
                write_json(OUT / "runs" / f"{stem}_random_forest_native.json", record)
                records.append(record)
                print(f"RF no SMOTE {target}/{split} repeat {iteration+1}/5: AUC={record['roc_auc']:.4f}", flush=True)
    return records


def export(paper_protocol_id, protocol_id, forests):
    summary, comparisons, metrics = [], [], []
    for target in TARGETS:
        for split_index, split in enumerate(SPLITS):
            groups = {"random_forest_native": sorted([r for r in forests if (r["target"], r["split"]) == (target, split)],
                                                     key=lambda r: r["iteration"])}
            for model in ("random_forest", "tabpfn_native"):
                groups[model] = [load_run(target, split, i, model, paper_protocol_id) for i in range(5)]
            for model in MODELS:
                group = groups[model]
                row = dict(target=target, split=split, model=model, repeats=len(group),
                           published_rf_auc=PUBLISHED_AUC[target][split_index])
                for metric in ("roc_auc", "average_precision", "brier", "ece"):
                    row[f"{metric}_mean"] = float(np.mean([r[metric] for r in group]))
                    row[f"{metric}_std"] = float(np.std([r[metric] for r in group], ddof=0))
                summary.append(row)
                metrics += [dict(target=target, split=split, model=model, iteration=r["iteration"],
                                 **{m: r[m] for m in ("roc_auc", "average_precision", "brier", "ece")}) for r in group]
            for candidate, reference in COMPARISONS:
                for a, b in zip(groups[candidate], groups[reference]):
                    if a["iteration"] != b["iteration"] or a["patient_ids"] != b["patient_ids"] or a["y_test"] != b["y_test"]:
                        raise ValueError("Unpaired patient predictions")
                comparisons.append(dict(target=target, split=split, candidate=candidate, reference=reference,
                                        **paired_auc_bootstrap(groups[candidate][0]["y_test"],
                                                               [r["probabilities"] for r in groups[candidate]],
                                                               [r["probabilities"] for r in groups[reference]])))
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)
    write_report(protocol_id, summary, comparisons)


def write_report(protocol_id, summary, comparisons):
    auc = {(r["target"], r["split"], r["model"]): r["roc_auc_mean"] for r in summary}
    lines = ["# No-SMOTE Random Forest control", "", f"Protocol: `{protocol_id}`. Supplementary to the frozen "
             "[paper comparison](../paper/REPORT.md), whose files are read but not modified.", "",
             "The frozen study's headline compares TabPFN without SMOTE against RF with SMOTE. That changes the model and the "
             "pipeline at once. This control fits the published RF hyperparameters on the same original training matrices "
             "without SMOTE, so the TabPFN-versus-RF difference below reflects the model alone.", "",
             "| Endpoint | Split | " + " | ".join(MODELS.values()) + " |", "|---|---|---:|---:|---:|"]
    for target in TARGETS:
        for split in SPLITS:
            lines.append(f"| {ENDPOINTS[target]} | {SPLIT_LABELS[split]} | "
                         + " | ".join(f"{auc[(target, split, m)]:.3f}" for m in MODELS) + " |")
    lines += ["", "Mean ROC AUC over five repetitions.", "", "## Paired differences", "",
              "| Endpoint | Split | Comparison | ΔAUC | 95% paired interval |", "|---|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {ENDPOINTS[row['target']]} | {SPLIT_LABELS[row['split']]} | {MODELS[row['candidate']]} vs "
                     f"{MODELS[row['reference']]} | {row['auc_difference']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |")
    lines += ["", "## Limits", "",
              "- Added after the frozen results were seen; it is a post-hoc control, not a prespecified comparison.",
              "- The published RF hyperparameters were searched by the authors for the SMOTE pipeline and are not retuned here. "
              "The [learning-curve study](../learning_curves/REPORT.md) contains an independently tuned no-SMOTE RF.",
              "- Intervals resample test patients conditional on the fixed training sets; unadjusted for multiple comparisons. "
              "The three test partitions overlap and were examined in earlier experiments."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / ".matplotlib"))
    paper_protocol_id = read_json(PAPER / "protocol.json")["protocol_id"]
    if not read_json(PAPER / "status.json")["complete"]:
        raise ValueError("The frozen paper comparison must be complete first")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    protocol = make_protocol(paper_protocol_id)
    protocol_id = fingerprint(protocol)
    path = OUT / "protocol.json"
    if path.exists() and read_json(path)["protocol_id"] != protocol_id:
        raise ValueError("Control protocol changed. Archive results/paper_rf_control first.")
    if not path.exists():
        write_json(path, {"protocol_id": protocol_id, "created_utc": datetime.now(timezone.utc).isoformat(), **protocol})
    export(paper_protocol_id, protocol_id, run_forests(paper_protocol_id, protocol_id))


if __name__ == "__main__":
    main()
