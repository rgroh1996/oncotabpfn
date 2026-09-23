"""Reproducible small-sample comparisons with resumable per-run artifacts."""
from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from data.dataset_builder import ensure_dataset
from src.common import DATA, ROOT, RESULTS, TARGETS, TARGET_LABELS, file_hash, fingerprint, read_json, safe_error, write_json
from src.pipeline import MODELS, PatientModel, calibration_bins, evaluation_splits, metric_scores, model_metadata, package_versions, stratified_order

METRICS = ("roc_auc", "average_precision", "brier", "ece")
METRIC_LABELS = ("ROC AUC ↑", "Average precision ↑", "Brier score ↓", "Calibration error ↓")
COLORS = {"tabpfn": "#008C82", "lightgbm": "#5173B8", "xgboost": "#D78A41"}
DISPLAY = {"tabpfn": "TabPFN-3.5", "lightgbm": "LightGBM", "xgboost": "XGBoost"}


def export_results(records: list[dict], output: Path, manifest: dict) -> pd.DataFrame:
    metrics = pd.DataFrame([{k: v for k, v in r.items() if k not in {"predictions", "model_metadata"}} for r in records])
    metrics.to_csv(output / "metrics.csv", index=False)
    prediction_rows = []
    for record in records:
        for prediction in record.get("predictions", []):
            prediction_rows.append({**{k: record[k] for k in ("target", "model", "size", "seed", "fold", "run_id", "source")}, **prediction})
    predictions = pd.DataFrame(prediction_rows)
    if len(predictions):
        predictions.to_parquet(output / "predictions.parquet", index=False)
    valid = metrics.loc[metrics.status.eq("complete")].copy()
    summary = (valid.groupby(["target", "model", "size"], sort=False)[list(METRICS)]
               .agg(["mean", "std", "count"])) if len(valid) else pd.DataFrame()
    if len(summary):
        summary.columns = ["_".join(c) for c in summary.columns]
        summary = summary.reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    paired = []
    if len(valid):
        for first, second in combinations(valid.model.unique(), 2):
            keys = ["target", "size", "seed", "fold"]
            pairs = valid.loc[valid.model.eq(first)].merge(valid.loc[valid.model.eq(second)], on=keys, suffixes=("_a", "_b"))
            for (target, size), group in pairs.groupby(["target", "size"]):
                for metric in METRICS:
                    difference = group[f"{metric}_a"] - group[f"{metric}_b"]
                    paired.append(dict(target=target, size=size, model_a=first, model_b=second, metric=metric,
                                       mean_difference=float(difference.mean()), repetitions=len(difference),
                                       std_difference=float(difference.std()) if len(difference) > 1 else 0.))
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    manifest["status_counts"] = metrics.status.value_counts().to_dict()
    manifest["comparison_complete"] = bool(len(metrics) and metrics.status.isin(["complete", "skipped"]).all())
    manifest["completed_models"] = sorted(valid.model.unique().tolist())
    write_json(output / "manifest.json", manifest)
    plot_results(valid, predictions, output, manifest)
    findings = ["# Observed benchmark findings", "", f"Source: **{manifest['source']}**. Split: {manifest['split']}.",
                "", f"Run statuses: {manifest['status_counts']}.", "",
                "Metrics use native model probabilities. Error bars describe variation across repetitions, not independent-patient confidence intervals.", ""]
    if "tabpfn" not in manifest["completed_models"]:
        findings += ["**No completed TabPFN-3.5 results. Its performance relative to the baselines has not been established.**", ""]
    if manifest["source"] == "synthetic":
        findings += ["**Synthetic demonstration only; these results provide no HANCOCK clinical evidence.**", ""]
    for _, row in summary.iterrows():
        if str(row["size"]) in {"50", "Full"}:
            findings.append(f"- {TARGET_LABELS[row.target]} · {DISPLAY[row.model]} · N={row['size']}: "
                            f"AUC {row.roc_auc_mean:.3f}, AP {row.average_precision_mean:.3f}, "
                            f"Brier {row.brier_mean:.3f}, ECE {row.ece_mean:.3f}.")
    findings += ["", "Endpoint eligibility excludes early-censored survival observations. Recurrence is documented occurrence over variable follow-up. "
                 "Repeated evaluations reuse patients; these are exploratory results from one center."]
    (output / "findings.md").write_text("\n".join(findings) + "\n")
    return metrics


def plot_results(valid, predictions, output, manifest):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    sizes = [str(s) for s in manifest["sizes"]]
    for row, target in enumerate(TARGETS):
        for col, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
            ax = axes[row, col]
            if len(valid):
                for model, group in valid.loc[valid.target.eq(target)].groupby("model"):
                    stats = group.groupby("size")[metric].agg(["mean", "std"]).reindex(sizes)
                    ax.errorbar(range(len(sizes)), stats["mean"], yerr=stats["std"].fillna(0),
                                marker="o", capsize=3, color=COLORS[model], label=DISPLAY[model])
            ax.set_xticks(range(len(sizes)), sizes)
            ax.set_xlabel("Eligible training patients")
            ax.set_title(f"{TARGET_LABELS[target]}\n{label}", loc="left")
            ax.grid(alpha=.15)
            if col < 2:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, fontsize=8)
    missing = " · TabPFN unavailable" if "tabpfn" not in manifest.get("completed_models", []) else ""
    fig.suptitle(f"OncoTabPFN | {manifest['source'].upper()} data | {manifest['split']} split{missing}\n"
                 "Untuned models · error bars: repetition SD", fontsize=15, fontweight="bold")
    fig.savefig(output / "benchmark_performance.png", dpi=180)
    plt.close(fig)
    if not len(predictions):
        return
    available_sizes = set(predictions["size"])
    chosen = "Full" if "Full" in available_sizes else next(str(s) for s in reversed(manifest["sizes"]) if str(s) in available_sizes)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for row, target in enumerate(TARGETS):
        subset = predictions.loc[predictions.target.eq(target) & predictions["size"].eq(chosen)]
        for model, group in subset.groupby("model"):
            # Average repeated held-out predictions per patient, never duplicate outcomes.
            patients = group.groupby("patient_id").agg(y=("y", "first"), p=("probability", "mean"))
            if patients.y.nunique() == 2:
                fpr, tpr, _ = roc_curve(patients.y, patients.p)
                axes[row, 0].plot(fpr, tpr, label=DISPLAY[model], color=COLORS[model])
            bins = calibration_bins(patients.y, patients.p)
            axes[row, 1].plot([b["probability"] for b in bins], [b["observed"] for b in bins],
                              marker="o", color=COLORS[model], label=DISPLAY[model])
        for col in range(2):
            axes[row, col].plot([0, 1], [0, 1], "--", color="#aaa")
            axes[row, col].set(xlim=(0, 1), ylim=(0, 1), title=TARGET_LABELS[target])
            if axes[row, col].get_legend_handles_labels()[0]:
                axes[row, col].legend()
        axes[row, 0].set(xlabel="False-positive rate", ylabel="True-positive rate")
        axes[row, 1].set(xlabel="Mean predicted probability", ylabel="Observed event fraction")
    fig.suptitle(f"Held-out discrimination and calibration · N={chosen} · {manifest['source'].upper()}\n"
                 "Repeated held-out probabilities averaged within patient", fontsize=14)
    fig.savefig(output / "roc_calibration.png", dpi=180)
    plt.close(fig)


def run_benchmark(df, output=RESULTS, models=MODELS, sizes=("50", "100", "200", "500", "Full"),
                  seeds=(42, 43, 44, 45, 46), split="official", pca_components=16, targets=TARGETS,
                  resume=True) -> pd.DataFrame:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    versions = package_versions()
    code_hash = fingerprint({name: file_hash(ROOT / name) for name in
                             ["src/pipeline.py", "src/benchmark.py", "data/dataset_builder.py"]})
    data_hash = fingerprint(pd.util.hash_pandas_object(df, index=True).tolist())
    manifest = dict(schema_version=1, source=str(df.data_source.iloc[0]), data_hash=data_hash, code_hash=code_hash,
                    versions=versions, split=split, sizes=list(sizes), seeds=list(seeds), models=list(models),
                    targets=list(targets), pca_components=pca_components)
    records, disabled_models = [], {}
    for target in targets:
        for seed in seeds:
            for fold, train, test in evaluation_splits(df, target, split, seed):
                order = stratified_order(train, target, seed)
                for size in sizes:
                    n = len(train) if str(size) == "Full" else int(size)
                    context = train.loc[order[:n]]
                    for name in models:
                        config = dict(target=target, model=name, size=str(size), seed=seed, fold=fold,
                                      source=manifest["source"], data_hash=data_hash, code_hash=code_hash,
                                      versions=versions, pca_components=pca_components, split=split)
                        run_id = fingerprint(config)
                        path = output / "runs" / f"{run_id}.json"
                        if resume and path.exists():
                            cached = read_json(path)
                            if cached["status"] == "complete":
                                records.append(cached)
                                continue
                        record = {k: config[k] for k in ("target", "model", "size", "seed", "fold", "source")}
                        record.update(run_id=run_id, n_train=len(context), n_test=len(test), status="pending", error=None,
                                      train_ids=context.patient_id.tolist(), test_ids=test.patient_id.tolist())
                        start = time.perf_counter()
                        if n > len(train) or n < 2 or context[target].nunique() < 2:
                            record.update(status="skipped", error=f"N={n} infeasible for {len(train)} eligible training patients and class balance")
                        elif name in disabled_models:
                            record.update(status="unavailable", error=disabled_models[name])
                        else:
                            try:
                                model = PatientModel(name, seed, pca_components).fit(context, context[target])
                                fit_end = time.perf_counter()
                                probability = model.predict_proba(test)
                                record.update(status="complete", fit_seconds=fit_end-start,
                                              predict_seconds=time.perf_counter()-fit_end,
                                              model_metadata=model_metadata(model.estimator), **metric_scores(test[target], probability),
                                              predictions=[dict(patient_id=pid, y=int(y), probability=float(p))
                                                           for pid, y, p in zip(test.patient_id, test[target], probability)])
                            except Exception as exc:
                                record.update(status="unavailable" if name == "tabpfn" else "failed", error=safe_error(exc))
                                if name == "tabpfn":
                                    disabled_models[name] = record["error"]
                        record["elapsed_seconds"] = time.perf_counter()-start
                        write_json(path, record)
                        records.append(record)
                        print(f"{target} | {name} | N={size} | seed={seed} fold={fold} | {record['status']}", flush=True)
    return export_results(records, output, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "hancock_processed.parquet")
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--sizes", nargs="+", default=["50", "100", "200", "500", "Full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--split", choices=["official", "out", "cv"], default="official")
    parser.add_argument("--pca-components", type=int, choices=[16, 32], default=16)
    parser.add_argument("--smoke", action="store_true", help="One seed, N=50; use a separate output directory")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    frame = ensure_dataset()[0] if args.data == DATA / "hancock_processed.parquet" else pd.read_parquet(args.data)
    metrics = run_benchmark(frame, args.output, args.models, ["50"] if args.smoke else args.sizes,
                            [42] if args.smoke else args.seeds, args.split, args.pca_components, resume=not args.no_resume)
    print(f"Artifacts saved to {args.output}; statuses: {metrics.status.value_counts().to_dict()}")
    if metrics.status.eq("failed").any():
        sys.exit(1)
