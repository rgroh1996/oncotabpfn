"""Run and report the prespecified paper-matched learning curves (resume-safe)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.common import read_json, safe_error, write_json
from src.learning_curve_protocol import (LABELS, MODELS, OUTPUT, SEEDS, SIZES,
                                         prepare, run_case, utcnow)
from src.paper_protocol import TARGETS, paired_auc_bootstrap

ENDPOINTS = {"survival_status": "Death status (paper endpoint)", "recurrence": "Recurrence (paper endpoint)"}


def report(output=OUTPUT):
    output = Path(output)
    with (output / ".report.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _report(output)


def _report(output=OUTPUT):
    output = Path(output)
    protocol = read_json(output / "protocol.json")
    amendment = read_json(output / "quota_amendment.json") if (output / "quota_amendment.json").exists() else None
    calibration_seeds = amendment["calibration_seeds"] if amendment else list(SEEDS)
    records = [read_json(p) for p in sorted((output / "runs").glob("*.json"))]
    if any(r["protocol_id"] != protocol["protocol_id"] or r["status"] != "complete" for r in records):
        raise ValueError("Invalid result provenance")
    expected = {(c["name"], m) for c in protocol["cases"] for m in MODELS}
    present = {(r["case"], r["model"]) for r in records}
    if len(present) != len(records) or not present <= expected:
        raise ValueError("Duplicate or unexpected runs")
    complete = present == expected
    status = dict(protocol_id=protocol["protocol_id"], complete=complete,
                  completed_runs=len(records), expected_runs=len(expected), updated_utc=utcnow())
    if not complete:
        write_json(output / "status.json", status)
    if not records:
        return
    rows, predictions = [], []
    for r in records:
        modes = ("raw", "calibrated") if r["seed"] in calibration_seeds else ("raw",)
        for calibration in modes:
            rows.append({**{k: r[k] for k in ("target", "size", "n", "seed", "model")},
                         "calibration": calibration, **r[f"{calibration}_metrics"],
                         "elapsed_seconds": r["elapsed_seconds"], "selected": r["selected"]})
        for i, patient in enumerate(r["patient_ids"]):
            predictions.append({**{k: r[k] for k in ("target", "size", "n", "seed", "model")},
                                "patient_id": patient, "y": r["y"][i], "raw": r["probabilities"][i],
                                "calibrated": r["calibrated_probabilities"][i] if r["seed"] in calibration_seeds else np.nan})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    score_names = ["roc_auc", "average_precision", "brier", "ece", "log_loss"]
    summary = metrics.groupby(["target", "size", "n", "model", "calibration"], sort=False)[score_names].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    calibration_summary = metrics.loc[metrics.seed.isin(calibration_seeds)].groupby(
        ["target", "size", "n", "model", "calibration"], sort=False)[score_names].agg(["mean", "std", "count"])
    calibration_summary.columns = ["_".join(c) for c in calibration_summary.columns]
    calibration_summary.reset_index().to_csv(output / "calibration_summary.csv", index=False)
    # A partial study may export metrics but must never declare a winner.
    if not complete:
        return
    indexed = {(r["target"], r["size"], r["seed"], r["model"]): r for r in records}
    comparisons = []
    for target in TARGETS:
        y = protocol["cohorts"][target]["test_y"]
        for size in (*[str(s) for s in SIZES], "Low-N mean (50,100,200)"):
            selected_sizes = ("50", "100", "200") if size.startswith("Low-N") else (size,)
            def matrix(model):
                return np.array([indexed[target, s, seed, model]["probabilities"] for s in selected_sizes for seed in SEEDS])
            for baseline in MODELS[1:]:
                comparison = paired_auc_bootstrap(y, matrix("tabpfn"), matrix(baseline), samples=2000, seed=20260922)
                comparisons.append(dict(target=target, size=size, reference=baseline, **comparison))
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    plot_curves(output, metrics, calibration_seeds)
    primary = comparisons.loc[comparisons["size"].str.startswith("Low-N")]
    text = ["# Paper-matched sample-efficiency experiment", "",
            f"Completed {len(records)} final model evaluations: two endpoints × five sizes × ten matched draws × six pipelines.", "",
            "Primary comparison: mean raw ROC-AUC over N=50,100,200, with equal weight for each size and seed. "
            "Positive differences favor TabPFN. Intervals are paired, class-stratified test-patient bootstrap intervals "
            "conditional on the fixed training draws, unadjusted across comparisons. This is exploratory.", "",
            "| Endpoint | Comparator | AUC difference | 95% interval |", "|---|---|---:|---:|"]
    if amendment:
        text[4:4] = ["**Quota amendment:** at the user's request, calibration is evaluated on fixed seeds 42–46 "
                     "for every model and endpoint. All ten seeds remain in the primary raw-AUC comparison. "
                     "Previously computed extra calibration predictions are retained locally but excluded from matched calibration summaries. "
                     "The change was driven by API allowance, not model scores. See quota_amendment.json.", ""]
    for r in primary.to_dict("records"):
        text.append(f"| {ENDPOINTS[r['target']]} | {LABELS[r['reference']]} | {r['auc_difference']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    text += ["", "## Mean raw AUC by training size", "", "| Endpoint | N | " + " | ".join(LABELS.values()) + " |",
             "|---|---|" + "---:|" * len(MODELS)]
    means = metrics.loc[metrics.calibration.eq("raw")].groupby(["target", "size", "model"]).roc_auc.mean()
    for target in TARGETS:
        for size in SIZES:
            text.append(f"| {ENDPOINTS[target]} | {size} | " + " | ".join(f"{means[target, str(size), model]:.4f}" for model in MODELS) + " |")
    text += ["", "## Interpretation of the primary comparison", ""]
    for target in TARGETS:
        low_means = {model: float(np.mean([means[target, s, model] for s in ("50", "100", "200")])) for model in MODELS}
        winner = max(low_means, key=low_means.get)
        rf = primary.loc[primary.target.eq(target) & primary.reference.eq("rf")].iloc[0]
        text.append(f"- {ENDPOINTS[target]}: {LABELS[winner]} has the highest low-N mean AUC ({low_means[winner]:.4f}); "
                    f"TabPFN scores {low_means['tabpfn']:.4f}. TabPFN minus no-SMOTE RF is {rf.auc_difference:+.4f} "
                    f"with interval [{rf.ci_low:+.4f}, {rf.ci_high:+.4f}], "
                    f"which {'includes' if rf.ci_low <= 0 <= rf.ci_high else 'excludes'} zero.")
    text += ["", "## Calibration and the SMOTE control", "",
             "The calibration plot shows raw and OOF-sigmoid Brier scores for every pipeline at every size. "
             "Brier score measures overall probability accuracy, including discrimination; it is not a pure calibration metric. "
             "ECE uses ten equal-width bins and is noisy on these small test sets. Sigmoid recalibration is not guaranteed to improve either metric.", "",
             "All models receive identical original patients and preprocessing matrices before the optional SMOTE step. "
             "Each baseline searches four prespecified candidates by three-fold training-only AUC. RF uses 300 trees, boosting 200. "
             "The RF control compares independently tuned pipelines with and without SMOTE; the earlier exact published forest reproduction remains in results/paper.", "",
             "Preprocessing is refit inside each N-patient subset and each inner training fold. Calibration fits a C=1 logistic sigmoid "
             "on the selected candidate's OOF probabilities and labels from that same N-patient subset; final models then use all N. "
             "No held-out test labels enter tuning or calibration. The raw AUC is the primary outcome, irrespective of calibration.", "",
             "## Limitations", ""]
    text.extend(f"- {note}" for note in protocol["limitations"])
    text += ["", "These results can establish an advantage within this experimental setting, not universal model superiority. "
             "All model and size comparisons, including unfavorable results, are retained in metrics.csv and paired_comparisons.csv.", ""]
    if amendment and (output / "quota_usage_latest.json").exists():
        start, end = read_json(output / "quota_usage_start.json"), read_json(output / "quota_usage_latest.json")
        text += ["## API usage", "", f"The amended continuation used {end['monthly_tokens_used']-start['monthly_tokens_used']:,} "
                 f"additional tokens, below the {amendment['max_additional_tokens']:,} cap. "
                 f"The recorded daily allowance remaining at completion was {end['daily_token_limit']-end['daily_tokens_used']:,} tokens.", ""]
    if (output / "numerical_verification.json").exists():
        audit = read_json(output / "numerical_verification.json")
        text += ["## Numerical audit", "", f"Refitted {audit['logistic_fits_independently_reproduced']} logistic models and "
                 f"{audit['sigmoid_fits_independently_reproduced']} sigmoid calibrators. Independent einsum probability and "
                 "objective-gradient checks verified the saved predictions despite matmul warnings in the frozen macOS environment. "
                 f"Maximum logistic probability difference: {audit['maximum_probability_difference']:.3g}. No results were changed.", ""]
    (output / "REPORT.md").write_text("\n".join(text))
    # Publish completion only once every file required by the dashboard exists.
    write_json(output / "status.json", status)


def plot_curves(output, metrics, calibration_seeds):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(zip(MODELS, ["#007f79", "#5269b4", "#a58ac5", "#d28232", "#b74353", "#647575"]))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, target in enumerate(TARGETS):
        for col, metric in enumerate(("roc_auc", "average_precision")):
            ax = axes[row, col]
            for model in MODELS:
                subset = metrics.loc[metrics.target.eq(target) & metrics.model.eq(model) & metrics.calibration.eq("raw")]
                grouped = subset.groupby("n")[metric].agg(["mean", "std"])
                x = grouped.index.to_numpy()
                ax.plot(x, grouped["mean"], marker="o", color=colors[model], label=LABELS[model], lw=2 if model == "tabpfn" else 1.3)
                ax.fill_between(x, grouped["mean"]-grouped["std"], grouped["mean"]+grouped["std"], color=colors[model], alpha=.08)
            ax.set_title(ENDPOINTS[target])
            ax.set_ylabel("ROC-AUC" if metric == "roc_auc" else "Average precision")
            ax.set_xlabel("Training patients (tuning and calibration included)")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("HANCOCK paper-matched learning curves · ten paired training draws\nShading: draw SD, not independent-patient confidence intervals", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "benchmark_performance.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey="row")
    for row, target in enumerate(TARGETS):
        for col, calibration in enumerate(("raw", "calibrated")):
            ax = axes[row, col]
            for model in MODELS:
                subset = metrics.loc[metrics.target.eq(target) & metrics.model.eq(model) & metrics.calibration.eq(calibration) & metrics.seed.isin(calibration_seeds)]
                mean = subset.groupby("n").brier.mean()
                ax.plot(mean.index, mean.values, marker="o", color=colors[model], label=LABELS[model])
            ax.set_title(f"{ENDPOINTS[target]} · {'raw' if col == 0 else 'OOF sigmoid'}")
            ax.set_ylabel("Brier score (lower is better)")
            ax.set_xlabel("Training patients")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Probability accuracy · {len(calibration_seeds)} matched draws · RF / SMOTE controls", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "calibration_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "baselines", "tabpfn", "report", "all"), default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=2, help="CPU baseline workers; each estimator uses one thread")
    args = parser.parse_args()
    if args.phase == "report":
        report(args.output)
        return
    protocol = prepare(args.output)
    print(f"Frozen protocol {protocol['protocol_id']}: {protocol['expected_runs']} final evaluations", flush=True)
    if args.phase == "prepare":
        report(args.output)
        return
    try:
        execute(args, protocol)
    finally:
        report(args.output)


def execute(args, protocol):
    models = MODELS if args.phase == "all" else (("tabpfn",) if args.phase == "tabpfn" else MODELS[1:])
    baseline_jobs = [(case, model) for case in protocol["cases"] for model in models if model != "tabpfn"]
    if baseline_jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(run_case, args.output, case, model, protocol["protocol_id"]) for case, model in baseline_jobs]
            for i, future in enumerate(as_completed(pending), 1):
                print(f"Baseline {i}/{len(pending)}: {future.result()}", flush=True)
                if i % 50 == 0:
                    report(args.output)
    if "tabpfn" in models:
        if (args.output / "quota_amendment.json").exists():
            from src.learning_curve_budget import run_budget
            run_budget(args.output)
            return
        for i, case in enumerate(protocol["cases"], 1):
            name = run_case(args.output, case, "tabpfn", protocol["protocol_id"])
            print(f"TabPFN {i}/{len(protocol['cases'])}: {name}", flush=True)
            if i % 10 == 0:
                report(args.output)
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        sys.exit(1)
