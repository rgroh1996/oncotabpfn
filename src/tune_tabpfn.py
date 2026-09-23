"""Prespecified training-only selection; one exploratory post-selection test per endpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.common import ROOT, RESULTS, file_hash, fingerprint, read_json, safe_error, write_json
from src.optimization import CONFIGS, SEED, THINKING_SECONDS, OptimizationModel, rank_configurations
from src.paper_protocol import TARGETS, load_features, load_partition, paired_auc_bootstrap, provenance
from src.pipeline import metric_scores, model_metadata

OUT = RESULTS / "optimization"


def prepare():
    features = load_features()
    cohort, partitions = {}, {}
    for target in TARGETS:
        train, test = load_partition(features, target, "in")
        cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(train, train.target))
        cohort[target] = dict(training_ids=train.patient_id.tolist(), reserved_test_ids=test.patient_id.tolist(),
                              folds=[dict(train_ids=train.iloc[a].patient_id.tolist(), validation_ids=train.iloc[b].patient_id.tolist()) for a, b in cv])
        partitions[target] = train, test, cv
    plan = dict(version=1, provenance=provenance(), seed=SEED, cv_folds=3, configurations=CONFIGS,
                implementation_sha256=file_hash(ROOT / "src/optimization.py"),
                thinking=dict(effort="medium", timeout_s=THINKING_SECONDS, metric="roc_auc"),
                selection="Highest mean three-fold validation ROC AUC; exact ties lower Brier then fixed config order",
                smote=False, final_seed=42, cohorts=cohort,
                evaluation="Official in-distribution test only after per-endpoint selection is saved; single final fit. Test previously inspected, so exploratory.",
                explanations="Fixed selected models; training-only background; held-out patient explanations do not drive selection",
                limitations="Published features inherit upstream cohort-wide preprocessing; CV selection scores are optimistic for the selected winner, not nested-CV generalization estimates")
    identifier = fingerprint(plan)
    if (OUT / "protocol.json").exists():
        if read_json(OUT / "protocol.json")["protocol_id"] != identifier:
            raise ValueError("Protocol changed; archive results/optimization before starting a new experiment")
    else:
        write_json(OUT / "protocol.json", dict(protocol_id=identifier, created_utc=datetime.now(timezone.utc).isoformat(), **plan))
    return identifier, partitions


def tune(protocol_id, partitions):
    for target in TARGETS:
        train, _, folds = partitions[target]
        records = []
        for config in CONFIGS:
            for fold, (train_idx, validation_idx) in enumerate(folds):
                path = OUT / "runs" / f"{target}_{config['name']}_{fold}.json"
                if path.exists():
                    cached = read_json(path)
                    if cached["protocol_id"] == protocol_id and cached["status"] == "complete":
                        records.append(cached)
                        print(f"Cached {target}/{config['name']}/{fold}", flush=True)
                        continue
                training, validation = train.iloc[train_idx], train.iloc[validation_idx]
                started = time.perf_counter()
                record = dict(protocol_id=protocol_id, target=target, config=config["name"], fold=fold,
                              seed=42+fold, n_train=len(training), n_validation=len(validation),
                              training_ids=training.patient_id.tolist(), validation_ids=validation.patient_id.tolist())
                try:
                    print(f"Fitting {target}/{config['name']}, fold {fold+1}/3", flush=True)
                    model = OptimizationModel(config, seed=42+fold).fit(training, training.target)
                    probability = model.predict(validation)
                    record.update(status="complete", **metric_scores(validation.target, probability),
                                  y=validation.target.tolist(), probabilities=probability.tolist(),
                                  model_metadata=model_metadata(model.estimator), n_features=len(model.columns),
                                  seconds=time.perf_counter()-started)
                    write_json(path, record)
                    print(f"  AUC={record['roc_auc']:.4f}, Brier={record['brier']:.4f}, {record['seconds']:.1f}s", flush=True)
                    records.append(record)
                except Exception as exc:
                    record.update(status="failed", error=safe_error(exc), seconds=time.perf_counter()-started)
                    write_json(path, record)
                    # Stop and preserve all successes; never silently drop failed candidates.
                    raise RuntimeError(record["error"]) from None
        ranked = rank_configurations(records)
        write_json(OUT / f"selection_{target}.json", dict(protocol_id=protocol_id, target=target,
                   selected=ranked[0]["config"], ranking=ranked, selected_utc=datetime.now(timezone.utc).isoformat(),
                   basis="Training-fold validation only; no test scores used"))
        print(f"Selected {target}: {ranked[0]['config']} (CV AUC {ranked[0]['roc_auc']:.4f})", flush=True)


def final_evaluation(protocol_id, partitions):
    # Require both decisions to be frozen before reading either final test outcome for scoring.
    selections = {t: read_json(OUT / f"selection_{t}.json") for t in TARGETS}
    for target in TARGETS:
        selection = selections[target]
        if selection["protocol_id"] != protocol_id:
            raise ValueError("Stale selection")
        path = OUT / f"test_{target}.json"
        if path.exists() and read_json(path)["protocol_id"] == protocol_id:
            if read_json(path)["selected"] != selection["selected"]:
                raise ValueError(f"Cached final test for {target} does not match the saved selection; archive it before rerunning")
            print(f"Cached final test: {target}", flush=True)
            continue
        config = next(c for c in CONFIGS if c["name"] == selection["selected"])
        train, test, _ = partitions[target]
        model_dir = OUT / "models" / target
        if (model_dir / "server_model.json").exists():
            # A saved model from an earlier selection must never be scored under the new name.
            saved = joblib.load(model_dir / "preprocessing.joblib")["config"]["name"]
            if saved != config["name"]:
                raise ValueError(f"Saved {target} model is {saved}, selection is {config['name']}; archive models/{target}")
            model = OptimizationModel.load(model_dir)
        else:
            print(f"Final fit {target}: {config['name']}", flush=True)
            model = OptimizationModel(config, seed=42, cache=True).fit(train, train.target)
            model.save(model_dir)
        probability = model.predict(test)
        reference = read_json(RESULTS / "paper/runs" / f"{target}_in_0_tabpfn_native.json")
        forest = read_json(RESULTS / "paper/runs" / f"{target}_in_0_random_forest.json")
        for original in (reference, forest):
            if original["patient_ids"] != test.patient_id.tolist() or original["y_test"] != test.target.tolist():
                raise ValueError("Reference patients/outcomes do not match")
        write_json(path, dict(protocol_id=protocol_id, target=target, selected=config["name"],
                             train_ids=train.patient_id.tolist(), patient_ids=test.patient_id.tolist(),
                             y=test.target.tolist(), probabilities=probability.tolist(), **metric_scores(test.target, probability),
                             baseline_metrics=metric_scores(test.target, reference["probabilities"]),
                             rf_metrics=metric_scores(test.target, forest["probabilities"]),
                             baseline_probabilities=reference["probabilities"],
                             paired_vs_baseline=paired_auc_bootstrap(test.target, [probability], [reference["probabilities"]]),
                             seed=42, evaluated_utc=datetime.now(timezone.utc).isoformat(), model_metadata=model_metadata(model.estimator)))
        print(f"Final {target}: AUC={metric_scores(test.target, probability)['roc_auc']:.4f}", flush=True)


def report(protocol_id):
    runs = [read_json(p) for p in sorted((OUT / "runs").glob("*.json"))]
    valid = [r for r in runs if r.get("status") == "complete" and r["protocol_id"] == protocol_id]
    if not valid:
        write_json(OUT / "status.json", dict(protocol_id=protocol_id, completed_cv_runs=0,
                                             expected_cv_runs=36, final_tests=0, tuning_complete=False))
        return
    metrics = pd.DataFrame([{k: r[k] for k in ("target", "config", "fold", "n_train", "n_validation", "roc_auc", "average_precision", "brier", "ece", "seconds")} for r in valid])
    metrics.to_csv(OUT / "cv_metrics.csv", index=False)
    summary = metrics.groupby(["target", "config"], sort=False)[["roc_auc", "average_precision", "brier", "ece"]].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / "cv_summary.csv", index=False)
    findings = ["# TabPFN optimization and interpretation", "",
                "Six fixed candidates, three folds per endpoint, official in-distribution training patients only. "
                "Selection maximizes mean validation AUC. All candidates omit SMOTE. Thinking uses medium effort, a 60-second fit budget and ROC-AUC objective.", "",
                "The paper feature tables are reused and inherit upstream cohort-wide encoding/blood preprocessing. "
                "Native handling preserves their numeric category codes and remaining missing values; it does not reconstruct unimputed raw blood measurements.", "",
                "| Endpoint | Selected configuration | Training CV AUC | Final test AUC | Untuned baseline AUC | ΔAUC [95% paired interval] |",
                "|---|---|---:|---:|---:|---:|"]
    final_rows = []
    for target in TARGETS:
        path = OUT / f"test_{target}.json"
        if not path.exists():
            continue
        row = read_json(path)
        selection = read_json(OUT / f"selection_{target}.json")
        d = row["paired_vs_baseline"]
        cv_auc = selection["ranking"][0]["roc_auc"]
        findings.append(f"| {target} | {row['selected']} | {cv_auc:.4f} | {row['roc_auc']:.4f} | {row['baseline_metrics']['roc_auc']:.4f} | {d['auc_difference']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] |")
        final_rows.append(dict(target=target, selected=row["selected"], cv_auc=cv_auc,
                               **{k: row[k] for k in ("roc_auc", "average_precision", "brier", "ece")},
                               baseline_auc=row["baseline_metrics"]["roc_auc"], **d))
    pd.DataFrame(final_rows).to_csv(OUT / "test_summary.csv", index=False)
    if len(final_rows) == 2 and all(r["auc_difference"] <= 0 for r in final_rows):
        findings[2:2] = ["**Observed result: neither selected configuration improved reserved-test ROC AUC over the untuned no-SMOTE baseline. "
                         "The validation-selected 19-feature clinical/pathology model is retained for interpretation; the original baseline is unchanged.**", ""]
    findings += ["", "The final comparison uses a single seed-42 fit per endpoint against the saved seed-42 no-SMOTE, paper-preprocessing baseline. "
                 "The baseline's name in the earlier paper experiment was `tabpfn_native`, meaning no SMOTE; here `native_default` instead means explicit native categorical preprocessing. "
                 "These are different configurations.", "",
                 "Final test sets were already inspected in the earlier experiment, so improvements are exploratory. "
                 "No test metric feeds selection. Selection CV scores are optimistic for the chosen winner and are not nested-CV estimates. "
                 "Paired bootstrap intervals are conditional on fixed training data and unadjusted for multiple comparisons. "
                 "Endpoints match released HANCOCK code, not five-year overall survival; no clinical or causal validation follows.", "",
                 "Feature attribution outputs are added separately in `explanations/`; they explain the selected model, never a surrogate baseline. "
                 "See `protocol.json` for prespecified settings and exact cohort/fold memberships.", "",
                 "Sources: [Thinking mode](https://docs.priorlabs.ai/capabilities/thinking-mode), "
                 "[interpretability](https://docs.priorlabs.ai/capabilities/interpretability), "
                 "[HANCOCK paper](https://doi.org/10.1038/s41467-025-62386-6)."]
    for target in TARGETS:
        path = OUT / "explanations" / target / "summary.json"
        if not path.exists():
            continue
        explained = read_json(path)
        findings += ["", f"## Exploratory explanations: {target}", "",
                     f"Model: `{explained['model']}`. SHAP summarizes {explained['explained_patients']} randomly sampled test patients "
                     f"relative to {explained['background_patients']} sampled training-background patients. "
                     f"Maximum probability reconstruction error: {explained['max_additivity_error']:.3g}. "
                     f"Feature-rank correlation between ordering repeats: {explained['ordering_rank_spearman']:.3f}; "
                     f"top-10 overlap: {explained['top10_ordering_overlap']}/10.", "",
                     "| Feature | Mean absolute SHAP (percentage points) |", "|---|---:|"]
        for feature in explained["top_features"][:8]:
            findings.append(f"| {feature['feature']} | {100*feature['mean_absolute_shap']:.3f} |")
        attributions = pd.read_csv(OUT / "explanations" / target / "shap_values.csv")
        leading = explained["top_features"][0]["feature"]
        influence = attributions.loc[attributions.feature.eq(leading), "shap_value"].abs()
        if influence.sum() > 0 and influence.max()/influence.sum() > .5:
            findings += ["", f"**Sample sensitivity:** one patient contributes {influence.max()/influence.sum():.1%} "
                         f"of the leading feature's absolute attribution (`{leading}`). This is not a reliable whole-cohort ranking."]
        findings += ["", f"![SHAP sample distribution](explanations/{target}/shap_beeswarm.png)", "",
                     "These describe model influence in a small explanation sample, not causal risk factors or whole-cohort importance. "
                     "Category values are published codes; masking breaks cross-feature dependencies. Background-sample sensitivity was not estimated. "
                     "The selected model omits blood, ICD and TMA inputs, so they receive no SHAP scores. No attribution was used for feature selection.", "",
                     f"![Held-out modality permutation](explanations/{target}/modality_importance.png)"]
    (OUT / "REPORT.md").write_text("\n".join(findings)+"\n")
    write_json(OUT / "status.json", dict(protocol_id=protocol_id, completed_cv_runs=len(valid), expected_cv_runs=36,
                                         final_tests=len(final_rows), tuning_complete=len(valid) == 36 and len(final_rows) == 2))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        subset = metrics.loc[metrics.target.eq(target)].groupby("config").roc_auc.agg(["mean", "std"]).reindex([c["name"] for c in CONFIGS])
        labels = ["Paper preprocessing", "Native categories", "Paper + Thinking", "Native + Thinking",
                  "Clinical + pathology", "Clinical + pathology + blood"]
        ax.errorbar(subset["mean"], range(len(subset)), xerr=subset["std"].fillna(0), fmt="o", color="#008c82", capsize=3)
        ax.set(title="Death status" if target == "survival_status" else "Recurrence", yticks=range(len(subset)),
               yticklabels=labels, xlabel="Training-only validation ROC AUC (mean ± fold SD)", xlim=(.55, .95))
        ax.grid(axis="x", alpha=.15)
        ax.invert_yaxis()
    fig.savefig(OUT / "cv_comparison.png", dpi=170)
    plt.close(fig)
    print(f"Exported {len(valid)}/36 CV runs and {len(final_rows)}/2 final tests", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "cv", "final", "report"), default="all")
    args = parser.parse_args()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    protocol_id, partitions = prepare()
    try:
        if args.phase in ("all", "cv"):
            tune(protocol_id, partitions)
        if args.phase in ("all", "final"):
            final_evaluation(protocol_id, partitions)
    finally:
        if any((OUT / "runs").glob("*.json")):
            report(protocol_id)


if __name__ == "__main__":
    main()
