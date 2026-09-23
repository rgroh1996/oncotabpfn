"""SHAP on original clinical columns and held-out modality permutation importance.

The selected model and fitted preprocessing remain fixed. Permutations/masking
happen BEFORE preprocessing, so each categorical variable is one SHAP feature.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results/.matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "results/optimization/explanations/cache/numba"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.common import ROOT, RESULTS, file_hash, fingerprint, read_json, safe_error, write_json
from src.optimization import SEED, OptimizationModel, feature_modalities
from src.paper_protocol import TARGETS, load_features, load_partition
from src.pipeline import metric_scores

OUT = RESULTS / "optimization/explanations"


class CachedPredictor:
    """Cache complete masked batches; retain progress across interrupted API jobs."""
    def __init__(self, model, directory, batch_size=512):
        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.calls = 0
        self.rows = 0

    def __call__(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.model.columns):
            raise ValueError("Explanation input must preserve original feature columns")
        results = []
        for offset in range(0, len(array), self.batch_size):
            batch = array[offset:offset+self.batch_size]
            # pandas hashing has stable NaN handling and preserves row/column order.
            digest = fingerprint(dict(shape=batch.shape, rows=pd.util.hash_pandas_object(pd.DataFrame(batch), index=False).tolist()))
            path = self.directory / f"{digest}.npy"
            if path.exists():
                probability = np.load(path, allow_pickle=False)
            else:
                probability = self.model.predict(batch, batch_size=self.batch_size)
                np.save(path, probability, allow_pickle=False)
                self.calls += 1
                self.rows += len(batch)
            if probability.shape != (len(batch),) or not np.isfinite(probability).all():
                raise ValueError("Invalid cached probabilities")
            results.append(probability)
        return np.concatenate(results) if results else np.empty(0)


def aggregate_importance(values, columns):
    return pd.DataFrame(dict(feature=columns, mean_absolute_shap=np.abs(values).mean(axis=0),
                             mean_signed_shap=values.mean(axis=0))).sort_values("mean_absolute_shap", ascending=False)


def explain_target(target, args):
    import shap
    root = RESULTS / "optimization"
    evaluation = read_json(root / f"test_{target}.json")
    selection = read_json(root / f"selection_{target}.json")
    if evaluation["selected"] != selection["selected"] or evaluation["protocol_id"] != selection["protocol_id"]:
        raise ValueError("Selected model and evaluation do not match")
    features = load_features()
    train, test = load_partition(features, target, "in")
    if train.patient_id.tolist() != evaluation["train_ids"] or test.patient_id.tolist() != evaluation["patient_ids"]:
        raise ValueError("Cohort changed since selection")
    model_dir = root / "models" / target
    model = OptimizationModel.load(model_dir)
    if model.config["name"] != selection["selected"]:
        raise ValueError("Wrong model loaded for explanation")
    rng = np.random.default_rng(SEED)
    background_idx = rng.choice(len(train), min(args.background, len(train)), replace=False)
    patient_idx = np.sort(rng.choice(len(test), min(args.patients, len(test)), replace=False))
    background = train.iloc[background_idx][model.columns].astype(float)
    patients = test.iloc[patient_idx][model.columns].astype(float)
    destination = OUT / target
    destination.mkdir(parents=True, exist_ok=True)
    plan = dict(protocol_id=evaluation["protocol_id"], target=target, selected=selection["selected"],
                seed=SEED, shap_version=version("shap"),
                server_model_sha256=file_hash(model_dir / "server_model.json"),
                preprocessing_sha256=file_hash(model_dir / "preprocessing.joblib"),
                implementation_sha256=file_hash(ROOT / "src/explain_tabpfn.py"),
                background_ids=train.iloc[background_idx].patient_id.tolist(),
                explained_ids=test.iloc[patient_idx].patient_id.tolist(),
                original_features=model.columns, permutation_orderings=args.orderings,
                permutation_repeats=args.permutation_repeats,
                method="shap.PermutationExplainer on original columns; independent marginal training background; event-probability scale",
                selection="Random held-out patients, no outcome or risk-based sampling; no feature selection from explanations",
                limits="Small exploratory SHAP sample/background; marginal masking breaks cross-feature correlations; importance is model-specific, not causal")
    identifier = fingerprint(plan)
    if (destination / "protocol.json").exists():
        if read_json(destination / "protocol.json")["explanation_id"] != identifier:
            raise ValueError("Explanation protocol changed; archive its destination before rerunning")
    else:
        write_json(destination / "protocol.json", dict(explanation_id=identifier, created_utc=datetime.now(timezone.utc).isoformat(), **plan))
    predictor = CachedPredictor(model, OUT / "cache" / identifier)
    observed = predictor(test[model.columns].to_numpy(dtype=float))
    drift = float(np.max(np.abs(observed - np.asarray(evaluation["probabilities"]))))
    if drift > 1e-4:
        raise ValueError(f"Reloaded model predictions changed by {drift}; do not explain a different fitted model")
    base_probability = float(predictor(background.to_numpy()).mean())
    all_values, all_values_second, row_baselines = [], [], []
    for i in range(len(patients)):
        patient_id = test.iloc[patient_idx[i]].patient_id
        cached_path = destination / f"patient_{patient_id}.npz"
        if cached_path.exists():
            with np.load(cached_path, allow_pickle=False) as record:
                values, second, baseline = record["values"], record["second"], float(record["baseline"])
        else:
            # Two independent seeded runs permit a limited ordering-sensitivity check.
            first = shap.PermutationExplainer(predictor, background, feature_names=model.columns, seed=SEED+i)
            # max_evals sets forward-and-reverse permutation paths, not API requests.
            budget = args.orderings * (2 * len(model.columns) + 1)
            a = first(patients.iloc[[i]], max_evals=budget, batch_size=512, silent=True)
            second_explainer = shap.PermutationExplainer(predictor, background, feature_names=model.columns, seed=SEED+1000+i)
            b = second_explainer(patients.iloc[[i]], max_evals=budget, batch_size=512, silent=True)
            values, second, baseline = a.values[0], b.values[0], float(a.base_values[0])
            if abs(float(b.base_values[0])-baseline) > 1e-5:
                raise ValueError("SHAP backgrounds differ between repeats")
            np.savez_compressed(cached_path, values=values, second=second, baseline=baseline)
        all_values.append(values)
        all_values_second.append(second)
        row_baselines.append(baseline)
        print(f"SHAP {target}: {i+1}/{len(patients)} patients explained", flush=True)
    first_values, second_values = np.asarray(all_values), np.asarray(all_values_second)
    values = (first_values+second_values)/2
    baselines = np.asarray(row_baselines)
    actual = observed[patient_idx]
    additivity = float(np.max(np.abs(values.sum(axis=1)+baselines-actual)))
    if additivity > 1e-4 or np.max(np.abs(baselines-base_probability)) > 1e-4:
        raise ValueError(f"SHAP probability reconstruction failed: {additivity}")
    importance = aggregate_importance(values, model.columns)
    importance.to_csv(destination / "feature_importance.csv", index=False)
    long_rows = []
    for i, patient_id in enumerate(test.iloc[patient_idx].patient_id):
        for j, name in enumerate(model.columns):
            long_rows.append(dict(patient_id=patient_id, feature=name, feature_value=patients.iloc[i, j],
                                  shap_value=values[i, j], ordering_difference=abs(first_values[i, j]-second_values[i, j]),
                                  base_probability=baselines[i], predicted_probability=actual[i]))
    pd.DataFrame(long_rows).to_csv(destination / "shap_values.csv", index=False)
    np.savez_compressed(destination / "shap_arrays.npz", values=values, first=first_values, second=second_values,
                        baseline=baselines, probability=actual, data=patients.to_numpy(),
                        patient_ids=test.iloc[patient_idx].patient_id.to_numpy(dtype=str), features=np.asarray(model.columns))
    explanation = shap.Explanation(values=values, base_values=baselines, data=patients.to_numpy(), feature_names=model.columns)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=16, show=False, plot_size=(12, 8))
    plt.title(f"{target}: exploratory SHAP · {len(patients)} held-out patients\nPositive values increase predicted event probability", fontsize=11)
    plt.savefig(destination / "shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    for i, patient_id in enumerate(test.iloc[patient_idx].patient_id):
        # Omit encoded feature values from waterfall labels; CSV records retain them explicitly.
        row = shap.Explanation(values=values[i], base_values=baselines[i], feature_names=model.columns)
        shap.plots.waterfall(row, max_display=12, show=False)
        plt.title(f"{target} · patient {patient_id}\nPublished feature codes; model attribution, not causation", fontsize=10)
        plt.savefig(destination / f"waterfall_{patient_id}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    permutation_rows = []
    baseline_scores = metric_scores(test.target, observed)
    perm_rng = np.random.default_rng(SEED)
    original = test[model.columns].to_numpy(dtype=float)
    for name, columns in feature_modalities().items():
        indices = [model.columns.index(c) for c in columns if c in model.columns]
        if not indices:
            continue
        for repeat in range(args.permutation_repeats):
            shuffled = original.copy()
            order = perm_rng.permutation(len(test))
            # Shuffle whole modality jointly, preserving within-modality relationships.
            shuffled[:, indices] = original[order][:, indices]
            scores = metric_scores(test.target, predictor(shuffled))
            permutation_rows.append(dict(modality=name, repeat=repeat,
                                         auc_drop=baseline_scores["roc_auc"]-scores["roc_auc"],
                                         brier_increase=scores["brier"]-baseline_scores["brier"]))
        print(f"Permutation {target}: {name} complete", flush=True)
    permutation = pd.DataFrame(permutation_rows)
    permutation.to_csv(destination / "modality_permutation.csv", index=False)
    stats = permutation.groupby("modality").auc_drop.agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.barh(stats.index, stats["mean"], xerr=stats["std"].fillna(0), capsize=4, color="#008c82")
    ax.axvline(0, color="black", lw=.8)
    ax.set(title=f"{target} · held-out modality importance", xlabel="AUC decrease after joint shuffling (mean ± shuffle SD)")
    fig.savefig(destination / "modality_importance.png", dpi=170)
    plt.close(fig)
    first_importance, second_importance = np.abs(first_values).mean(axis=0), np.abs(second_values).mean(axis=0)
    correlation = float(spearmanr(first_importance, second_importance).statistic)
    top_count = min(10, len(model.columns))
    first_top = set(np.argsort(first_importance)[-top_count:])
    second_top = set(np.argsort(second_importance)[-top_count:])
    summary = dict(explanation_id=identifier, target=target, model=selection["selected"], complete=True,
                   explained_patients=len(patients), background_patients=len(background),
                   patient_ids=test.iloc[patient_idx].patient_id.tolist(),
                   event="Death status" if target == "survival_status" else "Recurrence",
                   feature_count=len(model.columns), max_additivity_error=additivity, max_reload_prediction_drift=drift,
                   base_probability=base_probability, ordering_rank_spearman=correlation,
                   top10_ordering_overlap=len(first_top & second_top),
                   mean_absolute_ordering_difference=float(np.abs(first_values-second_values).mean()),
                   api_calls_this_invocation=predictor.calls, api_rows_this_invocation=predictor.rows,
                   top_features=importance.head(10).to_dict(orient="records"),
                   caveats=["SHAP ranking summarizes a small held-out sample, not the entire cohort.",
                            "Background consists of a small random training sample; rankings may change with background or seed.",
                            "Categorical values are the published encoded values; one-hot indicators are never masked separately.",
                            "Marginal masking and modality shuffling can break cross-feature correlations.",
                            "These are model attributions, not causal effects or treatment recommendations.",
                            "No feature was selected or removed using these held-out explanations."])
    write_json(destination / "summary.json", summary)
    print(f"Explanation complete: {target}, additivity error {additivity:.2g}, ordering rank correlation {correlation:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=12)
    parser.add_argument("--background", type=int, default=8)
    parser.add_argument("--orderings", type=int, default=2)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.patients, args.background, args.orderings, args.permutation_repeats) < 1:
        parser.error("All budgets must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        try:
            explain_target(target, args)
        except Exception as exc:
            write_json(OUT / "last_error.json", dict(target=target, error=safe_error(exc)))
            raise RuntimeError(safe_error(exc)) from None


if __name__ == "__main__":
    main()
