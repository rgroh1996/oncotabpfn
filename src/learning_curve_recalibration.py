"""Sensitivity analysis: refit every learning-curve sigmoid calibrator with a nonnegative slope.

The frozen protocol's C=1 sigmoid is unconstrained. When a selected candidate's out-of-fold scores
are anti-correlated with the labels in a small subset, it learns a negative slope and reverses the
test ranking. This script leaves the frozen study untouched. It refits the same penalized objective
with slope >= 0 from the saved OOF and raw test probabilities (no API calls) and writes a matched
comparison to results/learning_curves/recalibration.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import expit

from src.common import RESULTS, ROOT, file_hash, read_json, write_json
from src.learning_curve_protocol import LABELS, logit, score_probabilities

SOURCE = RESULTS / "learning_curves"
OUT = SOURCE / "recalibration"
SMALL = ("50", "100", "200")


def fit_sigmoid(oof, y, nonnegative, C=1.):
    """sklearn's lbfgs objective: 0.5 * slope^2 + C * log loss; the intercept is unpenalized.

    Strictly convex in two parameters, so damped Newton is exact. If the free optimum has a negative
    slope, the constrained optimum lies on slope = 0, where the intercept has a closed form.
    """
    x, y = logit(oof).ravel(), np.asarray(y, dtype=float)

    def objective(w):
        z = w[0] * x + w[1]
        return .5 * w[0] ** 2 + C * np.sum(np.logaddexp(0, z) - y * z)

    w = np.zeros(2)
    for _ in range(100):
        p = expit(w[0] * x + w[1])
        gradient = np.array([w[0] + C * (p - y) @ x, C * (p - y).sum()])
        if np.abs(gradient).max() < 1e-10 * max(1., len(y)):
            break
        v = C * p * (1 - p)
        hessian = np.array([[1 + v @ (x * x), v @ x], [v @ x, v.sum()]])
        step, current = np.linalg.solve(hessian, gradient), objective(w)
        while objective(w - step) > current and np.abs(step).max() > 1e-14:
            step /= 2
        w = w - step
    else:
        raise RuntimeError("Calibration fit did not converge")
    if nonnegative and w[0] < 0:
        rate = y.mean()
        w = np.array([0., np.log(rate / (1 - rate))])
    return w


def main():
    protocol = read_json(SOURCE / "protocol.json")
    seeds = set(read_json(SOURCE / "quota_amendment.json")["calibration_seeds"])
    rows, fits = [], []
    for path in sorted((SOURCE / "runs").glob("*.json")):
        run = read_json(path)
        if run["protocol_id"] != protocol["protocol_id"] or run["status"] != "complete":
            raise ValueError(f"Invalid frozen run: {path.name}")
        if run["seed"] not in seeds:
            continue
        test = logit(run["probabilities"]).ravel()
        # Reproduce the frozen calibrator first, so the constrained refit changes only the bound.
        # sklearn stops at tol=1e-4, so compare probabilities rather than exact coefficients.
        free = fit_sigmoid(run["oof_probabilities"], run["training_y"], nonnegative=False)
        if not np.allclose(expit(free[0] * test + free[1]), run["calibrated_probabilities"], atol=1e-3):
            raise ValueError(f"Frozen calibrator not reproduced: {path.name}")
        slope, intercept = fit_sigmoid(run["oof_probabilities"], run["training_y"], nonnegative=True)
        constrained = expit(slope * test + intercept)
        key = dict(target=run["target"], size=run["size"], n=run["n"], model=run["model"], seed=run["seed"])
        fits.append(dict(**key, frozen_slope=run["calibration"]["slope"], constrained_slope=slope,
                         constrained_intercept=intercept, inverted=run["calibration"]["slope"] <= 0))
        for mode, probabilities in (("raw", run["probabilities"]), ("frozen_sigmoid", run["calibrated_probabilities"]),
                                    ("nonnegative_sigmoid", constrained)):
            rows.append(dict(**key, calibration=mode, **score_probabilities(run["y"], probabilities)))
    metrics, fits = pd.DataFrame(rows), pd.DataFrame(fits)
    if len(fits) != 300:
        raise ValueError(f"Expected 300 matched calibration runs, found {len(fits)}")
    OUT.mkdir(exist_ok=True)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    fits.to_csv(OUT / "calibrator_fits.csv", index=False)
    small = metrics.loc[metrics["size"].isin(SMALL)]
    summary = small.groupby(["target", "model", "calibration"])[["roc_auc", "brier", "ece"]].mean().reset_index()
    summary.to_csv(OUT / "summary_small_n.csv", index=False)
    inverted = fits.loc[fits.inverted]
    write_json(OUT / "summary.json", dict(
        source_protocol_id=protocol["protocol_id"], matched_runs=len(fits), inverted_frozen_fits=len(inverted),
        inverted_by_model=inverted.model.value_counts().to_dict(),
        implementation_sha256=file_hash(ROOT / "src/learning_curve_recalibration.py")))
    lines = ["# Calibration sensitivity analysis", "",
             f"Supplementary to the frozen [learning-curve study](../REPORT.md) (`{protocol['protocol_id']}`), which is not modified.", "",
             f"The frozen sigmoid calibrator has an unconstrained slope. In **{len(inverted)} of {len(fits)}** matched calibration runs "
             "it learned a slope ≤ 0 from small, noisy out-of-fold scores, which reverses the test ranking. "
             "Here every calibrator is refitted on the same saved out-of-fold probabilities with the same C=1 objective, "
             "constrained to a nonnegative slope. Unconstrained refits reproduce every frozen calibrator first.", "",
             "| Endpoint | Model | Size | Seed | Frozen slope |", "|---|---|---:|---:|---:|"]
    for row in inverted.sort_values(["target", "model", "n", "seed"]).itertuples():
        lines.append(f"| {row.target} | {LABELS[row.model]} | {row.size} | {row.seed} | {row.frozen_slope:+.3f} |")
    lines += ["", "## Mean over N=50, 100, 200 and five matched seeds", "",
              "| Endpoint | Model | Raw Brier | Frozen sigmoid Brier | Nonnegative sigmoid Brier | Raw ECE | Frozen ECE | Nonnegative ECE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    pivot = summary.pivot_table(index=["target", "model"], columns="calibration", values=["brier", "ece"])
    for (target, model), row in pivot.iterrows():
        lines.append(f"| {target} | {LABELS[model]} | " + " | ".join(
            f"{row[(metric, mode)]:.4f}" for metric in ("brier", "ece") for mode in ("raw", "frozen_sigmoid", "nonnegative_sigmoid")) + " |")
    lines += ["", "A slope bounded at zero yields a constant prediction for that run: the calibrator reports no usable signal "
              "instead of reversing the ranking. The raw-AUC primary outcome of the frozen study does not use calibration and is unchanged. "
              "This analysis was added after the frozen results were seen."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"{len(inverted)} inverted frozen fits of {len(fits)}; wrote {OUT}")


if __name__ == "__main__":
    main()
