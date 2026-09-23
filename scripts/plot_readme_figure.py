"""Render the README hero figure from the frozen learning-curve metrics (no API calls).

Reads results/learning_curves/metrics.csv and writes results/learning_curves/readme_learning_curves.png.
The study's own figures are left untouched.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import RESULTS

SOURCE = RESULTS / "learning_curves"
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e8e7e3"
FOCUS = {"tabpfn": ("TabPFN-3.5", "#2a78d6"), "rf": ("Tuned RF", "#eb6834")}
CONTEXT = {"rf_smote": "Tuned RF + SMOTE", "lightgbm": "Tuned LightGBM", "xgboost": "Tuned XGBoost",
           "logistic": "Tuned logistic"}
CONTEXT_COLOR = "#b4b3ad"
ENDPOINTS = {"survival_status": "Death status", "recurrence": "Recurrence"}
SIZES = ["50", "100", "200", "500", "Full"]


def spread(values, gap):
    """Nudge label y-positions apart by at least `gap`, keeping their order and centre."""
    order = np.argsort(values)
    placed = np.asarray(values, dtype=float)[order].copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(placed)):
            overlap = gap - (placed[i] - placed[i - 1])
            if overlap > 1e-9:
                placed[i - 1] -= overlap / 2
                placed[i] += overlap / 2
                moved = True
        if not moved:
            break
    result = np.empty_like(placed)
    result[order] = placed
    return result


def main():
    metrics = pd.read_csv(SOURCE / "metrics.csv", dtype={"size": str})
    raw = metrics.loc[metrics.calibration.eq("raw")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor=SURFACE, sharey=True)
    x = np.arange(len(SIZES))
    for ax, (target, title) in zip(axes, ENDPOINTS.items()):
        ax.set_facecolor(SURFACE)
        data = raw.loc[raw.target.eq(target)]
        full_n = int(data.loc[data["size"].eq("Full"), "n"].iloc[0])
        ax.axvspan(-.35, 2.35, color="#f1f0ed", zorder=0, lw=0)
        ax.text(1, .895, "Primary comparison · 50–200 patients", ha="center", va="top", fontsize=9, color=INK_2)
        ends = []
        models = [*CONTEXT, *reversed(list(FOCUS))]  # Focus series drawn last, on top.
        for model in models:
            grouped = data.loc[data.model.eq(model)].groupby("size").roc_auc.agg(["mean", "std"]).reindex(SIZES)
            if model in FOCUS:
                label, color = FOCUS[model]
                if model == "tabpfn":
                    ax.fill_between(x, grouped["mean"] - grouped["std"], grouped["mean"] + grouped["std"],
                                    color=color, alpha=.13, lw=0, zorder=2)
                ax.plot(x, grouped["mean"], color=color, lw=2.6 if model == "tabpfn" else 2, zorder=4,
                        marker="o", ms=7, mec=SURFACE, mew=1.5)
                ends.append((grouped["mean"].iloc[-1], label, color, "bold" if model == "tabpfn" else "normal"))
            else:
                ax.plot(x, grouped["mean"], color=CONTEXT_COLOR, lw=1.3, zorder=3, marker="o", ms=4.5,
                        mec=SURFACE, mew=1)
                ends.append((grouped["mean"].iloc[-1], CONTEXT[model], INK_3, "normal"))
        # Direct labels at the right edge, nudged apart; text stays in ink tones, identity via the swatch line.
        positions = spread([e[0] for e in ends], .0135)
        for (value, label, color, weight), y in zip(ends, positions):
            ax.plot([4.08, 4.2, 4.3], [value, y, y], color=color if color != INK_3 else CONTEXT_COLOR, lw=1, clip_on=False)
            ax.text(4.36, y, label, va="center", fontsize=9.5, fontweight=weight,
                    color=INK if weight == "bold" else INK_2, clip_on=False)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
        ax.set_xticks(x, [*SIZES[:-1], f"All ({full_n})"])
        ax.set_xlim(-.35, 4.05)
        ax.set_xlabel("Training patients")
        ax.grid(axis="y", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("ROC-AUC on held-out test patients")
    axes[0].set_ylim(.55, .9)
    fig.text(.012, .965, "Untuned TabPFN beats tuned boosting on small cohorts, with no clear difference from a tuned forest",
             fontsize=14.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(.012, .905, "HANCOCK official in-distribution test split · mean of ten matched training draws · "
             "blue band: ±1 SD across draws for TabPFN", fontsize=10, color=INK_2, ha="left", va="top")
    fig.text(.012, .02, "Baselines choose among four configurations by cross-validation inside each patient budget; "
             "TabPFN uses its default settings. Source: results/learning_curves.", fontsize=8.5, color=INK_3, ha="left")
    fig.subplots_adjust(left=.07, right=.87, top=.78, bottom=.16, wspace=.38)
    out = SOURCE / "readme_learning_curves.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
