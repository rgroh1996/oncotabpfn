"""Independently reproduce logistic probabilities without BLAS matrix products.

The frozen NumPy/macOS stack emits matmul floating-point warnings. This audit
refits every logistic model and checks predictions and objective gradients via
einsum, without changing any study results or optimization tolerances.
"""
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.common import read_json, write_json
from src.learning_curve_protocol import GRIDS, OUTPUT, logit


def reference_probability(x, y, test, c):
    n = len(y)
    penalty = 1 / (c * n)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True):
        fitted = LogisticRegression(C=c, solver="lbfgs", max_iter=3000).fit(x, y)
        original = fitted.predict_proba(test)[:, 1]
    assert fitted.n_iter_[0] < 3000
    w, intercept = fitted.coef_[0], fitted.intercept_[0]
    z = np.einsum("ij,j->i", x, w, optimize=False) + intercept
    residual = expit(z) - y
    loss = np.mean(np.logaddexp(0, z) - y * z) + .5 * penalty * np.sum(w * w)
    gradient = np.r_[np.einsum("ij,i->j", x, residual, optimize=False) / n + penalty*w, residual.mean()]
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    # sklearn's default L-BFGS-B gradient tolerance is 1e-4.
    assert np.max(np.abs(gradient)) < 1.1e-4, np.max(np.abs(gradient))
    independent = expit(np.einsum("ij,j->i", test, w, optimize=False) + intercept)
    np.testing.assert_allclose(independent, original, atol=1e-12, rtol=1e-12)
    return independent


def main():
    protocol = read_json(OUTPUT / "protocol.json")
    maximum = 0.
    checked = 0
    for case in protocol["cases"]:
        with np.load(OUTPUT / "inputs" / f"{case['name']}.npz", allow_pickle=False) as arrays:
            for candidate, params in enumerate(GRIDS["logistic"]):
                for fold in range(3):
                    record = read_json(OUTPUT / "inner_cache" / f"{case['name']}_logistic_c{candidate}_f{fold}.json")
                    fit = arrays[f"fit_indices_{fold}"]
                    p = reference_probability(arrays[f"fit_{fold}"], arrays["y"][fit], arrays[f"val_{fold}"], params["C"])
                    error = float(np.max(np.abs(p-record["probabilities"])))
                    assert error < 1e-10, (case["name"], candidate, fold, error)
                    maximum = max(maximum, error)
                    checked += 1
            record = read_json(OUTPUT / "runs" / f"{case['name']}_logistic.json")
            c = GRIDS["logistic"][record["selected"]]["C"]
            p = reference_probability(arrays["train"], arrays["y"], arrays["test"], c)
            error = float(np.max(np.abs(p-record["probabilities"])))
            assert error < 1e-10, (case["name"], "final", error)
            maximum = max(maximum, error)
            checked += 1
    calibration_checks, calibration_maximum = 0, 0.
    for path in (OUTPUT / "runs").glob("*.json"):
        record = read_json(path)
        if not record.get("calibration_available", True):
            continue
        p = reference_probability(logit(record["oof_probabilities"]), np.array(record["training_y"]), logit(record["probabilities"]), 1.)
        error = float(np.max(np.abs(p-record["calibrated_probabilities"])))
        assert error < 1e-10, (path.name, "calibration", error)
        calibration_maximum = max(calibration_maximum, error)
        calibration_checks += 1
    result = dict(complete=True, protocol_id=protocol["protocol_id"],
                  logistic_fits_independently_reproduced=checked, maximum_probability_difference=maximum,
                  sigmoid_fits_independently_reproduced=calibration_checks, maximum_calibrated_probability_difference=calibration_maximum,
                  method="Reproducible sklearn refits; independent einsum probability and L2-objective gradient checks; gradient infinity norm below 1.1e-4",
                  probabilities_unchanged=True)
    write_json(OUTPUT / "numerical_verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
