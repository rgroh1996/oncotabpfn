"""Prespecified matched-label-budget study on the frozen HANCOCK paper tables."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.common import ROOT, RESULTS, file_hash, fingerprint, read_json, write_json
from src.paper_protocol import TARGETS, load_features, load_partition, provenance, setup_preprocessor
from src.pipeline import make_estimator, metric_scores, model_metadata

OUTPUT = RESULTS / "learning_curves"
SIZES = (50, 100, 200, 500, "Full")
SEEDS = tuple(range(42, 52))
MODELS = ("tabpfn", "rf", "rf_smote", "lightgbm", "xgboost", "logistic")
LABELS = {"tabpfn": "TabPFN-3.5", "rf": "Tuned RF", "rf_smote": "Tuned RF + SMOTE",
          "lightgbm": "Tuned LightGBM", "xgboost": "Tuned XGBoost", "logistic": "Tuned logistic"}
GRIDS = {
    "tabpfn": [{}],
    "rf": [dict(max_features="sqrt", min_samples_leaf=1, max_depth=None),
           dict(max_features="sqrt", min_samples_leaf=3, max_depth=None),
           dict(max_features=.5, min_samples_leaf=5, max_depth=8),
           dict(max_features="log2", min_samples_leaf=2, max_depth=12)],
    "lightgbm": [dict(num_leaves=7, min_child_samples=3, reg_lambda=1., learning_rate=.03),
                 dict(num_leaves=15, min_child_samples=5, reg_lambda=1., learning_rate=.05),
                 dict(num_leaves=7, min_child_samples=10, reg_lambda=10., learning_rate=.05),
                 dict(num_leaves=31, min_child_samples=5, reg_lambda=10., learning_rate=.03)],
    "xgboost": [dict(max_depth=2, min_child_weight=1, reg_lambda=1., learning_rate=.03),
                dict(max_depth=3, min_child_weight=1, reg_lambda=1., learning_rate=.05),
                dict(max_depth=2, min_child_weight=3, reg_lambda=10., learning_rate=.05),
                dict(max_depth=5, min_child_weight=3, reg_lambda=10., learning_rate=.03)],
    "logistic": [dict(C=c) for c in (.01, .1, 1., 10.)],
}
GRIDS["rf_smote"] = GRIDS["rf"]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def nested_order(y, seed):
    """Interleave shuffled classes to keep every prefix within one event of its quota."""
    y = np.asarray(y, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Two binary classes required")
    rng = np.random.default_rng(seed)
    queues = [rng.permutation(np.flatnonzero(y == c)) for c in (0, 1)]
    taken = [0, 0]
    order = []
    for length in range(1, len(y) + 1):
        want_positive = int(np.floor(length * y.mean() + .5))
        c = int(want_positive > taken[1])
        order.append(int(queues[c][taken[c]]))
        taken[c] += 1
    return np.asarray(order)


def folds_for(y, seed):
    return [(a, b) for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)]


def score_probabilities(y, p):
    from sklearn.metrics import log_loss
    scores = metric_scores(y, p)
    scores["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    return scores


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(oof, y, test):
    """Regularized sigmoid learned exclusively from this subset's OOF predictions."""
    estimator = LogisticRegression(C=1., solver="lbfgs", max_iter=2000).fit(logit(oof), y)
    return estimator.predict_proba(logit(test))[:, 1], {
        "slope": float(estimator.coef_[0, 0]), "intercept": float(estimator.intercept_[0]), "C": 1.}


def estimator_for(model, params, seed):
    if model == "tabpfn":
        return make_estimator("tabpfn", seed)
    if model in ("rf", "rf_smote"):
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed, **params)
    if model == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, verbosity=-1, n_jobs=1, random_state=seed, **params)
    if model == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, tree_method="hist", n_jobs=1, random_state=seed, **params)
    if model == "logistic":
        return LogisticRegression(max_iter=3000, solver="lbfgs", **params)
    raise ValueError(model)


def resample_training(x, y, seed):
    from imblearn.over_sampling import SMOTE
    minority = int(np.bincount(y, minlength=2).min())
    if minority < 2:
        raise ValueError("SMOTE requires two minority patients in each training fold")
    return SMOTE(random_state=seed, k_neighbors=min(5, minority - 1)).fit_resample(x, y)


def case_name(target, size, seed):
    return f"{target}_n{size}_s{seed}"


def prepare(output=OUTPUT):
    """Freeze all label budgets, folds, matrices and tuning choices before execution."""
    output = Path(output)
    source_files = ("src/learning_curve_protocol.py", "src/paper_protocol.py", "src/pipeline.py")
    settings = dict(schema=1, split="in", targets=TARGETS, sizes=SIZES, seeds=SEEDS,
                    models=MODELS, grids=GRIDS, inner_folds=3, forest_trees=300, boosting_trees=200,
                    primary="Mean raw ROC-AUC across N=50,100,200, equally weighted, then averaged across ten seeds",
                    calibration="C=1 sigmoid on selected candidate OOF predictions; same N-patient label budget; final estimator refit on all N",
                    selection="Highest mean three-fold validation AUC; ties use first prespecified candidate",
                    bootstrap="2000 paired stratified held-out patient resamples; conditional on these ten training draws; unadjusted exploratory intervals",
                    limitations=["Official test patients were evaluated in previous experiments; this is exploratory, not a new untouched test set.",
                                 "The published feature tables contain cohort-wide blood imputation and categorical encoding.",
                                 "Death status is not five-year overall survival; recurrence preserves the released code's living-patient exception.",
                                 "The no-SMOTE and SMOTE RF pipelines are tuned independently with the same grid; they are pipeline-level controls.",
                                 "Four candidates per baseline are a bounded search, not exhaustive optimization. TabPFN is untuned.",
                                 "Calibration reuses selected-candidate OOF scores; these training scores are not unbiased evaluation estimates.",
                                 "Full repeats share all training patients. Repetitions and sizes do not create independent test cohorts.",
                                 "Only the official in-distribution split is studied here. No external-validation claim is supported."],
                    provenance=provenance(), extra_packages={p: version(p) for p in ("lightgbm", "xgboost")},
                    implementation={p: file_hash(ROOT / p) for p in source_files})
    protocol_id = fingerprint(settings)
    path = output / "protocol.json"
    if path.exists():
        existing = read_json(path)
        if existing["protocol_id"] != protocol_id:
            raise ValueError("Frozen protocol changed; use a separate output directory")
        for name, digest in existing["input_hashes"].items():
            if file_hash(output / name) != digest:
                raise ValueError(f"Changed frozen input: {name}")
        return existing
    if (output / "runs").exists() and list((output / "runs").glob("*.json")):
        raise ValueError("Refusing to prepare over existing evaluations")
    features = load_features()
    cohorts, cases, hashes = {}, [], {}
    (output / "inputs").mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        train, test = load_partition(features, target, "in")
        columns = list(features.drop(columns="patient_id").columns)
        cohorts[target] = dict(training_ids=train.patient_id.tolist(), test_ids=test.patient_id.tolist(),
                               test_y=test.target.tolist(), features=columns)
        for seed in SEEDS:
            order = nested_order(train.target.to_numpy(), seed)
            for size in SIZES:
                n = len(train) if size == "Full" else size
                if n > len(train):
                    raise ValueError("Prespecified sample exceeds training pool")
                subset = train.iloc[order[:n]].reset_index(drop=True)
                y = subset.target.to_numpy(dtype=int)
                name = case_name(target, size, seed)
                arrays = {"y": y, "test_y": test.target.to_numpy(dtype=int)}
                preprocessor = setup_preprocessor(columns)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
                    arrays["train"] = preprocessor.fit_transform(subset[columns])
                    arrays["test"] = preprocessor.transform(test[columns])
                    for fold, (fit, val) in enumerate(folds_for(y, seed)):
                        prep = setup_preprocessor(columns)
                        arrays[f"fit_{fold}"] = prep.fit_transform(subset.iloc[fit][columns])
                        arrays[f"val_{fold}"] = prep.transform(subset.iloc[val][columns])
                        arrays[f"fit_indices_{fold}"] = fit
                        arrays[f"val_indices_{fold}"] = val
                for key, value in arrays.items():
                    if not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite prepared input: {name}/{key}")
                relative = f"inputs/{name}.npz"
                np.savez_compressed(output / relative, **arrays)
                hashes[relative] = file_hash(output / relative)
                cases.append(dict(name=name, target=target, size=str(size), n=n, seed=seed,
                                  training_ids=subset.patient_id.tolist(), test_ids=test.patient_id.tolist(),
                                  inner_folds=[dict(fit_indices=a.tolist(), val_indices=b.tolist()) for a, b in folds_for(y, seed)]))
    protocol = dict(**settings, protocol_id=protocol_id, prepared_utc=utcnow(), cohorts=cohorts,
                    cases=cases, input_hashes=hashes, expected_runs=len(cases) * len(MODELS))
    write_json(path, protocol)
    return protocol


def run_case(output, case, model, protocol_id):
    output = Path(output)
    path = output / "runs" / f"{case['name']}_{model}.json"
    if path.exists():
        record = read_json(path)
        if record["protocol_id"] != protocol_id or record["status"] != "complete":
            raise ValueError("Invalid cached result")
        return path.name
    started = time.perf_counter()
    started_utc = utcnow()
    with np.load(output / "inputs" / f"{case['name']}.npz", allow_pickle=False) as inputs:
        data = dict(inputs)
    y = data["y"]
    scores, oofs = [], []
    with threadpool_limits(limits=1):
        for candidate, params in enumerate(GRIDS[model]):
            oof = np.full(len(y), np.nan)
            fold_scores = []
            for fold in range(3):
                fit, val = data[f"fit_indices_{fold}"], data[f"val_indices_{fold}"]
                cache = output / "inner_cache" / f"{case['name']}_{model}_c{candidate}_f{fold}.json"
                if cache.exists():
                    saved = read_json(cache)
                    if saved["protocol_id"] != protocol_id or saved["validation_indices"] != val.tolist():
                        raise ValueError("Invalid inner cache")
                    p = np.asarray(saved["probabilities"])
                else:
                    xfit, yfit = data[f"fit_{fold}"], y[fit]
                    if model == "rf_smote":
                        xfit, yfit = resample_training(xfit, yfit, case["seed"] + fold)
                    estimator = estimator_for(model, params, case["seed"] + fold)
                    estimator.fit(xfit, yfit)
                    p = estimator.predict_proba(data[f"val_{fold}"])[:, 1]
                    score_probabilities(y[val], p)  # Reject malformed API predictions before caching.
                    write_json(cache, dict(protocol_id=protocol_id, validation_indices=val.tolist(),
                                           probabilities=p.tolist(), completed_utc=utcnow()))
                oof[val] = p
                fold_scores.append(float(roc_auc_score(y[val], p)))
            if not np.isfinite(oof).all():
                raise ValueError("OOF predictions do not cover the complete label budget")
            scores.append(dict(candidate=candidate, parameters=params, fold_auc=fold_scores,
                               mean_auc=float(np.mean(fold_scores))))
            oofs.append(oof)
        selected = int(np.argmax([s["mean_auc"] for s in scores]))
        selection_utc = utcnow()
        xfit, yfit = data["train"], y
        if model == "rf_smote":
            xfit, yfit = resample_training(xfit, yfit, case["seed"])
        estimator = estimator_for(model, GRIDS[model][selected], case["seed"])
        estimator.fit(xfit, yfit)
        p = estimator.predict_proba(data["test"])[:, 1]
        calibrated, calibration = calibrate(oofs[selected], y, p)
    metadata = model_metadata(estimator)
    if "resolved_config" in metadata:
        metadata["requested_config"] = metadata.pop("resolved_config")
    record = dict(protocol_id=protocol_id, status="complete", case=case["name"],
                  target=case["target"], size=case["size"], n=case["n"], seed=case["seed"], model=model,
                  started_utc=started_utc, selection_utc=selection_utc, completed_utc=utcnow(),
                  elapsed_seconds=time.perf_counter()-started, candidates=scores, selected=selected,
                  training_ids=case["training_ids"], patient_ids=case["test_ids"],
                  oof_probabilities=oofs[selected].tolist(), training_y=y.tolist(),
                  y=data["test_y"].tolist(), probabilities=p.tolist(), calibrated_probabilities=calibrated.tolist(),
                  raw_metrics=score_probabilities(data["test_y"], p),
                  calibrated_metrics=score_probabilities(data["test_y"], calibrated), calibration=calibration,
                  estimator=metadata)
    write_json(path, record)
    return path.name
