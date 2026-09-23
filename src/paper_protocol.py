"""HANCOCK v1.0.0 Figure 2 protocol, adapted from the authors' Apache-2.0 code.

Original: ankilab/HANCOCK_MultimodalDataset, commit
521b99b03a94008b28df5c3df4aa5f82aa14b25a. See vendor/.../LICENSE.
Changes: omit visualization-only UMAP; add validation and explicit RNG arguments.
"""
from __future__ import annotations

import ast
from importlib.metadata import version
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import DATA, ROOT, file_hash, read_json

COMMIT = "521b99b03a94008b28df5c3df4aa5f82aa14b25a"
VENDOR = ROOT / "vendor/HANCOCK_MultimodalDataset"
FEATURE_FILES = ("clinical.csv", "pathological.csv", "blood.csv", "icd_codes.csv", "tma_cell_density.csv")
SPLITS = ("in", "out", "Oropharynx")
TARGETS = ("survival_status", "recurrence")
SPLIT_LABELS = dict(zip(SPLITS, ("In distribution", "Out of distribution", "Oropharynx")))
PUBLISHED_AUC = {"survival_status": (.79, .78, .71), "recurrence": (.79, .71, .69)}
PINNED = {"numpy": "2.0.2", "pandas": "2.2.3", "scikit-learn": "1.5.1", "imbalanced-learn": "0.12.3"}


def literal_lists(path: Path) -> dict:
    """Read published feature order without importing plotting/UMAP dependencies."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = ast.literal_eval(node.value)
    return out


def provenance() -> dict:
    revision = subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Expected frozen HANCOCK commit {COMMIT}, found {revision}")
    paths = [VENDOR / "features" / name for name in (*FEATURE_FILES, "targets.csv")]
    paths += [VENDOR / name for name in (
        "feature_extraction/extract_tabular_features.py", "feature_extraction/extract_tma_features.py",
        "data_exploration/umap_embedding.py", "multimodal_machine_learning/execution/outcome_prediction.py",
    )]
    for path in paths:
        original = subprocess.check_output(["git", "-C", str(VENDOR), "show", f"{COMMIT}:{path.relative_to(VENDOR)}"])
        if path.read_bytes() != original:
            raise ValueError(f"Modified paper source: {path.name}")
    manifest = read_json(DATA / "paper/manifest.json")
    if manifest["source"] != "official":
        raise ValueError("Paper replication requires official data")
    for split in SPLITS:
        path = DATA / "paper" / f"dataset_split_{split}.json"
        if file_hash(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Changed official split: {split}")
        paths.append(path)
    packages = {name: version(name) for name in (*PINNED, "scipy", "tabpfn-client", "matplotlib", "joblib", "threadpoolctl")}
    for name, expected in PINNED.items():
        if packages[name] != expected:
            raise ValueError(f"Use .venv-paper: {name} must be {expected}, got {packages[name]}")
    return {"commit": revision, "packages": packages,
            "files": {str(p.relative_to(ROOT)): file_hash(p) for p in paths}}


def setup_preprocessor(columns):
    lists = literal_lists(VENDOR / "feature_extraction/extract_tabular_features.py")
    tma = literal_lists(VENDOR / "feature_extraction/extract_tma_features.py")
    categorical = [c for c in lists["NOMINAL_FEATURES"] if c in columns]
    numeric = [c for c in lists["BLOOD_FEATURES"] + tma["TMA_FEATURES"] + lists["DISCRETE_FEATURES"] + lists["ORDINAL_FEATURES"] if c in columns]
    remaining = [c for c in columns if c not in categorical and c not in numeric]
    return ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]), categorical),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), numeric),
        ("encoded", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), remaining),
    ], remainder="passthrough", verbose=False)


def eligibility(frame, target):
    if target == "recurrence":
        # Intentionally preserve the released code's living-patient exception.
        return ((frame.recurrence == "yes") & (frame.days_to_recurrence <= 365 * 3)) | (
            (frame.recurrence == "no") & ((frame.days_to_last_information > 365 * 3) | (frame.survival_status == "living")))
    if target == "survival_status":
        return frame.survival_status_with_cause != "deceased not tumor specific"
    raise ValueError(target)


def load_features():
    frames = [pd.read_csv(VENDOR / "features" / name, dtype={"patient_id": str}) for name in FEATURE_FILES]
    for frame in frames:
        if frame.patient_id.duplicated().any():
            raise ValueError("Duplicate modality patient")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="patient_id", how="outer", validate="one_to_one")
    return merged.reset_index(drop=True)


def load_partition(features, target, split):
    assignments = pd.read_json(DATA / "paper" / f"dataset_split_{split}.json", dtype={"patient_id": str})[["patient_id", "dataset"]]
    targets = pd.read_csv(VENDOR / "features/targets.csv", dtype={"patient_id": str})
    if assignments.patient_id.duplicated().any() or set(assignments.dataset) != {"training", "test"}:
        raise ValueError("Invalid official split")
    if set(assignments.patient_id) != set(targets.patient_id) or len(targets) != 763:
        raise ValueError("Incomplete official cohort")
    joined = assignments.merge(targets, on="patient_id", how="inner", validate="one_to_one")
    selected = joined.loc[eligibility(joined, target)].copy()
    selected[target] = selected[target].map({"no": 0, "yes": 1} if target == "recurrence" else {"living": 0, "deceased": 1})
    if selected[target].isna().any():
        raise ValueError("Unknown target label")
    partitions = []
    for subset in ("training", "test"):
        frame = selected.loc[selected.dataset.eq(subset), ["patient_id", target]].rename(columns={target: "target"})
        frame = frame.merge(features, on="patient_id", how="inner", validate="one_to_one")
        if frame.target.nunique() != 2:
            raise ValueError("Both outcome classes required")
        partitions.append(frame)
    train, test = partitions
    if set(train.patient_id) & set(test.patient_id):
        raise ValueError("Patient overlap")
    if len(train) + len(test) != len(selected):
        raise ValueError("Patients lost during feature join")
    return train, test


def published_forest(target, split, random_state):
    params = dict(n_estimators=1200, min_samples_split=2, min_samples_leaf=1,
                  max_leaf_nodes=1000, max_features="sqrt", max_depth=20, criterion="gini")
    if target == "recurrence" and split == "in":
        params.update(n_estimators=1600, max_features="log2", max_depth=30)
    elif target == "recurrence" and split == "out":
        params.update(n_estimators=800, max_leaf_nodes=100, max_depth=80, criterion="log_loss")
    elif target == "survival_status" and split == "in":
        params.update(n_estimators=1000, min_samples_split=5, max_features="log2", max_depth=None, criterion="entropy")
    if target not in TARGETS or split not in SPLITS:
        raise ValueError("Unsupported endpoint or split")
    return RandomForestClassifier(**params, random_state=random_state)


def paired_auc_bootstrap(y, candidate, reference, samples=2000, seed=20260921):
    """Resample held-out patients, keeping all five repetitions paired.

    AUC equals mean pairwise ranking between positive and negative patients.
    Multinomial class-stratified weights avoid 20k repeated sklearn metric calls.
    """
    y = np.asarray(y, dtype=int)
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[1] != len(y):
        raise ValueError("Expected aligned repetitions x patients")
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both classes required")
    def rankings(probabilities):
        differences = probabilities[:, positive, None] - probabilities[:, None, negative]
        return ((differences > 0) + .5 * (differences == 0)).mean(axis=0)
    delta = rankings(candidate) - rankings(reference)
    rng = np.random.default_rng(seed)
    pos_weights = rng.multinomial(len(positive), np.full(len(positive), 1 / len(positive)), size=samples) / len(positive)
    neg_weights = rng.multinomial(len(negative), np.full(len(negative), 1 / len(negative)), size=samples) / len(negative)
    differences = np.einsum("bi,ij,bj->b", pos_weights, delta, neg_weights, optimize=True)
    low, high = np.quantile(differences, [.025, .975])
    return {"auc_difference": float(delta.mean()), "ci_low": float(low), "ci_high": float(high),
            "bootstrap_samples": samples, "bootstrap_seed": seed}
