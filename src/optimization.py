"""Training-only TabPFN selection and persisted prediction pipelines.

The original published feature tables and outcome definitions remain fixed.
No outcome, identifier, test membership, or event time is a model input.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common import load_token
from src.paper_protocol import FEATURE_FILES, VENDOR, literal_lists, setup_preprocessor

SEED = 20260921
THINKING_SECONDS = 60
CONFIGS = (
    dict(name="paper_default", preprocessing="paper", thinking=False, modalities="all"),
    dict(name="native_default", preprocessing="native", thinking=False, modalities="all"),
    dict(name="paper_thinking", preprocessing="paper", thinking=True, modalities="all"),
    dict(name="native_thinking", preprocessing="native", thinking=True, modalities="all"),
    dict(name="native_clinical_pathology", preprocessing="native", thinking=False, modalities="clinical_pathology"),
    dict(name="native_clinical_pathology_blood", preprocessing="native", thinking=False, modalities="clinical_pathology_blood"),
)


def feature_modalities():
    names = ("Clinical", "Pathology", "Blood", "ICD codes", "TMA cell densities")
    return {name: [c for c in pd.read_csv(VENDOR / "features" / filename, nrows=0).columns if c != "patient_id"]
            for name, filename in zip(names, FEATURE_FILES)}


def selected_columns(config):
    groups = feature_modalities()
    names = list(groups) if config["modalities"] == "all" else ["Clinical", "Pathology"]
    if config["modalities"] == "clinical_pathology_blood":
        names.append("Blood")
    return [col for name in names for col in groups[name]]


def native_categorical_indices(columns):
    lists = literal_lists(VENDOR / "feature_extraction/extract_tabular_features.py")
    categorical = set(lists["NOMINAL_FEATURES"] + lists["BINARY_FEATURES"] + feature_modalities()["ICD codes"])
    return [i for i, name in enumerate(columns) if name in categorical]


def authenticate():
    import tabpfn_client
    token = load_token()
    if not token:
        raise RuntimeError("TABPFN_TOKEN is required")
    tabpfn_client.set_access_token(token)


class OptimizationModel:
    """Accept published original columns, preserving one clinical feature per column."""
    def __init__(self, config, seed=42, cache=False):
        self.config = dict(config)
        self.seed = seed
        self.cache = cache
        self.columns = selected_columns(config)
        self.preprocessor = None

    def fit(self, frame, y):
        from tabpfn_client import TabPFNClassifier
        from tabpfn_client.constants import ModelVersion
        authenticate()
        x = frame[self.columns].astype(float)
        kwargs = dict(random_state=self.seed)
        if self.config["preprocessing"] == "paper":
            self.preprocessor = setup_preprocessor(self.columns)
            x = self.preprocessor.fit_transform(x)
        else:
            kwargs["categorical_features_indices"] = native_categorical_indices(self.columns)
        if self.config["thinking"]:
            kwargs.update(thinking_effort="medium", thinking_timeout_s=THINKING_SECONDS, thinking_metric="roc_auc")
        elif self.cache:
            kwargs["fit_mode"] = "fit_with_cache"
        self.estimator = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, **kwargs)
        self.estimator.fit(x, np.asarray(y, dtype=int))
        return self

    def predict(self, frame, batch_size=512):
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(np.asarray(frame), columns=self.columns)
        x = frame[self.columns].astype(float)
        if self.preprocessor is not None:
            x = self.preprocessor.transform(x)
        results = []
        for offset in range(0, len(x), batch_size):
            batch = x.iloc[offset:offset+batch_size] if isinstance(x, pd.DataFrame) else x[offset:offset+batch_size]
            results.append(self.estimator.predict_proba(batch)[:, 1])
        return np.concatenate(results) if results else np.empty(0)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.estimator.save_model(directory / "server_model.json")
        joblib.dump(dict(config=self.config, seed=self.seed, cache=self.cache, columns=self.columns,
                         preprocessor=self.preprocessor), directory / "preprocessing.joblib")

    @classmethod
    def load(cls, directory):
        from tabpfn_client import TabPFNClassifier
        authenticate()
        directory = Path(directory)
        # Only load locally generated trusted artifacts, never uploaded pickle files.
        state = joblib.load(directory / "preprocessing.joblib")
        model = cls(state["config"], state["seed"], state["cache"])
        model.columns, model.preprocessor = state["columns"], state["preprocessor"]
        model.estimator = TabPFNClassifier.load_model(directory / "server_model.json")
        return model


def rank_configurations(records):
    """Require all three folds; break exact AUC ties by lower Brier, then fixed order."""
    scores = []
    for index, config in enumerate(CONFIGS):
        rows = [r for r in records if r["config"] == config["name"] and r.get("status") == "complete"]
        if len(rows) != 3 or sorted(r["fold"] for r in rows) != [0, 1, 2]:
            continue
        scores.append(dict(config=config["name"], roc_auc=float(np.mean([r["roc_auc"] for r in rows])),
                           brier=float(np.mean([r["brier"] for r in rows])), order=index))
    if not scores:
        raise ValueError("No candidate has all three complete validation folds")
    return sorted(scores, key=lambda r: (-r["roc_auc"], r["brier"], r["order"]))
