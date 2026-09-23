"""Leakage-resistant preprocessing, model adapters and calibrated research inference."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.dataset_builder import NUMERIC_FEATURES, CATEGORICAL_FEATURES, engineer_biomarkers
from src.common import RESULTS, load_token, write_json

MODELS = ("tabpfn", "lightgbm", "xgboost")


class ModelUnavailable(RuntimeError):
    pass


def make_estimator(name: str, seed: int = 42):
    if name == "tabpfn":
        token = load_token()
        if not token:
            raise ModelUnavailable("Set TABPFN_TOKEN in .env or Streamlit secrets to enable TabPFN-3.5 API inference.")
        import tabpfn_client
        from tabpfn_client import TabPFNClassifier
        from tabpfn_client.constants import ModelVersion
        tabpfn_client.set_access_token(token)
        if not hasattr(ModelVersion, "V3_5") or not hasattr(TabPFNClassifier, "create_default_for_version"):
            raise ModelUnavailable("Upgrade tabpfn-client: this installation cannot explicitly select TabPFN-3.5.")
        return TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, random_state=seed)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=2, verbosity=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=2)
    raise ValueError(f"Unknown model: {name}")


def model_metadata(estimator) -> dict:
    metadata = {"class": type(estimator).__name__, "parameters": estimator.get_params(deep=False)}
    # The SDK may resolve defaults on the server; record available fitted config too.
    config = estimator._get_tabpfn_config() if hasattr(estimator, "_get_tabpfn_config") else None
    if config is not None:
        metadata["resolved_config"] = config.model_dump() if hasattr(config, "model_dump") else str(config)
    return metadata


def package_versions() -> dict:
    return {p: version(p) for p in ("tabpfn-client", "lightgbm", "xgboost", "scikit-learn", "pandas", "numpy")}


class FeaturePreprocessor:
    """Fit on one context only. IDs/outcomes are never passed to an estimator."""
    def __init__(self, pca_components: int = 16):
        if pca_components not in {16, 32}:
            raise ValueError("PCA components must be 16 or 32")
        self.pca_components = pca_components

    def _frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = engineer_biomarkers(frame)
        out = pd.DataFrame(index=frame.index)
        for c in self.numeric:
            out[c] = pd.to_numeric(frame[c], errors="coerce") if c in frame else np.nan
        for c in self.categorical:
            out[c] = frame[c].fillna("Unknown").astype(str) if c in frame else "Unknown"
        return out.replace([np.inf, -np.inf], np.nan)

    def fit(self, frame: pd.DataFrame):
        engineered = engineer_biomarkers(frame)
        self.numeric = [c for c in NUMERIC_FEATURES if c in engineered and engineered[c].notna().any()]
        self.categorical = [c for c in CATEGORICAL_FEATURES if c in frame and frame[c].notna().any()]
        transformers = []
        if self.numeric:
            transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), self.numeric))
        if self.categorical:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical))
        if not transformers:
            raise ValueError("Training context has no usable predictors")
        self.tabular = ColumnTransformer(transformers)
        self.tabular.fit(self._frame(frame))
        self.embedding_columns = sorted(c for c in frame if c.startswith("embedding_"))
        self.pca = None
        if self.embedding_columns:
            matrix = frame[self.embedding_columns].to_numpy(dtype=float)
            available = np.isfinite(matrix).any(axis=1)
            valid_columns = np.isfinite(matrix[available]).any(axis=0)
            self.embedding_columns = [c for c, valid in zip(self.embedding_columns, valid_columns) if valid]
            if available.sum() >= 2 and self.embedding_columns:
                matrix = frame.loc[available, self.embedding_columns]
                self.embedding_imputer = SimpleImputer(strategy="median")
                matrix = self.embedding_imputer.fit_transform(matrix)
                self.embedding_scaler = StandardScaler()
                matrix = self.embedding_scaler.fit_transform(matrix)
                rank = int(np.linalg.matrix_rank(matrix))
                k = min(self.pca_components, len(matrix)-1, matrix.shape[1], rank)
                if k:
                    self.pca = PCA(n_components=k, svd_solver="full").fit(matrix)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.tabular.transform(self._frame(frame))
        if self.pca is not None:
            values = frame.reindex(columns=self.embedding_columns).astype(float)
            available = values.notna().any(axis=1).to_numpy()
            embedded = self.embedding_imputer.transform(values)
            embedded = self.pca.transform(self.embedding_scaler.transform(embedded))
            embedded[~available] = 0
            matrix = np.column_stack([matrix, embedded, available.astype(float)])
        return np.asarray(matrix, dtype=np.float64)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


class PatientModel:
    def __init__(self, name="tabpfn", seed=42, pca_components=16):
        self.preprocessor = FeaturePreprocessor(pca_components)
        self.estimator = make_estimator(name, seed)
        self.name = name

    def fit(self, frame: pd.DataFrame, y):
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Both outcome classes are required in the training context")
        self.estimator.fit(pd.DataFrame(self.preprocessor.fit_transform(frame)), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.estimator.predict_proba(pd.DataFrame(self.preprocessor.transform(frame))))
        index = list(self.estimator.classes_).index(1)
        positive = probabilities[:, index]
        if not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
            raise ValueError("Model returned invalid probabilities")
        return positive


def calibration_bins(y, probability, bins=10) -> list[dict]:
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    indices = np.minimum((probability * bins).astype(int), bins-1)
    return [dict(bin=i, count=int((indices == i).sum()),
                 probability=float(probability[indices == i].mean()),
                 observed=float(y[indices == i].mean())) for i in range(bins) if (indices == i).any()]


def metric_scores(y, probability) -> dict:
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if len(y) != len(probability) or not len(y) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Metrics require aligned finite probabilities in [0, 1]")
    both = len(np.unique(y)) == 2
    bins = calibration_bins(y, probability)
    return dict(roc_auc=float(roc_auc_score(y, probability)) if both else None,
                average_precision=float(average_precision_score(y, probability)) if both else None,
                brier=float(brier_score_loss(y, probability)),
                ece=float(sum(b["count"] * abs(b["probability"]-b["observed"]) for b in bins) / len(y)))


def evaluation_splits(df: pd.DataFrame, target: str, mode="official", seed=42):
    eligible = df.loc[df[target].notna()].copy()
    if len(eligible) == 0:
        raise ValueError(f"No eligible patients for {target}")
    if mode in {"official", "out"}:
        column = "split_out" if mode == "out" else "split_in"
        if column not in eligible or not eligible[column].isin(["training", "test"]).all():
            raise ValueError("A complete valid official split is required; select --split cv explicitly if absent")
        train = eligible.loc[eligible[column].eq("training")]
        test = eligible.loc[eligible[column].eq("test")]
        if not len(train) or not len(test):
            raise ValueError("The endpoint has an empty training or test partition")
        yield 0, train, test
    elif mode == "cv":
        if eligible[target].value_counts().min() < 5 or eligible[target].nunique() < 2:
            raise ValueError("Five-fold stratification requires at least five patients per outcome class")
        for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(eligible, eligible[target])):
            yield fold, eligible.iloc[train], eligible.iloc[test]
    else:
        raise ValueError(f"Unknown split mode {mode}")


def stratified_order(frame: pd.DataFrame, target: str, seed=42) -> list:
    """One ordered stream: every prefix preserves approximate prevalence and nesting."""
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(frame.index[frame[target].eq(c)])) for c in (0, 1)]
    if not all(queues):
        raise ValueError("Training partition must contain both classes")
    total = np.array([len(q) for q in queues])
    used = np.zeros(2, dtype=int)
    ordered = []
    for step in range(len(frame)):
        deficit = (step + 1) * total / total.sum() - used
        deficit[used == total] = -np.inf
        cls = int(np.argmax(deficit))
        ordered.append(queues[cls][used[cls]])
        used[cls] += 1
    return ordered


class SigmoidCalibration:
    def fit(self, p, y):
        if len(np.unique(y)) < 2:
            raise ValueError("Probability calibration requires both classes")
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(self._logit(p), y)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p), 1e-6, 1-1e-6)
        return np.log(p / (1-p)).reshape(-1, 1)

    def predict(self, p):
        return self.model.predict_proba(self._logit(p))[:, list(self.model.classes_).index(1)]


class ConformalClassifier:
    def __init__(self, alpha=.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def fit(self, p, y):
        p, y = np.asarray(p), np.asarray(y, dtype=int)
        if not len(y):
            raise ValueError("A nonempty independent conformal calibration set is required")
        scores = np.where(y == 1, 1-p, p)
        rank = math.ceil((len(y)+1)*(1-self.alpha))
        self.threshold = float(np.sort(scores)[rank-1]) if rank <= len(y) else float("inf")
        self.n_calibration = len(y)
        return self

    def predict(self, p):
        p = np.asarray(p)
        return np.column_stack([p <= self.threshold, (1-p) <= self.threshold])


def dashboard_partition(train: pd.DataFrame, seed=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Random, label-independent partition preserves exchangeability for conformal calibration.
    context, calibration = train_test_split(train, test_size=.4, random_state=seed)
    probability, conformal = train_test_split(calibration, test_size=.5, random_state=seed+1)
    return context, probability, conformal


@dataclass
class DashboardModel:
    target: str
    model: PatientModel
    calibrator: SigmoidCalibration
    conformal: ConformalClassifier
    test_ids: list[str]
    evaluation: dict

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.calibrator.predict(self.model.predict_proba(frame))
        return p, self.conformal.predict(p)


def fit_dashboard_model(df: pd.DataFrame, target: str, model_name="tabpfn", seed=42,
                        pca_components=16, artifact_dir: Path | None = RESULTS / "dashboard") -> DashboardModel:
    _, train, test = next(evaluation_splits(df, target))
    context, probability, conformal_frame = dashboard_partition(train, seed)
    model = PatientModel(model_name, seed, pca_components).fit(context, context[target])
    combined = pd.concat([probability, conformal_frame, test], ignore_index=True)
    raw = model.predict_proba(combined)
    n_p, n_c = len(probability), len(conformal_frame)
    calibrator = SigmoidCalibration().fit(raw[:n_p], probability[target])
    p = calibrator.predict(raw[n_p:])
    conformal = ConformalClassifier().fit(p[:n_c], conformal_frame[target])
    sets = conformal.predict(p[n_c:])
    y_test = test[target].to_numpy(dtype=int)
    evaluation = dict(target=target, source=str(df.data_source.iloc[0]), model=model_name,
                      counts={"context": len(context), "probability_calibration": n_p, "conformal_calibration": n_c, "test": len(test)},
                      partitions={name: part.patient_id.tolist() for name, part in
                                  [("context", context), ("probability_calibration", probability), ("conformal_calibration", conformal_frame), ("test", test)]},
                      native_metrics=metric_scores(y_test, raw[n_p+n_c:]), calibrated_metrics=metric_scores(y_test, p[n_c:]),
                      calibration_bins=calibration_bins(y_test, p[n_c:]),
                      conformal_coverage=float(sets[np.arange(len(y_test)), y_test].mean()),
                      conformal_mean_set_size=float(sets.sum(axis=1).mean()), nominal_coverage=.9,
                      model_metadata=model_metadata(model.estimator), versions=package_versions())
    if artifact_dir is not None:
        write_json(artifact_dir / f"{target}_{model_name}.json", evaluation)
    return DashboardModel(target, model, calibrator, conformal, test.patient_id.tolist(), evaluation)
