"""Build a patient table without fitting any data-dependent preprocessing."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.common import DATA, TARGETS, file_hash, read_json, write_json

HORIZON_DAYS = 5 * 365.25
# Explicit allowlist: no event, treatment or follow-up fields can become features.
LAB_SPECS = {
    "Leukocytes": ("leukocytes", "count"), "Lymphocytes": ("lymphocytes", "count"),
    "Neutrophils": ("neutrophils", "count"), "Granulocytes": ("granulocytes", "count"),
    "Platelets": ("platelets", "count"), "Thrombocytes": ("platelets", "count"),
    "Monocytes": ("monocytes", "count"), "Eosinophils": ("eosinophils", "count"),
    "Basophils": ("basophils", "count"), "CRP": ("crp", "mg/l"),
    "Hemoglobin": ("hemoglobin", "g/dl"), "Hematocrit": ("hematocrit", "%"),
    "Erythrocytes": ("erythrocytes", "red_count"),
    "Sodium": ("sodium", "mmol/l"), "Potassium": ("potassium", "mmol/l"),
    "Calcium": ("calcium", "mmol/l"), "Chloride": ("chloride", "mmol/l"),
    "Magnesium": ("magnesium", "mmol/l"), "Creatinine": ("creatinine", "mg/dl"),
    "Glucose": ("glucose", "mg/dl"), "Urea": ("urea", "mg/dl"),
    "INR": ("inr", "ratio"), "PT": ("pt", "%"), "aPPT": ("aptt", "s"),
    "Thrombin time": ("thrombin_time", "s"), "MCV": ("mcv", "fl"),
    "MCH": ("mch", "pg"), "MHCH": ("mchc", "g/dl"),
    "RDW": ("rdw", "%"), "MPV": ("mpv", "fl"), "PDW": ("pdw", "fl"),
    "PLCR": ("plcr", "%"), "Glomerular filtration rate": ("egfr", "ml/min"),
}
LAB_COLUMNS = list(dict.fromkeys(v[0] for v in LAB_SPECS.values()))
NUMERIC_FEATURES = ["age", "smoking_pack_years", *LAB_COLUMNS, "nlr", "plr", "sii", "crp_elevated",
                    "blood_available", "pathology_available"]
CATEGORICAL_FEATURES = ["sex", "smoking_status", "alcohol_consumption", "primary_tumor_site",
                        "t_stage", "n_stage", "m_stage", "grade", "hpv_p16"]


def normalize_grade(value) -> str:
    """Official HANCOCK codes HPV-associated ungraded tumors as hpv_association_p16."""
    if value in ("hpv_association_p16", "HPV_OSCC"):
        return "HPV_OSCC"
    return value if value in ("G1", "G2", "G3") else "Unknown"


def normalize_stage(value, prefix: str) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).upper().strip()
    if prefix == "T" and "TIS" in text:
        return "Tis"
    match = re.fullmatch(rf"[PCY]*{prefix}([0-4])[ABC]?", text)
    if match and int(match[1]) <= (4 if prefix == "T" else 3):
        return prefix + match[1]
    return "Unknown"


def convert_lab(value, unit, kind: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(value) or value < 0:
        return np.nan
    normalized = "" if pd.isna(unit) else str(unit).lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    if kind == "count":
        factor = {"x10^3/ul": 1, "10^3/ul": 1, "10^9/l": 1, "x10^9/l": 1, "/ul": .001}.get(normalized)
    elif kind == "red_count":
        factor = {"x10^6/ul": 1, "10^6/ul": 1, "10^12/l": 1}.get(normalized)
    elif kind == "ratio":
        factor = 1 if normalized in {"", "1", "ratio"} else None
    else:
        factor = 1 if normalized == kind else None
        if kind == "mg/l" and normalized == "mg/dl":
            factor = 10
        if kind == "g/dl" and normalized == "g/l":
            factor = .1
    return value * factor if factor is not None else np.nan


def aggregate_blood(records: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    accepted, rejected = [], 0
    for row in records.to_dict("records"):
        spec = LAB_SPECS.get(row.get("analyte_name"))
        if spec is None:
            continue
        day = pd.to_numeric(row.get("days_before_first_treatment"), errors="coerce")
        value = convert_lab(row.get("value"), row.get("unit"), spec[1])
        if not pd.notna(day) or not 0 <= day <= 14 or pd.isna(value):
            rejected += 1
            continue
        accepted.append(dict(patient_id=row["patient_id"], analyte=spec[0], day=day, value=value))
    if not accepted:
        return pd.DataFrame(columns=["patient_id", *LAB_COLUMNS]), {"rejected_measurements": rejected}
    table = pd.DataFrame(accepted)
    table = table.groupby(["patient_id", "analyte", "day"], as_index=False).value.median()
    nearest = table.sort_values("day").drop_duplicates(["patient_id", "analyte"])
    values = nearest.pivot(index="patient_id", columns="analyte", values="value")
    days = nearest.pivot(index="patient_id", columns="analyte", values="day").add_suffix("_days_before_treatment")
    return values.join(days).reset_index(), {"rejected_measurements": rejected, "accepted_measurements": len(accepted)}


def engineer_biomarkers(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name in ("lymphocytes", "neutrophils", "platelets", "crp", "crp_reference_upper"):
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    denominator = frame.lymphocytes.where(frame.lymphocytes > 0)
    frame["nlr"] = frame.neutrophils.where(frame.neutrophils >= 0) / denominator
    frame["plr"] = frame.platelets.where(frame.platelets >= 0) / denominator
    frame["sii"] = frame.platelets.where(frame.platelets >= 0) * frame.neutrophils.where(frame.neutrophils >= 0) / denominator
    frame["crp_elevated"] = np.where(frame.crp.notna() & frame.crp_reference_upper.notna(),
                                     (frame.crp > frame.crp_reference_upper).astype(float), np.nan)
    for name in ("nlr", "plr", "sii"):
        frame[name] = frame[name].replace([np.inf, -np.inf], np.nan)
    return frame


def make_targets(clinical: pd.DataFrame) -> pd.DataFrame:
    days = pd.to_numeric(clinical.days_to_last_information, errors="coerce")
    status = clinical.survival_status
    known = status.isin(["living", "deceased"]) & days.ge(0)
    death = known & status.eq("deceased") & days.le(HORIZON_DAYS)
    survived = known & days.ge(HORIZON_DAYS) & ~death
    mortality = pd.Series(np.nan, index=clinical.index)
    mortality.loc[death] = 1
    mortality.loc[survived] = 0
    recurrence = clinical.recurrence.map({"yes": 1., "no": 0.})
    return pd.DataFrame({"target_survival_5y": mortality, "target_recurrence": recurrence}, index=clinical.index)


def build_dataset(raw_dir: Path = DATA / "raw", output: Path = DATA / "hancock_processed.parquet",
                  embeddings: Path | None = None) -> tuple[pd.DataFrame, dict]:
    from data.download_hancock import REQUIRED, validate_payload
    payload = {name: (raw_dir / name).read_bytes() for name in REQUIRED}
    validate_payload(payload)
    manifest = read_json(raw_dir / "manifest.json")
    for name in REQUIRED:
        if file_hash(raw_dir / name) != manifest["files"].get(name):
            raise ValueError(f"Raw source changed: {name}; refresh acquisition before rebuilding")
    clinical = pd.DataFrame(read_json(raw_dir / "clinical_data.json"))
    pathological = pd.DataFrame(read_json(raw_dir / "pathological_data.json"))
    blood = pd.DataFrame(read_json(raw_dir / "blood_data.json"))
    values, blood_report = aggregate_blood(blood)
    df = clinical[["patient_id", "days_to_last_information", "survival_status", "recurrence", "days_to_recurrence"]].copy()
    df["age"] = pd.to_numeric(clinical.age_at_initial_diagnosis, errors="coerce")
    for column in ("sex", "smoking_status"):
        df[column] = clinical[column]
    df["m_stage"] = clinical.primarily_metastasis.map({"no": "M0", "yes": "M1"}).fillna("Unknown")
    df["smoking_pack_years"] = np.nan
    df["alcohol_consumption"] = pd.Series(None, index=df.index, dtype=object)
    path = pathological.set_index("patient_id")
    path["t_stage"] = path.pT_stage.map(lambda v: normalize_stage(v, "T"))
    path["n_stage"] = path.pN_stage.map(lambda v: normalize_stage(v, "N"))
    grading = path["grading"] if "grading" in path else path["grading_hpv"]
    path["grade"] = grading.map(normalize_grade)
    path["hpv_p16"] = path.hpv_association_p16.fillna("not_tested")
    path["primary_tumor_site"] = path.primary_tumor_site.replace({"Oral_Cavity": "Oral cavity"})
    path["pathology_available"] = 1.
    df = df.join(path[["primary_tumor_site", "t_stage", "n_stage", "grade", "hpv_p16", "pathology_available"]], on="patient_id", validate="one_to_one")
    df = df.merge(values, on="patient_id", how="left", validate="one_to_one")
    for column in LAB_COLUMNS:
        if column not in df:
            df[column] = np.nan
    df["blood_available"] = df[LAB_COLUMNS].notna().any(axis=1).astype(float)
    df["pathology_available"] = df.pathology_available.fillna(0.)
    references = read_json(raw_dir / "blood_data_reference_ranges.json")
    crp = next((r for r in references if r["analyte_name"] == "CRP"), {})
    df["crp_reference_upper"] = [convert_lab(crp.get(f"normal_{sex}_max"), crp.get("unit"), "mg/l")
                                  for sex in df.sex]
    df = engineer_biomarkers(df)
    targets = make_targets(clinical)
    for target in TARGETS:
        df[target] = targets[target].to_numpy()
        df[f"{target}_eligible"] = df[target].notna()
    for suffix in ("in", "out"):
        splits = pd.DataFrame(read_json(raw_dir / f"dataset_split_{suffix}.json")).set_index("patient_id")
        df[f"split_{suffix}"] = df.patient_id.map(splits.dataset)
    df["data_source"] = manifest["source"]
    embedding_info = None
    if embeddings is not None and embeddings.exists():
        vectors = pd.read_parquet(embeddings)
        if "patient_id" not in vectors or vectors.patient_id.duplicated().any():
            raise ValueError("Embeddings require unique patient_id values")
        if not vectors.patient_id.map(lambda x: isinstance(x, str)).all():
            raise ValueError("Embedding identifiers must be strings, preserving leading zeros")
        if not set(vectors.patient_id) <= set(df.patient_id):
            raise ValueError("Embeddings contain identifiers outside this cohort")
        columns = [c for c in vectors if c != "patient_id"]
        if not columns or not all(pd.api.types.is_numeric_dtype(vectors[c]) for c in columns):
            raise ValueError("Embedding columns must all be numeric")
        if np.isinf(vectors[columns].to_numpy()).any():
            raise ValueError("Embeddings contain infinite values")
        vectors = vectors.rename(columns={c: f"embedding_{i:04d}" for i, c in enumerate(columns)})
        df = df.merge(vectors, on="patient_id", how="left", validate="one_to_one")
        embedding_info = {"sha256": file_hash(embeddings), "dimensions": len(columns)}
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    report = dict(schema_version=1, source=manifest["source"], rows=len(df), raw_manifest_sha256=file_hash(raw_dir / "manifest.json"),
                  processed_sha256=file_hash(output), embeddings=embedding_info, blood=blood_report,
                  endpoint_counts={t: {"eligible": int(df[t].notna().sum()), "positive": int(df[t].eq(1).sum()),
                                        "negative": int(df[t].eq(0).sum()), "excluded": int(df[t].isna().sum())} for t in TARGETS},
                  missing_fraction={c: float(df[c].isna().mean()) for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES},
                  unsupported_source_fields=["smoking_pack_years", "alcohol_consumption", "neutrophils", "high_sensitivity_crp"],
                  notes=["NLR/SII require true neutrophils; granulocytes are not substituted.",
                         "Laboratory ratios may combine nearest measurements from different days in the 14-day window.",
                         "Five-year labels exclude early censoring and may be subject to selection bias.",
                         "Pathological features imply a postoperative assessment; endpoint time origin remains diagnosis."])
    write_json(output.with_suffix(".report.json"), report)
    return df, report


def ensure_dataset() -> tuple[pd.DataFrame, dict]:
    from data.download_hancock import download
    download()
    path = DATA / "hancock_processed.parquet"
    embeddings = DATA / "embeddings.parquet"
    if path.exists() and path.with_suffix(".report.json").exists():
        report = read_json(path.with_suffix(".report.json"))
        expected_embedding = file_hash(embeddings) if embeddings.exists() else None
        if (report["raw_manifest_sha256"] == file_hash(DATA / "raw/manifest.json") and
            report["processed_sha256"] == file_hash(path) and
            (report.get("embeddings") or {}).get("sha256") == expected_embedding):
            return pd.read_parquet(path), report
    return build_dataset(embeddings=embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATA / "raw")
    parser.add_argument("--output", type=Path, default=DATA / "hancock_processed.parquet")
    parser.add_argument("--embeddings", type=Path, default=DATA / "embeddings.parquet")
    args = parser.parse_args()
    frame, report = build_dataset(args.raw_dir, args.output, args.embeddings)
    print(f"Saved {len(frame)} patients × {len(frame.columns)} columns to {args.output}")
    print(report["endpoint_counts"])
