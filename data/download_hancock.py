"""Download official FAU archives, or generate a schema-compatible research demo."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.common import DATA, file_hash, read_json, safe_error, write_json

BASE_URL = "https://data.fau.de/public/24/87/322108724/"
ARCHIVES = ("StructuredData.zip", "DataSplits_DataDictionaries.zip")
REQUIRED = ("clinical_data.json", "pathological_data.json", "blood_data.json",
            "blood_data_reference_ranges.json", "dataset_split_in.json", "dataset_split_out.json")
DICTIONARIES = ("DataDictionary_blood.csv", "DataDictionary_clinical.csv", "DataDictionary_pathological.csv")


def validate_payload(files: dict[str, bytes]) -> None:
    for name in REQUIRED:
        if name not in files:
            raise ValueError(f"Archive is missing {name}")
        records = json.loads(files[name])
        if not isinstance(records, list) or not records:
            raise ValueError(f"{name}: expected a nonempty list")
        required = {"analyte_name", "normal_male_max"} if "reference" in name else {"patient_id"}
        if name == "clinical_data.json":
            required |= {"survival_status", "days_to_last_information", "recurrence"}
        if name == "blood_data.json":
            required |= {"value", "unit", "analyte_name", "days_before_first_treatment"}
        if "split" in name:
            required |= {"dataset"}
        if not all(isinstance(r, dict) and required <= r.keys() for r in records):
            raise ValueError(f"{name}: unsupported record schema")
    clinical = json.loads(files["clinical_data.json"])
    ids = [r["patient_id"] for r in clinical]
    if len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids):
        raise ValueError("Patient identifiers must be unique strings")
    for name in ("pathological_data.json", "dataset_split_in.json", "dataset_split_out.json"):
        records = json.loads(files[name])
        record_ids = [r["patient_id"] for r in records]
        if len(record_ids) != len(set(record_ids)) or not set(record_ids) <= set(ids):
            raise ValueError(f"{name}: duplicate or unknown patient identifiers")
        if "split" in name and (set(record_ids) != set(ids) or
                                any(r["dataset"] not in {"training", "test"} for r in records)):
            raise ValueError(f"{name}: invalid split partition")


def official_payload() -> tuple[dict[str, bytes], dict]:
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5,
                                                          status_forcelist=[429, 502, 503, 504])))
    files, archives = {}, {}
    for name in ARCHIVES:
        url = BASE_URL + name
        response = session.get(url, timeout=(10, 45))
        response.raise_for_status()
        if len(response.content) > 50_000_000:
            raise ValueError("Unexpected archive size")
        archives[name] = {"url": url, "sha256": hashlib.sha256(response.content).hexdigest()}
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for info in archive.infolist():
                path = Path(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe archive member")
                if path.name in REQUIRED + DICTIONARIES:
                    if info.file_size > 50_000_000 or path.name in files:
                        raise ValueError("Oversized or duplicate archive member")
                    files[path.name] = archive.read(info)
    validate_payload(files)
    return files, archives


def synthetic_payload(seed: int = 42) -> dict[str, bytes]:
    """Synthetic functionality fixture, not a statistically faithful cohort replica."""
    rng = np.random.default_rng(seed)
    clinical, pathology, blood = [], [], []
    labs = {
        "Leukocytes": ("x10^3/µl", "26464-8", 4., 11.),
        "Lymphocytes": ("x10^3/µl", "26474-7", 1., 4.),
        "Granulocytes": ("x10^3/µl", "30394-1", 1.5, 8.),
        "Platelets": ("x10^3/µl", "26515-7", 150., 400.),
        "CRP": ("mg/l", "1988-5", 0., 5.),
        "Hemoglobin": ("g/dl", "718-7", 12., 17.),
        "Sodium": ("mmol/l", "2951-2", 135., 145.),
        "Potassium": ("mmol/l", "2823-3", 3.5, 5.1),
        "Creatinine": ("mg/dl", "2160-0", .5, 1.3),
        "INR": (None, "34714-6", .8, 1.2),
    }
    for i in range(1, 764):
        pid = f"{i:03d}"
        age = int(np.clip(rng.normal(62, 11), 28, 92))
        t, n = int(rng.choice([1, 2, 3, 4], p=[.22, .3, .17, .31])), int(rng.choice(4, p=[.45, .2, .28, .07]))
        site = str(rng.choice(["Oral_Cavity", "Oropharynx", "Hypopharynx", "Larynx", "CUP"], p=[.26, .4, .1, .235, .005]))
        hpv = "positive" if site == "Oropharynx" and rng.random() < .4 else "negative"
        inflammation = rng.normal(.15*t, .6)
        death_time = float(rng.exponential(3400 * np.exp(-.018*(age-60)-.22*(t-2)-.16*n+.55*(hpv == "positive")-.2*inflammation)))
        censor_time = float(rng.uniform(180, 4300))
        death = death_time <= censor_time
        followup = int(min(death_time, censor_time))
        recur_time = float(rng.exponential(4800 * np.exp(-.3*(t-2)-.22*n+.45*(hpv == "positive"))))
        recurrence = recur_time <= followup
        row = dict(patient_id=pid, year_of_initial_diagnosis=int(rng.integers(2004, 2019)),
                   age_at_initial_diagnosis=age, sex=str(rng.choice(["male", "female"], p=[.76, .24])),
                   smoking_status=str(rng.choice(["smoker", "former", "non-smoker"], p=[.53, .27, .2])),
                   primarily_metastasis="yes" if rng.random() < .02 else "no",
                   survival_status="deceased" if death else "living",
                   survival_status_with_cause="deceased" if death else "living",
                   days_to_last_information=followup, first_treatment_intent="curative",
                   first_treatment_modality="local surgery", days_to_first_treatment=int(rng.integers(0, 45)),
                   adjuvant_treatment_intent="curative", adjuvant_radiotherapy="yes" if t > 2 else "no",
                   adjuvant_radiotherapy_modality=None, adjuvant_systemic_therapy="no",
                   adjuvant_systemic_therapy_modality=None, adjuvant_radiochemotherapy="no",
                   recurrence="yes" if recurrence else "no", days_to_recurrence=int(recur_time) if recurrence else None,
                   progress_1="no", days_to_progress_1=None, progress_2=None, days_to_progress_2=None)
        for k in range(1, 5):
            row[f"metastasis_{k}_locations"] = None
            row[f"days_to_metastasis_{k}"] = None
        clinical.append(row)
        pathology.append(dict(patient_id=pid, primary_tumor_site=site, pT_stage=f"pT{t}", pN_stage=f"pN{n}",
                              grading="HPV_OSCC" if hpv == "positive" else str(rng.choice(["G1", "G2", "G3"], p=[.1, .6, .3])),
                              hpv_association_p16=hpv, number_of_positive_lymph_nodes=n,
                              number_of_resected_lymph_nodes=int(rng.integers(max(n, 5), 40)),
                              perinodal_invasion="no", lymphovascular_invasion_L="no", vascular_invasion_V="no",
                              perineural_invasion_Pn="no", resection_status="R0", resection_status_carcinoma_in_situ="CIS_absent",
                              carcinoma_in_situ="no", closest_resection_margin_in_cm="0.5", histologic_type="SCC_Conventional-Keratinizing",
                              infiltration_depth_in_mm=float(rng.uniform(1, 20))))
        if rng.random() < .06:
            continue
        lymph = float(rng.lognormal(.4-.15*inflammation, .35))
        gran = float(rng.lognormal(1.4+.2*inflammation, .3))
        vals = [lymph+gran+float(rng.uniform(.3, 1)), lymph, gran, float(rng.lognormal(5.5+.08*inflammation, .25)),
                float(rng.lognormal(1.1+inflammation, 1)), float(rng.normal(13.8, 1.6)), float(rng.normal(139, 3)),
                float(rng.normal(4.2, .35)), float(rng.lognormal(-.1, .25)), float(rng.normal(1, .08))]
        for (analyte, (unit, loinc, _, _)), value in zip(labs.items(), vals):
            if rng.random() < .1:
                continue
            blood.append(dict(patient_id=pid, value=round(value, 3), unit=unit, analyte_name=analyte,
                              LOINC_code=loinc, LOINC_name=analyte, group="Synthetic laboratory panel",
                              days_before_first_treatment=int(rng.integers(0, 15))))
    refs = [dict(analyte_name=a, unit=u, LOINC_name=a, group="Synthetic laboratory panel",
                 normal_male_min=lo, normal_male_max=hi, normal_female_min=lo, normal_female_max=hi)
            for a, (u, _, lo, hi) in labs.items()]
    ids = [r["patient_id"] for r in clinical]
    test_ids = set(rng.choice(ids, size=153, replace=False))
    out_ids = {r["patient_id"] for r in pathology if r["primary_tumor_site"] == "Oropharynx"}
    objects = {"clinical_data.json": clinical, "pathological_data.json": pathology, "blood_data.json": blood,
               "blood_data_reference_ranges.json": refs,
               "dataset_split_in.json": [dict(patient_id=i, dataset="test" if i in test_ids else "training") for i in ids],
               "dataset_split_out.json": [dict(patient_id=i, dataset="test" if i in out_ids else "training") for i in ids]}
    return {name: json.dumps(value, allow_nan=False).encode() for name, value in objects.items()}


def download(raw_dir: Path = DATA / "raw", source: str = "auto", seed: int = 42, refresh: bool = False) -> dict:
    if refresh and source == "auto":
        # An auto refresh could silently replace real data with the synthetic fallback.
        raise ValueError("--refresh requires an explicit --source real or --source synthetic")
    manifest_path = raw_dir / "manifest.json"
    if manifest_path.exists() and not refresh:
        manifest = read_json(manifest_path)
        if source != "auto" and manifest["source"] != source:
            raise ValueError("Existing source differs; use --refresh to replace it explicitly")
        if not all((raw_dir / n).exists() and file_hash(raw_dir / n) == digest
                   for n, digest in manifest["files"].items()):
            raise ValueError("Raw files changed or are incomplete; use --refresh to reacquire")
        validate_payload({n: (raw_dir / n).read_bytes() for n in REQUIRED})
        return manifest
    if raw_dir.exists() and any((raw_dir / n).exists() for n in REQUIRED) and not refresh:
        raise ValueError("Unmanaged raw files found; use a separate --raw-dir or --refresh")
    reason, archives = None, {}
    if source != "synthetic":
        try:
            files, archives = official_payload()
            actual = "real"
        except (requests.RequestException, ValueError, zipfile.BadZipFile, KeyError) as exc:
            if source == "real":
                raise
            reason = safe_error(exc)
            print(f"Official acquisition unavailable: {reason}\nUsing SYNTHETIC demonstration data.", file=sys.stderr)
            files, actual = synthetic_payload(seed), "synthetic"
    else:
        files, actual = synthetic_payload(seed), "synthetic"
    validate_payload(files)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        temporary = raw_dir / (name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(raw_dir / name)
    # Prevent stale dictionaries from a previous source being attributed to synthetic data.
    for name in DICTIONARIES:
        if name not in files and (raw_dir / name).exists():
            (raw_dir / name).unlink()
    manifest = dict(schema_version=1, source=actual, seed=seed if actual == "synthetic" else None,
                    acquired_at=datetime.now(timezone.utc).isoformat(), archives=archives, fallback_reason=reason,
                    patient_count=len(json.loads(files["clinical_data.json"])),
                    files={n: hashlib.sha256(b).hexdigest() for n, b in files.items()})
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto")
    parser.add_argument("--raw-dir", type=Path, default=DATA / "raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = download(args.raw_dir, args.source, args.seed, args.refresh)
    print(f"Ready: {result['patient_count']} patients; source={result['source']}; {args.raw_dir}")
