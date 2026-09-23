import json

import numpy as np
import pandas as pd
import pytest
import requests

from data.dataset_builder import (HORIZON_DAYS, aggregate_blood, build_dataset, convert_lab,
                                  engineer_biomarkers, make_targets, normalize_grade,
                                  normalize_stage)
from data.download_hancock import download, synthetic_payload, validate_payload


def test_fallback_schema_and_determinism(synthetic_cohort):
    first = synthetic_payload(42)
    assert first == synthetic_payload(42)
    assert first != synthetic_payload(43)
    validate_payload(first)
    frame, report, root = synthetic_cohort
    assert len(frame) == 763 and frame.patient_id.is_unique
    assert frame.patient_id.iloc[0] == "001"
    assert report["source"] == "synthetic"
    assert frame.nlr.isna().all() and frame.sii.isna().all()
    assert frame.plr.notna().any()
    assert (root / "processed.parquet").exists()
    assert frame.target_survival_5y.isna().any()


def test_auto_refresh_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="explicit --source"):
        download(tmp_path / "raw", source="auto", refresh=True)


def test_acquisition_failure_has_explicit_provenance(tmp_path, monkeypatch):
    def unavailable():
        raise requests.ConnectionError("simulated network outage")
    monkeypatch.setattr("data.download_hancock.official_payload", unavailable)
    result = download(tmp_path / "auto", source="auto")
    assert result["source"] == "synthetic"
    assert "simulated network outage" in result["fallback_reason"]
    with pytest.raises(requests.ConnectionError):
        download(tmp_path / "real", source="real")
    with pytest.raises(ValueError, match="Existing source differs"):
        download(tmp_path / "auto", source="real")


def test_five_year_labels_and_unknown_recurrence():
    clinical = pd.DataFrame({"survival_status": ["deceased", "deceased", "deceased", "living", "living", "unknown", "deceased"],
                             "days_to_last_information": [100, HORIZON_DAYS, HORIZON_DAYS+1, 100, HORIZON_DAYS, 3000, -2],
                             "recurrence": ["yes", "no", "no", "no", None, "unknown", "no"]})
    labels = make_targets(clinical)
    assert labels.target_survival_5y.iloc[:3].tolist() == [1, 1, 0]
    assert np.isnan(labels.target_survival_5y.iloc[3])
    assert labels.target_survival_5y.iloc[4] == 0
    assert labels.target_survival_5y.iloc[5:].isna().all()
    assert labels.target_recurrence.iloc[0] == 1
    assert labels.target_recurrence.iloc[4:6].isna().all()


@pytest.mark.parametrize("value,prefix,expected", [("pT4b", "T", "T4"), ("pTis", "T", "Tis"),
                                                    ("TX", "T", "Unknown"), ("pN2c", "N", "N2"),
                                                    ("NX", "N", "Unknown"), (None, "T", "Unknown")])
def test_stages(value, prefix, expected):
    assert normalize_stage(value, prefix) == expected


@pytest.mark.parametrize("value,expected", [("hpv_association_p16", "HPV_OSCC"), ("HPV_OSCC", "HPV_OSCC"),
                                            ("G2", "G2"), ("GX", "Unknown"), (None, "Unknown")])
def test_grades(value, expected):
    assert normalize_grade(value) == expected


def test_laboratory_units_timing_and_duplicates():
    assert convert_lab(2, "mg/dl", "mg/l") == 20
    assert convert_lab(4500, "/µl", "count") == 4.5
    assert np.isnan(convert_lab(30, "%", "count"))
    assert np.isnan(convert_lab(-1, "mg/l", "mg/l"))
    rows = pd.DataFrame([dict(patient_id="001", analyte_name="CRP", unit="mg/l", value=v,
                              days_before_first_treatment=d) for d, v in [(2, 10), (2, 20), (5, 100), (-1, 999), (20, 999)]])
    result, report = aggregate_blood(rows)
    assert result.crp.iloc[0] == 15
    assert result.crp_days_before_treatment.iloc[0] == 2
    assert report["rejected_measurements"] == 2


def test_ratios_and_crp_boundary():
    frame = engineer_biomarkers(pd.DataFrame({"lymphocytes": [2., 0., np.nan, 2.], "neutrophils": [6., 6., 6., np.nan],
                                               "platelets": [200.] * 4, "crp": [5., 5.1, np.nan, 3.],
                                               "crp_reference_upper": [5.] * 4, "granulocytes": [9.] * 4}))
    assert frame.loc[0, ["nlr", "plr", "sii"]].tolist() == [3, 100, 600]
    assert frame.loc[1:2, ["nlr", "plr", "sii"]].isna().all().all()
    assert np.isnan(frame.loc[3, "nlr"])
    assert frame.crp_elevated.iloc[:2].tolist() == [0., 1.]
    assert np.isnan(frame.crp_elevated.iloc[2])


def test_corrupted_raw_files_are_rejected(tmp_path):
    download(tmp_path / "raw", source="synthetic")
    path = tmp_path / "raw/clinical_data.json"
    rows = json.loads(path.read_text())
    rows[0]["age_at_initial_diagnosis"] = 99
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Raw source changed"):
        build_dataset(tmp_path / "raw", tmp_path / "processed.parquet")


def test_duplicate_ids_rejected():
    files = synthetic_payload()
    records = json.loads(files["clinical_data.json"])
    records[1]["patient_id"] = records[0]["patient_id"]
    files["clinical_data.json"] = json.dumps(records).encode()
    with pytest.raises(ValueError, match="unique"):
        validate_payload(files)
