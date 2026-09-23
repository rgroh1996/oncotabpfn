"""Documented quota amendment: ten discrimination draws, five calibration draws."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.common import ROOT, file_hash, fingerprint, load_token, read_json, safe_error, write_json
from src.learning_curve_protocol import OUTPUT, SEEDS, prepare, run_case, score_probabilities, utcnow
from src.pipeline import make_estimator, model_metadata

CALIBRATION_SEEDS = SEEDS[:5]
TOKEN_COST_PER_CALL = 10_000
MAX_ADDITIONAL_TOKENS = 1_100_000


def calls_remaining(output, protocol):
    calls = 0
    for case in protocol["cases"]:
        if (output / "runs" / f"{case['name']}_tabpfn.json").exists():
            continue
        calls += 1
        if case["seed"] in CALIBRATION_SEEDS:
            calls += sum(not (output / "inner_cache" / f"{case['name']}_tabpfn_c0_f{fold}.json").exists() for fold in range(3))
    return calls


def prepare_amendment(output=OUTPUT):
    output = Path(output)
    protocol = prepare(output)
    path = output / "quota_amendment.json"
    settings = dict(protocol_id=protocol["protocol_id"], calibration_seeds=list(CALIBRATION_SEEDS),
                    discrimination_seeds=list(SEEDS), max_additional_tokens=MAX_ADDITIONAL_TOKENS,
                    estimated_tokens_per_predict=TOKEN_COST_PER_CALL,
                    expected_final_runs=600, expected_raw_metric_rows=600, expected_calibrated_metric_rows=300,
                    reason="User requested lower token use because the API limit cannot be raised. No performance results were used to choose seeds.",
                    rule="Retain ten matched draws for raw discrimination; report calibration only on fixed seeds 42-46 for every model and endpoint. Preserve surplus completed calibration predictions as audit records, excluded from the matched calibration summary.",
                    implementation_sha256=file_hash(ROOT / "src/learning_curve_budget.py"))
    if path.exists():
        amendment = read_json(path)
        if amendment["amendment_id"] != fingerprint(settings):
            raise ValueError("Frozen quota amendment changed")
        return protocol, amendment
    amendment = dict(**settings, amendment_id=fingerprint(settings), amended_utc=utcnow(),
                     completed_tabpfn_runs_before_amendment=len(list((output / "runs").glob("*tabpfn.json"))),
                     remaining_predict_calls=calls_remaining(output, protocol))
    amendment["estimated_remaining_tokens"] = amendment["remaining_predict_calls"] * TOKEN_COST_PER_CALL
    if amendment["estimated_remaining_tokens"] > MAX_ADDITIONAL_TOKENS:
        raise ValueError("Amended study exceeds the fixed remaining-token budget")
    write_json(path, amendment)
    return protocol, amendment


def run_raw_case(output, case, protocol_id, amendment_id):
    """Untuned TabPFN needs no inner selection fits for a raw held-out prediction."""
    output = Path(output)
    path = output / "runs" / f"{case['name']}_tabpfn.json"
    if path.exists():
        record = read_json(path)
        if record["protocol_id"] != protocol_id or record["status"] != "complete":
            raise ValueError("Invalid cached result")
        return path.name
    started, started_utc = time.perf_counter(), utcnow()
    with np.load(output / "inputs" / f"{case['name']}.npz", allow_pickle=False) as data:
        estimator = make_estimator("tabpfn", case["seed"])
        estimator.fit(data["train"], data["y"])
        p = estimator.predict_proba(data["test"])[:, 1]
        metadata = model_metadata(estimator)
        if "resolved_config" in metadata:
            metadata["requested_config"] = metadata.pop("resolved_config")
        record = dict(protocol_id=protocol_id, amendment_id=amendment_id, status="complete",
                      case=case["name"], target=case["target"], size=case["size"], n=case["n"],
                      seed=case["seed"], model="tabpfn", started_utc=started_utc,
                      selection_utc=started_utc, completed_utc=utcnow(), elapsed_seconds=time.perf_counter()-started,
                      candidates=[], selected=0, calibration_available=False,
                      training_ids=case["training_ids"], patient_ids=case["test_ids"],
                      training_y=data["y"].tolist(), y=data["test_y"].tolist(),
                      probabilities=p.tolist(), raw_metrics=score_probabilities(data["test_y"], p), estimator=metadata)
    write_json(path, record)
    return path.name


def usage():
    from tabpfn_client import set_access_token
    from tabpfn_client.client import ServiceClient
    token = load_token()
    set_access_token(token)
    return ServiceClient.get_api_usage(token)


def run_budget(output=OUTPUT):
    output = Path(output)
    protocol, amendment = prepare_amendment(output)
    from src.learning_curves import report
    start_path = output / "quota_usage_start.json"
    if not start_path.exists():
        write_json(start_path, usage())
    start = read_json(start_path)
    try:
        for case in protocol["cases"]:
            if (output / "runs" / f"{case['name']}_tabpfn.json").exists():
                continue
            current = usage()
            used = current["monthly_tokens_used"] - start["monthly_tokens_used"]
            next_calls = 1
            if case["seed"] in CALIBRATION_SEEDS:
                next_calls += sum(not (output / "inner_cache" / f"{case['name']}_tabpfn_c0_f{fold}.json").exists() for fold in range(3))
            next_cost = next_calls * TOKEN_COST_PER_CALL
            if used < 0:
                raise RuntimeError("Monthly quota counter reset; inspect budget before resuming")
            if used + next_cost > MAX_ADDITIONAL_TOKENS:
                raise RuntimeError("Reached the amended 1.1-million additional-token cap; completed predictions are cached")
            if current["daily_tokens_used"] + next_cost > current["daily_token_limit"]:
                raise RuntimeError("Daily quota cannot accommodate the next complete case; completed predictions are cached")
            if current["monthly_tokens_used"] + next_cost > current["monthly_token_limit"]:
                raise RuntimeError("Monthly quota cannot accommodate the next complete case; completed predictions are cached")
            if case["seed"] in CALIBRATION_SEEDS:
                name = run_case(output, case, "tabpfn", protocol["protocol_id"])
            else:
                name = run_raw_case(output, case, protocol["protocol_id"], amendment["amendment_id"])
            print(f"Completed {name}; remaining prediction calls: {calls_remaining(output, protocol)}", flush=True)
            report(output)
    finally:
        report(output)
        write_json(output / "quota_usage_latest.json", usage())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run"), default="run")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "prepare":
        _, amendment = prepare_amendment(args.output)
        print({k: amendment[k] for k in ("amendment_id", "remaining_predict_calls", "estimated_remaining_tokens", "max_additional_tokens")})
    else:
        run_budget(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        sys.exit(1)
