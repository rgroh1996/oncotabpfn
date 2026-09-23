"""OncoTabPFN — local precision oncology research workbench."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.dataset_builder import LAB_COLUMNS, engineer_biomarkers, ensure_dataset
from src.common import DATA, RESULTS, TARGETS, TARGET_LABELS, fingerprint, load_token, read_json, safe_error
from src.pipeline import fit_dashboard_model
from app.research_view import render_learning_curves, render_optimization_view

st.set_page_config(page_title="OncoTabPFN · Precision oncology", page_icon="◉", layout="wide")
st.markdown("""<style>
  .block-container {max-width:1440px;padding-top:2rem;padding-bottom:3rem}
  [data-testid="stSidebar"] {background:#102B3A}
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color:#E5F2F0}
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#A5C3CA}
  [data-testid="stSidebar"] [data-testid="stLinkButton"] a {background:transparent;border:1px solid #3E6272}
  [data-testid="stSidebar"] [data-testid="stLinkButton"] a:hover {border-color:#8FC1BC}
  h1 {font-weight:650;letter-spacing:-1.4px}
  h2, h3 {letter-spacing:-.5px}
  [data-testid="stMetric"] {background:white;border:1px solid #E0E7EE;border-radius:12px;padding:16px 20px}
  [data-testid="stMetricLabel"] p {color:#65778A;font-size:13px}
  [data-testid="stMetricValue"] {font-weight:600;font-size:1.8rem}
  .eyebrow {font-size:11px;letter-spacing:2px;color:#007F78;font-weight:700;margin-bottom:12px}
  .subhead {font-size:16px;color:#61758A;max-width:780px;margin-bottom:24px;line-height:1.6}
  .pill {display:inline-block;border:1px solid #C5DED9;border-radius:30px;padding:5px 12px;background:#E8F5F1;color:#176C62;font-size:12px;font-weight:600}
  .placeholder {height:204px;display:flex;align-items:center;justify-content:center;flex-direction:column;background:linear-gradient(135deg,#F1F8F7,#FAFCFE);border-radius:12px;border:1px dashed #C6D9D7;margin:12px 0 18px}
  .placeholder strong {font-size:38px;color:#89A5AC;font-weight:400}
  .placeholder span {font-size:13px;color:#657C87;margin-top:12px}
  .risk {display:inline-block;border-radius:20px;padding:6px 13px;font-size:12px;font-weight:600;margin:4px 0 12px}
  .low {color:#176C62;background:#E4F3ED}.intermediate {color:#936413;background:#FFF0CE}.high {color:#A63F49;background:#FBE5E8}
  div[data-testid="stTabs"] button p {font-weight:600}
  .footer {font-size:12px;color:#7C8D9E;border-top:1px solid #DDE6EB;padding-top:18px;margin-top:32px}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_cohort(manifest_stamp: float, embedding_stamp: float):
    return ensure_dataset()


@st.cache_resource(show_spinner=False, ttl=3600)
def get_model(data_hash: str, target: str, credential_fingerprint: str, _frame: pd.DataFrame):
    return fit_dashboard_model(_frame, target)


def gauge(probability: float, color: str, title: str):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=100*probability,
                                number={"suffix": "%", "font": {"size": 44, "color": "#17283C"}, "valueformat": ".1f"},
                                gauge={"axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#ccc"},
                                       "bar": {"color": color, "thickness": .65}, "bgcolor": "#EAF0F3", "borderwidth": 0},
                                title={"text": title, "font": {"size": 13, "color": "#61758A"}}))
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)", font_family="Arial")
    return fig


def risk_band(p: float) -> tuple[str, str]:
    if p < .2:
        return "low", "Low event risk"
    if p <= .5:
        return "intermediate", "Intermediate event risk"
    return "high", "High event risk"


def render_prediction(target, result):
    survival = target == TARGETS[0]
    st.subheader("Five-year survival" if survival else "Locoregional recurrence")
    st.caption("From initial diagnosis" if survival else "During recorded follow-up · variable duration")
    if result is None:
        st.markdown('<div class="placeholder"><strong>— %</strong><span>Submit a patient assessment to generate a prediction</span></div>', unsafe_allow_html=True)
        return
    if "error" in result:
        st.warning(result["error"])
        return
    p = result["probability"]
    st.plotly_chart(gauge(1-p if survival else p, "#008C82" if survival else "#5273B8",
                          "Estimated survival probability" if survival else "Estimated recurrence probability"), width="stretch")
    category, label = risk_band(p)
    st.markdown(f'<span class="risk {category}">{label}</span>', unsafe_allow_html=True)
    names = ["Alive beyond five years", "Death within five years"] if survival else ["No recorded recurrence", "Recorded recurrence"]
    selected = [names[i] for i, included in enumerate(result["outcome_set"]) if included]
    st.markdown("**90% conformal outcome set**")
    if len(selected) == 2:
        st.info("Both outcomes remain plausible. The model cannot narrow the outcome set at this coverage level.")
    elif selected:
        st.success(selected[0])
    else:
        st.warning("Empty outcome set: neither outcome meets the conformal inclusion threshold. Treat this as an abstention.")
    evaluation = result["evaluation"]
    st.caption(f"Held-out coverage {evaluation['conformal_coverage']:.1%} · "
               f"mean set size {evaluation['conformal_mean_set_size']:.2f} · "
               f"{evaluation['counts']['test']} eligible test patients")


def select_value(label, choices, current, key):
    current = current if pd.notna(current) and current in choices else choices[-1]
    return st.selectbox(label, choices, index=choices.index(current), key=key)


def optional_number(label, current, maximum, key, step=.1):
    return st.number_input(label, min_value=0., max_value=max(maximum, float(current)) if pd.notna(current) else maximum,
                           value=float(current) if pd.notna(current) else None, step=step, key=key,
                           placeholder="Not measured")


try:
    if st.secrets.get("TABPFN_TOKEN"):
        os.environ["TABPFN_TOKEN"] = str(st.secrets["TABPFN_TOKEN"])
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    pass

with st.sidebar:
    st.markdown("## ◉ OncoTabPFN")
    st.caption("PRECISION ONCOLOGY WORKBENCH")
    st.divider()
    st.markdown("### Head & neck cancer")
    st.caption("Clinical records · blood panels · pathology")
    st.markdown("### Model")
    st.markdown("**TabPFN-3.5**")
    token = load_token()
    st.caption("Prior Labs API · token configured" if token else "Prior Labs API · token needed")
    if not token:
        st.caption("Set TABPFN_TOKEN in the project's .env file, then refresh this page.")
    st.divider()
    st.markdown("### Research use")
    st.caption("An exploratory assessment for research and tumor-board demonstrations. Risk bands are illustrative; no treatment recommendations are generated.")
    st.link_button("HANCOCK cohort ↗", "https://www.hancock.research.uni-erlangen.org/download")
    st.link_button("Get an API key ↗", "https://platform.priorlabs.ai/account/api-keys")

try:
    manifest_file = DATA / "raw/manifest.json"
    embedding_file = DATA / "embeddings.parquet"
    with st.spinner("Loading the cohort and verifying data provenance…"):
        df, report = load_cohort(manifest_file.stat().st_mtime if manifest_file.exists() else 0,
                                 embedding_file.stat().st_mtime if embedding_file.exists() else 0)
except Exception as exc:
    st.error(f"Unable to load cohort: {safe_error(exc)}")
    st.code("python data/download_hancock.py\npython data/dataset_builder.py")
    st.stop()

source = report["source"]
st.markdown('<div class="eyebrow">ONCOLOGY / MULTIMODAL RESEARCH</div>', unsafe_allow_html=True)
st.title("A clearer view of patient risk.")
st.markdown('<div class="subhead">Explore survival and recurrence through clinical history, pathological staging, and blood biomarkers — with uncertainty in view.</div>', unsafe_allow_html=True)
if source == "synthetic":
    st.warning("SYNTHETIC DEMONSTRATION · These are generated patients, not the official HANCOCK cohort. Results have no clinical evidentiary value.")
else:
    st.markdown('<span class="pill">● Official HANCOCK cohort · FAU / Uniklinikum Erlangen</span>', unsafe_allow_html=True)
st.write("")
cards = st.columns(4)
for column, label, value in zip(cards, ["Cohort patients", "Five-year outcome eligible", "Recurrence outcome eligible", "Patients with blood data"],
                               [len(df), report["endpoint_counts"][TARGETS[0]]["eligible"], report["endpoint_counts"][TARGETS[1]]["eligible"], int(df.blood_available.sum())]):
    column.metric(label, f"{value:,}")
st.write("")
assessment, paper_tab, optimization_tab, methods = st.tabs(["Patient assessment", "Paper replication", "Optimization & explanations", "Cohort & methodology"])

with assessment:
    left, right = st.columns([.95, 1.65], gap="large")
    with left:
        st.subheader("Patient profile")
        mode = st.radio("Assessment source", ["Cohort patient", "New patient"], horizontal=True)
        if mode == "Cohort patient":
            heldout = df.loc[df.split_in.eq("test")].set_index("patient_id", drop=False)
            pid = st.selectbox("Held-out patient", heldout.index.tolist(),
                               format_func=lambda p: f"{p} · {heldout.loc[p, 'primary_tumor_site']} · age {int(heldout.loc[p, 'age'])}")
            base = heldout.loc[pid].copy()
            st.caption("This patient belongs to the official test partition and is excluded from every dashboard training context.")
        else:
            pid = "manual"
            base = pd.Series({c: np.nan for c in df.columns}, dtype=object)
            base.update(pd.Series({"age": 60, "sex": "Unknown", "t_stage": "Unknown", "n_stage": "Unknown", "m_stage": "Unknown",
                                   "grade": "Unknown", "primary_tumor_site": "Unknown", "hpv_p16": "not_tested",
                                   "blood_available": 0., "pathology_available": 1.}))
        prefix = f"{mode}_{pid}"
        with st.form("assessment_form"):
            patient = base.copy()
            patient["age"] = st.slider("Age at diagnosis", 18, 100, int(np.clip(base.age, 18, 100)), key=prefix+"age")
            c1, c2 = st.columns(2)
            with c1:
                patient["sex"] = select_value("Sex", ["female", "male", "Unknown"], base.sex, prefix+"sex")
                patient["t_stage"] = select_value("Pathological T stage", ["T1", "T2", "T3", "T4", "Tis", "T0", "Unknown"], base.t_stage, prefix+"t")
                patient["m_stage"] = select_value("Metastasis at diagnosis", ["M0", "M1", "Unknown"], base.m_stage, prefix+"m")
            with c2:
                patient["grade"] = select_value("Histological grade", ["G1", "G2", "G3", "HPV_OSCC", "Unknown"], base.grade, prefix+"grade")
                patient["n_stage"] = select_value("Pathological N stage", ["N0", "N1", "N2", "N3", "Unknown"], base.n_stage, prefix+"n")
                patient["hpv_p16"] = select_value("HPV / p16", ["positive", "negative", "not_tested"], base.hpv_p16, prefix+"hpv")
            patient["primary_tumor_site"] = select_value("Primary tumor site", ["Oral cavity", "Oropharynx", "Hypopharynx", "Larynx", "CUP", "Unknown"], base.primary_tumor_site, prefix+"site")
            patient["smoking_status"] = select_value("Smoking status", ["non-smoker", "former", "smoker", "Unknown"], base.get("smoking_status"), prefix+"smoking")
            with st.expander("Pretreatment laboratory measurements", expanded=True):
                a, b = st.columns(2)
                for col, name, label, maximum in [(a, "crp", "CRP · mg/L", 1000.), (b, "leukocytes", "Leukocytes · 10⁹/L", 300.),
                                                  (a, "lymphocytes", "Lymphocytes · 10⁹/L", 200.), (b, "platelets", "Platelets · 10⁹/L", 2000.),
                                                  (a, "neutrophils", "Neutrophils · 10⁹/L", 200.)]:
                    with col:
                        patient[name] = optional_number(label, base.get(name), maximum, prefix+name)
                st.caption("Blank means not measured. NLR, PLR and SII are recalculated from these counts. HANCOCK has no dedicated neutrophil measurement, so its trained models cannot learn an NLR/SII effect.")
            submitted = st.form_submit_button("Generate risk assessment", type="primary", width="stretch", disabled=not bool(token))
        patient["crp_reference_upper"] = 5. if patient["sex"] in {"male", "female"} else np.nan
        patient["blood_available"] = float(any(pd.notna(patient.get(c)) for c in LAB_COLUMNS))
        # Every cohort patient has a pathology record, so 0 would be a value the model never saw.
        patient["pathology_available"] = 1.
        patient_frame = engineer_biomarkers(pd.DataFrame([patient]))
        if not token:
            st.info("Add TABPFN_TOKEN to .env to generate API predictions. Patient exploration and benchmarks are available now.")
        request_key = fingerprint({"patient": pid, "source": source, "data_hash": report["processed_sha256"],
                                   "values": patient_frame.astype(str).to_dict("records")})
        if submitted:
            results = {}
            started = time.perf_counter()
            for target in TARGETS:
                try:
                    with st.spinner(f"Computing {TARGET_LABELS[target].lower()}…"):
                        model = get_model(report["processed_sha256"], target, fingerprint(token), df)
                        probabilities, sets = model.predict(patient_frame)
                        results[target] = dict(probability=float(probabilities[0]), outcome_set=sets[0].tolist(), evaluation=model.evaluation)
                except Exception as exc:
                    results[target] = dict(error=safe_error(exc))
            st.session_state["assessment"] = dict(key=request_key, results=results, seconds=time.perf_counter()-started)
    with right:
        st.subheader("Risk & uncertainty")
        st.caption("Postoperative research assessment · calibrated probabilities · held-out conformal calibration")
        stored = st.session_state.get("assessment", {})
        active = stored if stored.get("key") == request_key else {}
        columns = st.columns(2)
        for column, target in zip(columns, TARGETS):
            with column, st.container(border=True):
                render_prediction(target, active.get("results", {}).get(target))
        if active:
            st.caption(f"TabPFN-3.5 API · total elapsed {active['seconds']:.1f}s · token configured")
        st.info("Conformal sets express which outcomes remain plausible at a nominal 90% marginal coverage level. They are not patient-specific probability intervals or treatment guidance.")
        with st.container(border=True):
            st.markdown("**Biomarker snapshot**")
            biomarker_cols = st.columns(3)
            for col, name, label in zip(biomarker_cols, ["plr", "nlr", "sii"], ["PLR", "NLR", "SII"]):
                value = patient_frame[name].iloc[0]
                col.metric(label, f"{value:,.1f}" if pd.notna(value) else "Not available")
            st.caption("PLR: platelets / lymphocytes · NLR: neutrophils / lymphocytes · SII: platelets × NLR")
        with st.expander("Interpret this assessment"):
            st.write("Low (<20%), intermediate (20–50%), and high (>50%) bands refer to event probabilities and use illustrative thresholds. The survival gauge shows the complement of mortality risk.")
            st.write("Five-year mortality is modeled only in patients whose five-year outcome can be established. Early censoring exclusions can bias estimates. Recurrence probabilities refer to recorded occurrence over unequal follow-up, not a fixed time horizon.")
            st.write("Manual changes are scenario explorations, not causal treatment simulations. Missing fields are handled using the training context. First inference includes context preparation and calibration; latency depends on the API.")

with paper_tab:
    render_learning_curves()
    st.divider()
    st.subheader("Does TabPFN improve on the HANCOCK paper?")
    st.caption("Figure 2 Random Forest experiment · official in-distribution, out-of-distribution and Oropharynx partitions · five repetitions")
    st.write("The primary comparison substitutes TabPFN-3.5 for the published Random Forest using the same features, preprocessing and SMOTE samples. A prespecified secondary variant uses the original training patients without SMOTE.")
    paper_dir = RESULTS / "paper"
    if (paper_dir / "status.json").exists() and (paper_dir / "summary.csv").exists():
        paper_status = read_json(paper_dir / "status.json")
        if not paper_status["complete"]:
            st.info(f"Comparison in progress: {paper_status['completed_runs']} of {paper_status['expected_runs']} runs saved.")
        else:
            st.success("All 90 runs completed: 30 published Random Forest runs and 60 TabPFN runs.")
        if (paper_dir / "paper_comparison.png").exists():
            st.image(str(paper_dir / "paper_comparison.png"))
        paper_summary = pd.read_csv(paper_dir / "summary.csv")
        st.dataframe(paper_summary, hide_index=True, width="stretch")
        paired_path = paper_dir / "paired_comparisons.csv"
        if paired_path.exists() and paired_path.stat().st_size > 1:
            st.markdown("**Paired AUC differences versus reproduced Random Forest**")
            st.dataframe(pd.read_csv(paired_path), hide_index=True, width="stretch")
            st.caption("Intervals resample held-out patients while retaining all repetitions together. These exploratory 95% intervals are not adjusted for multiple comparisons. Splits overlap; repetition SD is not a confidence interval.")
        if (paper_dir / "REPORT.md").exists():
            st.download_button("Download paper comparison report", (paper_dir / "REPORT.md").read_bytes(), "HANCOCK_comparison.md", "text/markdown")
        st.download_button("Download paper comparison metrics", (paper_dir / "metrics.csv").read_bytes(), "HANCOCK_paper_metrics.csv", "text/csv")
    else:
        st.info("No paper-comparison results available yet.")
    with st.expander("How these endpoints differ from the patient dashboard"):
        st.write("The paper's death-status endpoint excludes known non-tumor deaths and has no five-year horizon. Recurrence uses the released code's eligibility rule, including living nonrecurrent patients with short follow-up. These definitions differ from the dashboard endpoints and from a strict three-year recurrence endpoint.")
        st.write("Published feature tables include CD3/CD8 cell densities and ICD codes, and inherit upstream cohort-wide preprocessing. The comparison uses the Figure 2 Random Forest, not the separate foundation-model/RFS experiments. These scores do not validate clinical use.")
        st.markdown("[HANCOCK paper](https://doi.org/10.1038/s41467-025-62386-6) · [Frozen author code](https://github.com/ankilab/HANCOCK_MultimodalDataset/tree/521b99b03a94008b28df5c3df4aa5f82aa14b25a)")

with optimization_tab:
    render_optimization_view()

with methods:
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.subheader("One patient. Several modalities.")
        st.write("Clinical records and pathological staging are joined to the closest available pretreatment blood measurements by patient ID. Optional patient-level slide embeddings are compressed using PCA fitted only on the training context.")
        st.markdown("**Observed availability**")
        missingness = pd.DataFrame({"Feature": list(report["missing_fraction"]), "Available": [1-x for x in report["missing_fraction"].values()]})
        fig = px.bar(missingness.sort_values("Available"), x="Available", y="Feature", orientation="h", color_discrete_sequence=["#008C82"])
        fig.update_layout(height=800, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Evidence, with boundaries.")
        st.markdown("**Endpoint eligibility**")
        st.dataframe(pd.DataFrame(report["endpoint_counts"]).T.rename(index=TARGET_LABELS), width="stretch")
        st.markdown("**Evaluation design**")
        st.write("Official in-distribution test patients are never included in dashboard context or calibration. Benchmark subsamples are nested and matched between TabPFN-3.5, LightGBM and XGBoost. All preprocessing and embedding PCA are fitted inside the training partition.")
        st.markdown("**Probability and outcome-set calibration**")
        st.write("For dashboard models, development patients are randomly partitioned into 60% model context, 20% sigmoid calibration, and 20% conformal calibration. Coverage is measured on the remaining official test set. Conformal guarantees require exchangeability and do not establish individual-patient or subgroup coverage.")
        st.markdown("**Limitations**")
        for note in report["notes"]:
            st.write("• " + note)
        st.write("• Single-center, retrospective research cohort; no external clinical validation. Risk estimates do not establish treatment benefit.")
        st.write("• Standard CRP, smoking status and granulocytes are measured; pack-years, alcohol intake, dedicated neutrophils and high-sensitivity CRP are unavailable.")
        st.markdown("**Data & model provenance**")
        st.json({"source": source, "patients": len(df), "processed_sha256": report["processed_sha256"],
                 "embedding_dimensions": (report.get("embeddings") or {}).get("dimensions", 0),
                 "requested_model": "TabPFN-3.5", "inference": "Prior Labs API"}, expanded=False)
        st.markdown("[HANCOCK publication · Nature Communications (2025)](https://doi.org/10.1038/s41467-025-62386-6)")

st.markdown('<div class="footer">OncoTabPFN · Research demonstrator · HANCOCK / FAU Erlangen-Nürnberg · No clinical treatment recommendations</div>', unsafe_allow_html=True)
