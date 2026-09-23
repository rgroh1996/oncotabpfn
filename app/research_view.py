"""Read-only optimization and explanation views; opening them never calls the API."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.common import RESULTS, read_json

LABELS = {"survival_status": "Death status · paper endpoint", "recurrence": "Recurrence · paper endpoint"}


def render_learning_curves():
    root = RESULTS / "learning_curves"
    st.subheader("Paper-matched sample efficiency")
    st.write("Six pipelines use the same patients and the paper's 79 original features. Ten nested training draws cover 50, 100, 200, 500 and all eligible training patients on the official in-distribution split.")
    if not (root / "status.json").exists():
        st.info("Run src/learning_curves.py in the paper environment to generate the comparison.")
        return
    status = read_json(root / "status.json")
    required = ("benchmark_performance.png", "calibration_comparison.png", "paired_comparisons.csv", "REPORT.md")
    if not status["complete"] or not all((root / name).exists() for name in required):
        st.info(f"Study in progress: {status['completed_runs']} / {status['expected_runs']} final evaluations saved. No winner is declared from partial results.")
        return
    st.success(f"All {status['completed_runs']} final evaluations completed, including training-only baseline selection and probability calibration.")
    if (root / "quota_amendment.json").exists():
        st.info("Token-saving amendment: discrimination uses all ten matched draws; calibration uses the same five fixed draws (seeds 42–46) for every model. No performance-based seed selection was used.")
    st.caption("These official test patients were used in previous experiments. Results are exploratory; they do not establish external generalization. The published feature tables retain upstream cohort-wide preprocessing.")
    st.image(str(root / "benchmark_performance.png"))
    st.caption("Shading shows variation across training draws, not confidence intervals. The Full repetitions share all training patients.")
    pairs = pd.read_csv(root / "paired_comparisons.csv")
    primary = pairs.loc[pairs["size"].str.startswith("Low-N")]
    st.markdown("**Primary comparison: mean raw AUC across 50, 100 and 200 patients**")
    st.dataframe(primary, hide_index=True, width="stretch")
    st.caption("Positive differences favor TabPFN. Intervals resample held-out patients with all model, size and training-draw predictions kept paired. They are conditional on these training draws and unadjusted across comparisons.")
    st.image(str(root / "calibration_comparison.png"))
    st.caption("Every sigmoid uses only out-of-fold predictions and labels from its N-patient training subset. RF with and without SMOTE use the same tuning grid, selected independently. Brier measures overall probability accuracy, not calibration alone.")
    with st.expander("All metrics and paired comparisons"):
        st.dataframe(pd.read_csv(root / "summary.csv"), hide_index=True, width="stretch")
        st.dataframe(pairs, hide_index=True, width="stretch")
    st.download_button("Download learning-curve metrics", (root / "metrics.csv").read_bytes(), "learning_curve_metrics.csv", "text/csv")
    st.download_button("Download learning-curve report", (root / "REPORT.md").read_bytes(), "learning_curve_report.md", "text/markdown")


def render_optimization_view():
    root = RESULTS / "optimization"
    st.subheader("Training-only optimization & model explanations")
    st.write("Six configurations compare preprocessing, Thinking mode and modality subsets. Selection uses three folds within the official training cohort. The selected model is then evaluated on the reserved in-distribution test patients.")
    if not (root / "status.json").exists():
        st.info("No optimization artifacts yet. Run python src/tune_tabpfn.py in the paper environment.")
        return
    status = read_json(root / "status.json")
    if status.get("tuning_complete"):
        st.success("36 validation runs and both post-selection evaluations completed.")
    else:
        st.info(f"Exported {status['completed_cv_runs']} / {status['expected_cv_runs']} validation runs; {status['final_tests']} / 2 final evaluations.")
    st.caption("These test patients were previously evaluated in the paper replication. Results remain exploratory. The selected model's validation score is not an unbiased estimate of its generalization performance.")
    if (root / "cv_comparison.png").exists():
        st.image(str(root / "cv_comparison.png"))
    if (root / "test_summary.csv").exists() and (root / "test_summary.csv").stat().st_size > 1:
        table = pd.read_csv(root / "test_summary.csv")
        st.dataframe(table, hide_index=True, width="stretch")
        if len(table) and table.auc_difference.le(0).all():
            st.info("Neither selected model improved test ROC AUC over the untuned no-SMOTE baseline in this experiment. Consult the paired intervals for uncertainty; the original baseline remains available in Paper replication.")
    with st.expander("Validation metrics and experiment settings"):
        if (root / "cv_metrics.csv").exists():
            st.dataframe(pd.read_csv(root / "cv_metrics.csv"), hide_index=True, width="stretch")
            st.download_button("Download optimization metrics", (root / "cv_metrics.csv").read_bytes(), "optimization_cv.csv", "text/csv")
        if (root / "REPORT.md").exists():
            st.download_button("Download optimization report", (root / "REPORT.md").read_bytes(), "optimization_report.md", "text/markdown")
    st.divider()
    target = st.selectbox("Explanation endpoint", list(LABELS), format_func=LABELS.get, key="explanation_target")
    directory = root / "explanations" / target
    if not (directory / "summary.json").exists():
        st.info("Explanations for this selected model are not yet available.")
        return
    summary = read_json(directory / "summary.json")
    explanation_protocol = read_json(directory / "protocol.json")
    if explanation_protocol["protocol_id"] != status["protocol_id"]:
        st.warning("Explanation artifacts belong to a different optimization run.")
        return
    st.markdown(f"**Explaining `{summary['model']}`**")
    st.caption(f"{summary['explained_patients']} randomly sampled held-out patients · {summary['background_patients']} training-background patients · original clinical features · positive SHAP values increase event probability")
    rows = pd.read_csv(directory / "shap_values.csv", dtype={"patient_id": str})
    leading = summary["top_features"][0]["feature"]
    influence = rows.loc[rows.feature.eq(leading), "shap_value"].abs()
    if influence.sum() > 0 and influence.max()/influence.sum() > .5:
        st.info(f"Sample sensitivity: one patient accounts for {influence.max()/influence.sum():.0%} of the leading feature's absolute attribution ({leading}). This ranking should not be generalized to the entire cohort.")
    left, right = st.columns(2)
    with left:
        importance = pd.read_csv(directory / "feature_importance.csv")
        top = importance.head(15).iloc[::-1].copy()
        top["Mean absolute contribution (percentage points)"] = 100 * top.mean_absolute_shap
        fig = px.bar(top, x="Mean absolute contribution (percentage points)", y="feature", orientation="h", color_discrete_sequence=["#008c82"])
        fig.update_layout(height=620, title="Feature influence in the explained sample", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.image(str(directory / "modality_importance.png"))
        st.caption("Each modality is shuffled jointly on all held-out test patients. Error bars show variation across shuffles, not patient confidence intervals. Correlated modalities may substitute for each other.")
        st.metric("Top-10 overlap between SHAP ordering repeats", f"{summary['top10_ordering_overlap']} / {min(10, summary['feature_count'])}")
        st.caption(f"Feature-rank correlation: {summary['ordering_rank_spearman']:.2f}. This checks ordering sensitivity only, not stability across new cohorts or background samples.")
    with st.expander("SHAP beeswarm and complete feature ranking"):
        st.image(str(directory / "shap_beeswarm.png"))
        st.caption("Colors reflect published numeric/encoded feature values. High category codes do not imply higher clinical severity.")
        st.dataframe(importance, hide_index=True, width="stretch")
    st.subheader("Why did the model predict this patient's risk?")
    patient_id = st.selectbox("Explained held-out patient", summary["patient_ids"], key=f"explained_patient_{target}")
    patient = rows.loc[rows.patient_id.eq(patient_id)]
    values = patient.iloc[0]
    columns = st.columns(2)
    columns[0].metric("Mean prediction on training background", f"{values.base_probability:.1%}")
    columns[1].metric("Patient's predicted event probability", f"{values.predicted_probability:.1%}")
    st.image(str(directory / f"waterfall_{patient_id}.png"))
    st.caption("The baseline is the model's mean prediction on the sampled background, not the observed population event rate. Contributions reconstruct the displayed model probability.")
    st.download_button("Download SHAP attributions", (directory / "shap_values.csv").read_bytes(), f"{target}_shap.csv", "text/csv")
    st.download_button("Download modality importance", (directory / "modality_permutation.csv").read_bytes(), f"{target}_modality_importance.csv", "text/csv")
    with st.expander("Interpretation limits"):
        for note in summary["caveats"]:
            st.write(note)
        st.write("The paper endpoints differ from the patient dashboard's five-year endpoint. Explanations describe the selected research model and do not justify treatment changes.")
