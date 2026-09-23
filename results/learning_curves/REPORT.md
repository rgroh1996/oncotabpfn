# Paper-matched sample-efficiency experiment

Completed 600 final model evaluations: two endpoints × five sizes × ten matched draws × six pipelines.

**Quota amendment:** at the user's request, calibration is evaluated on fixed seeds 42–46 for every model and endpoint. All ten seeds remain in the primary raw-AUC comparison. Previously computed extra calibration predictions are retained locally but excluded from matched calibration summaries. The change was driven by API allowance, not model scores. See quota_amendment.json.

Primary comparison: mean raw ROC-AUC over N=50,100,200, with equal weight for each size and seed. Positive differences favor TabPFN. Intervals are paired, class-stratified test-patient bootstrap intervals conditional on the fixed training draws, unadjusted across comparisons. This is exploratory.

| Endpoint | Comparator | AUC difference | 95% interval |
|---|---|---:|---:|
| Death status (paper endpoint) | Tuned RF | -0.0203 | [-0.0613, +0.0194] |
| Death status (paper endpoint) | Tuned RF + SMOTE | +0.0359 | [+0.0063, +0.0642] |
| Death status (paper endpoint) | Tuned LightGBM | +0.0507 | [+0.0126, +0.0861] |
| Death status (paper endpoint) | Tuned XGBoost | +0.0553 | [+0.0132, +0.0943] |
| Death status (paper endpoint) | Tuned logistic | +0.0416 | [-0.0065, +0.0937] |
| Recurrence (paper endpoint) | Tuned RF | +0.0163 | [-0.0177, +0.0486] |
| Recurrence (paper endpoint) | Tuned RF + SMOTE | +0.0469 | [+0.0211, +0.0706] |
| Recurrence (paper endpoint) | Tuned LightGBM | +0.0664 | [+0.0301, +0.1002] |
| Recurrence (paper endpoint) | Tuned XGBoost | +0.0656 | [+0.0229, +0.1061] |
| Recurrence (paper endpoint) | Tuned logistic | +0.0703 | [+0.0168, +0.1298] |

## Mean raw AUC by training size

| Endpoint | N | TabPFN-3.5 | Tuned RF | Tuned RF + SMOTE | Tuned LightGBM | Tuned XGBoost | Tuned logistic |
|---|---|---:|---:|---:|---:|---:|---:|
| Death status (paper endpoint) | 50 | 0.6921 | 0.7151 | 0.6653 | 0.6385 | 0.6542 | 0.6416 |
| Death status (paper endpoint) | 100 | 0.7305 | 0.7492 | 0.6855 | 0.6769 | 0.6647 | 0.6908 |
| Death status (paper endpoint) | 200 | 0.7525 | 0.7718 | 0.7167 | 0.7076 | 0.6904 | 0.7179 |
| Death status (paper endpoint) | 500 | 0.8239 | 0.8104 | 0.7732 | 0.7440 | 0.7835 | 0.7477 |
| Death status (paper endpoint) | Full | 0.8342 | 0.8278 | 0.7797 | 0.7709 | 0.7933 | 0.7548 |
| Recurrence (paper endpoint) | 50 | 0.6593 | 0.6229 | 0.5940 | 0.5769 | 0.5786 | 0.5910 |
| Recurrence (paper endpoint) | 100 | 0.7222 | 0.7142 | 0.6786 | 0.6440 | 0.6470 | 0.6446 |
| Recurrence (paper endpoint) | 200 | 0.7544 | 0.7498 | 0.7225 | 0.7157 | 0.7135 | 0.6895 |
| Recurrence (paper endpoint) | 500 | 0.7849 | 0.7637 | 0.7728 | 0.7188 | 0.7391 | 0.7691 |
| Recurrence (paper endpoint) | Full | 0.7885 | 0.7720 | 0.7850 | 0.7343 | 0.7457 | 0.7890 |

## Interpretation of the primary comparison

- Death status (paper endpoint): Tuned RF has the highest low-N mean AUC (0.7453); TabPFN scores 0.7250. TabPFN minus no-SMOTE RF is -0.0203 with interval [-0.0613, +0.0194], which includes zero.
- Recurrence (paper endpoint): TabPFN-3.5 has the highest low-N mean AUC (0.7120); TabPFN scores 0.7120. TabPFN minus no-SMOTE RF is +0.0163 with interval [-0.0177, +0.0486], which includes zero.

## Calibration and the SMOTE control

The calibration plot shows raw and OOF-sigmoid Brier scores for every pipeline at every size. Brier score measures overall probability accuracy, including discrimination; it is not a pure calibration metric. ECE uses ten equal-width bins and is noisy on these small test sets. Sigmoid recalibration is not guaranteed to improve either metric.

All models receive identical original patients and preprocessing matrices before the optional SMOTE step. Each baseline searches four prespecified candidates by three-fold training-only AUC. RF uses 300 trees, boosting 200. The RF control compares independently tuned pipelines with and without SMOTE; the earlier exact published forest reproduction remains in results/paper.

Preprocessing is refit inside each N-patient subset and each inner training fold. Calibration fits a C=1 logistic sigmoid on the selected candidate's OOF probabilities and labels from that same N-patient subset; final models then use all N. No held-out test labels enter tuning or calibration. The raw AUC is the primary outcome, irrespective of calibration.

## Limitations

- Official test patients were evaluated in previous experiments; this is exploratory, not a new untouched test set.
- The published feature tables contain cohort-wide blood imputation and categorical encoding.
- Death status is not five-year overall survival; recurrence preserves the released code's living-patient exception.
- The no-SMOTE and SMOTE RF pipelines are tuned independently with the same grid; they are pipeline-level controls.
- Four candidates per baseline are a bounded search, not exhaustive optimization. TabPFN is untuned.
- Calibration reuses selected-candidate OOF scores; these training scores are not unbiased evaluation estimates.
- Full repeats share all training patients. Repetitions and sizes do not create independent test cohorts.
- Only the official in-distribution split is studied here. No external-validation claim is supported.

These results can establish an advantage within this experimental setting, not universal model superiority. All model and size comparisons, including unfavorable results, are retained in metrics.csv and paired_comparisons.csv.

## API usage

The amended continuation used 1,000,000 additional tokens, below the 1,100,000 cap. The recorded daily allowance remaining at completion was 530,000 tokens.

## Numerical audit

Refitted 1300 logistic models and 575 sigmoid calibrators. Independent einsum probability and objective-gradient checks verified the saved predictions despite matmul warnings in the frozen macOS environment. Maximum logistic probability difference: 2.94e-15. No results were changed.
