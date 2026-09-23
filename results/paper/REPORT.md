# HANCOCK paper comparison

Completed runs: 90/90. Protocol: `6b092e8b8a9439e46cf6`.

This reproduces the released Figure 2 Random Forest experiment, then replaces RF with untuned TabPFN-3.5. The primary comparison retains identical SMOTE-augmented matrices. A prespecified secondary comparison omits SMOTE for TabPFN.

| Endpoint | Split | Model | ROC AUC, mean ± SD |
|---|---|---|---:|
| Death status (paper exclusions) | In distribution | Published RF + SMOTE | 0.7909 ± 0.0047 |
| Death status (paper exclusions) | In distribution | TabPFN-3.5 + SMOTE (primary) | 0.6943 ± 0.0189 |
| Death status (paper exclusions) | In distribution | TabPFN-3.5, no SMOTE (secondary) | 0.8359 ± 0.0015 |
| Death status (paper exclusions) | Out of distribution | Published RF + SMOTE | 0.7797 ± 0.0034 |
| Death status (paper exclusions) | Out of distribution | TabPFN-3.5 + SMOTE (primary) | 0.7401 ± 0.0080 |
| Death status (paper exclusions) | Out of distribution | TabPFN-3.5, no SMOTE (secondary) | 0.7896 ± 0.0014 |
| Death status (paper exclusions) | Oropharynx | Published RF + SMOTE | 0.7107 ± 0.0101 |
| Death status (paper exclusions) | Oropharynx | TabPFN-3.5 + SMOTE (primary) | 0.6911 ± 0.0051 |
| Death status (paper exclusions) | Oropharynx | TabPFN-3.5, no SMOTE (secondary) | 0.7682 ± 0.0025 |
| Recurrence (released-code eligibility) | In distribution | Published RF + SMOTE | 0.7890 ± 0.0064 |
| Recurrence (released-code eligibility) | In distribution | TabPFN-3.5 + SMOTE (primary) | 0.7186 ± 0.0088 |
| Recurrence (released-code eligibility) | In distribution | TabPFN-3.5, no SMOTE (secondary) | 0.7893 ± 0.0018 |
| Recurrence (released-code eligibility) | Out of distribution | Published RF + SMOTE | 0.7149 ± 0.0046 |
| Recurrence (released-code eligibility) | Out of distribution | TabPFN-3.5 + SMOTE (primary) | 0.6770 ± 0.0075 |
| Recurrence (released-code eligibility) | Out of distribution | TabPFN-3.5, no SMOTE (secondary) | 0.7265 ± 0.0014 |
| Recurrence (released-code eligibility) | Oropharynx | Published RF + SMOTE | 0.6872 ± 0.0069 |
| Recurrence (released-code eligibility) | Oropharynx | TabPFN-3.5 + SMOTE (primary) | 0.6260 ± 0.0042 |
| Recurrence (released-code eligibility) | Oropharynx | TabPFN-3.5, no SMOTE (secondary) | 0.6796 ± 0.0015 |

## Paired differences versus reproduced RF

95% intervals resample test patients, retaining all five repetitions together. They describe test-sample uncertainty conditional on these training sets; they are exploratory, unadjusted for multiple comparisons. Repetitions and the three overlapping test partitions are not independent cohorts.

| Endpoint | Split | TabPFN variant | ΔAUC | 95% paired interval |
|---|---|---|---:|---:|
| Death status (paper exclusions) | In distribution | TabPFN-3.5 + SMOTE (primary) | -0.0966 | [-0.1837, -0.0104] |
| Death status (paper exclusions) | In distribution | TabPFN-3.5, no SMOTE (secondary) | +0.0449 | [+0.0016, +0.0929] |
| Death status (paper exclusions) | Out of distribution | TabPFN-3.5 + SMOTE (primary) | -0.0396 | [-0.0878, +0.0121] |
| Death status (paper exclusions) | Out of distribution | TabPFN-3.5, no SMOTE (secondary) | +0.0099 | [-0.0353, +0.0577] |
| Death status (paper exclusions) | Oropharynx | TabPFN-3.5 + SMOTE (primary) | -0.0196 | [-0.0852, +0.0475] |
| Death status (paper exclusions) | Oropharynx | TabPFN-3.5, no SMOTE (secondary) | +0.0575 | [+0.0067, +0.1070] |
| Recurrence (released-code eligibility) | In distribution | TabPFN-3.5 + SMOTE (primary) | -0.0704 | [-0.1327, -0.0154] |
| Recurrence (released-code eligibility) | In distribution | TabPFN-3.5, no SMOTE (secondary) | +0.0003 | [-0.0364, +0.0340] |
| Recurrence (released-code eligibility) | Out of distribution | TabPFN-3.5 + SMOTE (primary) | -0.0379 | [-0.0924, +0.0148] |
| Recurrence (released-code eligibility) | Out of distribution | TabPFN-3.5, no SMOTE (secondary) | +0.0116 | [-0.0362, +0.0623] |
| Recurrence (released-code eligibility) | Oropharynx | TabPFN-3.5 + SMOTE (primary) | -0.0612 | [-0.1077, -0.0153] |
| Recurrence (released-code eligibility) | Oropharynx | TabPFN-3.5, no SMOTE (secondary) | -0.0076 | [-0.0571, +0.0462] |

## Reproduction checks and limits

Published RF AUCs (rounded): death status 0.79 / 0.78 / 0.71; recurrence 0.79 / 0.71 / 0.69 (in / out / Oropharynx). Our reproduced RF is the paired comparator; published rounded numbers are reference values only.

- Official, real 763-patient cohort; endpoint eligibility reduces the train/test counts. No synthetic fallback.
- Frozen clinical, pathology, hematology, ICD and CD3/CD8 TMA feature tables; train-fitted one-hot encoding, imputation and scaling. No UMAP model input.
- Reused feature tables inherit the authors' upstream cohort-wide encoding/blood imputation; this is a replication, not a fully re-engineered leakage-free preprocessing study.
- Death status excludes known non-tumor-specific deaths, retains unknown causes, and has no five-year horizon.
- Recurrence uses released-code eligibility: positive within 1095 days, or nonrecurrent with >1095 days follow-up OR living status. The living exception differs from the prose. This is not a strict censoring-aware three-year endpoint.
- Frozen RF hyperparameters are used without rerunning search. Original core dependency versions and both advancing RNG streams are preserved. Ancillary dependencies/platform can still affect exact numerical reproduction.
- TabPFN uses five seeds with v3.5_default through tabpfn-client 0.6.0, without tuning, feature selection or probability recalibration. Server-resolved defaults/weights are not fully pinned by the client.
- Brier/ECE are descriptive raw-probability metrics; SMOTE changes class prevalence. No claim of calibrated clinical probabilities follows from this comparison.
- This compares Figure 2 RF, not the paper's separate foundation-model/RFS experiments. No external validation or treatment-effect inference.

Sources: [paper](https://doi.org/10.1038/s41467-025-62386-6), [frozen authors' code](https://github.com/ankilab/HANCOCK_MultimodalDataset/tree/521b99b03a94008b28df5c3df4aa5f82aa14b25a). Full provenance, versions, feature names, patient memberships, per-run predictions and configuration are saved alongside this report.
