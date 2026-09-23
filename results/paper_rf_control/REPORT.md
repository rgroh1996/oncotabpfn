# No-SMOTE Random Forest control

Protocol: `a5c2be575d46a8f1b814`. Supplementary to the frozen [paper comparison](../paper/REPORT.md), whose files are read but not modified.

The frozen study's headline compares TabPFN without SMOTE against RF with SMOTE. That changes the model and the pipeline at once. This control fits the published RF hyperparameters on the same original training matrices without SMOTE, so the TabPFN-versus-RF difference below reflects the model alone.

| Endpoint | Split | Published RF + SMOTE | Published RF, no SMOTE | TabPFN-3.5, no SMOTE |
|---|---|---:|---:|---:|
| Death status | In distribution | 0.791 | 0.825 | 0.836 |
| Death status | Out of distribution | 0.780 | 0.774 | 0.790 |
| Death status | Oropharynx | 0.711 | 0.709 | 0.768 |
| Recurrence | In distribution | 0.789 | 0.783 | 0.789 |
| Recurrence | Out of distribution | 0.715 | 0.740 | 0.727 |
| Recurrence | Oropharynx | 0.687 | 0.692 | 0.680 |

Mean ROC AUC over five repetitions.

## Paired differences

| Endpoint | Split | Comparison | ΔAUC | 95% paired interval |
|---|---|---|---:|---:|
| Death status | In distribution | TabPFN-3.5, no SMOTE vs Published RF, no SMOTE | +0.0104 | [-0.0371, +0.0610] |
| Death status | In distribution | Published RF, no SMOTE vs Published RF + SMOTE | +0.0346 | [-0.0184, +0.0966] |
| Death status | Out of distribution | TabPFN-3.5, no SMOTE vs Published RF, no SMOTE | +0.0155 | [-0.0347, +0.0671] |
| Death status | Out of distribution | Published RF, no SMOTE vs Published RF + SMOTE | -0.0057 | [-0.0657, +0.0551] |
| Death status | Oropharynx | TabPFN-3.5, no SMOTE vs Published RF, no SMOTE | +0.0593 | [+0.0112, +0.1057] |
| Death status | Oropharynx | Published RF, no SMOTE vs Published RF + SMOTE | -0.0017 | [-0.0598, +0.0550] |
| Recurrence | In distribution | TabPFN-3.5, no SMOTE vs Published RF, no SMOTE | +0.0067 | [-0.0460, +0.0611] |
| Recurrence | In distribution | Published RF, no SMOTE vs Published RF + SMOTE | -0.0064 | [-0.0534, +0.0439] |
| Recurrence | Out of distribution | TabPFN-3.5, no SMOTE vs Published RF, no SMOTE | -0.0135 | [-0.0637, +0.0355] |
| Recurrence | Out of distribution | Published RF, no SMOTE vs Published RF + SMOTE | +0.0251 | [-0.0169, +0.0711] |
| Recurrence | Oropharynx | TabPFN-3.5, no SMOTE vs Published RF, no SMOTE | -0.0128 | [-0.0666, +0.0382] |
| Recurrence | Oropharynx | Published RF, no SMOTE vs Published RF + SMOTE | +0.0052 | [-0.0573, +0.0689] |

## Limits

- Added after the frozen results were seen; it is a post-hoc control, not a prespecified comparison.
- The published RF hyperparameters were searched by the authors for the SMOTE pipeline and are not retuned here. The [learning-curve study](../learning_curves/REPORT.md) contains an independently tuned no-SMOTE RF.
- Intervals resample test patients conditional on the fixed training sets; unadjusted for multiple comparisons. The three test partitions overlap and were examined in earlier experiments.
