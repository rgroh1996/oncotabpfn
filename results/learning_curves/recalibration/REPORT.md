# Calibration sensitivity analysis

Supplementary to the frozen [learning-curve study](../REPORT.md) (`38eb9ea63b87ca137237`), which is not modified.

The frozen sigmoid calibrator has an unconstrained slope. In **11 of 300** matched calibration runs it learned a slope ≤ 0 from small, noisy out-of-fold scores, which reverses the test ranking. Here every calibrator is refitted on the same saved out-of-fold probabilities with the same C=1 objective, constrained to a nonnegative slope. Unconstrained refits reproduce every frozen calibrator first.

| Endpoint | Model | Size | Seed | Frozen slope |
|---|---|---:|---:|---:|
| recurrence | Tuned LightGBM | 50 | 42 | -0.106 |
| recurrence | Tuned LightGBM | 50 | 44 | -0.124 |
| recurrence | Tuned LightGBM | 100 | 42 | -0.067 |
| recurrence | Tuned LightGBM | 100 | 44 | -0.058 |
| recurrence | Tuned RF + SMOTE | 50 | 42 | -0.236 |
| recurrence | TabPFN-3.5 | 50 | 42 | -0.247 |
| recurrence | TabPFN-3.5 | 50 | 44 | -0.080 |
| recurrence | Tuned XGBoost | 50 | 42 | -0.195 |
| recurrence | Tuned XGBoost | 50 | 45 | -0.029 |
| recurrence | Tuned XGBoost | 100 | 42 | -0.139 |
| survival_status | Tuned LightGBM | 100 | 43 | -0.020 |

## Mean over N=50, 100, 200 and five matched seeds

| Endpoint | Model | Raw Brier | Frozen sigmoid Brier | Nonnegative sigmoid Brier | Raw ECE | Frozen ECE | Nonnegative ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| recurrence | Tuned LightGBM | 0.1789 | 0.1721 | 0.1717 | 0.1233 | 0.0579 | 0.0579 |
| recurrence | Tuned logistic | 0.1836 | 0.1755 | 0.1755 | 0.0753 | 0.0487 | 0.0487 |
| recurrence | Tuned RF | 0.1611 | 0.1642 | 0.1642 | 0.0835 | 0.0659 | 0.0659 |
| recurrence | Tuned RF + SMOTE | 0.1725 | 0.1701 | 0.1698 | 0.1016 | 0.0681 | 0.0648 |
| recurrence | TabPFN-3.5 | 0.1502 | 0.1571 | 0.1566 | 0.0737 | 0.0672 | 0.0620 |
| recurrence | Tuned XGBoost | 0.1771 | 0.1739 | 0.1737 | 0.1045 | 0.0549 | 0.0551 |
| survival_status | Tuned LightGBM | 0.1586 | 0.1445 | 0.1444 | 0.1214 | 0.0443 | 0.0440 |
| survival_status | Tuned logistic | 0.1512 | 0.1409 | 0.1409 | 0.0910 | 0.0518 | 0.0518 |
| survival_status | Tuned RF | 0.1322 | 0.1343 | 0.1343 | 0.0550 | 0.0617 | 0.0614 |
| survival_status | Tuned RF + SMOTE | 0.1554 | 0.1423 | 0.1423 | 0.1244 | 0.0449 | 0.0449 |
| survival_status | TabPFN-3.5 | 0.1385 | 0.1400 | 0.1400 | 0.0568 | 0.0671 | 0.0671 |
| survival_status | Tuned XGBoost | 0.1545 | 0.1452 | 0.1452 | 0.0968 | 0.0447 | 0.0447 |

A slope bounded at zero yields a constant prediction for that run: the calibrator reports no usable signal instead of reversing the ranking. The raw-AUC primary outcome of the frozen study does not use calibration and is unchanged. This analysis was added after the frozen results were seen.
