# Observed benchmark findings

Source: **real**. Split: official.

Run statuses: {'complete': 135, 'skipped': 15}.

Metrics use native model probabilities. Error bars describe variation across repetitions, not independent-patient confidence intervals.

- Five-year mortality · TabPFN-3.5 · N=50: AUC 0.702, AP 0.689, Brier 0.222, ECE 0.077.
- Five-year mortality · LightGBM · N=50: AUC 0.573, AP 0.541, Brier 0.274, ECE 0.198.
- Five-year mortality · XGBoost · N=50: AUC 0.595, AP 0.560, Brier 0.319, ECE 0.278.
- Five-year mortality · TabPFN-3.5 · N=Full: AUC 0.749, AP 0.785, Brier 0.195, ECE 0.100.
- Five-year mortality · LightGBM · N=Full: AUC 0.719, AP 0.720, Brier 0.229, ECE 0.178.
- Five-year mortality · XGBoost · N=Full: AUC 0.665, AP 0.650, Brier 0.289, ECE 0.246.
- Locoregional recurrence · TabPFN-3.5 · N=50: AUC 0.635, AP 0.449, Brier 0.182, ECE 0.109.
- Locoregional recurrence · LightGBM · N=50: AUC 0.552, AP 0.293, Brier 0.211, ECE 0.151.
- Locoregional recurrence · XGBoost · N=50: AUC 0.552, AP 0.286, Brier 0.245, ECE 0.221.
- Locoregional recurrence · TabPFN-3.5 · N=Full: AUC 0.717, AP 0.572, Brier 0.135, ECE 0.055.
- Locoregional recurrence · LightGBM · N=Full: AUC 0.701, AP 0.621, Brier 0.137, ECE 0.125.
- Locoregional recurrence · XGBoost · N=Full: AUC 0.684, AP 0.569, Brier 0.149, ECE 0.121.

Endpoint eligibility excludes early-censored survival observations. Recurrence is documented occurrence over variable follow-up. Repeated evaluations reuse patients; these are exploratory results from one center.
