"""
Statistical Validation Protocol (Section IV-E and Table VII in Paper):
- Non-Parametric Wilcoxon Signed-Rank Test
- Paired Student's t-Test
- 10,000-Resample Non-Parametric Bootstrap Confidence Intervals (95% CI)
- Cohen's d Effect Size
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple

def bootstrap_ci(
    data: np.ndarray, num_resamples: int = 10000, ci: float = 0.95, random_seed: int = 42
) -> Tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval."""
    rng = np.random.default_rng(random_seed)
    n = len(data)
    boot_means = []
    for _ in range(num_resamples):
        resample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(resample))
    
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return round(lower, 4), round(upper, 4)

def calculate_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates paired Cohen's d effect size."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(np.mean(diff) / std_diff)

def run_paired_statistical_tests(
    proposed_scores: List[float], baseline_scores: List[float], metric_name: str
) -> Dict[str, Any]:
    """
    Executes complete paired statistical hypothesis test suite (N = 20, df = 19):
    1. Wilcoxon Signed-Rank Test
    2. Paired Student's t-Test
    3. 10,000-resample Bootstrap 95% CIs
    4. Cohen's d effect size
    """
    prop = np.array(proposed_scores, dtype=np.float64)
    base = np.array(baseline_scores, dtype=np.float64)
    diff = prop - base

    prop_mean, prop_std = float(np.mean(prop)), float(np.std(prop, ddof=1))
    base_mean, base_std = float(np.mean(base)), float(np.std(base, ddof=1))
    diff_mean, diff_std = float(np.mean(diff)), float(np.std(diff, ddof=1))

    # Bootstrap 95% CIs
    prop_ci = bootstrap_ci(prop)
    base_ci = bootstrap_ci(base)
    diff_ci = bootstrap_ci(diff)

    # 1. Wilcoxon Signed-Rank Test
    # Handle ties / zero differences gracefully
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) > 0:
        try:
            w_res = stats.wilcoxon(prop, base, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat, wilcoxon_p = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    else:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    # 2. Paired Student's t-test
    try:
        t_res = stats.ttest_rel(prop, base)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
    except Exception:
        t_stat, t_p = 0.0, 1.0

    # 3. Cohen's d
    d = calculate_cohens_d(prop, base)

    # Significance label
    sig_label = "ns"
    if t_p < 0.001 or wilcoxon_p < 0.001:
        sig_label = "***"
    elif t_p < 0.01 or wilcoxon_p < 0.01:
        sig_label = "**"
    elif t_p < 0.05 or wilcoxon_p < 0.05:
        sig_label = "*"

    return {
        "metric": metric_name,
        "proposed_mean": round(prop_mean, 4),
        "proposed_std": round(prop_std, 4),
        "proposed_95_boot_ci": list(prop_ci),
        "baseline_mean": round(base_mean, 4),
        "baseline_std": round(base_std, 4),
        "baseline_95_boot_ci": list(base_ci),
        "diff_mean": round(diff_mean, 4),
        "diff_95_boot_ci": list(diff_ci),
        "wilcoxon_statistic": round(wilcoxon_stat, 2),
        "wilcoxon_p_value": wilcoxon_p,
        "paired_t_statistic": round(t_stat, 4),
        "paired_t_p_value": t_p,
        "cohens_d": round(d, 4),
        "significance": sig_label
    }
