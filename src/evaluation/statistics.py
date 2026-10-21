"""
Statistical Rigor & Significance Testing Framework.
Implements:
1. Multi-seed execution and mean +/- std aggregation.
2. Friedman test across models.
3. Nemenyi post-hoc critical difference calculation.
4. Wilcoxon signed-rank test for pairwise statistical comparisons.
5. Optuna hyperparameter tuning strictly on validation folds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import friedmanchisquare, wilcoxon
import optuna

# Studentized range critical values q_alpha / sqrt(2) for alpha = 0.05
NEMENYI_Q_ALPHA_005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268
}

def compute_friedman_and_nemenyi(
    scores_dict: Dict[str, List[float]],
    higher_is_better: bool = True
) -> Dict[str, Any]:
    """
    Computes Friedman test and Nemenyi Critical Difference (CD) across models.
    scores_dict: {model_name: [score_seed1, score_seed2, ...]}
    """
    model_names = list(scores_dict.keys())
    k = len(model_names)
    n_seeds = len(scores_dict[model_names[0]])

    matrix = np.array([scores_dict[m] for m in model_names]).T  # shape (n_seeds, k)

    # Compute ranks per seed (1 = best)
    ranks = np.zeros_like(matrix)
    for i in range(n_seeds):
        row = matrix[i]
        if higher_is_better:
            # higher score -> rank 1
            ranks[i] = pd.Series(-row).rank().values
        else:
            ranks[i] = pd.Series(row).rank().values

    mean_ranks = np.mean(ranks, axis=0)
    rank_dict = {model_names[j]: float(mean_ranks[j]) for j in range(k)}

    # Friedman test
    try:
        stat, p_val = friedmanchisquare(*[matrix[:, j] for j in range(k)])
    except Exception:
        stat, p_val = 0.0, 1.0

    # Iman-Davenport correction: less conservative than standard chi2 Friedman test
    # F_F = ( (N - 1) * chi2_F ) / ( N * (k - 1) - chi2_F ) ~ F(k-1, (k-1)*(N-1))
    from scipy.stats import f as f_dist
    df1 = k - 1
    df2 = (k - 1) * (n_seeds - 1)
    denom = (n_seeds * (k - 1) - float(stat))
    if denom > 1e-9:
        f_stat_id = ((n_seeds - 1) * float(stat)) / denom
        p_val_id = float(1.0 - f_dist.cdf(f_stat_id, df1, df2))
    else:
        f_stat_id = float('inf')
        p_val_id = 0.0

    # Nemenyi CD
    q_val = NEMENYI_Q_ALPHA_005.get(k, 3.0)
    cd = q_val * np.sqrt((k * (k + 1.0)) / (6.0 * n_seeds))

    return {
        'friedman_stat': float(stat),
        'friedman_p_value': float(p_val),
        'iman_davenport_f_stat': float(f_stat_id),
        'iman_davenport_p_value': float(p_val_id),
        'df1': df1,
        'df2': df2,
        'mean_ranks': rank_dict,
        'nemenyi_cd': float(cd),
        'k_models': k,
        'n_evaluations': n_seeds,
        'assumption_note': (
            f"Evaluated across {n_seeds} seeds on the same benchmark partition. "
            f"Assumptions of asymptotic chi2 distribution are conservative with N={n_seeds} and k={k}; "
            f"Iman-Davenport F_F is reported for robust validation."
        )
    }

def compute_pairwise_wilcoxon(
    proposed_scores: List[float],
    baseline_scores: List[float]
) -> Dict[str, Any]:
    """
    Computes Wilcoxon signed-rank test, effect sizes (Cliff's Delta, Rank-Biserial r, Cohen's d),
    and 95% bootstrap confidence intervals between proposed method and baseline across seeds.
    
    CRITICAL STATISTICAL RIGOR NOTE:
    For N = 5 paired observations, the minimum possible two-sided p-value for the Wilcoxon
    signed-rank test is 2 * (1/2)^5 = 1/16 = 0.0625. Thus, at alpha = 0.05, a sample of 5 seeds
    can NEVER achieve statistical significance under a two-tailed test, even with uniform dominance.
    Reporting effect sizes and uncertainty intervals is mandatory.
    """
    diffs = np.array(proposed_scores) - np.array(baseline_scores)
    n = len(diffs)
    if np.all(diffs == 0):
        return {
            'stat': 0.0,
            'p_value': 1.0,
            'mean_diff': 0.0,
            'rank_biserial_r': 0.0,
            'cliffs_delta': 0.0,
            'cohens_d': 0.0,
            'ci_95_lower': 0.0,
            'ci_95_upper': 0.0,
            'is_significant_alpha_005': False
        }
    try:
        stat, p_val = wilcoxon(proposed_scores, baseline_scores)
        stat = float(stat)
        p_val = float(p_val)
    except Exception:
        stat, p_val = 0.0, 1.0

    # Rank-biserial correlation r = 1 - 2W / S
    S = n * (n + 1) / 2.0
    rank_biserial = 1.0 - (2.0 * stat / S) if S > 0 else 0.0

    # Cliff's Delta
    all_diffs = [xi - yj for xi in proposed_scores for yj in baseline_scores]
    n_pos = sum(1 for d in all_diffs if d > 0)
    n_neg = sum(1 for d in all_diffs if d < 0)
    cliffs_delta = (n_pos - n_neg) / len(all_diffs) if len(all_diffs) > 0 else 0.0

    # Cohen's d for paired differences
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 1e-9
    cohens_d = mean_diff / (std_diff + 1e-9)

    # 95% Bootstrap CI
    rng = np.random.RandomState(42)
    boot_diffs = [np.mean(diffs[rng.choice(n, size=n, replace=True)]) for _ in range(2000)]
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        'stat': stat,
        'p_value': p_val,
        'mean_diff': mean_diff,
        'rank_biserial_r': float(rank_biserial),
        'cliffs_delta': float(cliffs_delta),
        'cohens_d': float(cohens_d),
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'is_significant_alpha_005': bool(p_val < 0.05)
    }

def compute_holm_bonferroni(
    p_values_dict: Dict[str, float]
) -> Dict[str, Dict[str, Any]]:
    """
    Applies Holm-Bonferroni step-down correction for family-wise error rate control.
    """
    sorted_items = sorted(p_values_dict.items(), key=lambda x: x[1])
    m = len(sorted_items)
    results = {}
    running_p = 0.0
    for idx, (name, raw_p) in enumerate(sorted_items):
        factor = m - idx
        adj_p = min(1.0, max(running_p, raw_p * factor))
        running_p = adj_p
        results[name] = {
            'raw_p': raw_p,
            'holm_p': adj_p,
            'is_significant_alpha_005': bool(adj_p < 0.05)
        }
    return results

def tune_lightgbm_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Optuna hyperparameter optimization strictly on validation fold (zero test leakage).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        import lightgbm as lgb
        from sklearn.metrics import average_precision_score

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'class_weight': 'balanced',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': seed
        }

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, preds)
        return score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
