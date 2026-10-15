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

    # Nemenyi CD
    q_val = NEMENYI_Q_ALPHA_005.get(k, 3.0)
    cd = q_val * np.sqrt((k * (k + 1.0)) / (6.0 * n_seeds))

    return {
        'friedman_stat': float(stat),
        'friedman_p_value': float(p_val),
        'mean_ranks': rank_dict,
        'nemenyi_cd': float(cd),
        'k_models': k,
        'n_evaluations': n_seeds
    }

def compute_pairwise_wilcoxon(
    proposed_scores: List[float],
    baseline_scores: List[float]
) -> Tuple[float, float]:
    """
    Computes Wilcoxon signed-rank test between proposed method and baseline.
    Returns (statistic, p_value).
    """
    diffs = np.array(proposed_scores) - np.array(baseline_scores)
    if np.all(diffs == 0):
        return 0.0, 1.0
    try:
        stat, p_val = wilcoxon(proposed_scores, baseline_scores)
        return float(stat), float(p_val)
    except Exception:
        return 0.0, 1.0

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
