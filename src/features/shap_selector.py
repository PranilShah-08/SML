"""
SHAP-Stability Feature Selection (Novelty 1).
Selects invariant, generalisable network flow features by penalizing features whose
SHAP attribution rank fluctuates across temporal days or distinct network distributions.
"""

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from scipy.stats import spearmanr
from typing import List, Tuple, Dict

class SHAPStabilitySelector:
    def __init__(self, n_features_to_select: int = 25, lambda_penalty: float = 2.0, seed: int = 42):
        self.n_features_to_select = n_features_to_select
        self.lambda_penalty = lambda_penalty
        self.seed = seed
        self.selected_indices: List[int] = []
        self.selected_names: List[str] = []
        self.stability_df: pd.DataFrame = pd.DataFrame()
        self.spearman_rank_corr: float = 0.0

    def fit(
        self,
        X_day1: np.ndarray,
        y_day1: np.ndarray,
        X_day2: np.ndarray,
        y_day2: np.ndarray,
        feature_names: List[str],
        sample_size: int = 1000
    ) -> 'SHAPStabilitySelector':
        """
        Fits LightGBM on Day 1 and Day 2 independently, extracts TreeSHAP values,
        and computes cross-day rank stability.
        """
        print(f"[SHAP-Stability] Evaluating cross-day feature attribution stability...")
        n_features = len(feature_names)

        # 1. Model Day 1 (e.g. Tuesday)
        clf1 = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=self.seed, verbose=-1, n_jobs=-1)
        clf1.fit(X_day1, y_day1)
        
        # 2. Model Day 2 (e.g. Wednesday)
        clf2 = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=self.seed, verbose=-1, n_jobs=-1)
        clf2.fit(X_day2, y_day2)

        # Compute SHAP values on background samples
        idx1 = np.random.RandomState(self.seed).choice(len(X_day1), size=min(sample_size, len(X_day1)), replace=False)
        idx2 = np.random.RandomState(self.seed).choice(len(X_day2), size=min(sample_size, len(X_day2)), replace=False)

        explainer1 = shap.TreeExplainer(clf1)
        explainer2 = shap.TreeExplainer(clf2)

        shap_vals1 = explainer1.shap_values(X_day1[idx1])
        shap_vals2 = explainer2.shap_values(X_day2[idx2])

        # Handle binary vs multiclass output format from TreeExplainer
        if isinstance(shap_vals1, list):
            sv1 = np.abs(shap_vals1[1]).mean(axis=0)
        elif len(shap_vals1.shape) == 3:
            sv1 = np.abs(shap_vals1[:, :, 1]).mean(axis=0)
        else:
            sv1 = np.abs(shap_vals1).mean(axis=0)

        if isinstance(shap_vals2, list):
            sv2 = np.abs(shap_vals2[1]).mean(axis=0)
        elif len(shap_vals2.shape) == 3:
            sv2 = np.abs(shap_vals2[:, :, 1]).mean(axis=0)
        else:
            sv2 = np.abs(shap_vals2).mean(axis=0)

        # Ranks: rank 1 is highest importance
        rank1 = pd.Series(sv1).rank(ascending=False).values
        rank2 = pd.Series(sv2).rank(ascending=False).values

        # Spearman rank correlation across temporal days
        corr, _ = spearmanr(rank1, rank2)
        self.spearman_rank_corr = float(corr)

        # Compute rank divergence and stability score
        rank_diff = np.abs(rank1 - rank2)
        combined_importance = (sv1 + sv2) / 2.0
        
        # Stability Score: penalizes large rank jumps across temporal distributions
        # Score = Mean_Importance / (1 + lambda * (rank_diff / n_features))
        stability_score = combined_importance / (1.0 + self.lambda_penalty * (rank_diff / float(n_features)))

        df = pd.DataFrame({
            'feature': feature_names,
            'shap_day1': sv1,
            'rank_day1': rank1,
            'shap_day2': sv2,
            'rank_day2': rank2,
            'rank_delta': rank_diff,
            'mean_shap': combined_importance,
            'stability_score': stability_score
        })

        df = df.sort_values(by='stability_score', ascending=False).reset_index(drop=True)
        self.stability_df = df

        # Select top K
        selected_features = df.head(self.n_features_to_select)['feature'].tolist()
        self.selected_names = selected_features
        self.selected_indices = [feature_names.index(f) for f in selected_features]

        print(f"[SHAP-Stability] Cross-day Spearman Rank Correlation: {self.spearman_rank_corr:.4f}")
        print(f"[SHAP-Stability] Selected {len(self.selected_indices)} stable features out of {n_features}.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.selected_indices:
            raise ValueError("Selector has not been fitted yet.")
        return X[:, self.selected_indices]
