"""
Drift Monitor & Selective Active Retraining (Novelty 3).
Combines:
1. Sequential ADWIN drift detection on conformal p-value streams
2. 1-Wasserstein divergence tracking on SHAP feature attributions
3. Budget-constrained Active Learning retraining on highest-uncertainty / zero-day flows
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import wasserstein_distance
from river.drift import ADWIN
import shap
import lightgbm as lgb

class DriftMonitor:
    def __init__(self, delta: float = 0.01, budget_retrain: int = 200, seed: int = 42):
        """
        Args:
            delta: ADWIN confidence parameter (smaller = fewer false alarms).
            budget_retrain: Number of flows to query oracle for selective active retraining.
        """
        self.adwin = ADWIN(delta=delta)
        self.budget_retrain = budget_retrain
        self.seed = seed
        self.drift_detected_indices: List[int] = []
        self.feature_drift_scores: Dict[str, float] = {}

    def monitor_stream(
        self,
        p_values_stream: np.ndarray,
        chunk_size: int = 500
    ) -> List[int]:
        """
        Feeds max conformal p-values sequentially through ADWIN.
        Returns indices of detected change-points.
        """
        self.drift_detected_indices = []
        max_p_vals = np.max(p_values_stream, axis=1)

        for i, val in enumerate(max_p_vals):
            self.adwin.update(float(val))
            if self.adwin.drift_detected:
                self.drift_detected_indices.append(i)

        print(f"[Drift Monitor] ADWIN detected {len(self.drift_detected_indices)} concept/covariate drift events in stream.")
        return self.drift_detected_indices

    def analyze_attribution_drift(
        self,
        model,
        X_base: np.ndarray,
        X_drifted: np.ndarray,
        feature_names: List[str],
        sample_n: int = 500
    ) -> pd.DataFrame:
        """
        Computes 1-Wasserstein distance between baseline SHAP distributions and drifted SHAP distributions.
        Identifies root-cause features behind the distribution shift ("Why Drift Happened").
        """
        print("[Drift Monitor] Computing 1-Wasserstein divergence on SHAP attribution distributions...")
        rng = np.random.RandomState(self.seed)
        
        idx_base = rng.choice(len(X_base), size=min(sample_n, len(X_base)), replace=False)
        idx_drift = rng.choice(len(X_drifted), size=min(sample_n, len(X_drifted)), replace=False)

        explainer = shap.TreeExplainer(model)
        shap_base = explainer.shap_values(X_base[idx_base])
        shap_drift = explainer.shap_values(X_drifted[idx_drift])

        # Normalize format
        def get_pos_shap(sv):
            if isinstance(sv, list):
                return sv[1]
            elif len(sv.shape) == 3:
                return sv[:, :, 1]
            return sv

        sb = get_pos_shap(shap_base)
        sd = get_pos_shap(shap_drift)

        drift_records = []
        for j, feat in enumerate(feature_names):
            w_dist = float(wasserstein_distance(sb[:, j], sd[:, j]))
            mean_base = float(np.mean(np.abs(sb[:, j])))
            mean_drift = float(np.mean(np.abs(sd[:, j])))
            pct_change = ((mean_drift - mean_base) / (mean_base + 1e-9)) * 100.0

            drift_records.append({
                'feature': feat,
                'wasserstein_drift': w_dist,
                'mean_shap_base': mean_base,
                'mean_shap_drift': mean_drift,
                'attribution_shift_pct': pct_change
            })

        drift_df = pd.DataFrame(drift_records).sort_values(by='wasserstein_drift', ascending=False).reset_index(drop=True)
        return drift_df

    def selective_active_retrain(
        self,
        model_factory,
        X_train_orig: np.ndarray,
        y_train_orig: np.ndarray,
        X_stream: np.ndarray,
        y_stream_oracle: np.ndarray,
        uncertainty_scores: np.ndarray,
        is_unknown_attack: np.ndarray
    ) -> Any:
        """
        Selects top-B most uncertain flows (prioritizing confirmed Unknown-Attack / Zero-Days)
        and fine-tunes / retrains model with strict budget compliance.
        """
        print(f"[Active Retrain] Querying oracle for top-{self.budget_retrain} most uncertain / zero-day flows...")
        
        # Priority score: High conformal uncertainty + priority boost for Zero-Day flows
        priority = uncertainty_scores.copy()
        priority[is_unknown_attack] += 5.0  # Strongly prioritize zero-day / unknown attacks

        top_indices = np.argsort(priority)[-self.budget_retrain:]

        X_active = X_stream[top_indices]
        y_active = y_stream_oracle[top_indices]

        # Combine with original training set
        X_augmented = np.vstack([X_train_orig, X_active])
        y_augmented = np.concatenate([y_train_orig, y_active])

        # Retrain model
        retrained_model = model_factory()
        retrained_model.fit(X_augmented, y_augmented)
        print(f"[Active Retrain] Model successfully retrained on {len(X_augmented):,} flows (budget: +{self.budget_retrain}).")
        return retrained_model, top_indices
