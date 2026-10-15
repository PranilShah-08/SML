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

        underlying_model = model.named_estimators_['rf'] if hasattr(model, 'named_estimators_') and 'rf' in model.named_estimators_ else (model.named_estimators_['lgbm'] if hasattr(model, 'named_estimators_') else model)
        explainer = shap.TreeExplainer(underlying_model)
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
        is_unknown_attack: Optional[np.ndarray] = None,
        strategy: str = 'hybrid'
    ) -> Tuple[Any, np.ndarray]:
        """
        Selects top-B flows using hybrid cluster diversity + conformal uncertainty
        (prioritizing zero-day clusters and high-uncertainty flows)
        and fine-tunes / retrains model with strict budget compliance.
        """
        print(f"[Active Retrain] Querying oracle for {self.budget_retrain} flows (Strategy: {strategy})...")
        from sklearn.cluster import MiniBatchKMeans

        if strategy == 'hybrid':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_stream_scaled = np.nan_to_num(scaler.fit_transform(X_stream), nan=0.0, posinf=1e4, neginf=-1e4)

            n_clusters = max(5, min(30, self.budget_retrain // 15))
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed, batch_size=min(1000, len(X_stream)))
            clusters = kmeans.fit_predict(X_stream_scaled)

            # 1. Cluster diversity portion (up to 70% of budget)
            div_indices = []
            rng = np.random.RandomState(self.seed)
            samples_per_cluster = max(1, int(round((self.budget_retrain * 0.7) / n_clusters)))
            for k in range(n_clusters):
                k_idx = np.where(clusters == k)[0]
                if len(k_idx) > 0:
                    div_indices.extend(rng.choice(k_idx, size=min(len(k_idx), samples_per_cluster), replace=False))

            # 2. Conformal uncertainty / unknown attack portion (remaining budget)
            priority = uncertainty_scores.copy()
            if is_unknown_attack is not None:
                priority[is_unknown_attack] += 5.0
            priority[div_indices] = -np.inf
            rem_budget = self.budget_retrain - len(div_indices)
            if rem_budget > 0:
                unc_indices = np.argsort(priority)[-rem_budget:].tolist()
                top_indices = np.array(list(set(div_indices + unc_indices)))
            else:
                top_indices = np.array(div_indices[:self.budget_retrain])
        else:
            priority = uncertainty_scores.copy()
            if is_unknown_attack is not None:
                priority[is_unknown_attack] += 5.0
            top_indices = np.argsort(priority)[-self.budget_retrain:]

        X_active = X_stream[top_indices]
        y_active = y_stream_oracle[top_indices]

        # Combine with original training set
        X_augmented = np.vstack([X_train_orig, X_active])
        y_augmented = np.concatenate([y_train_orig, y_active])

        # Retrain model
        retrained_model = model_factory()
        retrained_model.fit(X_augmented, y_augmented)
        print(f"[Active Retrain] Model successfully retrained on {len(X_augmented):,} flows (budget: +{len(top_indices)}).")
        return retrained_model, top_indices
