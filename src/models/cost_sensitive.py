"""
Two-Stage Cost-Sensitive Intrusion Detector (Layer [3]).
Stage 1: Ultra-fast binary Benign/Attack filter calibrated at fixed target FPR (0.1% or 0.01%).
Stage 2: Granular multiclass attack-family classifier executed strictly on flagged anomalies.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin

class TwoStageDetector(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        target_fpr: float = 0.001,
        stage1_params: Optional[Dict[str, Any]] = None,
        stage2_params: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        self.target_fpr = target_fpr
        self.stage1_params = stage1_params or {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.05,
            'class_weight': 'balanced',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': seed
        }
        self.stage2_params = stage2_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'class_weight': 'balanced',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': seed
        }
        self.seed = seed
        self.stage1_clf = lgb.LGBMClassifier(**self.stage1_params)
        self.stage2_clf = lgb.LGBMClassifier(**self.stage2_params)
        self.operational_threshold: float = 0.5
        self.classes_: List[str] = []
        self.attack_classes_: List[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train_binary: np.ndarray,
        y_train_multi: Optional[np.ndarray] = None,
        X_cal_benign: Optional[np.ndarray] = None
    ) -> 'TwoStageDetector':
        """
        Trains Stage 1 Binary Detector and Stage 2 Multiclass Attack-Family Detector.
        """
        if y_train_multi is None:
            y_train_multi = np.array(['ATTACK' if y == 1 else 'BENIGN' for y in y_train_binary])

        self.classes_ = sorted(list(np.unique(y_train_multi)))
        self.attack_classes_ = [c for c in self.classes_ if c != 'BENIGN']

        print(f"[Two-Stage] Fitting Stage 1 Binary Detector (LightGBM)...")
        self.stage1_clf.fit(X_train, y_train_binary)

        # Calibrate operational threshold at target_fpr on benign validation flows
        if X_cal_benign is not None and len(X_cal_benign) > 0:
            cal_scores = self.stage1_clf.predict_proba(X_cal_benign)[:, 1]
            self.operational_threshold = float(np.quantile(cal_scores, 1.0 - self.target_fpr))
        else:
            train_benign_scores = self.stage1_clf.predict_proba(X_train[y_train_binary == 0])[:, 1]
            self.operational_threshold = float(np.quantile(train_benign_scores, 1.0 - self.target_fpr))

        print(f"[Two-Stage] Operational threshold for FPR <= {self.target_fpr*100:.2f}%: {self.operational_threshold:.6f}")

        # Train Stage 2 on attack flows
        print(f"[Two-Stage] Fitting Stage 2 Attack-Family Classifier...")
        attack_mask = (y_train_binary == 1)
        if np.sum(attack_mask) > 0 and len(np.unique(y_train_multi[attack_mask])) > 1:
            self.stage2_clf.fit(X_train[attack_mask], y_train_multi[attack_mask])
        else:
            self.stage2_clf.fit(X_train, y_train_multi)

        return self

    def predict_proba_binary(self, X: np.ndarray) -> np.ndarray:
        return self.stage1_clf.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns full joint probability matrix P(Y=c | X).
        P(Y=BENIGN | X) = 1 - P(Attack | X)
        P(Y=c | X) = P(Attack | X) * P_stage2(Y=c | Attack, X)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        benign_idx = self.classes_.index('BENIGN')

        s1_probs = self.stage1_clf.predict_proba(X)[:, 1]  # P(Attack)
        full_probs = np.zeros((n_samples, n_classes), dtype=float)
        full_probs[:, benign_idx] = 1.0 - s1_probs

        if hasattr(self.stage2_clf, 'predict_proba') and len(self.attack_classes_) > 0:
            s2_probs = self.stage2_clf.predict_proba(X)
            s2_classes = list(self.stage2_clf.classes_)

            for s2_c_idx, c_name in enumerate(s2_classes):
                if c_name in self.classes_ and c_name != 'BENIGN':
                    target_c_idx = self.classes_.index(c_name)
                    full_probs[:, target_c_idx] = s1_probs * s2_probs[:, s2_c_idx]
        else:
            # Fallback if only 1 attack class
            for c_idx, c_name in enumerate(self.classes_):
                if c_name != 'BENIGN':
                    full_probs[:, c_idx] = s1_probs / max(1, len(self.attack_classes_))

        # Re-normalize rows to guarantee valid probability simplex
        row_sums = full_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return full_probs / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return np.array([self.classes_[i] for i in pred_indices])

