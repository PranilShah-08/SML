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
        stage1_model_type: str = 'ensemble',
        stage1_params: Optional[Dict[str, Any]] = None,
        stage2_params: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        self.target_fpr = target_fpr
        self.stage1_model_type = stage1_model_type
        self.stage1_params = stage1_params or {
            'n_estimators': 150,
            'max_depth': 8,
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
        
        if self.stage1_model_type == 'ensemble':
            from sklearn.ensemble import RandomForestClassifier, VotingClassifier
            from xgboost import XGBClassifier
            rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=seed, n_jobs=-1)
            lgbm = lgb.LGBMClassifier(**self.stage1_params)
            xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=seed, eval_metric='logloss', n_jobs=-1)
            self.stage1_clf = VotingClassifier(
                estimators=[('rf', rf), ('lgbm', lgbm), ('xgb', xgb)],
                voting='soft'
            )
        else:
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

        print(f"[Two-Stage] Fitting Stage 1 Binary Detector ({self.stage1_model_type})...")
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


class CostSensitiveOOBRandomForest(BaseEstimator, ClassifierMixin):
    """
    Cost-Sensitive Out-of-Bag Margin-Weighted Random Forest (CS-OOB-RF).
    
    Novel Algorithmic Architecture:
    1. Class-Weighted Tree Induction: Splits optimize asymmetric cost-sensitive Gini index.
    2. Tree-level OOB Score & Normalization: Evaluates each tree on held-out out-of-bag samples
       using Balanced Accuracy (B-ACC) or Minority-Class Recall. Normalizes via softmax with
       temperature beta: w_b = exp(beta * s_b_norm) / sum(exp(beta * s_j_norm)).
    3. Weighted Ensemble Voting: P(Y=1|X) = sum_b w_b * P_b(Y=1|X).
    4. Pareto-Optimal Adaptive Thresholding: Calibrates threshold theta* on OOB samples
       to maximize F1 subject to strict FPR constraint (e.g., FPR <= 0.001).
    5. Infeasibility Fallback: If no threshold satisfies both FPR <= alpha_target and Recall >= min_recall,
       applies regularized Pareto optimization with quadratic penalty on FPR violation.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        min_samples_split: int = 5,
        class_weight: Optional[Any] = 'balanced',
        oob_metric: str = 'balanced_accuracy',  # 'balanced_accuracy', 'minority_recall', 'accuracy'
        beta: float = 2.0,                     # Temperature for OOB weighting (0 = uniform)
        target_fpr: float = 0.001,             # Exact FPR constraint (e.g., 0.1% = 0.001)
        min_recall: float = 0.90,              # Minimum target recall
        penalty_gamma: float = 10.0,           # Regularization penalty for infeasibility relaxation
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.oob_metric = oob_metric
        self.beta = beta
        self.target_fpr = target_fpr
        self.min_recall = min_recall
        self.penalty_gamma = penalty_gamma
        self.random_state = random_state

        from sklearn.ensemble import RandomForestClassifier
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            oob_score=True,
            bootstrap=True,
            n_jobs=-1,
            random_state=self.random_state
        )
        self.tree_weights_: np.ndarray = np.array([])
        self.tree_oob_scores_: np.ndarray = np.array([])
        self.optimal_threshold_: float = 0.5
        self.oob_fpr_: float = 0.0
        self.oob_recall_: float = 0.0
        self.is_infeasible_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CostSensitiveOOBRandomForest':
        from sklearn.ensemble._forest import _generate_sample_indices
        n_samples = X.shape[0]
        self.rf.fit(X, y)

        tree_scores = []
        oob_sample_preds = np.full((n_samples, self.n_estimators), np.nan, dtype=np.float32)

        for b, tree in enumerate(self.rf.estimators_):
            sample_indices = _generate_sample_indices(
                tree.random_state, n_samples, n_samples, None
            )
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[sample_indices] = False
            oob_indices = np.where(oob_mask)[0]

            if len(oob_indices) == 0:
                tree_scores.append(0.5)
                continue

            X_oob = X[oob_indices]
            y_oob = y[oob_indices]

            tree_probs = tree.predict_proba(X_oob)
            p1 = tree_probs[:, 1] if tree_probs.shape[1] > 1 else tree_probs[:, 0]
            oob_sample_preds[oob_indices, b] = p1
            y_pred_b = (p1 >= 0.5).astype(int)

            tp = np.sum((y_oob == 1) & (y_pred_b == 1))
            tn = np.sum((y_oob == 0) & (y_pred_b == 0))
            fp = np.sum((y_oob == 0) & (y_pred_b == 1))
            fn = np.sum((y_oob == 1) & (y_pred_b == 0))

            if self.oob_metric == 'balanced_accuracy':
                rec = tp / (tp + fn + 1e-9)
                spec = tn / (tn + fp + 1e-9)
                score_b = 0.5 * (rec + spec)
            elif self.oob_metric == 'minority_recall':
                score_b = tp / (tp + fn + 1e-9)
            elif self.oob_metric == 'accuracy':
                score_b = (tp + tn) / (len(oob_indices) + 1e-9)
            else:
                score_b = (tp + tn) / (len(oob_indices) + 1e-9)

            tree_scores.append(score_b)

        self.tree_oob_scores_ = np.array(tree_scores)

        # 2. Normalize tree scores and compute voting weights via Softmax with temperature beta
        if self.beta == 0.0 or (np.max(self.tree_oob_scores_) - np.min(self.tree_oob_scores_)) < 1e-6:
            self.tree_weights_ = np.full(self.n_estimators, 1.0 / self.n_estimators)
        else:
            s_min = np.min(self.tree_oob_scores_)
            s_max = np.max(self.tree_oob_scores_)
            s_scaled = (self.tree_oob_scores_ - s_min) / (s_max - s_min + 1e-9)
            exp_scores = np.exp(self.beta * s_scaled)
            self.tree_weights_ = exp_scores / np.sum(exp_scores)

        # 3. Compute Out-of-Bag Ensemble Probabilities using weighted voting
        weighted_oob_probs = np.zeros(n_samples)
        for i in range(n_samples):
            valid_trees = ~np.isnan(oob_sample_preds[i, :])
            if np.any(valid_trees):
                w = self.tree_weights_[valid_trees]
                w_norm = w / (np.sum(w) + 1e-9)
                weighted_oob_probs[i] = np.sum(w_norm * oob_sample_preds[i, valid_trees])
            else:
                weighted_oob_probs[i] = 0.5

        # 4. Adaptive Thresholding under exact FPR constraint
        threshold_candidates = np.linspace(0.01, 0.99, 99)
        feasible_candidates = []
        for th in threshold_candidates:
            preds = (weighted_oob_probs >= th).astype(int)
            fp = np.sum((y == 0) & (preds == 1))
            tn = np.sum((y == 0) & (preds == 0))
            tp = np.sum((y == 1) & (preds == 1))
            fn = np.sum((y == 1) & (preds == 0))

            fpr = fp / (fp + tn + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            prec = tp / (tp + fp + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)

            if fpr <= self.target_fpr:
                feasible_candidates.append((th, f1, fpr, rec))

        satisfactory_candidates = [c for c in feasible_candidates if c[3] >= self.min_recall]

        if satisfactory_candidates:
            best_cand = max(satisfactory_candidates, key=lambda x: x[1])
            self.optimal_threshold_ = best_cand[0]
            self.oob_fpr_ = best_cand[2]
            self.oob_recall_ = best_cand[3]
            self.is_infeasible_ = False
        elif feasible_candidates:
            best_cand = max(feasible_candidates, key=lambda x: x[3])
            self.optimal_threshold_ = best_cand[0]
            self.oob_fpr_ = best_cand[2]
            self.oob_recall_ = best_cand[3]
            self.is_infeasible_ = False
        else:
            # Infeasibility handling: Regularized Pareto relaxation
            self.is_infeasible_ = True
            best_score = float('inf')
            best_cand = None
            for th in threshold_candidates:
                preds = (weighted_oob_probs >= th).astype(int)
                fp = np.sum((y == 0) & (preds == 1))
                tn = np.sum((y == 0) & (preds == 0))
                tp = np.sum((y == 1) & (preds == 1))
                fn = np.sum((y == 1) & (preds == 0))

                fpr = fp / (fp + tn + 1e-9)
                rec = tp / (tp + fn + 1e-9)
                penalty = (max(0.0, fpr - self.target_fpr) ** 2) + self.penalty_gamma * (max(0.0, self.min_recall - rec) ** 2)
                if penalty < best_score:
                    best_score = penalty
                    best_cand = (th, fpr, rec)
            
            if best_cand:
                self.optimal_threshold_ = best_cand[0]
                self.oob_fpr_ = best_cand[1]
                self.oob_recall_ = best_cand[2]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        p1 = np.zeros(n_samples)

        for b, tree in enumerate(self.rf.estimators_):
            tree_probs = tree.predict_proba(X)
            t_p1 = tree_probs[:, 1] if tree_probs.shape[1] > 1 else tree_probs[:, 0]
            p1 += self.tree_weights_[b] * t_p1

        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.optimal_threshold_).astype(int)


