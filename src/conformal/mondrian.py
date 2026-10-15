"""
Per-Class (Mondrian) Conformal Uncertainty Layer (Novelty 2).
Provides finite-sample valid conformal p-values, tri-state operational triage
(auto-decide vs analyst escalation), and statistically guaranteed Zero-Day / Unknown Attack detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

class MondrianConformalPredictor:
    def __init__(self, epsilon: float = 0.05, zero_day_threshold: float = 0.05):
        """
        Args:
            epsilon: Conformal error rate / significance level (0.05 -> 95% coverage guarantee).
            zero_day_threshold: Threshold below which a flow is rejected by all known profiles.
        """
        self.epsilon = epsilon
        self.zero_day_threshold = zero_day_threshold
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.calibration_scores: Dict[int, np.ndarray] = {}

    def fit(self, probas_cal: np.ndarray, y_cal: np.ndarray, classes: List[str]) -> 'MondrianConformalPredictor':
        """
        Calibrates Mondrian non-conformity distributions per class.
        For BENIGN: non-conformity score is s(x) = P(Attack | x).
        For Attack class c: non-conformity score is s(x, c) = 1.0 - P(Y = c | x).
        """
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        benign_idx = self.class_to_idx['BENIGN']

        # Filter calibration samples to known classes
        valid_mask = np.isin(y_cal, self.classes)
        probas_filtered = probas_cal[valid_mask]
        y_filtered = y_cal[valid_mask]

        if len(y_filtered) > 0 and isinstance(y_filtered[0], str):
            y_indices = np.array([self.class_to_idx[y] for y in y_filtered])
        else:
            y_indices = np.array(y_filtered, dtype=int)

        self.calibration_scores = {}
        for c_idx, c_name in enumerate(self.classes):
            mask = (y_indices == c_idx)
            if np.sum(mask) == 0:
                self.calibration_scores[c_idx] = np.array([1.0])
                continue

            if c_name == 'BENIGN':
                # For Benign, non-conformity is how much it deviates toward attack: 1 - P(BENIGN)
                scores = 1.0 - probas_filtered[mask, benign_idx]
            else:
                # For Attack family c, non-conformity is 1 - P(c)
                scores = 1.0 - probas_filtered[mask, c_idx]

            self.calibration_scores[c_idx] = np.sort(scores)

        print(f"[Conformal] Calibrated Mondrian non-conformity distributions across {len(self.classes)} known classes.")
        return self

    def compute_p_values(self, probas_test: np.ndarray) -> np.ndarray:
        """
        Computes finite-sample valid conformal p-values for every candidate class hypothesis.
        """
        n_samples = probas_test.shape[0]
        n_classes = len(self.classes)
        benign_idx = self.class_to_idx['BENIGN']
        p_values = np.zeros((n_samples, n_classes), dtype=float)

        for c_idx, c_name in enumerate(self.classes):
            scores_c = self.calibration_scores[c_idx]
            n_cal_c = len(scores_c)

            if c_name == 'BENIGN':
                test_scores = 1.0 - probas_test[:, benign_idx]
            else:
                test_scores = 1.0 - probas_test[:, c_idx]

            counts = n_cal_c - np.searchsorted(scores_c, test_scores, side='left')
            p_values[:, c_idx] = (1.0 + counts) / (n_cal_c + 1.0)

        return p_values

    def predict_triage(
        self,
        probas_test: np.ndarray,
        auto_decide_p_min: float = 0.20
    ) -> Dict[str, Any]:
        """
        Performs tri-state operational triage:
        - 'AUTO_DECIDE': Confident benign or single-class attack with high p-value margin.
        - 'SOC_ANALYST': Multi-class attack ambiguity or borderline confidence.
        - 'UNKNOWN_ATTACK': Rejected by BENIGN hypothesis AND rejected by all known attack profiles.
        """
        p_values = self.compute_p_values(probas_test)
        n_samples = probas_test.shape[0]
        benign_idx = self.class_to_idx['BENIGN']

        prediction_sets = []
        triage_decisions = []
        is_unknown_attack = []
        conformal_entropy = []

        for i in range(n_samples):
            p_i = p_values[i]
            # Valid classes in prediction set at error level epsilon
            pred_set = [self.classes[c] for c in range(len(self.classes)) if p_i[c] > self.epsilon]
            prediction_sets.append(pred_set)

            # Entropy of conformal p-values
            p_sum = np.sum(p_i)
            if p_sum > 0:
                p_norm = p_i / p_sum
                entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))
            else:
                entropy = 0.0
            conformal_entropy.append(entropy)

            # TRI-STATE OPERATIONAL LOGIC:
            # 1. Is flow accepted as normal BENIGN traffic?
            p_benign = p_i[benign_idx]
            max_attack_p = np.max([p_i[c] for c in range(len(self.classes)) if c != benign_idx]) if len(self.classes) > 1 else 0.0

            if p_benign > self.epsilon and max_attack_p <= self.epsilon:
                # High-confidence normal background flow
                triage_decisions.append('AUTO_DECIDE')
                is_unknown_attack.append(False)
            elif p_benign <= self.epsilon and max_attack_p <= self.zero_day_threshold:
                # Rejected by BENIGN and rejected by every known attack profile!
                triage_decisions.append('UNKNOWN_ATTACK')
                is_unknown_attack.append(True)
            elif max_attack_p > self.epsilon and len(pred_set) == 1 and max_attack_p >= auto_decide_p_min:
                # Clean, unambiguous known attack
                triage_decisions.append('AUTO_DECIDE')
                is_unknown_attack.append(False)
            else:
                # Ambiguous collision, multiple candidate classes, or borderline margin
                triage_decisions.append('SOC_ANALYST')
                is_unknown_attack.append(False)

        return {
            'p_values': p_values,
            'prediction_sets': prediction_sets,
            'triage_decisions': np.array(triage_decisions),
            'is_unknown_attack': np.array(is_unknown_attack),
            'conformal_entropy': np.array(conformal_entropy),
            'classes': self.classes
        }
