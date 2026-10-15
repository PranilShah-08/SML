"""
Operational Evaluation Metrics for High-Throughput Network Intrusion Detection.
Implements:
1. Recall at target ultra-low False Positive Rates (FPR = 0.1% and 0.01%)
2. PR-AUC (Precision-Recall Area Under Curve)
3. Multiclass Macro-F1
4. Bayesian Operational Precision across attack base-rate priors (10^-2, 10^-3, 10^-4)
5. Secondary metrics: Accuracy, ROC-AUC, Latency (microseconds per flow)
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    confusion_matrix
)

def compute_recall_at_target_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    """
    Computes Recall at a strictly enforced ultra-low target False Positive Rate (e.g. 0.001 or 0.0001).
    Returns (recall_at_fpr, operational_threshold).
    """
    benign_scores = y_score[y_true == 0]
    attack_scores = y_score[y_true == 1]

    if len(benign_scores) == 0 or len(attack_scores) == 0:
        return 0.0, 0.5

    # Threshold strictly enforcing FPR <= target_fpr on benign traffic
    threshold = float(np.quantile(benign_scores, 1.0 - target_fpr))
    
    # Calculate recall at this exact threshold
    recall = float(np.mean(attack_scores >= threshold))
    return recall, threshold

def compute_bayesian_precision(tpr: float, fpr: float, prior_attack: float) -> float:
    """
    Calculates Bayesian Operational Precision P(Attack | Alert) via Axelsson's Base-Rate Theorem:
    P(A | Alert) = (TPR * P(A)) / (TPR * P(A) + FPR * (1 - P(A)))
    """
    numerator = tpr * prior_attack
    denominator = numerator + fpr * (1.0 - prior_attack)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)

def evaluate_intrusion_detector(
    y_true_binary: np.ndarray,
    y_score_binary: np.ndarray,
    y_true_multi: Optional[np.ndarray] = None,
    y_pred_multi: Optional[np.ndarray] = None,
    latency_per_sample_us: float = 0.0
) -> Dict[str, Any]:
    """
    Evaluates detector across all primary operational and secondary statistical metrics.
    """
    # 1. Primary: Recall @ FPR = 0.1% and 0.01%
    rec_01, thresh_01 = compute_recall_at_target_fpr(y_true_binary, y_score_binary, target_fpr=0.001)
    rec_001, thresh_001 = compute_recall_at_target_fpr(y_true_binary, y_score_binary, target_fpr=0.0001)

    # 2. Primary: PR-AUC
    try:
        pr_auc = float(average_precision_score(y_true_binary, y_score_binary))
    except Exception:
        pr_auc = 0.0

    # 3. Primary: Multiclass Macro-F1
    if y_true_multi is not None and y_pred_multi is not None:
        macro_f1 = float(f1_score(y_true_multi, y_pred_multi, average='macro', zero_division=0))
    else:
        # Binary fallback if multiclass predictions not supplied
        bin_pred = (y_score_binary >= thresh_01).astype(int)
        macro_f1 = float(f1_score(y_true_binary, bin_pred, average='macro', zero_division=0))

    # 4. Primary: Bayesian Precision at operational priors (10^-2, 10^-3, 10^-4)
    # Using the operational threshold at FPR = 0.1% (target_fpr = 10^-3)
    bayes_p_1e2 = compute_bayesian_precision(tpr=rec_01, fpr=0.001, prior_attack=1e-2)
    bayes_p_1e3 = compute_bayesian_precision(tpr=rec_01, fpr=0.001, prior_attack=1e-3)
    bayes_p_1e4 = compute_bayesian_precision(tpr=rec_001, fpr=0.0001, prior_attack=1e-4)

    # Secondary Metrics
    try:
        roc_auc = float(roc_auc_score(y_true_binary, y_score_binary))
    except Exception:
        roc_auc = 0.5
    
    bin_pred_01 = (y_score_binary >= thresh_01).astype(int)
    acc = float(accuracy_score(y_true_binary, bin_pred_01))

    return {
        # Primary Operational Metrics (Lead with these)
        'recall_at_fpr_0.1%': rec_01,
        'recall_at_fpr_0.01%': rec_001,
        'pr_auc': pr_auc,
        'macro_f1': macro_f1,
        'bayesian_precision_1e-2': bayes_p_1e2,
        'bayesian_precision_1e-3': bayes_p_1e3,
        'bayesian_precision_1e-4': bayes_p_1e4,
        # Secondary Metrics
        'roc_auc': roc_auc,
        'accuracy': acc,
        'latency_us': latency_per_sample_us,
        'threshold_fpr_0.1%': thresh_01,
        'threshold_fpr_0.01%': thresh_001
    }
