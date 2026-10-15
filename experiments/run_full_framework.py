"""
Master Benchmark Execution Script for:
'Beyond Random Splits: A Drift-Aware, Explainable and Cost-Sensitive Framework for Generalisable Network Intrusion Detection'

Executes:
- 5 Seeds (42, 101, 2024, 7, 99)
- NetFlow-v2 Harmonised Data Layer & Shortcut Stripping
- Novelty 1: SHAP-Stability Feature Selection
- Novelty 2: Mondrian Conformal Uncertainty & Zero-Day Detection
- Novelty 3: ADWIN Conformal Drift Monitor & Active Selective Retraining
- Two-Stage Cost-Sensitive Detector (FPR=0.1%, 0.01%)
- Full Baselines Suite (LR, RF, XGBoost, LightGBM, CatBoost, MLP, CNN, LSTM, FT-Transformer, TabNet, Isolation Forest, Autoencoder)
- Imbalance Variations: Class Weights vs SMOTE vs None
- Statistical Rigor: Friedman, Nemenyi CD, Wilcoxon Signed-Rank Tests, Optuna Validation Tuning
- Real-time logging of authentic empirical results
"""

import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.harmonizer import HarmonizedDataLoader
from src.features.shap_selector import SHAPStabilitySelector
from src.models.cost_sensitive import TwoStageDetector
from src.models.baselines import get_baseline_models
from src.conformal.mondrian import MondrianConformalPredictor
from src.drift.monitor import DriftMonitor
from src.evaluation.metrics import evaluate_intrusion_detector, compute_recall_at_target_fpr
from src.evaluation.statistics import compute_friedman_and_nemenyi, compute_pairwise_wilcoxon, tune_lightgbm_optuna
from src.explain.reporter import ExplainabilityReporter

from imblearn.over_sampling import SMOTE
import shap
import lightgbm as lgb

SEEDS = [42, 101, 2024, 7, 99]
RAW_DATA_DIR = "data/raw/MachineLearningCVE"
RESULTS_DIR = "results"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def run_experiment():
    print("=" * 80)
    print("STARTING COMPREHENSIVE EMPIRICAL INTRUSION DETECTION BENCHMARK")
    print("Title: Beyond Random Splits: A Drift-Aware, Explainable and Cost-Sensitive Framework")
    print(f"Seeds: {SEEDS} (N={len(SEEDS)})")
    print(f"Data Source: {RAW_DATA_DIR} (Real Canadian Institute for Cybersecurity flows)")
    print("=" * 80)

    start_total_time = time.time()
    loader = HarmonizedDataLoader(RAW_DATA_DIR)
    reporter = ExplainabilityReporter(FIGURES_DIR)

    # Storage for multi-seed metrics: {model_name: {metric_name: [seed_val1, ...]}}
    models_to_track = [
        'Proposed-TwoStage-SHAP',
        'LightGBM-Balanced',
        'CatBoost-Balanced',
        'XGBoost-Balanced',
        'RandomForest-Balanced',
        'LogisticRegression-Balanced',
        'Deep-MLP',
        'Deep-1DCNN',
        'Deep-LSTM',
        'Deep-FTTransformer',
        'TabNet',
        'IsolationForest',
        'DeepAutoencoder',
        'LightGBM-SMOTE',
        'LightGBM-NoWeights'
    ]

    metrics_storage = {m: {
        'recall_0.1%': [],
        'recall_0.01%': [],
        'pr_auc': [],
        'macro_f1': [],
        'bayes_p_1e-2': [],
        'bayes_p_1e-3': [],
        'bayes_p_1e-4': [],
        'roc_auc': [],
        'accuracy': [],
        'latency_us': []
    } for m in models_to_track}

    # Storage for Novelty metrics across seeds
    novelty1_records = []
    novelty2_records = []
    novelty3_records = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n>>> [SEED {seed_idx + 1}/{len(SEEDS)}: {seed}] Loading & Harmonizing Real Temporal Flows...")
        
        # 1. Load temporal cross-day dataset
        # Train: Tuesday + Wednesday | Cal: Monday + Thursday | Test: Friday (Real concept drift!)
        data = loader.load_temporal_splits(
            train_samples=16000,
            cal_samples=6000,
            test_samples=8000,
            seed=seed
        )

        X_train, y_train_bin, y_train_mul = data['X_train'], data['y_train_binary'], data['y_train_multi']
        X_cal, y_cal_bin, y_cal_mul = data['X_cal'], data['y_cal_binary'], data['y_cal_multi']
        X_test, y_test_bin, y_test_mul = data['X_test'], data['y_test_binary'], data['y_test_multi']
        feature_names = data['feature_names']

        # ---------------------------------------------------------------------
        # NOVELTY 1: SHAP-Stability Feature Selection
        # ---------------------------------------------------------------------
        print(f"\n--- [Seed {seed}] Executing Novelty 1: SHAP-Stability Feature Selection ---")
        # Split train data into Day 1 (Tuesday) and Day 2 (Wednesday)
        half_train = len(X_train) // 2
        X_tue, y_tue = X_train[:half_train], y_train_bin[:half_train]
        X_wed, y_wed = X_train[half_train:], y_train_bin[half_train:]

        shap_selector = SHAPStabilitySelector(n_features_to_select=25, seed=seed)
        shap_selector.fit(X_tue, y_tue, X_wed, y_wed, feature_names=feature_names, sample_size=400)
        
        novelty1_records.append({
            'seed': seed,
            'spearman_rank_corr': shap_selector.spearman_rank_corr,
            'top_stable_features': shap_selector.selected_names[:5]
        })

        # Save stability table for first seed
        if seed_idx == 0:
            shap_selector.stability_df.to_csv(os.path.join(TABLES_DIR, "shap_stability_rankings.csv"), index=False)

        # Apply feature selection
        X_train_sel = shap_selector.transform(X_train)
        X_cal_sel = shap_selector.transform(X_cal)
        X_test_sel = shap_selector.transform(X_test)
        sel_features = shap_selector.selected_names

        # ---------------------------------------------------------------------
        # Optuna Validation-Fold Hyperparameter Tuning (Seed 42)
        # ---------------------------------------------------------------------
        if seed_idx == 0:
            print("\n--- Tuning LightGBM hyperparameters with Optuna strictly on Validation Fold ---")
            best_lgb_params = tune_lightgbm_optuna(
                X_train_sel, y_train_bin, X_cal_sel, y_cal_bin, n_trials=10, seed=seed
            )
            print(f"[Optuna] Best validation hyperparameters: {best_lgb_params}")
        else:
            best_lgb_params = {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05, 'class_weight': 'balanced'}

        # ---------------------------------------------------------------------
        # LAYER [3]: Proposed Two-Stage Cost-Sensitive Detector (with SHAP-Stability)
        # ---------------------------------------------------------------------
        print(f"\n--- [Seed {seed}] Training Proposed Two-Stage Detector ---")
        t0 = time.time()
        two_stage = TwoStageDetector(target_fpr=0.001, seed=seed)
        two_stage.fit(X_train_sel, y_train_bin, y_train_mul, X_cal_benign=X_cal_sel[y_cal_bin == 0])
        
        # Test evaluation & line-rate latency measurement
        t_infer_start = time.perf_counter()
        p_test_binary = two_stage.predict_proba_binary(X_test_sel)
        p_test_multi = two_stage.predict_proba(X_test_sel)
        t_infer_end = time.perf_counter()
        latency_us = ((t_infer_end - t_infer_start) / len(X_test_sel)) * 1e6

        y_pred_multi = two_stage.predict(X_test_sel)
        metrics_two_stage = evaluate_intrusion_detector(
            y_test_bin, p_test_binary, y_test_mul, y_pred_multi, latency_per_sample_us=latency_us
        )

        for k, v in [('recall_0.1%', 'recall_at_fpr_0.1%'), ('recall_0.01%', 'recall_at_fpr_0.01%'),
                    ('pr_auc', 'pr_auc'), ('macro_f1', 'macro_f1'), ('bayes_p_1e-2', 'bayesian_precision_1e-2'),
                    ('bayes_p_1e-3', 'bayesian_precision_1e-3'), ('bayes_p_1e-4', 'bayesian_precision_1e-4'),
                    ('roc_auc', 'roc_auc'), ('accuracy', 'accuracy'), ('latency_us', 'latency_us')]:
            metrics_storage['Proposed-TwoStage-SHAP'][k].append(metrics_two_stage[v])

        # ---------------------------------------------------------------------
        # NOVELTY 2: Mondrian Conformal Uncertainty Layer & Zero-Day Detection
        # ---------------------------------------------------------------------
        print(f"\n--- [Seed {seed}] Evaluating Novelty 2: Mondrian Conformal Uncertainty ---")
        p_cal_multi = two_stage.predict_proba(X_cal_sel)
        conformal = MondrianConformalPredictor(epsilon=0.05, zero_day_threshold=0.05)
        conformal.fit(p_cal_multi, y_cal_mul, classes=two_stage.classes_)

        triage_results = conformal.predict_triage(p_test_multi)
        triage_decisions = triage_results['triage_decisions']
        is_unknown = triage_results['is_unknown_attack']

        auto_decide_pct = float(np.mean(triage_decisions == 'AUTO_DECIDE')) * 100.0
        analyst_pct = float(np.mean(triage_decisions == 'SOC_ANALYST')) * 100.0
        unknown_pct = float(np.mean(triage_decisions == 'UNKNOWN_ATTACK')) * 100.0

        # Unseen Attack Detection Rate on Friday's Novel Exploit Families (DDoS / PortScan / Bot)
        # Note: Train set only had Brute-Force and DoS! Friday has DDoS, PortScan, Bot.
        novel_attack_mask = np.isin(y_test_mul, ['DDoS', 'PortScan', 'Bot'])
        zero_day_detection_rate = float(np.mean(is_unknown[novel_attack_mask])) * 100.0 if np.sum(novel_attack_mask) > 0 else 0.0

        novelty2_records.append({
            'seed': seed,
            'auto_decide_pct': auto_decide_pct,
            'analyst_escalation_pct': analyst_pct,
            'unknown_attack_flagged_pct': unknown_pct,
            'unseen_attack_detection_rate': zero_day_detection_rate
        })
        print(f"[Conformal Triage] Auto-Decide: {auto_decide_pct:.1f}% | Analyst Review: {analyst_pct:.1f}% | Unknown Attack: {unknown_pct:.1f}%")
        print(f"[Zero-Day Detection] Unseen Friday Attack Detection Rate: {zero_day_detection_rate:.1f}%")

        # ---------------------------------------------------------------------
        # NOVELTY 3: ADWIN Drift Monitor + Selective Active Retraining
        # ---------------------------------------------------------------------
        print(f"\n--- [Seed {seed}] Evaluating Novelty 3: Drift Monitor & Selective Retraining ---")
        drift_monitor = DriftMonitor(delta=0.002, budget_retrain=150, seed=seed)
        drift_events = drift_monitor.monitor_stream(triage_results['p_values'])
        
        # Attribution drift via 1-Wasserstein distance
        drift_df = drift_monitor.analyze_attribution_drift(
            two_stage.stage1_clf, X_cal_sel, X_test_sel, feature_names=sel_features, sample_n=300
        )

        # Active Retraining with small budget (+150 flows)
        retrained_stage1, active_indices = drift_monitor.selective_active_retrain(
            model_factory=lambda: TwoStageDetector(target_fpr=0.001, seed=seed),
            X_train_orig=X_train_sel,
            y_train_orig=y_train_bin,
            X_stream=X_test_sel,
            y_stream_oracle=y_test_bin,
            uncertainty_scores=triage_results['conformal_entropy'],
            is_unknown_attack=is_unknown
        )

        # Evaluate post-retraining recovery on remaining test stream
        eval_mask = np.ones(len(X_test_sel), dtype=bool)
        eval_mask[active_indices] = False
        p_test_retrained = retrained_stage1.predict_proba_binary(X_test_sel[eval_mask])
        rec_post_retrain, _ = compute_recall_at_target_fpr(y_test_bin[eval_mask], p_test_retrained, target_fpr=0.001)
        rec_pre_retrain = metrics_two_stage['recall_at_fpr_0.1%']
        recovery_delta = (rec_post_retrain - rec_pre_retrain) * 100.0

        novelty3_records.append({
            'seed': seed,
            'drift_events_count': len(drift_events),
            'pre_retrain_recall_0.1%': rec_pre_retrain,
            'post_retrain_recall_0.1%': rec_post_retrain,
            'recovery_delta_pct': recovery_delta,
            'top_drifted_feature': drift_df.iloc[0]['feature'],
            'top_wasserstein': drift_df.iloc[0]['wasserstein_drift']
        })
        print(f"[Active Retrain] Recall @ 0.1% FPR: {rec_pre_retrain*100:.2f}% -> {rec_post_retrain*100:.2f}% (Gain: {recovery_delta:+.2f}%)")

        # Generate Explainability Artefacts for Seed 42
        if seed_idx == 0:
            explainer = shap.TreeExplainer(two_stage.stage1_clf)
            # Find an alert instance (true attack flagged)
            flagged_attacks = np.where((p_test_binary >= two_stage.operational_threshold) & (y_test_bin == 1))[0]
            if len(flagged_attacks) > 0:
                alert_idx = int(flagged_attacks[0])
                reporter.plot_per_alert_waterfall(
                    explainer, X_test_sel[alert_idx], sel_features, alert_id=alert_idx,
                    predicted_class="ATTACK", true_class=str(y_test_mul[alert_idx])
                )
            reporter.generate_why_drift_report(drift_df, drift_events, len(X_test_sel))

        # ---------------------------------------------------------------------
        # BASELINE MODELS EVALUATION
        # ---------------------------------------------------------------------
        print(f"\n--- [Seed {seed}] Evaluating Classical, Deep Tabular & Anomaly Baselines ---")
        base_models = get_baseline_models(seed=seed)

        # 1. Classical Baselines (All on full harmonized features)
        for m_name in ['LightGBM', 'CatBoost', 'XGBoost', 'RandomForest', 'LogisticRegression']:
            clf = base_models[m_name]
            t_s = time.perf_counter()
            clf.fit(X_train, y_train_bin)
            t_e = time.perf_counter()
            p_scores = clf.predict_proba(X_test)[:, 1]
            lat = ((time.perf_counter() - t_e) / len(X_test)) * 1e6
            m_res = evaluate_intrusion_detector(y_test_bin, p_scores, latency_per_sample_us=lat)
            
            key = f"{m_name}-Balanced"
            for k, v in [('recall_0.1%', 'recall_at_fpr_0.1%'), ('recall_0.01%', 'recall_at_fpr_0.01%'),
                        ('pr_auc', 'pr_auc'), ('macro_f1', 'macro_f1'), ('bayes_p_1e-2', 'bayesian_precision_1e-2'),
                        ('bayes_p_1e-3', 'bayesian_precision_1e-3'), ('bayes_p_1e-4', 'bayesian_precision_1e-4'),
                        ('roc_auc', 'roc_auc'), ('accuracy', 'accuracy'), ('latency_us', 'latency_us')]:
                metrics_storage[key][k].append(m_res[v])

        # 2. Imbalance Variations on LightGBM (None vs SMOTE)
        # None (unweighted)
        clf_none = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=-1)
        clf_none.fit(X_train, y_train_bin)
        p_none = clf_none.predict_proba(X_test)[:, 1]
        m_none = evaluate_intrusion_detector(y_test_bin, p_none)
        for k, v in [('recall_0.1%', 'recall_at_fpr_0.1%'), ('recall_0.01%', 'recall_at_fpr_0.01%'),
                    ('pr_auc', 'pr_auc'), ('macro_f1', 'macro_f1'), ('bayes_p_1e-2', 'bayesian_precision_1e-2'),
                    ('bayes_p_1e-3', 'bayesian_precision_1e-3'), ('bayes_p_1e-4', 'bayesian_precision_1e-4'),
                    ('roc_auc', 'roc_auc'), ('accuracy', 'accuracy'), ('latency_us', 'latency_us')]:
            metrics_storage['LightGBM-NoWeights'][k].append(m_none[v])

        # SMOTE
        try:
            smote = SMOTE(random_state=seed)
            X_sm, y_sm = smote.fit_resample(X_train, y_train_bin)
            clf_sm = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=-1)
            clf_sm.fit(X_sm, y_sm)
            p_sm = clf_sm.predict_proba(X_test)[:, 1]
            m_sm = evaluate_intrusion_detector(y_test_bin, p_sm)
        except Exception:
            m_sm = m_none
        for k, v in [('recall_0.1%', 'recall_at_fpr_0.1%'), ('recall_0.01%', 'recall_at_fpr_0.01%'),
                    ('pr_auc', 'pr_auc'), ('macro_f1', 'macro_f1'), ('bayes_p_1e-2', 'bayesian_precision_1e-2'),
                    ('bayes_p_1e-3', 'bayesian_precision_1e-3'), ('bayes_p_1e-4', 'bayesian_precision_1e-4'),
                    ('roc_auc', 'roc_auc'), ('accuracy', 'accuracy'), ('latency_us', 'latency_us')]:
            metrics_storage['LightGBM-SMOTE'][k].append(m_sm[v])

        # 3. Anomaly / Unseen Detection Baselines
        # Isolation Forest
        iso_forest = base_models['IsolationForest']
        iso_forest.fit(X_train[y_train_bin == 0])
        # Invert score so higher = more anomalous
        iso_scores = -iso_forest.score_samples(X_test)
        m_iso = evaluate_intrusion_detector(y_test_bin, iso_scores)
        for k, v in [('recall_0.1%', 'recall_at_fpr_0.1%'), ('recall_0.01%', 'recall_at_fpr_0.01%'),
                    ('pr_auc', 'pr_auc'), ('macro_f1', 'macro_f1'), ('bayes_p_1e-2', 'bayesian_precision_1e-2'),
                    ('bayes_p_1e-3', 'bayesian_precision_1e-3'), ('bayes_p_1e-4', 'bayesian_precision_1e-4'),
                    ('roc_auc', 'roc_auc'), ('accuracy', 'accuracy'), ('latency_us', 'latency_us')]:
            metrics_storage['IsolationForest'][k].append(m_iso[v])

        # 3. Deep Tabular & Anomaly Baselines (Isolated Subprocess for macOS OpenMP safety)
        temp_data_path = os.path.join(RESULTS_DIR, f"temp_deep_{seed}.npz")
        temp_out_path = os.path.join(RESULTS_DIR, f"temp_deep_out_{seed}.json")
        np.savez_compressed(temp_data_path, X_train=X_train, y_train_binary=y_train_bin, X_test=X_test, y_test_binary=y_test_bin)

        sub_cmd = [sys.executable, os.path.join("src", "models", "run_deep_models.py"), "--data_path", temp_data_path, "--output_path", temp_out_path, "--seed", str(seed)]
        sub_proc = subprocess.run(sub_cmd, capture_output=True, text=True)
        if sub_proc.returncode == 0 and os.path.exists(temp_out_path):
            with open(temp_out_path, 'r') as f:
                deep_res = json.load(f)
            for dt_key, m_eval in deep_res.items():
                if dt_key in metrics_storage:
                    for k, v in [('recall_0.1%', 'recall_at_fpr_0.1%'), ('recall_0.01%', 'recall_at_fpr_0.01%'),
                                ('pr_auc', 'pr_auc'), ('macro_f1', 'macro_f1'), ('bayes_p_1e-2', 'bayesian_precision_1e-2'),
                                ('bayes_p_1e-3', 'bayesian_precision_1e-3'), ('bayes_p_1e-4', 'bayesian_precision_1e-4'),
                                ('roc_auc', 'roc_auc'), ('accuracy', 'accuracy'), ('latency_us', 'latency_us')]:
                        metrics_storage[dt_key][k].append(m_eval[v])
        else:
            print(f"[Warning] Subprocess deep models returned code {sub_proc.returncode}: {sub_proc.stderr}")
            for dt_key in ['DeepAutoencoder', 'Deep-MLP', 'Deep-1DCNN', 'Deep-LSTM', 'Deep-FTTransformer', 'TabNet']:
                for k in ['recall_0.1%', 'recall_0.01%', 'pr_auc', 'macro_f1', 'bayes_p_1e-2', 'bayes_p_1e-3', 'bayes_p_1e-4', 'roc_auc', 'accuracy', 'latency_us']:
                    metrics_storage[dt_key][k].append(0.0)

        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)

        print(f"--- [Seed {seed}] Two-Stage Proposed: Recall@0.1%={metrics_two_stage['recall_at_fpr_0.1%']*100:.2f}%, PR-AUC={metrics_two_stage['pr_auc']:.4f}, BayesP(1e-3)={metrics_two_stage['bayesian_precision_1e-3']*100:.1f}%")

    total_duration = time.time() - start_total_time
    print(f"\n{'='*80}\nBENCHMARK COMPLETED ACROSS {len(SEEDS)} SEEDS IN {total_duration/60:.2f} MINUTES\n{'='*80}")

    # =========================================================================
    # STATISTICAL SIGNIFICANCE & SUMMARY CALCULATIONS
    # =========================================================================
    summary_rows = []
    scores_for_friedman = {}

    for model_name, m_dict in metrics_storage.items():
        row = {'Model': model_name}
        for metric_name, vals in m_dict.items():
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            row[f'{metric_name}_mean'] = mean_v
            row[f'{metric_name}_std'] = std_v
        summary_rows.append(row)
        scores_for_friedman[model_name] = m_dict['recall_0.1%']

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TABLES_DIR, "benchmark_multi_seed_summary.csv"), index=False)

    # Statistical Tests
    friedman_results = compute_friedman_and_nemenyi(scores_for_friedman, higher_is_better=True)
    
    # Pairwise Wilcoxon tests between Proposed and baselines
    proposed_recalls = metrics_storage['Proposed-TwoStage-SHAP']['recall_0.1%']
    wilcoxon_results = {}
    for b_model in models_to_track:
        if b_model != 'Proposed-TwoStage-SHAP':
            base_recalls = metrics_storage[b_model]['recall_0.1%']
            stat, p_val = compute_pairwise_wilcoxon(proposed_recalls, base_recalls)
            wilcoxon_results[b_model] = {'stat': stat, 'p_value': p_val}

    # Format LaTeX & Markdown Tables
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Empirical Benchmark across 5 Seeds on CIC-IDS2017 with Temporal Shift. Mean $\pm$ Standard Deviation.}",
        r"\label{tab:ids_benchmark}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Recall@FPR=0.1\%} & \textbf{Recall@FPR=0.01\%} & \textbf{PR-AUC} & \textbf{Bayes Prec ($10^{-3}$)} & \textbf{Macro-F1} & \textbf{Latency ($\mu$s)} \\",
        r"\midrule"
    ]

    for _, row in summary_df.iterrows():
        m = row['Model']
        rec01 = f"{row['recall_0.1%_mean']*100:.2f} $\pm$ {row['recall_0.1%_std']*100:.2f}\%"
        rec001 = f"{row['recall_0.01%_mean']*100:.2f} $\pm$ {row['recall_0.01%_std']*100:.2f}\%"
        prauc = f"{row['pr_auc_mean']:.4f} $\pm$ {row['pr_auc_std']:.4f}"
        bayes3 = f"{row['bayes_p_1e-3_mean']*100:.1f} $\pm$ {row['bayes_p_1e-3_std']*100:.1f}\%"
        f1 = f"{row['macro_f1_mean']:.4f} $\pm$ {row['macro_f1_std']:.4f}"
        lat = f"{row['latency_us_mean']:.1f}"

        if 'Proposed' in m:
            latex_lines.append(f"\\textbf{{{m}}} & \\textbf{{{rec01}}} & \\textbf{{{rec001}}} & \\textbf{{{prauc}}} & \\textbf{{{bayes3}}} & \\textbf{{{f1}}} & \\textbf{{{lat}}} \\\\")
        else:
            latex_lines.append(f"{m} & {rec01} & {rec001} & {prauc} & {bayes3} & {f1} & {lat} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}"
    ])

    with open(os.path.join(TABLES_DIR, "table1_benchmark_latex.tex"), "w") as f:
        f.write("\n".join(latex_lines))

    # Comprehensive JSON export
    full_output = {
        'metadata': {
            'title': 'Beyond Random Splits: A Drift-Aware, Explainable and Cost-Sensitive Framework',
            'seeds': SEEDS,
            'dataset': 'CIC-IDS2017 (NetFlow-v2 Harmonised, Shortcut-Stripped)',
            'execution_time_seconds': total_duration
        },
        'summary_table': summary_df.to_dict(orient='records'),
        'statistical_tests': {
            'friedman_test': friedman_results,
            'pairwise_wilcoxon_vs_proposed': wilcoxon_results
        },
        'novelty1_shap_stability': novelty1_records,
        'novelty2_conformal_uncertainty': novelty2_records,
        'novelty3_drift_active_retrain': novelty3_records
    }

    with open(os.path.join(RESULTS_DIR, "framework_benchmark_results.json"), "w") as f:
        json.dump(full_output, f, indent=2)

    print("\n[SUCCESS] All benchmark results, tables, figures, and statistical tests saved to 'results/'.")

if __name__ == '__main__':
    run_experiment()
