# Beyond Random Splits: A Drift-Aware, Explainable and Cost-Sensitive Framework for Generalisable Network Intrusion Detection

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LightGBM 4.7](https://img.shields.io/badge/LightGBM-4.7-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost 1.2](https://img.shields.io/badge/CatBoost-1.2-yellow.svg)](https://catboost.ai/)
[![SHAP 0.51](https://img.shields.io/badge/SHAP-0.51-FF6F00.svg)](https://shap.readthedocs.io/)
[![Benchmark: CIC-IDS2017](https://img.shields.io/badge/Dataset-CIC--IDS2017-0052CC.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Statistical Testing: Friedman p<1e-6](https://img.shields.io/badge/Friedman%20Test-p%20<%201.2e--7-success.svg)](#4-statistical-significance-testing)
[![Protocol: 5 Seeds Repeated](https://img.shields.io/badge/Protocol-5%20Seeds%20Mean%20%C2%B1%20Std-blue.svg)](#3-comprehensive-empirical-results-5-seed-protocol)

---

## Table of Contents
1. [Overview & Core Research Problem](#1-overview--core-research-problem)
2. [End-to-End System Architecture (6 Layers)](#2-end-to-end-system-architecture-6-layers)
   - [Layer 1: Harmonised Feature Layer (NetFlow-v2)](#layer-1-harmonised-feature-layer)
   - [Layer 2: SHAP-Stability Feature Selection (Novelty 1)](#layer-2-shap-stability-feature-selection-novelty-1)
   - [Layer 3: Two-Stage Cost-Sensitive Detector](#layer-3-two-stage-cost-sensitive-detector)
   - [Layer 4: Mondrian Conformal Uncertainty & Zero-Day Layer (Novelty 2)](#layer-4-mondrian-conformal-uncertainty-layer-novelty-2)
   - [Layer 5: Dual-Signal Drift Monitor & Active Selective Retraining (Novelty 3)](#layer-5-drift-monitor--selective-retraining-novelty-3)
   - [Layer 6: Explanation & Forensic Incident Reporting](#layer-6-explanation--forensic-incident-reporting)
3. [Comprehensive Empirical Results (5-Seed Protocol)](#3-comprehensive-empirical-results-5-seed-protocol)
   - [3.1 Multi-Seed Operational Benchmark Table (15 Architectures)](#31-multi-seed-operational-benchmark-table)
   - [3.2 The Base-Rate Fallacy in Telemetry (Axelsson's Law)](#32-the-base-rate-fallacy-in-telemetry)
4. [Statistical Significance Testing](#4-statistical-significance-testing)
   - [4.1 Friedman Global Non-Parametric Test](#41-friedman-global-test)
   - [4.2 Nemenyi Post-Hoc Test & Critical Difference](#42-nemenyi-post-hoc-test)
   - [4.3 Pairwise Wilcoxon Signed-Rank Comparisons](#43-pairwise-wilcoxon-tests)
5. [Repository Structure & Code Manifest](#5-repository-structure--code-manifest)
6. [Reproduction & Execution Guide](#6-reproduction--execution-guide)

---

## 1. Overview & Core Research Problem

High-throughput Network Intrusion Detection Systems (NIDS) operating on enterprise backbones (10 Gbps to 100 Gbps) face fundamental statistical and operational challenges:
1. **Shortcut Learning**: Classifiers overfit to synthetic testbed artifacts (destination ports, IP subnets, timestamps, TCP window sizes) rather than learning invariant transport semantics.
2. **Distributional Fragility (Beyond Random Splits)**: Evaluating models on random train/test splits leaks packet-burst autocorrelation, creating artificially inflated metrics ($>99\%$) that collapse under temporal shift.
3. **The Base-Rate Fallacy (Axelsson, 2000)**: In real telemetry where attacks represent $\le 0.1\%$ ($P(A) \le 10^{-3}$), even a low $0.5\%$ False Positive Rate causes $>85\%$ of all triggered alerts to be false alarms, paralyzing Security Operations Centers (SOCs).
4. **Overconfident Uncertainty Calibration**: Softmax probabilities fail to calibrate under domain shift, masking zero-day threats.

This repository implements an end-to-end, mathematically rigorous framework evaluated across **5 independent seeds** (`42, 101, 2024, 7, 99`) on genuine **[CIC-IDS2017](file:///Users/pranil/Desktop/SML/data/raw/MachineLearningCVE)** flow records under chronological temporal shift (Train: Tuesday+Wednesday $\to$ Cal: Monday+Thursday $\to$ Test: Friday).

---

## 2. End-to-End System Architecture (6 Layers)

```
        Raw Bidirectional Flows (CIC-IDS2017)
                        │
 ┌──────────────────────▼───────────────────────┐
 │ [1] Harmonised Feature Layer                 │
 │     • NetFlow-v2 feature standardization     │
 │     • Elimination of spurious shortcuts      │
 │       (Destination Port, Init_Win, min_seg)  │
 └──────────────────────┬───────────────────────┘
                        │
 ┌──────────────────────▼───────────────────────┐
 │ [2] SHAP-Stability Feature Selection (NOVELTY 1)
 │     • Cross-day TreeSHAP attribution ranking │
 │     • Pareto-optimal stability penalization: │
 │       Score = Mean_SHAP / (1 + λ * ΔRank / D)│
 └──────────────────────┬───────────────────────┘
                        │
 ┌──────────────────────▼───────────────────────┐
 │ [3] Two-Stage Cost-Sensitive Detector        │
 │     • Stage 1: Binary Filter (LightGBM/FT-Tr)│
 │       Calibrated at target FPR <= 0.1%, 0.01%│
 │     • Stage 2: Multiclass Attack Classifier  │
 └──────────────────────┬───────────────────────┘
                        │
 ┌──────────────────────▼───────────────────────┐
 │ [4] Mondrian Conformal Uncertainty (NOVELTY 2│
 │     • Per-class non-conformity calibration   │
 │     • Tri-State Triage Decision Policy:      │
 │       - Auto-Decide (Confident Flow)         │
 │       - SOC Analyst (Ambiguous Multi-Class)  │
 │       - Unknown Attack (Zero-Day Anomaly)    │
 └──────────────────────┬───────────────────────┘
                        │
 ┌──────────────────────▼───────────────────────┐
 │ [5] Drift Monitor & Active Retrain (NOVELTY 3│
 │     • ADWIN change-point detection on p-vals │
 │     • 1-Wasserstein shift on SHAP values     │
 │     • Budgeted Active Retraining (+150 flows)│
 └──────────────────────┬───────────────────────┘
                        │
 ┌──────────────────────▼───────────────────────┐
 │ [6] Explainable Output & Forensics           │
 │     • Local SHAP Waterfall Plots per Alert   │
 │     • Root-Cause "Why Drift Happened" Report │
 └──────────────────────────────────────────────┘
```

---

## 3. Comprehensive Empirical Results (5-Seed Protocol)

### 3.1 Multi-Seed Operational Benchmark Table
Results evaluated across 5 repeated seeds under chronological cross-day shift (Tuesday/Wednesday train $\to$ Friday test). Metrics report **Mean $\pm$ Standard Deviation**:

| Architecture | Recall @ FPR=0.1% | Recall @ FPR=0.01% | PR-AUC | Bayes Prec ($10^{-3}$) | Multiclass Macro-F1 | Latency ($\mu$s/flow) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep-FTTransformer** | **31.18 $\pm$ 0.24%** | **31.09 $\pm$ 0.24%** | **0.8267 $\pm$ 0.0541** | **23.79 $\pm$ 0.14%** | **0.6509 $\pm$ 0.0016** | 61.3 $\mu$s |
| **Deep-LSTM** | 31.11 $\pm$ 0.41% | 29.98 $\pm$ 1.64% | 0.7572 $\pm$ 0.0492 | 23.74 $\pm$ 0.24% | 0.6504 $\pm$ 0.0028 | 2.8 $\mu$s |
| **Deep-MLP** | 30.53 $\pm$ 0.70% | 29.34 $\pm$ 1.28% | 0.8243 $\pm$ 0.0481 | 23.41 $\pm$ 0.42% | 0.6463 $\pm$ 0.0048 | 0.15 $\mu$s |
| **Deep-1DCNN** | 30.61 $\pm$ 0.22% | 27.67 $\pm$ 4.12% | 0.8378 $\pm$ 0.0257 | 23.45 $\pm$ 0.13% | 0.6469 $\pm$ 0.0016 | 74.1 $\mu$s |
| **CatBoost-Balanced** | 17.00 $\pm$ 8.66% | 7.57 $\pm$ 2.69% | 0.8304 $\pm$ 0.0392 | 14.08 $\pm$ 6.27% | 0.5386 $\pm$ 0.0707 | **0.097 $\mu$s** |
| **TabNet** | 20.45 $\pm$ 12.55% | 18.88 $\pm$ 13.63% | 0.7945 $\pm$ 0.0390 | 16.01 $\pm$ 9.44% | 0.5623 $\pm$ 0.1064 | 8.1 $\mu$s |
| **Proposed-TwoStage-SHAP** | 14.00 $\pm$ 13.09% | 8.28 $\pm$ 8.55% | 0.6983 $\pm$ 0.0520 | 11.15 $\pm$ 9.88% | 0.1348 $\pm$ 0.0030 | 2.0 $\mu$s |
| **LogisticRegression-Balanced** | 9.71 $\pm$ 0.68% | 9.71 $\pm$ 0.68% | 0.7870 $\pm$ 0.0117 | 8.85 $\pm$ 0.57% | 0.4783 $\pm$ 0.0063 | **0.024 $\mu$s** |
| **LightGBM-NoWeights** | 9.45 $\pm$ 9.04% | 1.19 $\pm$ 1.61% | 0.7675 $\pm$ 0.0273 | 8.04 $\pm$ 7.31% | 0.4725 $\pm$ 0.0814 | 0.0 $\mu$s |
| **XGBoost-Balanced** | 6.57 $\pm$ 12.03% | 0.90 $\pm$ 1.77% | 0.7703 $\pm$ 0.0105 | 5.13 $\pm$ 9.18% | 0.4409 $\pm$ 0.1031 | 0.32 $\mu$s |
| **LightGBM-SMOTE** | 2.51 $\pm$ 3.04% | 2.25 $\pm$ 2.81% | 0.7443 $\pm$ 0.0845 | 2.37 $\pm$ 2.85% | 0.4090 $\pm$ 0.0313 | 0.0 $\mu$s |
| **LightGBM-Balanced** | 2.17 $\pm$ 2.59% | 1.81 $\pm$ 2.22% | 0.7465 $\pm$ 0.0393 | 2.07 $\pm$ 2.45% | 0.4060 $\pm$ 0.0267 | 1.05 $\mu$s |
| **RandomForest-Balanced** | 0.00 $\pm$ 0.00% | 0.00 $\pm$ 0.00% | 0.8039 $\pm$ 0.0143 | 0.00 $\pm$ 0.00% | 0.3834 $\pm$ 0.0001 | 1.51 $\mu$s |
| **IsolationForest** | 0.00 $\pm$ 0.00% | 0.00 $\pm$ 0.00% | 0.4765 $\pm$ 0.0102 | 0.00 $\pm$ 0.00% | 0.3835 $\pm$ 0.0000 | 0.0 $\mu$s |
| **DeepAutoencoder** | 0.00 $\pm$ 0.00% | 0.00 $\pm$ 0.00% | 0.5863 $\pm$ 0.0123 | 0.00 $\pm$ 0.00% | 0.3835 $\pm$ 0.0000 | 93.2 $\mu$s |

---

## 4. Statistical Significance Testing

### 4.1 Friedman Global Test
- **Friedman Statistic**: $\chi_F^2 = 60.01$ ($df = 14$)
- **p-value**: $p = 1.17 \times 10^{-7}$
- **Conclusion**: Decisively rejects the null hypothesis ($p < 0.001$) that all 15 architectures have identical performance under temporal concept shift.

### 4.2 Nemenyi Post-Hoc Test & Critical Difference
- **Critical Difference**: $CD = 8.49$ ($\alpha = 0.05, k=15, N=5$)
- **Top Average Ranks**:
  1. `Deep-FTTransformer`: **Rank 1.2** (Top Ranked Architecture)
  2. `Deep-LSTM`: **Rank 2.3**
  3. `Deep-MLP`: **Rank 3.1**
  4. `Deep-1DCNN`: **Rank 3.8**
  5. `TabNet`: **Rank 6.8**
  6. `CatBoost-Balanced`: **Rank 7.0**
  7. `Proposed-TwoStage-SHAP`: **Rank 8.1**
  ...
  15. `RandomForest-Balanced`: **Rank 13.6**

---

## 5. Repository Structure & Code Manifest

```
/Users/pranil/Desktop/SML/
├── src/
│   ├── data/
│   │   └── harmonizer.py         # NetFlow-v2 feature standardization & shortcut stripping
│   ├── features/
│   │   └── shap_selector.py      # Novelty 1: SHAP-stability rank concordance selector
│   ├── models/
│   │   ├── cost_sensitive.py     # Two-Stage Cost-Sensitive Detector (FPR=0.1% threshold)
│   │   ├── baselines.py          # Classical models, class-weights, and deep autoencoders
│   │   ├── deep_tabular.py       # MLP, 1D-CNN, LSTM, FT-Transformer, TabNet
│   │   └── run_deep_models.py    # Isolated sub-process runner for macOS OpenMP safety
│   ├── conformal/
│   │   └── mondrian.py           # Novelty 2: Mondrian conformal p-values & zero-day detection
│   ├── drift/
│   │   └── monitor.py            # Novelty 3: ADWIN on p-values, 1-Wasserstein SHAP drift & active retrain
│   ├── evaluation/
│   │   ├── metrics.py            # Recall@FPR, Bayesian precision, PR-AUC, Macro-F1
│   │   └── statistics.py         # Friedman test, Nemenyi CD, Wilcoxon tests, Optuna tuner
│   └── explain/
│       └── reporter.py           # Local SHAP waterfall generator & Why Drift Happened reporter
├── experiments/
│   ├── run_full_framework.py     # 5-seed master benchmark driver
│   └── plot_benchmark_visualizations.py # Publication-quality figure generator
├── results/
│   ├── figures/                  # Publication figures (Nemenyi CD, Recall@FPR, Bayesian curve)
│   ├── tables/                   # Summary CSVs, LaTeX Table 1, and SHAP rankings
│   ├── why_drift_happened_report.md # Generated forensic drift incident report
│   └── framework_benchmark_results.json # Full machine-readable experimental telemetry
├── RESEARCH_PAPER_REPORT.md      # Complete academic paper manuscript
└── requirements.txt              # Pinned reproducible virtual environment
```

---

## 6. Reproduction & Execution Guide

To reproduce the full 5-seed empirical benchmark and regenerate all tables and figures from scratch:

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Run the 5-seed master benchmark
python experiments/run_full_framework.py

# 3. Generate publication-grade visualizations
python experiments/plot_benchmark_visualizations.py
```
