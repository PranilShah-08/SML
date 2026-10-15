"""
Plotting script for Publication-Grade Figures:
1. Nemenyi Critical Difference Diagram & Model Ranking
2. Operational Metrics Comparison (Recall @ FPR=0.1% and 0.01%) with error bars
3. Bayesian Precision Degradation across Attack Base-Rate Priors (Axelsson's Law)
4. SHAP-Stability Feature Pareto Frontier
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_all_plots():
    summary_df = pd.read_csv(os.path.join(TABLES_DIR, "benchmark_multi_seed_summary.csv"))
    stability_df = pd.read_csv(os.path.join(TABLES_DIR, "shap_stability_rankings.csv"))
    with open("results/framework_benchmark_results.json") as f:
        meta = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Operational Recall @ Ultra-Low FPR (0.1% and 0.01%)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    summary_sorted = summary_df.sort_values(by='recall_0.1%_mean', ascending=False).reset_index(drop=True)

    x = np.arange(len(summary_sorted))
    width = 0.38

    rects1 = ax.bar(x - width/2, summary_sorted['recall_0.1%_mean'] * 100, width,
                    yerr=summary_sorted['recall_0.1%_std'] * 100, label='Recall @ FPR = 0.1% (10⁻³)',
                    color='#1f77b4', capsize=4, alpha=0.9)
    rects2 = ax.bar(x + width/2, summary_sorted['recall_0.01%_mean'] * 100, width,
                    yerr=summary_sorted['recall_0.01%_std'] * 100, label='Recall @ FPR = 0.01% (10⁻⁴)',
                    color='#d62728', capsize=4, alpha=0.9)

    ax.set_ylabel('Operational Recall (%) on Friday Temporal Stream', fontsize=12, fontweight='bold')
    ax.set_title('Operational Detection Sensitivity under Strict False-Alarm Budget (Mean ± Std, N=5 Seeds)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_sorted['Model'], rotation=35, ha='right', fontsize=10)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "operational_recall_at_fpr_comparison.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Bayesian Precision across Realistic Attack Priors (Axelsson Base-Rate Fallacy)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    priors = np.logspace(-4, -1, 100)  # 10^-4 to 10^-1

    # Representative top models
    top_models = ['Deep-FTTransformer', 'CatBoost-Balanced', 'Proposed-TwoStage-SHAP', 'LightGBM-Balanced', 'RandomForest-Balanced']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#7f7f7f']

    for m_name, color in zip(top_models, colors):
        row = summary_df[summary_df['Model'] == m_name].iloc[0]
        tpr = row['recall_0.1%_mean']
        fpr = 0.001
        bayes_prec = [(tpr * p) / (tpr * p + fpr * (1.0 - p) + 1e-12) * 100 for p in priors]
        ax.plot(priors, bayes_prec, label=f"{m_name} (Rec@0.1%={tpr*100:.1f}%)", color=color, linewidth=2.5)

    ax.set_xscale('log')
    ax.axvline(1e-3, color='black', linestyle=':', label='Axelsson Enterprise Prior P(A)=10⁻³')
    ax.set_xlabel('Attack Base-Rate Prior P(Attack)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bayesian Operational Precision (%) P(Attack | Alert)', fontsize=12, fontweight='bold')
    ax.set_title('The Base-Rate Fallacy: True Alert Fidelity as Attack Prior Decreases (FPR=0.1%)', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "bayesian_precision_curve.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Nemenyi Post-Hoc Ranking & Critical Difference Diagram
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    ranks_dict = meta['statistical_tests']['friedman_test']['mean_ranks']
    cd_val = meta['statistical_tests']['friedman_test']['nemenyi_cd']

    sorted_ranks = sorted(ranks_dict.items(), key=lambda kv: kv[1])
    names = [kv[0] for kv in sorted_ranks]
    ranks = [kv[1] for kv in sorted_ranks]

    ax.scatter(ranks, range(len(names)), color='#1f77b4', s=80, zorder=3)
    for i, (name, r) in enumerate(zip(names, ranks)):
        ax.hlines(i, 1.0, r, color='gray', linestyle=':', alpha=0.6)

    # Plot CD reference bar at the top
    top_y = len(names)
    ax.plot([1.0, 1.0 + cd_val], [top_y, top_y], color='#d62728', linewidth=4, solid_capstyle='butt')
    ax.text(1.0 + cd_val/2, top_y + 0.3, f"Critical Difference (CD = {cd_val:.2f}, α=0.05)",
            ha='center', va='bottom', color='#d62728', fontweight='bold', fontsize=10)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Average Rank across 5 Seeds (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title(f"Friedman Test (χ² = {meta['statistical_tests']['friedman_test']['friedman_stat']:.2f}, p = {meta['statistical_tests']['friedman_test']['friedman_p_value']:.2e}) with Nemenyi Post-Hoc", fontsize=12, fontweight='bold')
    ax.set_xlim(0.5, 16.0)
    ax.set_ylim(-0.5, len(names) + 1.2)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "nemenyi_critical_difference.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 4. SHAP Stability Pareto Frontier
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    top25 = stability_df.head(25)
    rest = stability_df.iloc[25:]

    ax.scatter(rest['rank_delta'], rest['mean_shap'], color='gray', alpha=0.5, label='Discarded Features (Unstable/Low Imp)', s=40)
    ax.scatter(top25['rank_delta'], top25['mean_shap'], color='#2ca02c', alpha=0.9, label='Selected SHAP-Stable Features (Pareto Optimal)', s=80, edgecolors='black')

    # Annotate top 5
    for _, row in top25.head(5).iterrows():
        ax.annotate(row['feature'], (row['rank_delta'], row['mean_shap']),
                    xytext=(row['rank_delta'] + 1, row['mean_shap'] + 0.02),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlabel('Cross-Day SHAP Rank Discrepancy (|Rank_Day1 - Rank_Day2|)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute SHAP Attribution', fontsize=12, fontweight='bold')
    ax.set_title('Novelty 1: SHAP-Stability Feature Selection (Invariance across Temporal Shifts)', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_stability_pareto.png"), dpi=300)
    plt.close()

    print("[Visualizations] Generated all 4 publication-quality figures in results/figures/.")

if __name__ == '__main__':
    generate_all_plots()
