"""Comprehensive Exploratory Data Analysis for Stage 1 Heat Exchanger Degradation.

Answers the 11 core scientific questions and generates all required plots to results/figures/
and statistical summaries to results/tables/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data_processing import (
    HISTORY_FEATURE_NAMES,
    PROJECT_ROOT,
    VALID_PREDICTORS_CURRENT,
    construct_targets,
    engineer_history_features,
    load_raw_dataset,
)

FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def run_eda():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[EDA] Loading data and preparing features...")
    raw_df = load_raw_dataset()
    df_targets = construct_targets(raw_df)
    df = engineer_history_features(df_targets)

    # -------------------------------------------------------------
    # 1 & 2. DI_fouling and DI_total Distributions
    # -------------------------------------------------------------
    print("[EDA] Plotting Target Distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # DI_fouling
    sns.histplot(df["DI_fouling"], bins=50, kde=True, ax=axes[0], color="#1f77b4", edgecolor="black")
    axes[0].set_title("Distribution of Primary Target: DI_fouling", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Fouling Degradation Index (1 - U_overall / U_clean)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].axvline(df["DI_fouling"].mean(), color="red", linestyle="--", label=f"Mean: {df['DI_fouling'].mean():.5f}")
    axes[0].axvline(df["DI_fouling"].median(), color="green", linestyle=":", label=f"Median: {df['DI_fouling'].median():.5f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # DI_total
    sns.histplot(df["DI_total"], bins=50, kde=True, ax=axes[1], color="#ff7f0e", edgecolor="black")
    axes[1].set_title("Distribution of Secondary Target: DI_total", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Total Thermal Degradation Index (1 - U_total / U_clean)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].axvline(df["DI_total"].mean(), color="red", linestyle="--", label=f"Mean: {df['DI_total'].mean():.4f}")
    axes[1].axvline(df["DI_total"].median(), color="green", linestyle=":", label=f"Median: {df['DI_total'].median():.4f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "target_distributions.png", dpi=300)
    plt.close(fig)

    # Target statistics table
    target_stats = pd.DataFrame({
        "Metric": ["Mean", "Std Dev", "Min", "25%", "Median (50%)", "75%", "95%", "Max", "Skewness", "Kurtosis"],
        "DI_fouling": [
            df["DI_fouling"].mean(),
            df["DI_fouling"].std(),
            df["DI_fouling"].min(),
            df["DI_fouling"].quantile(0.25),
            df["DI_fouling"].median(),
            df["DI_fouling"].quantile(0.75),
            df["DI_fouling"].quantile(0.95),
            df["DI_fouling"].max(),
            df["DI_fouling"].skew(),
            df["DI_fouling"].kurtosis(),
        ],
        "DI_total": [
            df["DI_total"].mean(),
            df["DI_total"].std(),
            df["DI_total"].min(),
            df["DI_total"].quantile(0.25),
            df["DI_total"].median(),
            df["DI_total"].quantile(0.75),
            df["DI_total"].quantile(0.95),
            df["DI_total"].max(),
            df["DI_total"].skew(),
            df["DI_total"].kurtosis(),
        ]
    })
    target_stats.to_csv(TABLES_DIR / "target_statistics.csv", index=False)
    print(f"[EDA] Target stats saved to {TABLES_DIR / 'target_statistics.csv'}")

    # -------------------------------------------------------------
    # 3, 4, 5, 6. DI vs Major Valid Predictors
    # -------------------------------------------------------------
    print("[EDA] Plotting Targets vs Major Valid Predictors...")
    sample_sub = df.sample(n=10000, random_state=42)
    major_preds = ["m_dot_measured_kg_s", "Re", "T_in_measured_C", "dP_Pa"]
    labels = ["Measured Mass Flow [kg/s]", "Reynolds Number (Re)", "Measured Inlet Temp [°C]", "Pressure Drop [Pa]"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, (pred, lbl) in enumerate(zip(major_preds, labels)):
        # DI_fouling
        sns.scatterplot(data=sample_sub, x=pred, y="DI_fouling", alpha=0.25, s=12, color="#1f77b4", ax=axes[0, i])
        sns.regplot(data=sample_sub, x=pred, y="DI_fouling", scatter=False, ax=axes[0, i], color="red", ci=None, line_kws={"linewidth": 1.5})
        axes[0, i].set_title(f"DI_fouling vs {pred}", fontsize=11, fontweight="bold")
        axes[0, i].set_xlabel(lbl, fontsize=9)
        axes[0, i].set_ylabel("DI_fouling", fontsize=9)
        axes[0, i].grid(True, alpha=0.3)

        # DI_total
        sns.scatterplot(data=sample_sub, x=pred, y="DI_total", alpha=0.25, s=12, color="#ff7f0e", ax=axes[1, i])
        sns.regplot(data=sample_sub, x=pred, y="DI_total", scatter=False, ax=axes[1, i], color="red", ci=None, line_kws={"linewidth": 1.5})
        axes[1, i].set_title(f"DI_total vs {pred}", fontsize=11, fontweight="bold")
        axes[1, i].set_xlabel(lbl, fontsize=9)
        axes[1, i].set_ylabel("DI_total", fontsize=9)
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "targets_vs_major_predictors.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 7 & 8. Correlation & Redundancy Analysis
    # -------------------------------------------------------------
    print("[EDA] Computing Correlation & Redundancy Matrices...")
    analysis_cols = VALID_PREDICTORS_CURRENT + ["cum_thermal_throughput", "DI_fouling", "DI_total"]
    pearson_corr = df[analysis_cols].corr(method="pearson")
    spearman_corr = df[analysis_cols].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
    
    sns.heatmap(pearson_corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("Pearson Linear Correlation Matrix", fontsize=12, fontweight="bold")
    
    sns.heatmap(spearman_corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Spearman Rank Correlation Matrix (Nonlinearities)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "correlation_matrices.png", dpi=300)
    plt.close(fig)

    pearson_corr.to_csv(TABLES_DIR / "pearson_correlation.csv")
    spearman_corr.to_csv(TABLES_DIR / "spearman_correlation.csv")

    # -------------------------------------------------------------
    # 9 & 10. Identifiability Check: Instantaneous State vs Cumulative Degradation
    # -------------------------------------------------------------
    print("[EDA] Plotting Identifiability Check (Multi-Trajectory Comparison)...")
    # Select 3 representative scenarios: Low Temp (T60_Q5.0), Mid Temp (T90_Q5.0), High Temp (T120_Q5.0)
    scenarios_to_plot = ["T60_Q5.0", "T90_Q5.0", "T120_Q5.0"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    
    for sc, col in zip(scenarios_to_plot, colors):
        sub = df[df["scenario_id"] == sc].sort_values("time_h")
        t = sub["time_h"]
        
        # 1. Measured Inlet Temperature (Instantaneous Operating Condition)
        axes[0].plot(t, sub["T_in_measured_C"], label=f"{sc} (T_in measured)", color=col, lw=0.8, alpha=0.7)
        axes[0].set_ylabel("T_in Measured [°C]", fontsize=10)
        axes[0].set_title("Instantaneous Measured Temperature (Cyclic Operating Point)", fontsize=11, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        
        # 2. Measured Pressure Drop (Instantaneous Hydraulic Condition)
        axes[1].plot(t, sub["dP_Pa"], label=f"{sc} (dP measured)", color=col, lw=0.8, alpha=0.7)
        axes[1].set_ylabel("dP [Pa]", fontsize=10)
        axes[1].set_title("Instantaneous Measured Pressure Drop (Unchanged by Fouling in V2 Hydraulics)", fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        # 3. Ground Truth DI_fouling (True Cumulative State)
        axes[2].plot(t, sub["DI_fouling"], label=f"{sc} DI_fouling", color=col, lw=1.5)
        # Mark CIP events
        cip_points = sub[sub["cip_event"] == 1]
        axes[2].scatter(cip_points["time_h"], cip_points["DI_fouling"], marker="v", color="black", s=45, zorder=5, label="CIP Event" if sc == scenarios_to_plot[0] else "")
        axes[2].set_ylabel("DI_fouling (Degradation)", fontsize=10)
        axes[2].set_xlabel("Operating Time [hours]", fontsize=11)
        axes[2].set_title("Actual Accumulated Fouling Degradation (State Evolution & CIP Resets)", fontsize=11, fontweight="bold")
        axes[2].grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "identifiability_check_trajectories.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 11. Operating Regime Analysis
    # -------------------------------------------------------------
    print("[EDA] Generating Operating Regime Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Flow vs Temperature Regime Grid
    regime_df = df.groupby(["T_in_C", "m_dot_nominal_kg_s"])["DI_fouling"].mean().unstack()
    sns.heatmap(regime_df, cmap="YlOrRd", annot=True, fmt=".5f", ax=axes[0], cbar_kws={"label": "Mean DI_fouling"})
    axes[0].set_title("Mean DI_fouling Across Operating Regimes (T_in vs Flow)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Nominal Mass Flow [kg/s]", fontsize=10)
    axes[0].set_ylabel("Nominal Inlet Temp [°C]", fontsize=10)

    # Max DI_fouling across Regimes
    regime_max = df.groupby(["T_in_C", "m_dot_nominal_kg_s"])["DI_fouling"].max().unstack()
    sns.heatmap(regime_max, cmap="magma", annot=True, fmt=".5f", ax=axes[1], cbar_kws={"label": "Max DI_fouling (Fouling Peak)"})
    axes[1].set_title("Peak DI_fouling Across Operating Regimes", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Nominal Mass Flow [kg/s]", fontsize=10)
    axes[1].set_ylabel("Nominal Inlet Temp [°C]", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "operating_regime_matrix.png", dpi=300)
    plt.close(fig)

    print("[EDA] Complete! All figures saved to results/figures/ and tables to results/tables/")


if __name__ == "__main__":
    run_eda()
