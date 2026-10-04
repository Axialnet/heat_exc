"""Model diagnostics, residual plots, feature importances, and SHAP explainability."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from src.data_processing import (
    PROJECT_ROOT,
    VALID_PREDICTORS_CURRENT,
    VALID_PREDICTORS_MODE_B,
    construct_targets,
    engineer_history_features,
    load_raw_dataset,
)

FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MODEL_RESULTS_DIR = PROJECT_ROOT / "results" / "model_results"
MODELS_DIR = PROJECT_ROOT / "models" / "stage1"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"


def run_diagnostics():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("[Diagnostics] Loading dataset and test split...")
    raw_df = load_raw_dataset()
    df = construct_targets(raw_df)
    df = engineer_history_features(df)

    with open(MODEL_RESULTS_DIR / "scenario_split.json", "r") as f:
        split_info = json.load(f)
    test_sc = split_info["test_scenarios"]

    df_test = df[df["scenario_id"].isin(test_sc)].copy()
    y_test_foul = df_test["DI_fouling"].values
    y_test_tot = df_test["DI_total"].values

    # Load predictions
    models_to_plot = [
        ("di_fouling_current_linear", "Linear Regression (Current Only)", "#1f77b4"),
        ("di_fouling_current_rf", "Random Forest (Current Only)", "#2ca02c"),
        ("di_fouling_current_xgb", "XGBoost (Current Only)", "#d62728"),
        ("di_fouling_history_linear", "Linear Regression (Current + History)", "#9467bd"),
        ("di_fouling_history_rf", "Random Forest (Current + History)", "#8c564b"),
        ("di_fouling_history_xgb", "XGBoost (Current + History)", "#e377c2"),
    ]

    preds_dict = {}
    for key, _, _ in models_to_plot:
        npy_path = PREDICTIONS_DIR / f"{key}_test_preds.npy"
        preds_dict[key] = np.load(npy_path)

    # -------------------------------------------------------------
    # 1. Actual vs Predicted Plots (DI_fouling)
    # -------------------------------------------------------------
    print("[Diagnostics] Generating Actual vs Predicted plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    sample_idx = np.random.choice(len(y_test_foul), size=min(10000, len(y_test_foul)), replace=False)

    for idx, (col_id, title, color) in enumerate(models_to_plot):
        ax = axes[idx]
        y_pred = preds_dict[col_id]
        
        ax.scatter(y_test_foul[sample_idx], y_pred[sample_idx], alpha=0.3, s=10, color=color)
        
        min_v = min(y_test_foul.min(), y_pred.min())
        max_v = max(y_test_foul.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=1.5, label="Ideal 1:1 Line")
        
        r2_val = 1.0 - np.sum((y_test_foul - y_pred)**2) / np.sum((y_test_foul - y_test_foul.mean())**2)
        mae_val = float(np.mean(np.abs(y_test_foul - y_pred)))
        
        ax.set_title(f"{title}\nR² = {r2_val:.4f} | MAE = {mae_val:.6f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual DI_fouling", fontsize=9)
        ax.set_ylabel("Predicted DI_fouling", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "actual_vs_predicted_comparison.png", dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. Residual vs Predicted & Residual Distribution Plots
    # -------------------------------------------------------------
    print("[Diagnostics] Generating Residual diagnostics...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    xgb_a_pred = preds_dict["di_fouling_current_xgb"]
    xgb_b_pred = preds_dict["di_fouling_history_xgb"]
    res_a = y_test_foul - xgb_a_pred
    res_b = y_test_foul - xgb_b_pred

    # Res vs Pred (Mode A)
    axes[0, 0].scatter(xgb_a_pred[sample_idx], res_a[sample_idx], alpha=0.25, s=10, color="#d62728")
    axes[0, 0].axhline(0, color="black", linestyle="--", lw=1.5)
    axes[0, 0].set_title("XGBoost (Mode A: Current Only) Residual vs Predicted", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Predicted DI_fouling", fontsize=10)
    axes[0, 0].set_ylabel("Residual (Actual - Predicted)", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Res vs Pred (Mode B)
    axes[0, 1].scatter(xgb_b_pred[sample_idx], res_b[sample_idx], alpha=0.25, s=10, color="#2ca02c")
    axes[0, 1].axhline(0, color="black", linestyle="--", lw=1.5)
    axes[0, 1].set_title("XGBoost (Mode B: Current + History) Residual vs Predicted", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Predicted DI_fouling", fontsize=10)
    axes[0, 1].set_ylabel("Residual (Actual - Predicted)", fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Residual Distribution (Mode A)
    sns.histplot(res_a, bins=50, kde=True, ax=axes[1, 0], color="#d62728", edgecolor="black")
    axes[1, 0].axvline(0, color="black", linestyle="--", lw=1.5)
    axes[1, 0].set_title(f"XGBoost Mode A Residual Distribution\nMean: {np.mean(res_a):.6f} | Std: {np.std(res_a):.6f}", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Residual", fontsize=10)
    axes[1, 0].set_ylabel("Count", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Residual Distribution (Mode B)
    sns.histplot(res_b, bins=50, kde=True, ax=axes[1, 1], color="#2ca02c", edgecolor="black")
    axes[1, 1].axvline(0, color="black", linestyle="--", lw=1.5)
    axes[1, 1].set_title(f"XGBoost Mode B Residual Distribution\nMean: {np.mean(res_b):.6f} | Std: {np.std(res_b):.6f}", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Residual", fontsize=10)
    axes[1, 1].set_ylabel("Count", fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "residual_diagnostics_comparison.png", dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. Uncertainty Quantification & Error Breakdown Table
    # -------------------------------------------------------------
    abs_err_a = np.abs(res_a)
    abs_err_b = np.abs(res_b)

    error_stats = pd.DataFrame({
        "Error Metric": ["Median Absolute Error (50%)", "75th Percentile AE", "90th Percentile AE", "95th Percentile AE", "99th Percentile AE", "Max Absolute Error", "Mean Absolute Error"],
        "Mode A: Current Only": [
            float(np.percentile(abs_err_a, 50)),
            float(np.percentile(abs_err_a, 75)),
            float(np.percentile(abs_err_a, 90)),
            float(np.percentile(abs_err_a, 95)),
            float(np.percentile(abs_err_a, 99)),
            float(np.max(abs_err_a)),
            float(np.mean(abs_err_a)),
        ],
        "Mode B: Current + History": [
            float(np.percentile(abs_err_b, 50)),
            float(np.percentile(abs_err_b, 75)),
            float(np.percentile(abs_err_b, 90)),
            float(np.percentile(abs_err_b, 95)),
            float(np.percentile(abs_err_b, 99)),
            float(np.max(abs_err_b)),
            float(np.mean(abs_err_b)),
        ]
    })
    error_stats.to_csv(TABLES_DIR / "prediction_uncertainty_quantification.csv", index=False)
    print(f"[Diagnostics] Saved uncertainty metrics to: {TABLES_DIR / 'prediction_uncertainty_quantification.csv'}")

    # -------------------------------------------------------------
    # 4. XGBoost Feature Importance
    # -------------------------------------------------------------
    print("[Diagnostics] Extracting XGBoost Feature Importances...")
    xgb_pkg = joblib.load(MODELS_DIR / "di_fouling_history_xgb.joblib")
    model = xgb_pkg["model"]
    feature_names = xgb_pkg["features"]

    importance_gain = model.get_booster().get_score(importance_type="gain")
    importance_weight = model.get_booster().get_score(importance_type="weight")

    # Booster maps f0, f1... to feature index
    mapped_gain = {}
    mapped_weight = {}
    for k, v in importance_gain.items():
        name = feature_names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k
        mapped_gain[name] = v
    for k, v in importance_weight.items():
        name = feature_names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k
        mapped_weight[name] = v

    imp_df = pd.DataFrame({
        "Feature": list(mapped_gain.keys()),
        "Gain": list(mapped_gain.values()),
        "Weight": [mapped_weight.get(k, 0) for k in mapped_gain.keys()],
    }).sort_values("Gain", ascending=False)

    imp_df.to_csv(TABLES_DIR / "xgboost_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=imp_df.head(10), x="Gain", y="Feature", palette="viridis", ax=ax)
    ax.set_title("XGBoost Native Feature Importance (Gain Metric)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gain (Relative improvement in objective per split)", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "xgboost_feature_importance_gain.png", dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------
    # 5. SHAP Analysis on Representative Sample (2,000 test observations)
    # -------------------------------------------------------------
    print("[Diagnostics] Running SHAP TreeExplainer on 2,000 test observations (fixed seed=42)...")
    shap_sample = df_test.sample(n=min(2000, len(df_test)), random_state=42)
    X_shap = shap_sample[feature_names].values

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)
    shap_values.feature_names = feature_names

    # SHAP Summary Beeswarm Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Summary Plot: Contribution to Predicted DI_fouling", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    # SHAP Summary Bar Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title("Mean |SHAP Value|: Top Contributors to Predicted DI_fouling", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    # SHAP Dependence Plots for Top 4 Features
    top_4_features = imp_df["Feature"].head(4).tolist()
    print(f"[Diagnostics] Generating SHAP Dependence Plots for {top_4_features}...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_4_features):
        feat_idx = feature_names.index(feat)
        shap_feat_vals = shap_values.values[:, feat_idx]
        feat_raw_vals = X_shap[:, feat_idx]

        ax = axes[i]
        scatter = ax.scatter(feat_raw_vals, shap_feat_vals, alpha=0.5, c=X_shap[:, feature_names.index("T_in_measured_C")], cmap="coolwarm", s=15)
        ax.set_title(f"SHAP Dependence: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("SHAP Contribution to DI_fouling", fontsize=10)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("T_in_measured_C [°C]", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "shap_dependence_top_features.png", dpi=200)
    plt.close(fig)

    print("[Diagnostics] Complete! All figures saved to results/figures/ and tables to results/tables/")


if __name__ == "__main__":
    run_diagnostics()
