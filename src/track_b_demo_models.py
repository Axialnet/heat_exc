"""
Track B — Showcase / Demonstration Model Pipeline.
Intentionally evaluates controlled target-proxy leakage features for demonstration:
  - DEMO-A: Lagged and rolling target-derived variables (Rf_lag1, Rf_roll24, DI_lag1, DI_roll24)
  - DEMO-B: Direct simulator-derived performance variables (Rf_m2K_W, U_overall_W_m2K, Q_W, thermal_efficiency, etc.)
  - DEMO-C: Direct engineered algebraic proxy (DI_proxy = 1 - U_overall/U_clean)

Evaluates on the exact same scenario split (60 Train, 10 Val, 10 Test).
Labels all outputs explicitly as 'Demonstration / Target-Proxy Leakage'.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import joblib

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "version_2" / "v2_physics_corrected.csv"
RESULTS_DIR = ROOT / "results"
DEMO_DIR = RESULTS_DIR / "demo_model"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [DEMO_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("TRACK B — SHOWCASE / DEMONSTRATION MODEL (TARGET-PROXY LEAKAGE)")
print("=" * 70)

# 1. Load Data
df = pd.read_csv(DATA_PATH)
df["DI_fouling"] = 1.0 - (df["U_overall_W_m2K"] / df["U_clean_W_m2K"])
df["DI_total"] = 1.0 - (df["U_total_W_m2K"] / df["U_clean_W_m2K"])

# 2. Scenario-Level Split (Identical to Track A)
unique_scenarios = df["scenario_id"].unique()
train_scenarios = unique_scenarios[:60]
val_scenarios = unique_scenarios[60:70]
test_scenarios = unique_scenarios[70:]

train_mask = df["scenario_id"].isin(train_scenarios)
val_mask = df["scenario_id"].isin(val_scenarios)
test_mask = df["scenario_id"].isin(test_scenarios)

y_train = df.loc[train_mask, "DI_fouling"].values
y_val = df.loc[val_mask, "DI_fouling"].values
y_test = df.loc[test_mask, "DI_fouling"].values

# 3. Engineer Demonstration Features (Controlled Leakage)
print("\n[1/4] Constructing Demonstration Leakage Feature Sets...")

# Base measured variables
base_feats = [
    "T_in_measured_C", "m_dot_measured_kg_s", "rho_kg_m3", "mu_Pa_s",
    "Re", "u_m_s", "tau_w_Pa", "dP_Pa"
]

# DEMO-A: Lagged & rolling target-derived variables (simulates autoregressive state availability)
demo_a_groups = []
for sc_id, group in df.groupby("scenario_id", sort=False):
    g = group.copy()
    g["Rf_lag1"] = g["Rf_m2K_W"].shift(1).fillna(0.0)
    g["Rf_roll24"] = g["Rf_m2K_W"].rolling(24, min_periods=1).mean()
    g["DI_fouling_lag1"] = g["DI_fouling"].shift(1).fillna(0.0)
    g["DI_fouling_roll24"] = g["DI_fouling"].rolling(24, min_periods=1).mean()
    demo_a_groups.append(g)
df_demo = pd.concat(demo_a_groups).reset_index(drop=True)

feats_demo_a = base_feats + ["Rf_lag1", "Rf_roll24", "DI_fouling_lag1", "DI_fouling_roll24"]

# DEMO-B: Direct simulator-derived state variables
feats_demo_b = base_feats + [
    "Rf_m2K_W", "U_overall_W_m2K", "U_clean_W_m2K", "Q_W", "Q_clean_W",
    "thermal_efficiency", "fouling_factor_TEMA"
]

# Convert categorical if any
df_demo["fouling_factor_TEMA_num"] = pd.to_numeric(df_demo["fouling_factor_TEMA"], errors="coerce").fillna(0)
feats_demo_b = [f if f != "fouling_factor_TEMA" else "fouling_factor_TEMA_num" for f in feats_demo_b]

# DEMO-C: Direct algebraic proxy
df_demo["DI_proxy"] = 1.0 - (df_demo["U_overall_W_m2K"] / df_demo["U_clean_W_m2K"])
feats_demo_c = base_feats + ["DI_proxy"]

DEMO_CONFIGS = {
    "DEMO-A (Lagged/Rolling Target State)": feats_demo_a,
    "DEMO-B (Direct Simulator Performance Variables)": feats_demo_b,
    "DEMO-C (Direct Engineered Proxy)": feats_demo_c
}

demo_results = []
demo_preds_test = {}

# 4. Train Demonstration Models
print("\n[2/4] Training Demonstration Models (XGBoost & Ridge)...")
for demo_name, feat_cols in DEMO_CONFIGS.items():
    print(f"\n--- {demo_name} (Feature Count: {len(feat_cols)}) ---")
    X_train = df_demo.loc[train_mask, feat_cols].values
    X_val = df_demo.loc[val_mask, feat_cols].values
    X_test = df_demo.loc[test_mask, feat_cols].values
    
    # Train XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=4,
        early_stopping_rounds=25,
        eval_metric="rmse"
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    p_train = model_xgb.predict(X_train)
    p_val = model_xgb.predict(X_val)
    p_test = model_xgb.predict(X_test)
    
    r2_tr = r2_score(y_train, p_train)
    r2_v = r2_score(y_val, p_val)
    r2_t = r2_score(y_test, p_test)
    mae_t = mean_absolute_error(y_test, p_test)
    rmse_t = root_mean_squared_error(y_test, p_test)
    
    print(f"  XGBoost -> Train R2: {r2_tr:.4f} | Val R2: {r2_v:.4f} | Test R2: {r2_t:.4f} | Test MAE: {mae_t:.2e}")
    
    key = demo_name.split(" ")[0]
    demo_preds_test[f"{key}_XGB"] = p_test
    
    demo_results.append({
        "Model_Track": "Demonstration / Target-Proxy Leakage",
        "Demonstration_Level": demo_name,
        "Algorithm": "XGBoost",
        "Feature_Count": len(feat_cols),
        "Train_R2": r2_tr,
        "Val_R2": r2_v,
        "Test_R2": r2_t,
        "Test_MAE": mae_t,
        "Test_RMSE": rmse_t,
        "Leakage_Mechanism": (
            "Autoregressive state leakage (lag-1 target & rolling Rf)" if "DEMO-A" in demo_name
            else "Direct simulator performance variables (Rf, U_overall, Q)" if "DEMO-B" in demo_name
            else "Direct algebraic identity (1 - U_overall / U_clean)"
        )
    })
    
    # Save model checkpoint
    joblib.dump(model_xgb, DEMO_DIR / f"model_{key.lower()}_xgb.joblib")

# Add Track A Production baseline for transparent comparison
track_a_rec = {
    "Model_Track": "Production / Leakage-Free",
    "Demonstration_Level": "TRACK A: CIP-Aware History (Production Standard)",
    "Algorithm": "XGBoost",
    "Feature_Count": 16,
    "Train_R2": 0.4281,
    "Val_R2": 0.2686,
    "Test_R2": 0.2457,
    "Test_MAE": 0.000124,
    "Test_RMSE": 0.000245,
    "Leakage_Mechanism": "None (Strictly causal operational history & maintenance logs)"
}
demo_results.insert(0, track_a_rec)

demo_df = pd.DataFrame(demo_results)
demo_df.to_csv(TABLES_DIR / "track_a_vs_track_b_comparison.csv", index=False)
demo_df.to_csv(DEMO_DIR / "demo_model_comparison.csv", index=False)

# Save demo test predictions
pred_export = pd.DataFrame({
    "scenario_id": df_demo.loc[test_mask, "scenario_id"].values,
    "time_h": df_demo.loc[test_mask, "time_h"].values,
    "DI_actual": y_test,
    "DI_pred_track_a_production": pd.read_csv(RESULTS_DIR / "predictions" / "test_predictions_best_model.csv")["DI_pred"].values,
    "DI_pred_demo_a_lagged": demo_preds_test["DEMO-A_XGB"],
    "DI_pred_demo_b_sim_vars": demo_preds_test["DEMO-B_XGB"],
    "DI_pred_demo_c_proxy": demo_preds_test["DEMO-C_XGB"]
})
pred_export.to_csv(DEMO_DIR / "demonstration_predictions_test.csv", index=False)

print("\n[3/4] Saved comparison tables and predictions.")

# 5. Publication Comparison Figures
print("\n[4/4] Generating Track A vs Track B Comparison Visualizations...")

# Fig 1: Track A vs Track B Comparison Bar Chart
fig, ax = plt.subplots(figsize=(11, 5.5))
colors = ["#2b5c8f", "#e67e22", "#d35400", "#c0392b"]
bars = ax.bar([r["Demonstration_Level"].split(" (")[0] for r in demo_results], 
              [r["Test_R2"] for r in demo_results], 
              color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, max(yval + 0.02, 0.05), f"R² = {yval:.4f}", 
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Test R²", fontsize=11, fontweight="bold")
ax.set_title("Track A (Production Leakage-Free) vs Track B (Showcase Demo Leakage)", fontsize=13, fontweight="bold")
ax.set_ylim(-0.05, 1.15)
ax.axhline(0, color="gray", linewidth=0.8)
ax.grid(axis="y", linestyle="--", alpha=0.6)
plt.xticks(rotation=10, ha="right", fontsize=9, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_a_vs_track_b_comparison.png", dpi=200)
fig.savefig(DEMO_DIR / "track_a_vs_track_b_comparison.png", dpi=200)
plt.close(fig)

# Fig 2: Actual vs Predicted across Demo Models
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sample_idx = np.random.choice(len(y_test), size=min(8000, len(y_test)), replace=False)

demo_plots = [
    ("DEMO-A (Lagged State Leakage)", demo_preds_test["DEMO-A_XGB"], axes[0], "#e67e22"),
    ("DEMO-B (Simulator Variables)", demo_preds_test["DEMO-B_XGB"], axes[1], "#d35400"),
    ("DEMO-C (Direct Algebraic Proxy)", demo_preds_test["DEMO-C_XGB"], axes[2], "#c0392b")
]

for title, preds, ax_sub, col in demo_plots:
    r2_val = r2_score(y_test, preds)
    ax_sub.scatter(y_test[sample_idx], preds[sample_idx], alpha=0.25, s=10, color=col)
    mv = max(y_test.max(), preds.max())
    ax_sub.plot([0, mv], [0, mv], "k--", linewidth=1.5, label="1:1 Reference")
    ax_sub.set_title(f"{title}\nTest R² = {r2_val:.4f}", fontsize=11, fontweight="bold")
    ax_sub.set_xlabel("Actual DI_fouling", fontsize=10)
    ax_sub.set_ylabel("Predicted DI_fouling", fontsize=10)
    ax_sub.legend(loc="upper left")
    ax_sub.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "demo_model_actual_vs_predicted.png", dpi=200)
fig.savefig(DEMO_DIR / "demo_model_actual_vs_predicted.png", dpi=200)
plt.close(fig)

print("\nTRACK B PIPELINE COMPLETED SUCCESSFULLY.")
print(demo_df[["Model_Track", "Demonstration_Level", "Test_R2", "Test_MAE"]].to_string(index=False))
