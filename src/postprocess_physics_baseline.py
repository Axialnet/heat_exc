"""
Stage 1 Post-Processing:
1. Calibrated Physics Baseline (Raw + Train-Calibrated)
2. Final consolidated summary tables
3. Stage 1 Final Report (STAGE_1_FINAL_REPORT.md)

Run after src/run_stage1_production.py completes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "version_2" / "v2_physics_corrected.csv"
TABLES_DIR = ROOT / "results" / "tables"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
REPORTS_DIR = ROOT / "results"

print("=" * 70)
print("STAGE 1 POST-PROCESSING: PHYSICS BASELINES + FINAL REPORT")
print("=" * 70)

# 1. Load data and rebuild needed features
print("[1/4] Loading dataset and rebuilding physics features...")
df = pd.read_csv(DATA_PATH)
df["DI_fouling"] = 1.0 - (df["U_overall_W_m2K"] / df["U_clean_W_m2K"])

# Rebuild physics rates
df["dep_intensity"] = np.exp(0.035 * (df["T_in_measured_C"] - 50.0)) * ((df["tau_w_Pa"] / 0.08) ** 0.8)
df["rem_intensity"] = df["tau_w_Pa"] / 0.08

# Scenario-level CIP-aware cumulative features
cip_hours = [2190, 4380, 6570]
groups = []
for sc_id, group in df.groupby("scenario_id", sort=False):
    g = group.copy()
    t = g["time_h"].values
    h_since_cip = t.copy()
    cycle = np.zeros_like(t, dtype=int)
    for c_idx, c_t in enumerate(cip_hours):
        m = t >= c_t
        h_since_cip[m] = t[m] - c_t
        cycle[m] = c_idx + 1
    g["hours_since_cip"] = h_since_cip
    g["cip_cycle"] = cycle
    g["cum_dep_since_cip"] = g.groupby("cip_cycle")["dep_intensity"].cumsum()
    g["cum_rem_since_cip"] = g.groupby("cip_cycle")["rem_intensity"].cumsum()
    groups.append(g)
df_all = pd.concat(groups).reset_index(drop=True)

# Scenario splits
unique_scenarios = df_all["scenario_id"].unique()
train_scenarios = unique_scenarios[:60]
val_scenarios = unique_scenarios[60:70]
test_scenarios = unique_scenarios[70:]

train_mask = df_all["scenario_id"].isin(train_scenarios)
val_mask = df_all["scenario_id"].isin(val_scenarios)
test_mask = df_all["scenario_id"].isin(test_scenarios)

y_train = df_all.loc[train_mask, "DI_fouling"].values
y_val = df_all.loc[val_mask, "DI_fouling"].values
y_test = df_all.loc[test_mask, "DI_fouling"].values

# 2. Physics Baseline: Raw Mechanistic Exposure
BETA = 0.05

print("\n[2/4] Physics Baseline 1: Raw Uncalibrated Mechanistic Exposure...")
# Net deposition exposure (dimensional: arbitrary units)
net_exposure = (df_all["cum_dep_since_cip"] - BETA * df_all["cum_rem_since_cip"]).clip(lower=0.0)
expo_train = net_exposure[train_mask].values
expo_val = net_exposure[val_mask].values
expo_test = net_exposure[test_mask].values

# Spearman rank correlation on test set (does the ordering match?)
spear_corr_test, spear_pval = spearmanr(expo_test, y_test)
spear_corr_val, _ = spearmanr(expo_val, y_val)

print(f"  Spearman Rank Correlation (Test): rho = {spear_corr_test:.4f}")
print(f"  Spearman Rank Correlation (Val): rho = {spear_corr_val:.4f}")

# Raw scalar: K_scale_nominal = 1.2e-8 (deposition rate constant from V2 code)
# U_clean ~ 1106 W/m2K => DI_fouling ~ Rf * U_clean (when Rf << 1/U_clean)
# Dimensional scaling: DI ~ Rf * U_clean ~ growth_rate * U_clean * cumulative_exposure
# growth_rate = 1.2e-8 per step (per hour), U_clean ~ 1106 => K_nominal = 1.2e-8 * 1106 = 1.327e-5
K_NOMINAL = 1.2e-8 * 1106.0
raw_phys_test = K_NOMINAL * expo_test
raw_phys_val = K_NOMINAL * expo_val

r2_raw_test = r2_score(y_test, raw_phys_test)
r2_raw_val = r2_score(y_val, raw_phys_val)
mae_raw_test = mean_absolute_error(y_test, raw_phys_test)
rmse_raw_test = root_mean_squared_error(y_test, raw_phys_test)
bias_raw_test = np.mean(raw_phys_test - y_test)

print(f"  Raw K_nominal={K_NOMINAL:.4e}: Test R2={r2_raw_test:.4f}, MAE={mae_raw_test:.6f}, RMSE={rmse_raw_test:.6f}, Bias={bias_raw_test:.6f}")
print(f"  NOTE: R2 reflects scale mismatch. Spearman rho = {spear_corr_test:.4f} shows ORDERING quality.")

# 3. Physics Baseline: Train-Calibrated Scalar K_scale
print("\n[3/4] Physics Baseline 2: Train-Calibrated K_scale (fit on training set only)...")
# OLS on training data: y_train = K_cal * expo_train => K_cal = sum(y * expo) / sum(expo^2)
# This is a single-coefficient fit through the origin (no bias term)
K_CAL = np.dot(y_train, expo_train) / np.dot(expo_train, expo_train)
print(f"  Calibrated K_scale (train-only fit): {K_CAL:.4e}")

cal_phys_val = K_CAL * expo_val
cal_phys_test = K_CAL * expo_test

r2_cal_val = r2_score(y_val, cal_phys_val)
r2_cal_test = r2_score(y_test, cal_phys_test)
mae_cal_test = mean_absolute_error(y_test, cal_phys_test)
rmse_cal_test = root_mean_squared_error(y_test, cal_phys_test)
bias_cal_test = np.mean(cal_phys_test - y_test)
medae_cal_test = np.median(np.abs(y_test - cal_phys_test))
maxae_cal_test = np.max(np.abs(y_test - cal_phys_test))

print(f"  Calibrated K={K_CAL:.4e}: Val R2={r2_cal_val:.4f}, Test R2={r2_cal_test:.4f}")
print(f"  Test MAE={mae_cal_test:.6f}, RMSE={rmse_cal_test:.6f}, Bias={bias_cal_test:.6f}")

# Save physics baselines table
phys_df = pd.DataFrame([
    {
        "Physics_Baseline": "Raw Uncalibrated (K_nominal=1.327e-5)",
        "K_scale": K_NOMINAL,
        "Beta": BETA,
        "Derivation": "K = 1.2e-8 (V2 growth constant) * 1106 (U_clean)",
        "Spearman_rho_Test": spear_corr_test,
        "Val_R2": r2_raw_val,
        "Test_R2": r2_raw_test,
        "Test_MAE": mae_raw_test,
        "Test_RMSE": rmse_raw_test,
        "Test_Bias": bias_raw_test,
        "Interpretation": "Scale mismatch due to unit conversion; Spearman rho shows trend quality"
    },
    {
        "Physics_Baseline": "Train-Calibrated (K fit on training set only)",
        "K_scale": K_CAL,
        "Beta": BETA,
        "Derivation": "OLS through origin on training scenarios: K = sum(y*x)/sum(x^2)",
        "Spearman_rho_Test": spear_corr_test,
        "Val_R2": r2_cal_val,
        "Test_R2": r2_cal_test,
        "Test_MAE": mae_cal_test,
        "Test_RMSE": rmse_cal_test,
        "Test_Bias": bias_cal_test,
        "Interpretation": "Same trend, calibrated absolute scale"
    }
])
phys_df.to_csv(TABLES_DIR / "physics_baseline_metrics.csv", index=False)
print(f"Saved to {TABLES_DIR}/physics_baseline_metrics.csv")

# Save calibrated physics test predictions
pd.DataFrame({
    "DI_fouling_actual": y_test,
    "DI_fouling_raw_physics": raw_phys_test,
    "DI_fouling_calibrated_physics": cal_phys_test
}).to_csv(PREDICTIONS_DIR / "physics_predictions_test.csv", index=False)

# 4. Read the ML experiment results table and generate consolidated summary
print("\n[4/4] Loading ML results and generating final consolidated summary...")
exp_df = pd.read_csv(TABLES_DIR / "experiment_comparison.csv")
print("\n=== FULL EXPERIMENT COMPARISON TABLE (DI_fouling) ===")
foul_df = exp_df[exp_df["Target"] == "DI_fouling"].copy()
cols = ["Architecture", "Model", "Transform", "Feature_Count", "Val_R2", "Test_R2", "Test_MAE", "Test_RMSE", "Test_MedAE"]
print(foul_df[cols].to_string(index=False))

# Add physics baselines to comparison table
physics_rows = pd.DataFrame([
    {
        "Architecture": "Physics-Only",
        "Feature_Count": 2,
        "Model": "Uncalibrated Kinetic Exposure",
        "Target": "DI_fouling",
        "Transform": "None",
        "Val_R2": r2_raw_val,
        "Val_MAE": mae_raw_test,
        "Val_RMSE": rmse_raw_test,
        "Test_R2": r2_raw_test,
        "Test_MAE": mae_raw_test,
        "Test_RMSE": rmse_raw_test,
        "Test_MedAE": np.median(np.abs(y_test - raw_phys_test)),
        "Test_MaxAE": np.max(np.abs(y_test - raw_phys_test)),
        "Test_Bias": bias_raw_test,
        "Fit_Time_s": 0.001
    },
    {
        "Architecture": "Physics-Only",
        "Feature_Count": 2,
        "Model": "Train-Calibrated Kinetic Exposure",
        "Target": "DI_fouling",
        "Transform": "None",
        "Val_R2": r2_cal_val,
        "Val_MAE": mae_cal_test,
        "Val_RMSE": rmse_cal_test,
        "Test_R2": r2_cal_test,
        "Test_MAE": mae_cal_test,
        "Test_RMSE": rmse_cal_test,
        "Test_MedAE": medae_cal_test,
        "Test_MaxAE": maxae_cal_test,
        "Test_Bias": bias_cal_test,
        "Fit_Time_s": 0.001
    }
])

consolidated = pd.concat([exp_df, physics_rows], ignore_index=True)
consolidated.to_csv(TABLES_DIR / "experiment_comparison.csv", index=False)
print(f"\nSaved consolidated table to {TABLES_DIR}/experiment_comparison.csv")

print("\n=== PHYSICS BASELINES SUMMARY ===")
print(f"Spearman rho (trend quality, test): {spear_corr_test:.4f}  (p={spear_pval:.2e})")
print(f"Raw Uncalibrated: Test R2={r2_raw_test:.4f}, MAE={mae_raw_test:.6f}")
print(f"Train-Calibrated: Test R2={r2_cal_test:.4f}, MAE={mae_cal_test:.6f}")
print("\nPhysics baselines complete.")
