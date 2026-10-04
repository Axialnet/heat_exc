"""Data processing, feature engineering, and scenario-aware data splitting for Stage 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "version_2" / "v2_physics_corrected.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# Valid Predictor Definitions
VALID_PREDICTORS_CURRENT = [
    "T_in_measured_C",
    "m_dot_measured_kg_s",
    "rho_kg_m3",
    "mu_Pa_s",
    "Re",
    "u_m_s",
    "tau_w_Pa",
    "dP_Pa",
]

HISTORY_FEATURE_NAMES = [
    "T_in_roll_mean_24h",
    "T_in_roll_std_24h",
    "T_in_roll_mean_168h",
    "m_dot_roll_mean_24h",
    "m_dot_roll_std_24h",
    "m_dot_roll_mean_168h",
    "dP_roll_mean_24h",
    "dP_roll_std_24h",
    "dP_roll_mean_168h",
    "T_in_diff_24h",
    "dP_diff_24h",
    "cum_thermal_throughput",
]

VALID_PREDICTORS_MODE_B = VALID_PREDICTORS_CURRENT + HISTORY_FEATURE_NAMES

EXCLUDED_COLUMNS_AUDIT = {
    "scenario_id": {"class": "3. IDENTIFIER / METADATA", "reason": "Categorical run ID; causes scenario trajectory memorization."},
    "time_h": {"class": "3. IDENTIFIER / METADATA", "reason": "Elapsed simulation clock; would cause fitting time trends rather than physical inference."},
    "T_true_C": {"class": "6. UNAVAILABLE AT DEPLOYMENT", "reason": "Unobservable simulation ground truth temperature."},
    "T_true_K": {"class": "6. UNAVAILABLE AT DEPLOYMENT", "reason": "Kelvin conversion of unobservable simulation ground truth."},
    "T_in_C": {"class": "6. UNAVAILABLE AT DEPLOYMENT", "reason": "Base cyclic load setpoint from V1, not a plant instrument reading."},
    "T_in_K": {"class": "6. UNAVAILABLE AT DEPLOYMENT", "reason": "Kelvin conversion of base cyclic load."},
    "m_dot_nominal_kg_s": {"class": "2. REFERENCE / CONSTANT", "reason": "Setpoint design flow; dynamic variations captured by measured flow."},
    "m_dot_true_kg_s": {"class": "6. UNAVAILABLE AT DEPLOYMENT", "reason": "Unobservable simulation ground truth flow rate."},
    "Rf_m2K_W": {"class": "5. LEAKAGE / TARGET-DERIVED", "reason": "Internal fouling state variable; direct target leakage."},
    "U_clean_W_m2K": {"class": "2. REFERENCE / CONSTANT", "reason": "Design clean heat transfer coefficient; used as denominator in DI target."},
    "U_overall_W_m2K": {"class": "5. LEAKAGE / TARGET-DERIVED", "reason": "Degraded HTC with fouling; used directly to construct DI_fouling."},
    "U_total_W_m2K": {"class": "5. LEAKAGE / TARGET-DERIVED", "reason": "Total degraded HTC; used directly to construct DI_total."},
    "Q_W": {"class": "5. LEAKAGE / TARGET-DERIVED", "reason": "Heat duty with fouling; directly proportional to U_overall."},
    "Q_clean_W": {"class": "2. REFERENCE / CONSTANT", "reason": "Clean heat duty baseline; static reference value."},
    "Q_total_W": {"class": "5. LEAKAGE / TARGET-DERIVED", "reason": "Total heat duty; directly proportional to U_total."},
    "thermal_efficiency": {"class": "5. LEAKAGE / TARGET-DERIVED", "reason": "Direct mathematical equivalent of 1 - DI_fouling."},
    "efficiency_total": {"class": "5. LEAKAGE / TARGET-DERIVED", "reason": "Direct mathematical equivalent of 1 - DI_total."},
    "fouling_factor_TEMA": {"class": "5. LEAKAGE / TARGET-DERIVED", "reason": "Binary classification label derived directly from Rf threshold."},
    "cip_event": {"class": "3. IDENTIFIER / METADATA", "reason": "Intervention schedule log, not an operational measurement."},
    "cip_effectiveness": {"class": "6. UNAVAILABLE / LEAKAGE", "reason": "Simulated cleaning effectiveness drawn randomly during CIP; post-hoc unobservable."},
    "R_wall_m2K_W": {"class": "6. UNAVAILABLE / LEAKAGE", "reason": "Wall corrosion resistance; internal simulation state."},
    "degradation_source": {"class": "5. LEAKAGE / METADATA", "reason": "Synthetic label derived from Rf and R_wall thresholds."},
}


def load_raw_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load raw V2 dataset without modifications."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(["scenario_id", "time_h"]).reset_index(drop=True)
    return df


def construct_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Construct physics-based condition indices DI_fouling and DI_total."""
    df = df.copy()
    # Primary Target: Cleanable thermal degradation from fouling
    df["DI_fouling"] = 1.0 - (df["U_overall_W_m2K"] / df["U_clean_W_m2K"])
    # Secondary Target: Total thermal degradation including corrosion
    df["DI_total"] = 1.0 - (df["U_total_W_m2K"] / df["U_clean_W_m2K"])
    return df


def engineer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer legitimate recent-history features derived ONLY from valid measured variables."""
    dfs = []
    for sc_id, group in df.groupby("scenario_id", sort=False):
        g = group.copy()
        t_in = g["T_in_measured_C"]
        m_dot = g["m_dot_measured_kg_s"]
        dp = g["dP_Pa"]

        # Rolling statistics (24h and 168h windows)
        g["T_in_roll_mean_24h"] = t_in.rolling(24, min_periods=1).mean()
        g["T_in_roll_std_24h"] = t_in.rolling(24, min_periods=1).std().fillna(0.0)
        g["T_in_roll_mean_168h"] = t_in.rolling(168, min_periods=1).mean()

        g["m_dot_roll_mean_24h"] = m_dot.rolling(24, min_periods=1).mean()
        g["m_dot_roll_std_24h"] = m_dot.rolling(24, min_periods=1).std().fillna(0.0)
        g["m_dot_roll_mean_168h"] = m_dot.rolling(168, min_periods=1).mean()

        g["dP_roll_mean_24h"] = dp.rolling(24, min_periods=1).mean()
        g["dP_roll_std_24h"] = dp.rolling(24, min_periods=1).std().fillna(0.0)
        g["dP_roll_mean_168h"] = dp.rolling(168, min_periods=1).mean()

        # Short-window changes
        g["T_in_diff_24h"] = t_in.diff(24).fillna(0.0)
        g["dP_diff_24h"] = dp.diff(24).fillna(0.0)

        # Cumulative operating thermal-hydraulic throughput
        g["cum_thermal_throughput"] = (m_dot * t_in).cumsum()

        dfs.append(g)

    return pd.concat(dfs, ignore_index=True)


def get_scenario_split(df: pd.DataFrame, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Perform a stratified grouped scenario split ensuring full operational envelope coverage."""
    scenarios = df["scenario_id"].unique()
    rng = np.random.default_rng(seed)
    
    # 80 scenarios total: 56 train (70%), 12 val (15%), 12 test (15%)
    # Stratify by flow rate to ensure balanced flow conditions in all splits
    train_scenarios = []
    val_scenarios = []
    test_scenarios = []
    
    # Extract unique nominal flows from scenario_id (e.g. Q3.0, Q4.0, Q5.0, Q6.0, Q7.0)
    flows = sorted(list({sc.split("_")[1] for sc in scenarios}))
    
    for flow in flows:
        sc_flow = [sc for sc in scenarios if sc.endswith(flow)]
        rng.shuffle(sc_flow)
        # For 16 temperatures in each flow: 11 train, 2 or 3 val, 2 or 3 test
        n_test = 2 if len(test_scenarios) + 2 <= 12 else 3
        n_val = 2 if len(val_scenarios) + 2 <= 12 else 3
        
        test_chunk = sc_flow[:n_test]
        val_chunk = sc_flow[n_test : n_test + n_val]
        train_chunk = sc_flow[n_test + n_val :]
        
        test_scenarios.extend(test_chunk)
        val_scenarios.extend(val_chunk)
        train_scenarios.extend(train_chunk)

    # Adjust exact counts to 56 / 12 / 12 if slight discrepancy
    all_selected = test_scenarios[:12]
    val_selected = val_scenarios[:12]
    remaining = [sc for sc in scenarios if sc not in all_selected and sc not in val_selected]
    
    return remaining, val_selected, all_selected


def save_feature_audit_table():
    """Save the complete 30-column feature & leakage audit table to results/tables/."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for col in VALID_PREDICTORS_CURRENT:
        records.append({
            "Column Name": col,
            "Classification": "1. VALID PREDICTOR",
            "Predictive Mode": "Model A & B",
            "Physical Rationale / Deployment Justification": "Measurable process or hydro-thermal variable available online."
        })
    for col in HISTORY_FEATURE_NAMES:
        records.append({
            "Column Name": col,
            "Classification": "1. VALID PREDICTOR (Engineered History)",
            "Predictive Mode": "Model B only",
            "Physical Rationale / Deployment Justification": "Rolling statistic / cumulative load derived strictly from valid measured variables."
        })
    for col, meta in EXCLUDED_COLUMNS_AUDIT.items():
        records.append({
            "Column Name": col,
            "Classification": meta["class"],
            "Predictive Mode": "EXCLUDED",
            "Physical Rationale / Deployment Justification": meta["reason"]
        })
    
    audit_df = pd.DataFrame(records)
    out_csv = TABLES_DIR / "feature_leakage_audit.csv"
    audit_df.to_csv(out_csv, index=False)
    print(f"Feature & Leakage audit saved to: {out_csv}")


if __name__ == "__main__":
    print("Executing data processing verification...")
    raw_df = load_raw_dataset()
    df_with_targets = construct_targets(raw_df)
    df_engineered = engineer_history_features(df_with_targets)
    train_sc, val_sc, test_sc = get_scenario_split(df_engineered)
    
    print(f"Total Rows: {len(df_engineered):,}")
    print(f"Train Scenarios: {len(train_sc)} ({len(train_sc)/80*100:.1f}%) -> {len(df_engineered[df_engineered['scenario_id'].isin(train_sc)]):,} rows")
    print(f"Val Scenarios: {len(val_sc)} ({len(val_sc)/80*100:.1f}%) -> {len(df_engineered[df_engineered['scenario_id'].isin(val_sc)]):,} rows")
    print(f"Test Scenarios: {len(test_sc)} ({len(test_sc)/80*100:.1f}%) -> {len(df_engineered[df_engineered['scenario_id'].isin(test_sc)]):,} rows")
    
    save_feature_audit_table()
