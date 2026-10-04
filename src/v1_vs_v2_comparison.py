"""Rigorous empirical comparison of Version 1 vs Version 2 valid predictors.

Evaluates XGBoost on:
1. Version 1 valid predictors (constant fluid properties, uncoupled hydraulics)
2. Version 2 valid predictors (temperature-coupled hydraulics: rho(T), mu(T), coupled Re, u, tau_w, dP)

Using the exact same target definition (DI_fouling) and scenario split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb

from src.data_processing import (
    PROJECT_ROOT,
    VALID_PREDICTORS_CURRENT,
    VALID_PREDICTORS_MODE_B,
    construct_targets,
    engineer_history_features,
    load_raw_dataset,
)

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MODEL_RESULTS_DIR = PROJECT_ROOT / "results" / "model_results"


def run_v1_vs_v2_comparison():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[V1 vs V2 Comparison] Loading split info...")
    split_path = MODEL_RESULTS_DIR / "scenario_split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Scenario split info not found at {split_path}. Run train_models.py first.")

    with open(split_path, "r") as f:
        split_data = json.load(f)

    train_sc = split_data["train_scenarios"]
    val_sc = split_data["val_scenarios"]
    test_sc = split_data["test_scenarios"]

    # Load V2
    print("[V1 vs V2 Comparison] Preparing V2 data...")
    df_v2 = load_raw_dataset(PROJECT_ROOT / "data" / "version_2" / "v2_physics_corrected.csv")
    df_v2 = construct_targets(df_v2)
    df_v2 = engineer_history_features(df_v2)

    # Load V1
    print("[V1 vs V2 Comparison] Preparing V1 data...")
    v1_csv = PROJECT_ROOT / "data" / "version_1" / "shell_tube_fouling_final.csv"
    df_v1 = pd.read_csv(v1_csv).sort_values(["scenario_id", "time_h"]).reset_index(drop=True)
    df_v1 = construct_targets(df_v1)
    df_v1 = engineer_history_features(df_v1)

    # Predictors in V1 (V1 does not have rho_kg_m3 and mu_Pa_s columns; fluid properties were static constants)
    v1_current_predictors = [c for c in VALID_PREDICTORS_CURRENT if c in df_v1.columns]
    v2_current_predictors = VALID_PREDICTORS_CURRENT

    print(f"  V1 Current Predictors ({len(v1_current_predictors)}): {v1_current_predictors}")
    print(f"  V2 Current Predictors ({len(v2_current_predictors)}): {v2_current_predictors} (includes dynamic rho(T) and mu(T))")

    # Evaluate on Mode A (Current only) and Mode B (Current + History)
    experiments = [
        ("V1 (Uncoupled Hydraulics)", df_v1, v1_current_predictors, "Mode A: Current Only"),
        ("V2 (Physics-Coupled Hydraulics)", df_v2, v2_current_predictors, "Mode A: Current Only"),
        ("V1 (Uncoupled + History)", df_v1, v1_current_predictors + [f for f in df_v1.columns if "roll" in f or "diff" in f or "cum" in f], "Mode B: Current + History"),
        ("V2 (Physics-Coupled + History)", df_v2, VALID_PREDICTORS_MODE_B, "Mode B: Current + History"),
    ]

    results = []

    for name, df_exp, feat_cols, mode_desc in experiments:
        print(f"\nTraining XGBoost on {name}...")
        train_exp = df_exp[df_exp["scenario_id"].isin(train_sc)].iloc[::4]
        val_exp = df_exp[df_exp["scenario_id"].isin(val_sc)].iloc[::4]
        test_exp = df_exp[df_exp["scenario_id"].isin(test_sc)]

        X_tr = train_exp[feat_cols].values
        y_tr = train_exp["DI_fouling"].values
        X_va = val_exp[feat_cols].values
        y_val = val_exp["DI_fouling"].values
        X_te = test_exp[feat_cols].values
        y_te = test_exp["DI_fouling"].values

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=25,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], verbose=False)

        preds = model.predict(X_te)
        r2 = r2_score(y_te, preds)
        mae = mean_absolute_error(y_te, preds)
        rmse = root_mean_squared_error(y_te, preds)

        results.append({
            "Dataset Version": name,
            "Input Mode": mode_desc,
            "Feature Count": len(feat_cols),
            "Test R2": r2,
            "Test MAE": mae,
            "Test RMSE": rmse,
            "Best Iteration": model.best_iteration,
        })
        print(f"  -> Test R²: {r2:.4f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")

    comp_df = pd.DataFrame(results)
    out_path = TABLES_DIR / "v1_vs_v2_model_comparison.csv"
    comp_df.to_csv(out_path, index=False)
    print(f"\n[V1 vs V2 Comparison] Saved results to: {out_path}")
    print(comp_df.to_string())


if __name__ == "__main__":
    run_v1_vs_v2_comparison()
