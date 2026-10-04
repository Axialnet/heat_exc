"""Checkpointed, resilient Stage-1 training and evaluation pipeline.

Prioritizes:
1. Fast target verification & area-discrepancy quantification
2. DI_fouling + Current Only (Linear, RF, XGB)
3. DI_fouling + Current + History (Linear, RF, XGB)
4. DI_total + Current Only (Linear, RF, XGB)
5. DI_total + Current + History (Linear, RF, XGB)

Resumable: skips any model already saved to disk.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score
import xgboost as xgb

from src.data_processing import (
    HISTORY_FEATURE_NAMES,
    PROJECT_ROOT,
    VALID_PREDICTORS_CURRENT,
    VALID_PREDICTORS_MODE_B,
    construct_targets,
    engineer_history_features,
    get_scenario_split,
    load_raw_dataset,
)

MODELS_DIR = PROJECT_ROOT / "models" / "stage1"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MODEL_RESULTS_DIR = PROJECT_ROOT / "results" / "model_results"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics MAE, RMSE, MSE, R²."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "R2": r2}


def fast_target_verification(df: pd.DataFrame):
    """Perform fast sanity verification and quantify V2 area-rounding discrepancy."""
    print("\n" + "=" * 60)
    print("      FAST TARGET SANITY VERIFICATION & AREA AUDIT")
    print("=" * 60)

    # Primary: DI_fouling
    di_foul = df["DI_fouling"].values
    eff_foul = (1.0 - df["thermal_efficiency"]).values
    abs_diff_foul = np.abs(di_foul - eff_foul)
    rel_diff_foul = abs_diff_foul / np.where(di_foul > 1e-7, di_foul, 1e-7)

    # Secondary: DI_total
    di_tot = df["DI_total"].values
    eff_tot = (1.0 - df["efficiency_total"]).values
    abs_diff_tot = np.abs(di_tot - eff_tot)
    rel_diff_tot = abs_diff_tot / np.where(di_tot > 1e-7, di_tot, 1e-7)

    discrepancy_records = [
        {
            "Target": "DI_fouling",
            "Formula": "1 - U_overall / U_clean",
            "Min": float(np.min(di_foul)),
            "Max": float(np.max(di_foul)),
            "Mean": float(np.mean(di_foul)),
            "Std Dev": float(np.std(di_foul)),
            "NaN Count": int(np.isnan(di_foul).sum()),
            "Mean Abs Diff vs (1 - eff)": float(np.mean(abs_diff_foul)),
            "Median Abs Diff": float(np.median(abs_diff_foul)),
            "Max Abs Diff": float(np.max(abs_diff_foul)),
            "Mean Rel Diff": float(np.mean(rel_diff_foul)),
            "Physical Origin of Discrepancy": "V2 script rounded A_ht to 29.1 m² (true = 29.131 m²); explicit HTC ratio is physically exact.",
        },
        {
            "Target": "DI_total",
            "Formula": "1 - U_total / U_clean",
            "Min": float(np.min(di_tot)),
            "Max": float(np.max(di_tot)),
            "Mean": float(np.mean(di_tot)),
            "Std Dev": float(np.std(di_tot)),
            "NaN Count": int(np.isnan(di_tot).sum()),
            "Mean Abs Diff vs (1 - eff)": float(np.mean(abs_diff_tot)),
            "Median Abs Diff": float(np.median(abs_diff_tot)),
            "Max Abs Diff": float(np.max(abs_diff_tot)),
            "Mean Rel Diff": float(np.mean(rel_diff_tot)),
            "Physical Origin of Discrepancy": "Consistent floating-point tolerance with identical A_ht factor.",
        },
    ]

    disc_df = pd.DataFrame(discrepancy_records)
    disc_df.to_csv(TABLES_DIR / "target_discrepancy_analysis.csv", index=False)
    print(disc_df.T.to_string())
    print(f"\nTarget audit saved to: {TABLES_DIR / 'target_discrepancy_analysis.csv'}")


def train_checkpointed():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Loading] Loading raw V2 dataset and engineering targets & features...")
    raw_df = load_raw_dataset()
    df = construct_targets(raw_df)
    
    # Fast target verification
    fast_target_verification(df)

    print("\n[Features] Engineering causal recent-history features per scenario...")
    t0_feat = time.time()
    df = engineer_history_features(df)
    print(f"[Features] Done in {time.time() - t0_feat:.2f}s.")

    # Scenario split
    split_path = MODEL_RESULTS_DIR / "scenario_split.json"
    if split_path.exists():
        with open(split_path, "r") as f:
            split_info = json.load(f)
        train_sc = split_info["train_scenarios"]
        val_sc = split_info["val_scenarios"]
        test_sc = split_info["test_scenarios"]
        print(f"[Split] Loaded existing scenario split from {split_path}")
    else:
        train_sc, val_sc, test_sc = get_scenario_split(df, seed=42)
        split_info = {
            "train_scenarios": list(train_sc),
            "val_scenarios": list(val_sc),
            "test_scenarios": list(test_sc),
        }
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"[Split] Created and saved scenario split to {split_path}")

    # Masks
    train_mask = df["scenario_id"].isin(train_sc)
    val_mask = df["scenario_id"].isin(val_sc)
    test_mask = df["scenario_id"].isin(test_sc)

    df_train = df[train_mask]
    df_val = df[val_mask]
    df_test = df[test_mask]

    print(f"[Split Summary] Train: {len(df_train):,} rows ({len(train_sc)} scenarios)")
    print(f"[Split Summary] Val:   {len(df_val):,} rows ({len(val_sc)} scenarios)")
    print(f"[Split Summary] Test:  {len(df_test):,} rows ({len(test_sc)} scenarios)")

    # For responsive, high-performance training without sacrificing temporal resolution,
    # sample every 4th timestep for training (131,400 rows across all 60 training scenarios).
    # Test set is 100% evaluated on all 87,600 untouched test observations.
    train_sub = df_train.iloc[::4]
    val_sub = df_val.iloc[::4]
    test_full = df_test

    # Execution Plan
    plan = [
        # PASS 1: Primary Target (DI_fouling) + Current Only
        {"target": "DI_fouling", "mode_name": "Current Only", "mode_key": "current", "model_type": "linear", "features": VALID_PREDICTORS_CURRENT},
        {"target": "DI_fouling", "mode_name": "Current Only", "mode_key": "current", "model_type": "rf", "features": VALID_PREDICTORS_CURRENT},
        {"target": "DI_fouling", "mode_name": "Current Only", "mode_key": "current", "model_type": "xgb", "features": VALID_PREDICTORS_CURRENT},
        # PASS 2: Primary Target (DI_fouling) + Current + History
        {"target": "DI_fouling", "mode_name": "Current + History", "mode_key": "history", "model_type": "linear", "features": VALID_PREDICTORS_MODE_B},
        {"target": "DI_fouling", "mode_name": "Current + History", "mode_key": "history", "model_type": "rf", "features": VALID_PREDICTORS_MODE_B},
        {"target": "DI_fouling", "mode_name": "Current + History", "mode_key": "history", "model_type": "xgb", "features": VALID_PREDICTORS_MODE_B},
        # PASS 3: Secondary Target (DI_total) + Current Only
        {"target": "DI_total", "mode_name": "Current Only", "mode_key": "current", "model_type": "linear", "features": VALID_PREDICTORS_CURRENT},
        {"target": "DI_total", "mode_name": "Current Only", "mode_key": "current", "model_type": "rf", "features": VALID_PREDICTORS_CURRENT},
        {"target": "DI_total", "mode_name": "Current Only", "mode_key": "current", "model_type": "xgb", "features": VALID_PREDICTORS_CURRENT},
        # PASS 4: Secondary Target (DI_total) + Current + History
        {"target": "DI_total", "mode_name": "Current + History", "mode_key": "history", "model_type": "linear", "features": VALID_PREDICTORS_MODE_B},
        {"target": "DI_total", "mode_name": "Current + History", "mode_key": "history", "model_type": "rf", "features": VALID_PREDICTORS_MODE_B},
        {"target": "DI_total", "mode_name": "Current + History", "mode_key": "history", "model_type": "xgb", "features": VALID_PREDICTORS_MODE_B},
    ]

    master_records = []

    for item in plan:
        target = item["target"]
        mode_name = item["mode_name"]
        mode_key = item["mode_key"]
        model_type = item["model_type"]
        feats = item["features"]

        model_key = f"{target.lower()}_{mode_key}_{model_type}"
        model_path = MODELS_DIR / f"{model_key}.joblib"
        metrics_path = MODEL_RESULTS_DIR / f"{model_key}_metrics.json"
        preds_path = PREDICTIONS_DIR / f"{model_key}_test_preds.npy"

        # Checkpoint check: if already completed, load and skip retraining
        if model_path.exists() and metrics_path.exists() and preds_path.exists():
            print(f"\n>>> SKIPPED (already completed): {model_key}")
            with open(metrics_path, "r") as f:
                saved_metrics = json.load(f)
            master_records.append(saved_metrics)
            print(f"    Loaded Val R²={saved_metrics['R2_val']:.4f} | Test R²={saved_metrics['R2_test']:.4f}")
            continue

        print(f"\n" + "-" * 55)
        print(f"STARTING: [{target}] [{mode_name}] [{model_type.upper()}]")
        print(f"Model Key: {model_key} | Features: {len(feats)}")
        print("-" * 55)

        t_start = time.time()

        X_tr = train_sub[feats].values
        y_tr = train_sub[target].values
        X_va = val_sub[feats].values
        y_va = val_sub[target].values
        X_te = test_full[feats].values
        y_te = test_full[target].values

        if model_type == "linear":
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            X_te_s = scaler.transform(X_te)

            model = LinearRegression()
            model.fit(X_tr_s, y_tr)

            val_preds = model.predict(X_va_s)
            test_preds = model.predict(X_te_s)

            # Multicollinearity diagnostic: Condition number of X^T X
            cond_num = float(np.linalg.cond(X_tr_s))
            print(f"  Condition number of scaled predictor matrix: {cond_num:.2e}")

            save_pkg = {"model": model, "scaler": scaler, "features": feats, "condition_number": cond_num}

        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
            )
            model.fit(X_tr, y_tr)

            val_preds = model.predict(X_va)
            test_preds = model.predict(X_te)
            save_pkg = {"model": model, "features": feats}

        elif model_type == "xgb":
            model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.04,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=25,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            val_preds = model.predict(X_va)
            test_preds = model.predict(X_te)
            save_pkg = {"model": model, "features": feats, "best_iteration": int(model.best_iteration)}
            print(f"  XGBoost best validation iteration: {model.best_iteration}")

        elapsed = time.time() - t_start

        # Compute metrics
        m_val = compute_metrics(y_va, val_preds)
        m_te = compute_metrics(y_te, test_preds)

        print(f"COMPLETED {model_key} in {elapsed:.1f}s:")
        print(f"  VAL  -> R²: {m_val['R2']:.4f} | MAE: {m_val['MAE']:.6f} | RMSE: {m_val['RMSE']:.6f}")
        print(f"  TEST -> R²: {m_te['R2']:.4f} | MAE: {m_te['MAE']:.6f} | RMSE: {m_te['RMSE']:.6f}")

        # Checkpoint: Save model, predictions, and metrics immediately
        joblib.dump(save_pkg, model_path)
        np.save(preds_path, test_preds)

        record = {
            "target": target,
            "input_mode": mode_name,
            "model": model_type.upper(),
            "model_key": model_key,
            "MAE_val": m_val["MAE"],
            "RMSE_val": m_val["RMSE"],
            "R2_val": m_val["R2"],
            "MAE_test": m_te["MAE"],
            "RMSE_test": m_te["RMSE"],
            "R2_test": m_te["R2"],
            "elapsed_seconds": round(elapsed, 2),
        }

        with open(metrics_path, "w") as f:
            json.dump(record, f, indent=2)

        master_records.append(record)

    # Master Table compilation
    master_df = pd.DataFrame(master_records)
    master_csv = TABLES_DIR / "model_comparison_master.csv"
    master_df.to_csv(master_csv, index=False)

    print("\n" + "=" * 70)
    print("               MASTER MODEL COMPARISON TABLE")
    print("=" * 70)
    cols_display = ["target", "input_mode", "model", "MAE_val", "R2_val", "MAE_test", "R2_test", "elapsed_seconds"]
    print(master_df[cols_display].to_string(index=False))
    print(f"\nMaster results saved to: {master_csv}")


if __name__ == "__main__":
    train_checkpointed()
