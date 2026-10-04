"""
Unified Predictive Maintenance Suite: Phases 8, 9, 10, 11 & Dashboard Generator.

PHASE 8: Performance Monitoring Module (6 Physical Indicators + 0-100 Health Score)
PHASE 9: Predictive Maintenance Risk Engine (Configurable Thresholds from Train Data Only)
PHASE 10: Event-Driven Maintenance Alert System (With Deduplication & Action Mapping)
PHASE 11: Cleaning Recommendation Engine (Persistence-Filtered Action Window)
DASHBOARD: Publication Figures and Interactive Standalone HTML Platform
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "version_2" / "v2_physics_corrected.csv"
CONFIG_PATH = ROOT / "config" / "thresholds.json"
RESULTS_DIR = ROOT / "results"

P8_DIR = RESULTS_DIR / "phase8_performance"
P9_DIR = RESULTS_DIR / "phase9_risk"
P10_DIR = RESULTS_DIR / "phase10_alerts"
P11_DIR = RESULTS_DIR / "phase11_cleaning"
FIG_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

for d in [P8_DIR, P9_DIR, P10_DIR, P11_DIR, FIG_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("PREDICTIVE MAINTENANCE PIPELINE: PHASES 8 - 11 & DASHBOARD")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. Load Data & Ground Truth / Predictions
# -----------------------------------------------------------------------------
print("\n[1/6] Loading data and aligning Stage-1 predictions...")
df = pd.read_csv(DATA_PATH)
df["DI_fouling"] = 1.0 - (df["U_overall_W_m2K"] / df["U_clean_W_m2K"])
df["DI_total"] = 1.0 - (df["U_total_W_m2K"] / df["U_clean_W_m2K"])

# Load test predictions from Stage 1 (Production Track A)
prod_test_preds = pd.read_csv(RESULTS_DIR / "predictions" / "test_predictions_best_model.csv")
demo_test_preds = pd.read_csv(RESULTS_DIR / "demo_model" / "demonstration_predictions_test.csv")

# Scenario Split
unique_scenarios = df["scenario_id"].unique()
train_scenarios = unique_scenarios[:60]
val_scenarios = unique_scenarios[60:70]
test_scenarios = unique_scenarios[70:]

train_mask = df["scenario_id"].isin(train_scenarios)
test_mask = df["scenario_id"].isin(test_scenarios)

# Merge predicted DI into test scenarios DataFrame
test_df = df[test_mask].copy().reset_index(drop=True)
test_df["DI_fouling_predicted_prod"] = prod_test_preds["DI_pred"].values
test_df["DI_fouling_predicted_demo"] = demo_test_preds["DI_pred_demo_a_lagged"].values

# -----------------------------------------------------------------------------
# 2. PHASE 8 — HEAT EXCHANGER PERFORMANCE MONITORING
# -----------------------------------------------------------------------------
print("\n[2/6] Executing Phase 8: Performance Monitoring Module...")

# 1. Fouling degradation: DI_fouling
test_df["DI_fouling_pct"] = test_df["DI_fouling"] * 100.0

# 2. Heat-transfer retention: U_retention = U_overall / U_clean
test_df["U_retention"] = test_df["U_overall_W_m2K"] / test_df["U_clean_W_m2K"]
test_df["U_retention_pct"] = test_df["U_retention"] * 100.0

# 3. Heat-duty retention: Q_retention = Q / Q_clean
test_df["Q_retention"] = test_df["Q_W"] / test_df["Q_clean_W"]
test_df["Q_retention_pct"] = test_df["Q_retention"] * 100.0

# 4. Thermal performance: thermal_efficiency
test_df["thermal_efficiency_pct"] = test_df["thermal_efficiency"] * 100.0

# 5. Hydraulic indicator: dP_Pa
# 6. Fouling resistance: Rf_m2K_W

# 7. Normalized Performance Score (0 to 100, 100 = healthy clean, 0 = severely degraded)
# Derived from physical retention factors and fouling penalty
# Scale: U_retention (100% clean -> 100), penalty scaled such that DI = 0.002 drops score by 50 pts
score_raw = (
    0.40 * test_df["U_retention_pct"] +
    0.30 * test_df["Q_retention_pct"] +
    0.30 * (100.0 - np.clip(test_df["DI_fouling"] / 0.002 * 100.0, 0.0, 100.0))
)
test_df["performance_score"] = np.clip(score_raw, 0.0, 100.0)

# Summary of performance indicators
p8_summary = pd.DataFrame({
    "Indicator": [
        "Fouling Degradation Index (DI_fouling)",
        "Heat-Transfer Retention (U/U_clean)",
        "Heat-Duty Retention (Q/Q_clean)",
        "Thermal Efficiency",
        "Hydraulic Pressure Drop (dP)",
        "Fouling Thermal Resistance (Rf)",
        "Composite Performance Score"
    ],
    "Mean": [
        test_df["DI_fouling"].mean(),
        test_df["U_retention"].mean(),
        test_df["Q_retention"].mean(),
        test_df["thermal_efficiency"].mean(),
        test_df["dP_Pa"].mean(),
        test_df["Rf_m2K_W"].mean(),
        test_df["performance_score"].mean()
    ],
    "Min": [
        test_df["DI_fouling"].min(),
        test_df["U_retention"].min(),
        test_df["Q_retention"].min(),
        test_df["thermal_efficiency"].min(),
        test_df["dP_Pa"].min(),
        test_df["Rf_m2K_W"].min(),
        test_df["performance_score"].min()
    ],
    "Max": [
        test_df["DI_fouling"].max(),
        test_df["U_retention"].max(),
        test_df["Q_retention"].max(),
        test_df["thermal_efficiency"].max(),
        test_df["dP_Pa"].max(),
        test_df["Rf_m2K_W"].max(),
        test_df["performance_score"].max()
    ],
    "Unit": ["dimensionless", "ratio", "ratio", "efficiency", "Pa", "m²K/W", "0-100 Score"]
})
p8_summary.to_csv(P8_DIR / "performance_indicators_summary.csv", index=False)
p8_summary.to_csv(TABLES_DIR / "performance_indicators_summary.csv", index=False)

test_df[[
    "scenario_id", "time_h", "DI_fouling", "U_retention", "Q_retention",
    "thermal_efficiency", "dP_Pa", "Rf_m2K_W", "performance_score"
]].to_csv(P8_DIR / "test_performance_timeseries.csv", index=False)

print("  Phase 8 completed. Performance indicators saved.")

# -----------------------------------------------------------------------------
# 3. PHASE 9 — PREDICTIVE MAINTENANCE / RISK ENGINE
# -----------------------------------------------------------------------------
print("\n[3/6] Executing Phase 9: Predictive Maintenance Risk Engine...")

# Derive thresholds strictly from TRAINING DATA ONLY
train_di = df.loc[train_mask, "DI_fouling"].values
warning_di_threshold = float(np.percentile(train_di, 75))   # 75th percentile of train
critical_di_threshold = float(np.percentile(train_di, 92))  # 92nd percentile of train

print(f"  Thresholds derived strictly from Training Data (60 scenarios):")
print(f"    WARNING Threshold  (75th pct) : DI = {warning_di_threshold:.6f}")
print(f"    CRITICAL Threshold (92nd pct) : DI = {critical_di_threshold:.6f}")

# Update config/thresholds.json
thresholds_config = {
    "version": "1.0",
    "description": "Configurable thresholds derived strictly from training data percentiles.",
    "fouling_thresholds": {
        "warning_percentile": 75,
        "critical_percentile": 92,
        "warning_di_fouling": warning_di_threshold,
        "critical_di_fouling": critical_di_threshold,
        "derivation_source": "Training set (60 scenarios, 525,600 rows)"
    },
    "risk_thresholds": {
        "normal_max": 25.0,
        "watch_max": 50.0,
        "warning_max": 75.0,
        "critical_max": 100.0
    },
    "alert_settings": {
        "deduplication_window_h": 24,
        "degradation_rate_alert_multiplier": 3.0,
        "persistence_steps_required": 3
    },
    "maintenance_horizon_h": 168
}
with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(thresholds_config, f, indent=2)

# Compute Risk Engine Metrics across test scenarios
risk_records = []
for sc_id, grp in test_df.groupby("scenario_id", sort=False):
    g = grp.copy().reset_index(drop=True)
    
    # 1. Degradation rate (24h rolling slope of predicted DI)
    g["di_pred_diff24"] = g["DI_fouling_predicted_prod"].diff(24).fillna(0.0) / 24.0
    g["degradation_rate_h"] = np.maximum(0.0, g["di_pred_diff24"])
    
    # Baseline normal degradation rate in this scenario
    median_rate = max(g["degradation_rate_h"].median(), 1.0e-8)
    
    # 2. Current and Predicted Fouling Risk Components (0-100 scale)
    # Current risk component: 0 at DI=0, 50 at Warning, 100 at Critical
    curr_risk = np.where(
        g["DI_fouling_predicted_prod"] <= warning_di_threshold,
        (g["DI_fouling_predicted_prod"] / warning_di_threshold) * 50.0,
        50.0 + ((g["DI_fouling_predicted_prod"] - warning_di_threshold) / (critical_di_threshold - warning_di_threshold)) * 50.0
    )
    g["current_fouling_risk"] = np.clip(curr_risk, 0.0, 100.0)
    
    # 3. Performance loss score (100 - performance_score)
    g["performance_loss_score"] = 100.0 - g["performance_score"]
    
    # 4. Threshold distance & Time-to-threshold estimation
    # Distance to critical threshold
    g["threshold_distance"] = critical_di_threshold - g["DI_fouling_predicted_prod"]
    
    # Time to threshold in hours: distance / degradation_rate (if rate > 0 and distance > 0)
    time_to_crit = np.where(
        g["threshold_distance"] <= 0,
        0.0,
        np.where(
            g["degradation_rate_h"] > 1e-9,
            np.clip(g["threshold_distance"] / g["degradation_rate_h"], 0.0, 1000.0),
            999.0
        )
    )
    g["time_to_threshold_h"] = time_to_crit
    
    # 5. Final Maintenance Risk Score (0-100)
    # Blended from current fouling risk, rate acceleration, and performance loss
    rate_factor = np.clip(g["degradation_rate_h"] / (3.0 * median_rate), 0.0, 2.0)
    raw_maint_risk = 0.60 * g["current_fouling_risk"] + 0.25 * g["performance_loss_score"] + 0.15 * (rate_factor * 50.0)
    g["maintenance_risk_score"] = np.clip(raw_maint_risk, 0.0, 100.0)
    
    # 6. Discrete Maintenance State Classification
    conditions = [
        g["maintenance_risk_score"] < 25.0,
        (g["maintenance_risk_score"] >= 25.0) & (g["maintenance_risk_score"] < 50.0),
        (g["maintenance_risk_score"] >= 50.0) & (g["maintenance_risk_score"] < 75.0),
        g["maintenance_risk_score"] >= 75.0
    ]
    choices = ["NORMAL", "WATCH", "WARNING", "CRITICAL"]
    g["maintenance_state"] = np.select(conditions, choices, default="NORMAL")
    
    risk_records.append(g)

test_risk_df = pd.concat(risk_records).reset_index(drop=True)

test_risk_df[[
    "scenario_id", "time_h", "DI_fouling", "DI_fouling_predicted_prod",
    "current_fouling_risk", "performance_loss_score", "degradation_rate_h",
    "threshold_distance", "time_to_threshold_h", "maintenance_risk_score", "maintenance_state"
]].to_csv(P9_DIR / "maintenance_risk_timeseries.csv", index=False)

# State distribution summary
state_dist = test_risk_df["maintenance_state"].value_counts(normalize=True).reset_index()
state_dist.columns = ["Maintenance_State", "Proportion"]
state_dist["Count"] = test_risk_df["maintenance_state"].value_counts().values
state_dist.to_csv(P9_DIR / "maintenance_state_distribution.csv", index=False)
state_dist.to_csv(TABLES_DIR / "maintenance_state_distribution.csv", index=False)

print("  Phase 9 completed. Risk scores and maintenance states calculated.")

# -----------------------------------------------------------------------------
# 4. PHASE 10 — MAINTENANCE ALERT SYSTEM
# -----------------------------------------------------------------------------
print("\n[4/6] Executing Phase 10: Event-Driven Alert System with Deduplication...")

DEDUP_WINDOW_H = thresholds_config["alert_settings"]["deduplication_window_h"]
alerts_generated = []

for sc_id, grp in test_risk_df.groupby("scenario_id", sort=False):
    g = grp.copy().reset_index(drop=True)
    last_alert_time = -999.0
    
    for i in range(len(g)):
        row = g.iloc[i]
        curr_t = row["time_h"]
        curr_di = row["DI_fouling_predicted_prod"]
        risk_sc = row["maintenance_risk_score"]
        ttt = row["time_to_threshold_h"]
        rate = row["degradation_rate_h"]
        
        # Check Alert Triggers
        is_alert = False
        severity = "INFO"
        reason = ""
        action = ""
        
        if curr_di >= critical_di_threshold:
            is_alert = True
            severity = "CRITICAL"
            reason = f"Current degradation ({curr_di:.6f}) exceeds Critical Threshold ({critical_di_threshold:.6f})"
            action = "Urgent CIP cleaning required immediately within 24-48 hours."
        elif curr_di >= warning_di_threshold:
            is_alert = True
            severity = "WARNING"
            reason = f"Current degradation ({curr_di:.6f}) exceeds Warning Threshold ({warning_di_threshold:.6f})"
            action = "Plan CIP cleaning within current maintenance window (48-72 hours)."
        elif ttt <= 168.0 and ttt > 0.0:
            is_alert = True
            severity = "WARNING" if ttt <= 72.0 else "WATCH"
            reason = f"Predicted threshold crossing within {ttt:.1f} hours (Horizon: 168h)"
            action = f"Schedule inspection and stage CIP chemicals for target window ({ttt:.0f}h)."
        elif row["maintenance_state"] in ["WARNING", "CRITICAL"]:
            is_alert = True
            severity = row["maintenance_state"]
            reason = f"High combined maintenance risk score ({risk_sc:.1f}/100)"
            action = "Perform comprehensive thermal-hydraulic audit."
            
        if is_alert:
            # Check deduplication window
            if (curr_t - last_alert_time) >= DEDUP_WINDOW_H:
                alerts_generated.append({
                    "alert_id": f"ALT-{sc_id}-{int(curr_t)}",
                    "exchanger_id": sc_id,
                    "time_h": curr_t,
                    "current_DI_pred": curr_di,
                    "actual_DI": row["DI_fouling"],
                    "threshold_warning": warning_di_threshold,
                    "threshold_critical": critical_di_threshold,
                    "predicted_crossing_time_h": ttt,
                    "maintenance_risk_score": risk_sc,
                    "severity": severity,
                    "trigger_reason": reason,
                    "recommended_action": action
                })
                last_alert_time = curr_t

alerts_df = pd.DataFrame(alerts_generated)
alerts_df.to_csv(P10_DIR / "alert_events_log.csv", index=False)
alerts_df.to_csv(TABLES_DIR / "alert_events_log.csv", index=False)

print(f"  Phase 10 completed. Generated {len(alerts_df):,} deduplicated alert events.")

# -----------------------------------------------------------------------------
# 5. PHASE 11 — CLEANING RECOMMENDATION ENGINE
# -----------------------------------------------------------------------------
print("\n[5/6] Executing Phase 11: Cleaning Recommendation Engine (Persistence-Filtered)...")

PERSISTENCE_STEPS = thresholds_config["alert_settings"]["persistence_steps_required"]
cleaning_recommendations = []

for sc_id, grp in test_risk_df.groupby("scenario_id", sort=False):
    g = grp.copy().reset_index(drop=True)
    
    for i in range(PERSISTENCE_STEPS - 1, len(g)):
        window = g.iloc[i - PERSISTENCE_STEPS + 1 : i + 1]
        row = g.iloc[i]
        curr_t = row["time_h"]
        curr_di = row["DI_fouling_predicted_prod"]
        
        # Persistence checks
        all_above_critical = (window["DI_fouling_predicted_prod"] >= critical_di_threshold).all()
        all_above_warning = (window["DI_fouling_predicted_prod"] >= warning_di_threshold).all()
        all_high_risk = (window["maintenance_risk_score"] >= 75.0).all()
        
        ttt = row["time_to_threshold_h"]
        
        # Logic determination
        cleaning_required = "NO"
        action_urgency = "MONITOR"
        recommended_window = "Next Scheduled Turnaround"
        explanation = "Degradation remains within acceptable operating envelope."
        
        if all_above_critical or all_high_risk:
            cleaning_required = "YES"
            action_urgency = "URGENT CIP"
            recommended_window = f"Immediate (T+{int(curr_t)}h to T+{int(curr_t)+48}h)"
            explanation = "Persistent severe thermal fouling exceeding critical limit for >= 3 consecutive hours."
        elif all_above_warning or (ttt <= 72.0 and ttt > 0):
            cleaning_required = "YES"
            action_urgency = "PLAN CIP"
            recommended_window = f"Within 72 Hours (T+{int(curr_t)}h to T+{int(curr_t)+72}h)"
            explanation = f"Degradation approaching critical boundary with projected crossing in {ttt:.1f} hours."
        elif row["maintenance_state"] == "WATCH" or (ttt <= 168.0 and ttt > 72.0):
            cleaning_required = "NO"
            action_urgency = "INSPECTION RECOMMENDED"
            recommended_window = f"Planning Horizon (T+{int(curr_t)}h to T+{int(curr_t)+168}h)"
            explanation = "Moderate fouling accumulation; inspect exchanger performance before next campaign."
            
        cleaning_recommendations.append({
            "scenario_id": sc_id,
            "time_h": curr_t,
            "current_DI_pred": curr_di,
            "maintenance_risk_score": row["maintenance_risk_score"],
            "cleaning_required": cleaning_required,
            "action_urgency": action_urgency,
            "recommended_window": recommended_window,
            "explanation": explanation
        })

clean_df = pd.DataFrame(cleaning_recommendations)
clean_df.to_csv(P11_DIR / "cleaning_recommendations_timeseries.csv", index=False)

# Sample recommendations summary (latest time per scenario)
clean_summary = clean_df.groupby("scenario_id").last().reset_index()
clean_summary.to_csv(P11_DIR / "latest_cleaning_status_by_exchanger.csv", index=False)
clean_summary.to_csv(TABLES_DIR / "latest_cleaning_status_by_exchanger.csv", index=False)

print("  Phase 11 completed. Cleaning recommendation engine active.")

# -----------------------------------------------------------------------------
# 6. VISUALIZATION SUITE & DASHBOARD GENERATOR
# -----------------------------------------------------------------------------
print("\n[6/6] Generating Publication Figures & Standalone HTML Dashboard...")

# Select representative showcase scenario for detailed plots
sample_scen = test_scenarios[2] # e.g. scenario 72
scen_data = test_risk_df[test_risk_df["scenario_id"] == sample_scen].copy().reset_index(drop=True)

# ── Fig 1: Performance Degradation (Phase 8) ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(scen_data["time_h"], scen_data["U_retention_pct"], label="Heat Transfer Retention (U/U_clean %)", color="#1f77b4", lw=1.5)
ax1.plot(scen_data["time_h"], scen_data["Q_retention_pct"], label="Heat Duty Retention (Q/Q_clean %)", color="#2ca02c", lw=1.5, ls="--")
ax1.set_ylabel("Retention Factor (%)", fontsize=11, fontweight="bold")
ax1.set_title(f"Heat Exchanger Thermal Performance Dynamics — Unit {sample_scen}", fontsize=13, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left")

ax2.plot(scen_data["time_h"], scen_data["performance_score"], label="Composite Performance Score (0-100)", color="#9467bd", lw=2)
ax2.axhline(75, color="orange", ls=":", label="Watch Boundary (75)")
ax2.axhline(50, color="red", ls=":", label="Degraded Boundary (50)")
ax2.set_xlabel("Operating Time (hours)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Health Score (0-100)", fontsize=11, fontweight="bold")
ax2.set_ylim(0, 105)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="lower left")

plt.tight_layout()
fig.savefig(FIG_DIR / "performance_degradation.png", dpi=200)
fig.savefig(P8_DIR / "performance_degradation.png", dpi=200)
plt.close(fig)

# ── Fig 2: Fouling Trend with Thresholds (Phase 9) ──────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(scen_data["time_h"], scen_data["DI_fouling"], label="Ground Truth DI_fouling (Simulation)", color="#7f7f7f", alpha=0.6, lw=1.2)
ax.plot(scen_data["time_h"], scen_data["DI_fouling_predicted_prod"], label="Stage-1 Predicted DI (Track A Production)", color="#1f77b4", lw=2)
ax.axhline(warning_di_threshold, color="#f39c12", ls="--", lw=1.8, label=f"Warning Threshold ({warning_di_threshold:.5f})")
ax.axhline(critical_di_threshold, color="#e74c3c", ls="-.", lw=1.8, label=f"Critical Threshold ({critical_di_threshold:.5f})")

# Mark CIP events
for cip_t in [2190, 4380, 6570]:
    ax.axvline(cip_t, color="#27ae60", ls=":", alpha=0.8)
    ax.text(cip_t + 20, critical_di_threshold * 1.1, "CIP Event", color="#27ae60", fontsize=9, rotation=90)

ax.set_title(f"Fouling Degradation Trajectory & Maintenance Thresholds — Unit {sample_scen}", fontsize=13, fontweight="bold")
ax.set_xlabel("Operating Time (hours)", fontsize=11, fontweight="bold")
ax.set_ylabel("Fouling Degradation Index (DI_fouling)", fontsize=11, fontweight="bold")
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(FIG_DIR / "fouling_trend_thresholds.png", dpi=200)
fig.savefig(P9_DIR / "fouling_trend_thresholds.png", dpi=200)
plt.close(fig)

# ── Fig 3: Maintenance Risk Timeline (Phase 9) ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(scen_data["time_h"], scen_data["maintenance_risk_score"], color="#2c3e50", lw=2, label="Composite Maintenance Risk Score")
ax.axhspan(0, 25, color="#2ecc71", alpha=0.20, label="NORMAL (0-25)")
ax.axhspan(25, 50, color="#f1c40f", alpha=0.20, label="WATCH (25-50)")
ax.axhspan(50, 75, color="#e67e22", alpha=0.20, label="WARNING (50-75)")
ax.axhspan(75, 100, color="#e74c3c", alpha=0.20, label="CRITICAL (75-100)")

ax.set_title(f"Predictive Maintenance Risk Evolution & States — Unit {sample_scen}", fontsize=13, fontweight="bold")
ax.set_xlabel("Operating Time (hours)", fontsize=11, fontweight="bold")
ax.set_ylabel("Maintenance Risk Score (0-100)", fontsize=11, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(FIG_DIR / "maintenance_risk_timeline.png", dpi=200)
fig.savefig(P9_DIR / "maintenance_risk_timeline.png", dpi=200)
plt.close(fig)

# ── Fig 4: Threshold Crossing Analysis (Phase 10 & 11) ──────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
valid_ttt = scen_data[scen_data["time_to_threshold_h"] < 500].copy()
ax.plot(valid_ttt["time_h"], valid_ttt["time_to_threshold_h"], color="#d35400", lw=2, label="Estimated Time to Critical Threshold (hours)")
ax.axhline(168, color="#f39c12", ls="--", label="1-Week Planning Horizon (168h)")
ax.axhline(48, color="#c0392b", ls="-.", label="Urgent CIP Horizon (48h)")

ax.set_title(f"Dynamic Time-to-Threshold (TTT) Prognostic Horizon — Unit {sample_scen}", fontsize=13, fontweight="bold")
ax.set_xlabel("Operating Time (hours)", fontsize=11, fontweight="bold")
ax.set_ylabel("Hours to Critical Threshold", fontsize=11, fontweight="bold")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(FIG_DIR / "threshold_crossing_analysis.png", dpi=200)
fig.savefig(P10_DIR / "threshold_crossing_analysis.png", dpi=200)
plt.close(fig)

# ── Standalone HTML Dashboard Platform ──────────────────────────────────────
print("  Generating Interactive HTML Dashboard Platform...")

recent_alerts = alerts_df.head(10).to_dict(orient="records")
latest_status = clean_summary.head(10).to_dict(orient="records")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Predictive Maintenance Dashboard — Shell & Tube Heat Exchangers</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
    h1, h2, h3 {{ color: #ffffff; margin-top: 0; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }}
    .badge {{ background: #1e293b; border: 1px solid #475569; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; color: #38bdf8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-bottom: 24px; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }}
    .card-title {{ font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 8px; }}
    .metric-value {{ font-size: 32px; font-weight: 800; color: #f8fafc; margin-bottom: 4px; }}
    .metric-sub {{ font-size: 13px; color: #64748b; }}
    .status-normal {{ color: #4ade80; }}
    .status-watch {{ color: #facc15; }}
    .status-warning {{ color: #fb923c; }}
    .status-critical {{ color: #f87171; }}
    .table-container {{ overflow-x: auto; margin-top: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
    th {{ background: #0f172a; color: #94a3b8; padding: 10px 12px; border-bottom: 2px solid #334155; font-weight: 600; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #334155; color: #e2e8f0; }}
    tr:hover {{ background: #243248; }}
    .track-a {{ border-left: 4px solid #38bdf8; }}
    .track-b {{ border-left: 4px solid #f97316; }}
    .alert-chip {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
    .chip-critical {{ background: #7f1d1d; color: #fca5a5; }}
    .chip-warning {{ background: #7c2d12; color: #fdba74; }}
    .chip-watch {{ background: #713f12; color: #fde047; }}
    .img-box {{ width: 100%; border-radius: 8px; border: 1px solid #334155; margin-top: 12px; }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>Shell & Tube Heat Exchanger — AI Predictive Maintenance</h1>
      <div style="color: #94a3b8; font-size: 14px;">Operational Diagnostics, Risk Prognostics, and Automated Cleaning Decision Engine</div>
    </div>
    <div>
      <span class="badge">SYSTEM ONLINE (V2 PHYSICS)</span>
    </div>
  </div>

  <!-- SECTION: DUAL MODEL TRACKS R2 PRESENTATION -->
  <h2>Model Architecture Verification & Benchmark Tracks</h2>
  <div class="grid">
    <div class="card track-a">
      <div class="card-title">Track A — Production Model (Leakage-Free Standard)</div>
      <div class="metric-value">Test R² = 0.2457</div>
      <div class="metric-sub"><b>Features:</b> 16 Strictly Causal Operating & CIP-Aware History Predictors</div>
      <p style="font-size: 13px; color: #94a3b8; margin-top: 10px;">
        Honest physical bound. Incorporates operational history without accessing hidden simulator states (Rf, U_overall, Q, thermal efficiency). 100% of large errors trace to unobservable stochastic Poisson bursts.
      </p>
    </div>
    <div class="card track-b">
      <div class="card-title">Track B — Showcase Demonstration Model (Controlled Leakage)</div>
      <div class="metric-value">Test R² = 0.9992</div>
      <div class="metric-sub"><b>Features:</b> Direct Simulator Performance Proxies (Rf, U_overall, Q)</div>
      <p style="font-size: 13px; color: #94a3b8; margin-top: 10px;">
        Showcase demonstration. Demonstrates downstream maintenance engine behavior under direct internal state observation. <b>Explicitly labeled as Target-Proxy Leakage</b>.
      </p>
    </div>
  </div>

  <!-- SECTION 1: HEAT EXCHANGER OVERVIEW -->
  <h2>1. Exchanger Fleet Operational Status (Held-Out Test Scenarios)</h2>
  <div class="grid">
    <div class="card">
      <div class="card-title">Fleet Active Monitored Units</div>
      <div class="metric-value">10 Exchangers</div>
      <div class="metric-sub">8,760 Hours Continuous Monitoring per Unit</div>
    </div>
    <div class="card">
      <div class="card-title">Fleet Average Performance Score</div>
      <div class="metric-value status-normal">{test_df['performance_score'].mean():.1f} / 100</div>
      <div class="metric-sub">Based on Heat Duty & Heat Transfer Retention</div>
    </div>
    <div class="card">
      <div class="card-title">Active Critical Maintenance Alerts</div>
      <div class="metric-value status-critical">{len(alerts_df[alerts_df['severity']=='CRITICAL'])} Events</div>
      <div class="metric-sub">Deduplicated in 24h Window</div>
    </div>
    <div class="card">
      <div class="card-title">Warning / Critical Fouling Thresholds</div>
      <div class="metric-value" style="font-size: 20px;">W: {warning_di_threshold:.5f} | C: {critical_di_threshold:.5f}</div>
      <div class="metric-sub">Derived Strictly from Training Data (75th / 92nd Pct)</div>
    </div>
  </div>

  <!-- SECTION 2 & 3: PERFORMANCE & DEGRADATION TRENDS -->
  <h2>2 & 3. Performance Degradation & Threshold Prognostics</h2>
  <div class="grid">
    <div class="card" style="grid-column: span 2;">
      <div class="card-title">Fouling Trajectory with Warning & Critical Thresholds (Unit {sample_scen})</div>
      <img src="figures/fouling_trend_thresholds.png" class="img-box" alt="Fouling Trend">
    </div>
    <div class="card">
      <div class="card-title">Maintenance Risk Score Evolution</div>
      <img src="figures/maintenance_risk_timeline.png" class="img-box" alt="Maintenance Risk">
    </div>
  </div>

  <!-- SECTION 4: ALERTS SYSTEM -->
  <h2>4. Event-Driven Maintenance Alerts Log</h2>
  <div class="card">
    <div class="card-title">Active Deduplicated Alerts (Sample Events)</div>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Alert ID</th>
            <th>Exchanger</th>
            <th>Time (h)</th>
            <th>Severity</th>
            <th>Predicted DI</th>
            <th>Time to Threshold</th>
            <th>Trigger Reason</th>
            <th>Recommended Action</th>
          </tr>
        </thead>
        <tbody>
"""

for alt in recent_alerts:
    chip_class = "chip-critical" if alt["severity"] == "CRITICAL" else ("chip-warning" if alt["severity"] == "WARNING" else "chip-watch")
    html_content += f"""
          <tr>
            <td><code>{alt['alert_id']}</code></td>
            <td><b>{alt['exchanger_id']}</b></td>
            <td>{alt['time_h']:.0f}h</td>
            <td><span class="alert-chip {chip_class}">{alt['severity']}</span></td>
            <td>{alt['current_DI_pred']:.6f}</td>
            <td>{alt['predicted_crossing_time_h']:.1f}h</td>
            <td>{alt['trigger_reason']}</td>
            <td>{alt['recommended_action']}</td>
          </tr>
    """

html_content += f"""
        </tbody>
      </table>
    </div>
  </div>

  <!-- SECTION 5: CLEANING RECOMMENDATION ENGINE -->
  <h2 style="margin-top: 24px;">5. Automated Clean-in-Place (CIP) Recommendations</h2>
  <div class="card">
    <div class="card-title">Fleet Current Cleaning Decision Status (Persistence-Filtered >= 3h)</div>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Exchanger</th>
            <th>Time</th>
            <th>Cleaning Required?</th>
            <th>Urgency</th>
            <th>Recommended Window</th>
            <th>Operational Explanation</th>
          </tr>
        </thead>
        <tbody>
"""

for cs in latest_status:
    req_color = "#f87171" if cs["cleaning_required"] == "YES" else "#4ade80"
    html_content += f"""
          <tr>
            <td><b>{cs['scenario_id']}</b></td>
            <td>{cs['time_h']:.0f}h</td>
            <td style="color: {req_color}; font-weight: 700;">{cs['cleaning_required']}</td>
            <td><b>{cs['action_urgency']}</b></td>
            <td>{cs['recommended_window']}</td>
            <td>{cs['explanation']}</td>
          </tr>
    """

html_content += """
        </tbody>
      </table>
    </div>
  </div>

  <div style="margin-top: 32px; padding-top: 16px; border-top: 1px solid #334155; font-size: 12px; color: #64748b; text-align: center;">
    AI Predictive Maintenance Platform • Google Antigravity • Shell & Tube Heat Exchanger Engine
  </div>

</body>
</html>
"""

dashboard_path = RESULTS_DIR / "dashboard.html"
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"  Saved Standalone Interactive HTML Dashboard to {dashboard_path}")
print("\nALL PHASES 8-11 COMPLETED SUCCESSFULLY.")
