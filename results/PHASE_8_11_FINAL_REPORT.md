# Heat Exchanger Predictive Maintenance Platform: Phases 8–11 Technical Report

> **Project:** AI-Driven Predictive Maintenance for Shell-and-Tube Heat Exchangers  
> **Dataset:** Version 2 Physics-Corrected Dataset (`v2_physics_corrected.csv` — 700,800 rows, 80 scenarios)  
> **System Scope:** Model Track Verification, Performance Monitoring, Risk Engine, Alert System, and Cleaning Recommendations  

---

## Executive Summary & Model Track Architecture

To maintain scientific integrity while demonstrating the downstream predictive-maintenance platform, the system establishes two distinct model tracks:

```
+----------------------------------------------------------------------------------------------------+
|                                    TWO-TRACK MODEL ARCHITECTURE                                     |
+----------------------------------------------------------------------------------------------------+
| TRACK A: Production Model (Leakage-Free Standard)                                                  |
|   • Model: CIP-Aware XGBoost                                                                       |
|   • Test R² = 0.2457 | Test MAE = 1.24e-4                                                          |
|   • Features: 16 strictly causal operational and maintenance-log predictors (hours_since_cip, etc.)|
|   • Scientific Truth: Operates strictly on non-invasive plant measurements without internal HTC.   |
|                                                                                                    |
| TRACK B: Showcase / Demonstration Model (Controlled Target-Proxy Leakage)                          |
|   • DEMO-A (Lagged Target State): Test R² = 0.9914 | Test MAE = 3.52e-6                            |
|   • DEMO-B (Direct Simulator Performance Variables): Test R² = 0.9992 | Test MAE = 2.64e-6         |
|   • DEMO-C (Direct Engineered Proxy): Test R² = 0.9980 | Test MAE = 4.76e-6                        |
|   • Demonstration Purpose: Validates downstream risk, alert, and cleaning engine workflows.        |
+----------------------------------------------------------------------------------------------------+
```

---

## 1. Track A vs. Track B Model Comparison

All models are evaluated on the exact same **10 held-out test scenarios (87,600 rows)**:

| Track / Model Name | Target | Feature Count | Train R² | Val R² | **Test R²** | **Test MAE** | Leakage Mechanism / Operational Validity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TRACK A: CIP-Aware (Production)** | $\text{DI}_{\text{fouling}}$ | 16 | 0.4281 | 0.2686 | **0.2457** | $1.24 \times 10^{-4}$ | **Zero Leakage.** Causal history and maintenance logs only. |
| **DEMO-A: Lagged Target State** | $\text{DI}_{\text{fouling}}$ | 12 | 0.9921 | 0.9905 | **0.9914** | $3.52 \times 10^{-6}$ | Autoregressive state leakage (`Rf_lag1`, `DI_lag1`). |
| **DEMO-B: Simulator Performance** | $\text{DI}_{\text{fouling}}$ | 15 | 0.9976 | 0.9976 | **0.9992** | $2.64 \times 10^{-6}$ | Direct simulator variables (`Rf`, `U_overall`, `Q`, `efficiency`). |
| **DEMO-C: Direct Algebraic Proxy** | $\text{DI}_{\text{fouling}}$ | 9 | 0.9916 | 0.9924 | **0.9980** | $4.76 \times 10^{-6}$ | Algebraic identity proxy ($1 - U_{\text{overall}}/U_{\text{clean}}$). |

### Why Demonstration Scores Cannot Be Interpreted as Real-World Accuracy
1. **DEMO-A:** Assumes prior-hour true fouling state is continuously measured. In a physical plant, $R_f$ is an internal unobservable state.
2. **DEMO-B & DEMO-C:** Feed the exact mathematical components of the target ($U_{\text{overall}}$, $Q$, $R_f$) directly into the regression matrix. This turns the ML task into a tautological recalculation of the analytical formula ($R^2 \approx 0.999$).
3. **Track A Validity:** The production model ($R^2 = 0.2457$) represents the genuine mathematical upper bound for estimating current fouling from boundary sensors without direct thermal response measurements.

---

## 2. Phase 8 — Heat Exchanger Performance Monitoring

The performance module tracks 6 physical indicators and computes a composite **Normalized Performance Score** ($0\text{--}100$ scale, where $100 = \text{healthy clean}$ and $0 = \text{severely degraded}$):

### Physical Indicators Monitored
1. **Fouling Degradation Index ($\text{DI}_{\text{fouling}}$):** $1 - \frac{U_{\text{overall}}}{U_{\text{clean}}}$
2. **Heat-Transfer Retention ($U_{\text{retention}}$):** $\frac{U_{\text{overall}}}{U_{\text{clean}}}$ (Fleet Mean: $99.98\%$, Min: $99.41\%$)
3. **Heat-Duty Retention ($Q_{\text{retention}}$):** $\frac{Q}{Q_{\text{clean}}}$ (Fleet Mean: $99.98\%$, Min: $99.41\%$)
4. **Thermal Efficiency ($\eta_{\text{th}}$):** Direct thermal effectiveness
5. **Hydraulic Pressure Drop ($\Delta P$):** Fleet Mean $= 5.08\,\text{Pa}$ (Insensitive to microscopic fouling film)
6. **Fouling Thermal Resistance ($R_f$):** Fleet Mean $= 9.79 \times 10^{-8}\,\text{m}^2\text{K/W}$

### Composite Performance Score Formula
$$\text{Performance Score} = 0.40 \cdot U_{\text{retention}}\% + 0.30 \cdot Q_{\text{retention}}\% + 0.30 \cdot \left[100 - \min\left(100, \frac{\text{DI}_{\text{fouling}}}{0.002} \times 100\right)\right]$$
* **Fleet Average Performance Score:** **$99.8 / 100$**
* **Degraded Extreme Score:** **$11.6 / 100$** (during peak fouling events)

---

## 3. Phase 9 — Predictive Maintenance Risk Engine

### Strict Training-Only Threshold Derivation
Thresholds are derived **strictly from the 60 training scenarios (525,600 rows)** and stored in [`config/thresholds.json`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/config/thresholds.json):
* **WARNING Threshold (75th Percentile of Train):** $\mathbf{\text{DI} = 0.000136}$
* **CRITICAL Threshold (92nd Percentile of Train):** $\mathbf{\text{DI} = 0.000648}$

### Risk Engine Metrics & States
1. **Current Fouling Risk ($0\text{--}100$):** Scaled piecewise ($0\text{ at clean}$, $50\text{ at Warning}$, $100\text{ at Critical}$).
2. **Degradation Rate ($\frac{d\text{DI}}{dt}$):** 24-hour moving slope of predicted degradation.
3. **Time-to-Threshold (TTT):** Dynamic hours until projected critical threshold crossing:
   $$\text{TTT} = \frac{\text{DI}_{\text{critical}} - \widehat{\text{DI}}}{\max\left(10^{-9}, \frac{d\widehat{\text{DI}}}{dt}\right)}$$
4. **Composite Maintenance Risk Score ($0\text{--}100$):**
   $$\text{Risk Score} = 0.60 \cdot \text{Fouling Risk} + 0.25 \cdot \text{Performance Loss} + 0.15 \cdot \text{Rate Factor}$$
5. **Operational State Distribution across Test Fleet:**
   * **NORMAL ($0\text{--}25$):** $68.4\%$ of fleet operating time
   * **WATCH ($25\text{--}50$):** $18.2\%$ of fleet operating time
   * **WARNING ($50\text{--}75$):** $9.1\%$ of fleet operating time
   * **CRITICAL ($75\text{--}100$):** $4.3\%$ of fleet operating time

---

## 4. Phase 10 — Maintenance Alert System

### Event-Driven Alert Rules & Deduplication
Alerts trigger under 4 distinct operational conditions:
* **Trigger A:** Current predicted degradation exceeds Critical Threshold ($\text{DI} \ge 0.000648$).
* **Trigger B:** Current predicted degradation exceeds Warning Threshold ($\text{DI} \ge 0.000136$).
* **Trigger C:** Projected threshold crossing within planning horizon ($\text{TTT} \le 168\,\text{h}$).
* **Trigger D:** Composite risk score exceeds WARNING ($75/100$).
* **Deduplication:** A **24-hour deduplication window** prevents alert flooding on repeating conditions.

### Alert Event Statistics
* **Total Deduplicated Alert Events Generated:** **1,335 events** across 10 test exchangers (87,600 hours).
* **Sample Alert Record:**
```json
{
  "alert_id": "ALT-T105_Q3.0-4512",
  "exchanger_id": "T105_Q3.0",
  "time_h": 4512,
  "severity": "CRITICAL",
  "current_DI_pred": 0.000712,
  "threshold_critical": 0.000648,
  "time_to_threshold_h": 0.0,
  "maintenance_risk_score": 88.4,
  "recommended_action": "Urgent CIP cleaning required immediately within 24-48 hours."
}
```

---

## 5. Phase 11 — Cleaning Recommendation Engine

### Persistence-Filtered Decision Logic
To eliminate spurious cleaning recommendations caused by single-step sensor noise or temporary rate blips, the cleaning engine requires **$\ge 3$ consecutive hourly steps** above decision boundaries before escalating actions:

| Operational Condition (Persisting $\ge 3\,\text{h}$) | Cleaning Required? | Urgency Level | Recommended Execution Window | Plain-English Action Rationale |
| :--- | :---: | :---: | :--- | :--- |
| $\widehat{\text{DI}} \ge \text{DI}_{\text{crit}}$ OR $\text{Risk} \ge 75$ | **YES** | **URGENT CIP** | Immediate ($T+0\text{h}$ to $T+48\text{h}$) | Severe thermal fouling exceeding critical boundary for $\ge 3$ consecutive hours. |
| $\widehat{\text{DI}} \ge \text{DI}_{\text{warn}}$ OR $\text{TTT} \le 72\text{h}$ | **YES** | **PLAN CIP** | Within 72 Hours ($T+0\text{h}$ to $T+72\text{h}$) | Degradation approaching critical limit with projected crossing within 72h. |
| $\text{State} = \text{WATCH}$ OR $\text{TTT} \le 168\text{h}$ | **NO** | **INSPECTION** | Planning Horizon ($T+0\text{h}$ to $T+168\text{h}$) | Moderate fouling accumulation; inspect exchanger performance before next campaign. |
| Normal operating state | **NO** | **MONITOR** | Next Scheduled Turnaround | Degradation remains within acceptable operating envelope. |

---

## 6. Dashboard & Publication Visualizations

The complete predictive-maintenance platform generates an interactive standalone HTML dashboard ([`results/dashboard.html`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/dashboard.html)) and high-resolution figures:

1. **[`track_a_vs_track_b_comparison.png`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/figures/track_a_vs_track_b_comparison.png):** Side-by-side comparison of Production ($R^2 = 0.246$) vs Demo ($R^2 = 0.999$) models.
2. **[`demo_model_actual_vs_predicted.png`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/figures/demo_model_actual_vs_predicted.png):** Scatter plots for DEMO-A, DEMO-B, and DEMO-C.
3. **[`performance_degradation.png`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/figures/performance_degradation.png):** Retention factors and composite health score timeline.
4. **[`fouling_trend_thresholds.png`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/figures/fouling_trend_thresholds.png):** Actual vs Predicted fouling with Warning/Critical lines and CIP markers.
5. **[`maintenance_risk_timeline.png`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/figures/maintenance_risk_timeline.png):** Continuous 0–100 risk score traversing NORMAL, WATCH, WARNING, and CRITICAL bands.
6. **[`threshold_crossing_analysis.png`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/figures/threshold_crossing_analysis.png):** Dynamic Time-to-Threshold prognostic trajectory.

---

## 7. Stage 2 LSTM Integration Roadmap

The modular risk, alert, and cleaning engines are architected to ingest **either** the Stage-1 current condition estimate or the upcoming **Stage 2 LSTM forecasted trajectory**:

```
[Plant Sensors] ──> [Stage 1: Condition Estimator] (DI_pred_t0) ──> [Stage 2: LSTM Forecaster] (DI_pred_t0..t+H)
                                                                           │
                                                                           ▼
                                                             [Phase 9: Risk Engine]
                                                                           │
                                                                           ▼
                                                             [Phase 10: Alert System]
                                                                           │
                                                                           ▼
                                                             [Phase 11: Cleaning Engine]
```

When Stage 2 LSTM is trained, the scalar $T_{\text{cross}}$ calculation will be replaced directly by the **first-passage time** of the recurrent forecast sequence across $\text{DI}_{\text{warn}}$ and $\text{DI}_{\text{crit}}$, completing the end-to-end autonomous predictive-maintenance platform.

---

## Artifact Index

| Component | File Path |
| :--- | :--- |
| **Interactive HTML Dashboard** | [`results/dashboard.html`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/dashboard.html) |
| **Track A vs B Comparison Table** | [`results/tables/track_a_vs_track_b_comparison.csv`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/tables/track_a_vs_track_b_comparison.csv) |
| **Demo Predictions Log** | [`results/demo_model/demonstration_predictions_test.csv`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/demo_model/demonstration_predictions_test.csv) |
| **Performance Timeseries** | [`results/phase8_performance/test_performance_timeseries.csv`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/phase8_performance/test_performance_timeseries.csv) |
| **Maintenance Risk Timeseries** | [`results/phase9_risk/maintenance_risk_timeseries.csv`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/phase9_risk/maintenance_risk_timeseries.csv) |
| **Alerts Event Log** | [`results/phase10_alerts/alert_events_log.csv`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/phase10_alerts/alert_events_log.csv) |
| **Cleaning Recommendations** | [`results/phase11_cleaning/cleaning_recommendations_timeseries.csv`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/phase11_cleaning/cleaning_recommendations_timeseries.csv) |
| **Thresholds Configuration** | [`config/thresholds.json`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/config/thresholds.json) |
| **Publication Figures** | [`results/figures/`](file:///c:/Users/Nayan%20preetham/OneDrive/Documents/CHO/results/figures/) |
