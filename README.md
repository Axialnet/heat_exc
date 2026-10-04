# AI-Driven Predictive Maintenance for Shell-and-Tube Heat Exchangers

## Overview

This project aims to develop an **AI-driven predictive maintenance system for shell-and-tube heat exchangers**, with a primary focus on **fouling prediction, performance degradation monitoring, maintenance decision-making, and cleaning-event recommendation**.

Heat exchangers gradually lose performance as fouling accumulates on heat-transfer surfaces. Conventional maintenance strategies often rely on fixed cleaning schedules or reactive maintenance after performance has already degraded.

This project takes a different approach:

> **Use physics-informed data and machine learning to predict fouling and heat-exchanger degradation early enough to support proactive maintenance decisions.**

The complete system is being developed as a pipeline:

```text
Heat Exchanger Operating Data
            │
            ▼
     Data Processing
            │
            ▼
 Physics-Based Feature Engineering
            │
            ▼
     Fouling Prediction
            │
            ▼
 Performance / Degradation Estimation
            │
            ▼
 Maintenance Decision Engine
            │
      ┌─────┴─────┐
      ▼           ▼
   Alerts     Cleaning Recommendation
      │           │
      └─────┬─────┘
            ▼
     Maintenance Schedule
            │
            ▼
       AI Dashboard
```

The ultimate goal is to move from **monitoring what has already happened** to **predicting what is likely to happen and deciding what action should be taken**.

---

# Problem Statement

Fouling in heat exchangers causes:

* Increased thermal resistance
* Reduced heat-transfer efficiency
* Increased pressure drop
* Increased energy consumption
* Reduced equipment performance
* Increased operating costs
* Unplanned shutdowns
* Frequent or unnecessary cleaning

A simple time-based maintenance strategy may clean an exchanger:

> "Every 6 months."

However, the actual fouling rate depends on operating conditions.

One exchanger may require cleaning earlier, while another may continue operating efficiently for much longer.

Therefore, a better maintenance strategy is:

> **Clean when predicted degradation and fouling indicate that cleaning is economically or operationally justified.**

This project aims to build the intelligence required to make that decision.

---

# Project Objective

The main objective is to develop an **AI-based predictive maintenance framework for shell-and-tube heat exchangers** capable of:

1. Understanding heat-exchanger operating conditions.
2. Estimating important thermal and hydraulic parameters.
3. Predicting fouling behavior.
4. Predicting future fouling progression.
5. Monitoring heat-exchanger performance.
6. Detecting abnormal degradation.
7. Estimating when maintenance may be required.
8. Generating maintenance alerts.
9. Recommending suitable cleaning events.
10. Providing a dashboard for monitoring and decision support.

The project therefore combines:

**Physics + Data Engineering + Machine Learning + Predictive Analytics + Maintenance Decision Support + Visualization**

---

# Project Architecture

The proposed system consists of multiple layers.

```text
┌───────────────────────────────────────────────┐
│              HEAT EXCHANGER DATA              │
│                                               │
│ Temperature | Flow | Pressure | Properties   │
│ Operating Conditions | Fouling Measurements  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│              DATA PROCESSING                  │
│                                               │
│ Cleaning | Missing Values | Outliers          │
│ Normalization | Data Validation               │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│          PHYSICS-BASED FEATURE ENGINEERING    │
│                                               │
│ Re | Pr | Nu | h | U | Q | ΔP | Rf | etc.    │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│             AI / ML PREDICTION                │
│                                               │
│ Fouling Prediction                            │
│ Performance Prediction                        │
│ Degradation Forecasting                       │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│          MAINTENANCE INTELLIGENCE             │
│                                               │
│ Risk Score                                    │
│ Remaining Useful Operating Time               │
│ Maintenance Threshold                         │
│ Cleaning Recommendation                       │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│             ALERT & SCHEDULING                │
│                                               │
│ Early Warning | Maintenance Alert             │
│ Cleaning Schedule | Priority                  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 AI DASHBOARD                  │
│                                               │
│ Live Status | Fouling Trend | Predictions     │
│ Risk | Alerts | Maintenance Recommendations  │
└───────────────────────────────────────────────┘
```

---

# Phase 1 — Dataset Development

The project begins with an existing shell-and-tube heat-exchanger fouling simulation dataset.

[Original Shell-and-Tube Heat Exchanger Fouling Simulation Dataset](https://www.kaggle.com/datasets/fdavidsantillan/shell-and-tube-heat-exchanger-fouling-simulation?utm_source=chatgpt.com)

The dataset forms the starting point for the development of the predictive-maintenance framework.

---

# Phase 2 — Version 1 Analysis

The original dataset is treated as **Version 1**.

The first stage involves understanding the available variables and determining whether they provide enough information for reliable fouling analysis.

The analysis includes:

* Parameter identification
* Physical interpretation
* Statistical analysis
* Distribution analysis
* Correlation analysis
* Relationship with fouling
* Identification of missing physical information

The key question is:

> **Does Version 1 contain enough information to represent the physical conditions governing fouling and heat-exchanger degradation?**

---

# Phase 3 — Development of Version 2

Version 1 is extended into **Version 2** using physics-based feature engineering.

Instead of simply adding arbitrary machine-learning features, additional variables are derived from the existing physical parameters using heat-transfer and fluid-flow relationships.

Examples include:

* Reynolds number
* Prandtl number
* Nusselt number
* Convective heat-transfer coefficient
* Overall heat-transfer coefficient
* Heat-transfer rate
* Temperature driving force
* Pressure-drop-related parameters
* Thermal resistance
* Fouling resistance
* Other physically meaningful operating indicators

The objective is to provide the ML model with a richer representation of the heat exchanger's physical state.

```text
VERSION 1
   │
   ├── Existing variables
   │
   ▼
Physical Analysis
   │
   ├── What information is available?
   ├── What information is missing?
   └── What can be derived?
   │
   ▼
Physics-Based Feature Engineering
   │
   ▼
VERSION 2
   │
   ├── Original parameters
   └── Derived physical parameters
```

---

# Phase 4 — Version 1 vs Version 2 Validation

The two versions are compared to determine whether the additional parameters provide meaningful information.

The comparison includes:

* Feature distributions
* Parameter ranges
* Correlations
* Feature relationships
* Fouling sensitivity
* Physical interpretability
* Redundancy
* ML relevance

The objective is not merely to demonstrate that Version 2 contains more columns.

The objective is to establish that:

> **Version 2 provides a more informative physical representation of the heat exchanger and its fouling behavior.**

---

# Phase 5 — Exploratory Data Analysis

A comprehensive EDA pipeline will be developed to understand the dataset before machine-learning modeling.

This includes:

### Univariate Analysis

Understanding the distribution and range of each variable.

### Bivariate Analysis

Studying relationships between:

* Operating conditions
* Thermal parameters
* Hydraulic parameters
* Fouling variables
* Performance indicators

### Multivariate Analysis

Understanding interactions between multiple operating variables.

### Correlation Analysis

Identifying relationships between features and potential target variables.

### Outlier Analysis

Detecting unusual operating conditions and potentially unreliable observations.

### Feature Redundancy

Identifying highly correlated or mathematically redundant variables before model development.

---

# Phase 6 — AI-Based Fouling Prediction

The central intelligence layer of the project is the development of machine-learning models capable of predicting fouling.

Potential prediction targets include:

* Fouling resistance
* Fouling factor
* Heat-transfer degradation
* Overall heat-transfer coefficient
* Performance degradation
* Fouling severity

The modeling pipeline will compare multiple algorithms.

Possible models include:

```text
Linear Regression
Random Forest
Gradient Boosting
XGBoost
Support Vector Regression
Artificial Neural Networks
```

Models will be evaluated using appropriate regression and forecasting metrics.

Examples include:

* MAE
* RMSE
* MAPE
* R²

The objective is not simply to maximize accuracy.

The model should also be:

* Physically meaningful
* Interpretable
* Robust
* Stable across operating conditions
* Useful for maintenance decisions

---

# Phase 7 — Fouling Forecasting

Predicting the current fouling state is useful, but predictive maintenance requires looking into the future.

Therefore, the system will be extended from:

```text
"What is the fouling level now?"
```

to:

```text
"What is the expected fouling level in the future?"
```

The model will generate a fouling trajectory such as:

```text
Fouling
  │
  │                         /
  │                      _/
  │                   __/
  │                __/
  │             __/
  │___________/________________ Time
              ↑
          Current State
```

This allows the system to estimate when fouling is expected to cross an operational or maintenance threshold.

---

# Phase 8 — Heat Exchanger Performance Monitoring

Fouling does not matter only because of the fouling value itself.

Its operational consequence is what ultimately matters.

The system will therefore monitor performance indicators such as:

* Overall heat-transfer coefficient
* Heat-transfer rate
* Thermal effectiveness
* Pressure drop
* Temperature approach
* Fouling resistance
* Performance degradation

A performance-health indicator can then be developed to represent the overall condition of the exchanger.

Conceptually:

```text
Healthy
   │
   ▼
Normal Operation
   │
   ▼
Early Fouling
   │
   ▼
Performance Degradation
   │
   ▼
Maintenance Required
```

---

# Phase 9 — Predictive Maintenance / Risk Engine

The predicted fouling and performance degradation will be converted into a maintenance-oriented risk indicator.

For example:

```text
                    HEAT EXCHANGER HEALTH

Current Fouling        ███████░░░  Moderate
Predicted Fouling      █████████░  High
Performance Loss       ██████░░░░  Moderate
Maintenance Risk       ████████░░  High
```

The system will classify the exchanger into operational states such as:

```text
NORMAL
WATCH
WARNING
CRITICAL
```

The exact thresholds will be determined using the physical behavior of the system and the intended maintenance criteria.

---

# Phase 10 — Maintenance Alert System

The predictive model will be connected to an alert mechanism.

Instead of waiting until the exchanger reaches a critical condition, the system will identify the approaching condition early.

Example:

```text
Current Condition
       │
       ▼
Fouling Prediction
       │
       ▼
Threshold Detection
       │
       ▼
Maintenance Risk
       │
       ▼
ALERT
```

Example alert:

> Heat Exchanger HX-01 is predicted to exceed the fouling threshold within the defined prediction horizon. Maintenance planning is recommended.

Different levels of alerts can be implemented:

```text
LOW       → Continue monitoring
MEDIUM    → Inspect operating condition
HIGH      → Plan maintenance
CRITICAL  → Cleaning / intervention required
```

---

# Phase 11 — Cleaning Event Recommendation

The system will go beyond predicting fouling.

It will use the prediction to recommend **when cleaning should be considered**.

Instead of:

```text
Clean every 6 months
```

the objective is:

```text
Predict fouling
      ↓
Estimate degradation
      ↓
Determine threshold crossing
      ↓
Estimate maintenance window
      ↓
Recommend cleaning event
```

The recommendation may consider:

* Current fouling
* Predicted fouling
* Rate of fouling accumulation
* Performance degradation
* Operating conditions
* Maintenance thresholds
* Expected future degradation

The output can be expressed as:

```text
Recommended Action:
PLAN CLEANING

Expected threshold crossing:
Within prediction horizon

Priority:
HIGH

Reason:
Predicted fouling is expected to cause unacceptable
heat-transfer performance degradation.
```

The system is therefore intended to support **condition-based maintenance rather than purely calendar-based maintenance**.

---

# Phase 12 — Maintenance Scheduling

Once a cleaning event is recommended, the system can generate a maintenance planning window.

The objective is not simply:

> "Clean now."

Instead, the system should answer:

> **When should cleaning be scheduled to balance equipment performance and maintenance requirements?**

The maintenance engine can therefore generate:

```text
Current Date
     │
     ├── Safe operating period
     │
     ├── Recommended maintenance window
     │
     └── Predicted critical threshold
```

This creates a bridge between the ML prediction and an actual maintenance decision.

---

# Phase 13 — AI Dashboard

A major component of the final project will be an interactive dashboard for monitoring heat-exchanger health.

The dashboard will provide a centralized interface for engineers or operators.

Potential dashboard sections include:

### System Overview

```text
Total Heat Exchangers
Healthy
Warning
Critical
Maintenance Due
```

### Individual Heat Exchanger Health

```text
HX-01
Health Status: WARNING

Current Fouling:       XX
Predicted Fouling:     XX
Performance Loss:      XX%
Maintenance Risk:      HIGH
```

### Fouling Trend

Historical and predicted fouling trajectories will be displayed together.

```text
Fouling
  │
  │                 Predicted
  │                    /
  │                 __/
  │              __/
  │           __/
  │        __/
  │_______/________________ Time
          ↑
        Today
```

### Performance Trend

The dashboard will display changes in heat-exchanger performance over time.

### Maintenance Alerts

The system will display active warnings and predicted maintenance requirements.

### Cleaning Recommendations

Each exchanger can have an automatically generated recommendation based on its predicted condition.

---

# Proposed Dashboard Architecture

```text
                  AI PREDICTIVE MAINTENANCE
                           DASHBOARD
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
   System Health          Fouling Prediction      Performance
        │                       │                       │
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                       Maintenance Risk
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
             Alerts                    Cleaning Recommendation
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                       Maintenance Schedule
```

---

# Phase 14 — Explainable AI

For an industrial predictive-maintenance system, prediction alone is not enough.

The system should also provide an explanation for why the model is predicting increasing fouling or maintenance risk.

Feature-importance and explainability techniques can be used to identify influential variables.

For example:

```text
Prediction: HIGH FOULING RISK

Major contributing factors:

1. Reduced flow velocity
2. Increased operating temperature
3. Increased thermal resistance
4. Reduced heat-transfer coefficient
5. Increasing historical fouling rate
```

Potential explainability methods include:

* Feature importance
* SHAP
* Partial dependence analysis
* Sensitivity analysis

This makes the system more useful to engineers because the output is not treated as an unexplained black-box prediction.

---

# Complete End-to-End Workflow

The complete project is designed to progress from raw physical data to a maintenance recommendation.

```text
┌──────────────────────────────┐
│ Existing Heat Exchanger Data │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Data Cleaning & Validation   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Version 1 Analysis           │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Identify Missing Information │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Physics-Based Derivation     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Version 2 Dataset             │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ V1 vs V2 Validation          │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Feature Engineering          │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ML Fouling Prediction        │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Future Fouling Forecast      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Performance Degradation      │
│ Monitoring                   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Maintenance Risk Engine      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Alert Generation             │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Cleaning Recommendation      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Maintenance Scheduling       │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ AI Predictive Maintenance    │
│ Dashboard                    │
└──────────────────────────────┘
```

---

# Repository Structure

```text
shell-tube-heat-exchanger-predictive-maintenance/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   └── version_1/
│   │
│   ├── processed/
│   │   └── version_2/
│   │
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_version_1_analysis.ipynb
│   ├── 03_parameter_derivation.ipynb
│   ├── 04_version_2_generation.ipynb
│   ├── 05_version_comparison.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_fouling_prediction.ipynb
│   ├── 08_fouling_forecasting.ipynb
│   ├── 09_model_interpretability.ipynb
│   └── 10_maintenance_prediction.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── parameter_derivation.py
│   ├── models.py
│   ├── forecasting.py
│   ├── maintenance_engine.py
│   ├── alert_engine.py
│   └── visualization.py
│
├── dashboard/
│   ├── app.py
│   ├── components/
│   └── assets/
│
├── models/
│   ├── fouling_model/
│   ├── forecasting_model/
│   └── maintenance_model/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── model_results/
│
├── docs/
│   ├── methodology.md
│   ├── version_1.md
│   ├── version_2.md
│   ├── parameter_derivation.md
│   ├── ml_methodology.md
│   └── maintenance_framework.md
│
├── requirements.txt
│
└── LICENSE
```

---

# Technology Stack

The project will use Python-based scientific computing and machine-learning tools.

### Data & Scientific Computing

```text
Python
NumPy
Pandas
SciPy
```

### Visualization

```text
Matplotlib
Seaborn
Plotly
```

### Machine Learning

```text
Scikit-learn
XGBoost
```

### Explainable AI

```text
SHAP
```

### Dashboard

The dashboard can be developed using a Python-based framework such as:

```text
Streamlit
```

or another suitable dashboard framework depending on the final deployment architecture.

---

# Data-to-Decision Philosophy

The core philosophy of the project is:

```text
DATA
 ↓
INFORMATION
 ↓
PHYSICAL UNDERSTANDING
 ↓
PREDICTION
 ↓
RISK
 ↓
DECISION
 ↓
ACTION
```

Most basic ML projects stop at:

```text
Data → Model → Prediction
```

This project aims to go further:

```text
Data
 ↓
Physics-informed features
 ↓
ML prediction
 ↓
Future condition
 ↓
Maintenance risk
 ↓
Recommended action
 ↓
Scheduled intervention
```

This makes the project a **predictive-maintenance decision-support system**, rather than simply a machine-learning model for predicting fouling.

---

# Expected Final System

The final system is intended to answer questions such as:

> What is the current condition of the heat exchanger?

> How much fouling has accumulated?

> Is fouling increasing rapidly?

> How will fouling evolve over the next prediction horizon?

> How much performance degradation is expected?

> When is the exchanger likely to reach a critical condition?

> Does the exchanger require maintenance?

> When should cleaning be scheduled?

> Why is the system recommending maintenance?

> Which operating parameters are contributing most to the predicted fouling?

These outputs will be integrated into a single dashboard so that the ML model produces actionable engineering information.

---

# Project Outcome

The final outcome of this project will be an **AI-driven predictive maintenance framework for shell-and-tube heat exchangers** that integrates:

```text
Physics-Based Modeling
        +
Data Engineering
        +
Machine Learning
        +
Fouling Forecasting
        +
Performance Monitoring
        +
Risk Prediction
        +
Maintenance Alerts
        +
Cleaning Recommendation
        +
Maintenance Scheduling
        +
Interactive Dashboard
```

The intended progression is:

**Version 1 Dataset → Physics-Informed Version 2 → ML Prediction → Fouling Forecast → Performance Degradation → Maintenance Risk → Alert → Cleaning Recommendation → Maintenance Schedule → Dashboard**

---

# Current Development Direction

The project is being developed progressively, beginning with the construction and validation of the physics-informed Version 2 dataset.

The subsequent development stages build upon this foundation to create the complete predictive-maintenance pipeline.

The end goal is not only to predict fouling, but to convert that prediction into an engineering decision:

> **Predict the problem before it becomes critical, determine when intervention is appropriate, and provide the information required to act.**

---

# Dataset Source

The initial dataset is based on the publicly available shell-and-tube heat-exchanger fouling simulation dataset:

[Kaggle — Shell-and-Tube Heat Exchanger Fouling Simulation](https://www.kaggle.com/datasets/fdavidsantillan/shell-and-tube-heat-exchanger-fouling-simulation?utm_source=chatgpt.com)

Appropriate attribution and licensing requirements of the original dataset will be retained.

---

# Authors

**NAYAN**
B.Tech Civil Engineering
Indian Institute of Technology (BHU), Varanasi


---

# Project Vision

The long-term vision of this project is to transform heat-exchanger maintenance from:

```text
Reactive Maintenance
        ↓
Time-Based Maintenance
        ↓
Condition Monitoring
        ↓
Predictive Maintenance
        ↓
AI-Assisted Maintenance Decision Making
```

The final objective is an intelligent system capable of continuously evaluating heat-exchanger condition, forecasting fouling and performance degradation, identifying maintenance risk, and assisting engineers in deciding **when and why maintenance should be performed**.
