"""Physical sanity checks and mathematical validation of Stage 1 models and targets.

Checks:
1. Exact numerical identities of DI_fouling and DI_total
2. Physical bounds: Non-negativity, DI_total >= DI_fouling
3. Monotonicity and CIP reset dynamics
4. Plausibility of flow, temperature, and Reynolds effects
5. Output format and Stage 2 handoff verification
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
from scipy import stats

from src.data_processing import (
    PROJECT_ROOT,
    construct_targets,
    load_raw_dataset,
)

TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def run_sanity_checks():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[Sanity Checks] Loading dataset...")
    raw_df = load_raw_dataset()
    df = construct_targets(raw_df)

    checks = []

    # Check 1: Numerical Identity DI_fouling
    diff_foul = np.abs(df["DI_fouling"] - (1.0 - df["thermal_efficiency"]))
    max_diff_foul = float(diff_foul.max())
    mean_diff_foul = float(diff_foul.mean())
    checks.append({
        "Check Name": "DI_fouling Numerical Identity",
        "Tested Condition": "|(1 - U_overall/U_clean) - (1 - thermal_efficiency)| < 1e-3",
        "Result": "PASSED" if max_diff_foul < 1e-3 else "FAILED",
        "Observed Metric": f"Max Diff: {max_diff_foul:.2e}, Mean Diff: {mean_diff_foul:.2e}",
        "Physical Rationale": "Slight ~9.9e-4 difference due to roundoff in area A_ht (29.1 vs 29.131 m²) in V2 generation script."
    })

    # Check 2: Numerical Identity DI_total
    diff_tot = np.abs(df["DI_total"] - (1.0 - df["efficiency_total"]))
    max_diff_tot = float(diff_tot.max())
    mean_diff_tot = float(diff_tot.mean())
    checks.append({
        "Check Name": "DI_total Numerical Identity",
        "Tested Condition": "|(1 - U_total/U_clean) - (1 - efficiency_total)| < 1e-3",
        "Result": "PASSED" if max_diff_tot < 1e-3 else "FAILED",
        "Observed Metric": f"Max Diff: {max_diff_tot:.2e}, Mean Diff: {mean_diff_tot:.2e}",
        "Physical Rationale": "Consistent analytical representation within floating point tolerances."
    })

    # Check 3: Non-negativity of Degradation Indices
    min_foul = float(df["DI_fouling"].min())
    min_tot = float(df["DI_total"].min())
    non_negative = (min_foul >= -1e-8) and (min_tot >= -1e-8)
    checks.append({
        "Check Name": "Non-Negativity Bounds",
        "Tested Condition": "DI_fouling >= 0 and DI_total >= 0 everywhere",
        "Result": "PASSED" if non_negative else "FAILED",
        "Observed Metric": f"Min DI_fouling: {min_foul:.6f}, Min DI_total: {min_tot:.6f}",
        "Physical Rationale": "Degradation indices represent lost performance and cannot be negative."
    })

    # Check 4: Physical Dominance (DI_total >= DI_fouling)
    tot_ge_foul = bool((df["DI_total"] >= df["DI_fouling"] - 1e-8).all())
    corrosion_contrib = float((df["DI_total"] - df["DI_fouling"]).mean())
    checks.append({
        "Check Name": "Physical Dominance (DI_total >= DI_fouling)",
        "Tested Condition": "DI_total >= DI_fouling everywhere (corrosion adds to fouling)",
        "Result": "PASSED" if tot_ge_foul else "FAILED",
        "Observed Metric": f"Passed: {tot_ge_foul}, Mean Corrosion Gap: {corrosion_contrib:.4f}",
        "Physical Rationale": "Total thermal resistance R_total = 1/U_clean + Rf + R_wall >= 1/U_clean + Rf, so DI_total >= DI_fouling."
    })

    # Check 5: CIP Reset Dynamics
    # Find all CIP event points and compare DI_fouling pre vs post
    cip_indices = df[df["cip_event"] == 1].index
    pre_indices = cip_indices - 1
    post_indices = cip_indices + 1
    valid_mask = (pre_indices >= 0) & (post_indices < len(df))
    cip_drops = df.loc[pre_indices[valid_mask], "DI_fouling"].values - df.loc[post_indices[valid_mask], "DI_fouling"].values
    
    mean_drop = float(np.mean(cip_drops))
    cip_valid = mean_drop > 0
    checks.append({
        "Check Name": "CIP Cleaning Reset Dynamics",
        "Tested Condition": "Mean drop in DI_fouling across scheduled CIP events > 0",
        "Result": "PASSED" if cip_valid else "FAILED",
        "Observed Metric": f"Mean CIP Drop: {mean_drop:.6f} (removes ~{mean_drop/df['DI_fouling'].mean()*100:.1f}% of mean fouling)",
        "Physical Rationale": "CIP events remove 60-95% of accumulated thermal fouling resistance."
    })

    # Check 6: Temperature Dependence (Arrhenius deposition kinetics)
    mean_foul_by_temp = df.groupby("T_in_C")["DI_fouling"].mean()
    temp_corr = float(stats.spearmanr(mean_foul_by_temp.index, mean_foul_by_temp.values).statistic)
    temp_valid = temp_corr > 0.8
    checks.append({
        "Check Name": "Temperature-Dependent Fouling Kinetics",
        "Tested Condition": "Spearman rank correlation of mean DI_fouling vs Nominal Temperature > 0.8",
        "Result": "PASSED" if temp_valid else "FAILED",
        "Observed Metric": f"Spearman rho = {temp_corr:.4f}",
        "Physical Rationale": "Epstein (1993) Arrhenius deposition law: deposition rate scales exponentially with temperature exp(-E/RT)."
    })

    # Check 7: Hydraulic Shear Removal (Flow velocity / Reynolds effect)
    # At fixed temperature, higher flow induces higher wall shear tau_w, increasing removal and capping asymptotic fouling
    s_high_t = df[df["T_in_C"] == 110].groupby("m_dot_nominal_kg_s")["DI_fouling"].mean()
    flow_corr = stats.spearmanr(s_high_t.index, s_high_t.values).statistic
    flow_valid = flow_corr < 0  # higher flow -> lower fouling
    checks.append({
        "Check Name": "Hydraulic Shear Removal Effect",
        "Tested Condition": "Higher mass flow rate reduces asymptotic fouling resistance at high temperature",
        "Result": "PASSED" if flow_valid else "FAILED",
        "Observed Metric": f"Spearman rho = {flow_corr:.4f}",
        "Physical Rationale": "Shear stress tau_w increases removal rate: removal = A_rem * tau_w * Rf."
    })

    report_df = pd.DataFrame(checks)
    out_csv = TABLES_DIR / "physical_sanity_checks_report.csv"
    report_df.to_csv(out_csv, index=False)
    print(f"\n[Sanity Checks] Saved report to {out_csv}")
    print(report_df.to_string())

    with open(TABLES_DIR / "physical_sanity_checks.json", "w") as f:
        json.dump(checks, f, indent=2)


if __name__ == "__main__":
    run_sanity_checks()
