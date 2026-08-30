from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
V1_PATH = ROOT.parent / "version 1" / "shell_tube_fouling_final.csv"
OUTPUT_PATH = ROOT / "v2_physics_corrected.csv"
METADATA_PATH = ROOT / "v2_metadata.json"


def load_v1() -> pd.DataFrame:
    df = pd.read_csv(V1_PATH)
    df = df.sort_values(["scenario_id", "time_h"]).reset_index(drop=True)
    return df


def build_operating_states(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    t_h = df["time_h"].to_numpy(dtype=float)

    weekly = 1.5 * np.sin(2 * np.pi * t_h / (24 * 7))
    daily = 0.7 * np.sin(2 * np.pi * t_h / 24)

    base_temp = df["T_in_C"].to_numpy(dtype=float)
    true_temp = base_temp + 0.25 * weekly + 0.15 * daily
    true_temp += rng.normal(0.0, 0.06, size=len(df))

    nominal_flow = df["m_dot_nominal_kg_s"].to_numpy(dtype=float)
    true_flow = nominal_flow * (
        1.0 + 0.015 * np.sin(2 * np.pi * t_h / (24 * 7)) + 0.010 * np.sin(2 * np.pi * t_h / 24)
    )
    true_flow += rng.normal(0.0, 0.008, size=len(df)) * nominal_flow
    true_flow = np.clip(true_flow, 0.8 * nominal_flow, 1.2 * nominal_flow)

    df = df.copy()
    df["T_true_C"] = true_temp
    df["T_true_K"] = df["T_true_C"] + 273.15
    df["m_dot_true_kg_s"] = true_flow

    return df


def recalculate_hydraulics(df: pd.DataFrame) -> pd.DataFrame:
    rho = 1000.0 - 0.12 * (df["T_true_C"] - 20.0)
    mu = 1e-3 * np.exp(-0.0025 * (df["T_true_C"] - 20.0))

    area_flow = 0.032
    diam = 0.05
    length = 3.0

    velocity = df["m_dot_true_kg_s"] / (rho * area_flow)
    reynolds = rho * velocity * diam / mu
    friction_factor = 0.0791 * reynolds ** (-0.25)
    tau_w = 0.35 * rho * velocity ** 2 * friction_factor
    dP = 2.1 * friction_factor * (length / diam) * 0.5 * rho * velocity ** 2

    df["rho_kg_m3"] = rho
    df["mu_Pa_s"] = mu
    df["u_m_s"] = velocity
    df["Re"] = reynolds
    df["tau_w_Pa"] = tau_w
    df["dP_Pa"] = dP
    return df


def simulate_dynamic_fouling(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = df.copy()
    df["Rf_m2K_W"] = 0.0

    for scenario_id, group in df.groupby("scenario_id", sort=False):
        idx = group.index.to_numpy()
        base_rf = group["Rf_m2K_W"].to_numpy(dtype=float)
        time_h = group["time_h"].to_numpy(dtype=float)
        temp_true = group["T_true_C"].to_numpy(dtype=float)
        tau_w = group["tau_w_Pa"].to_numpy(dtype=float)
        rf = np.zeros(len(group), dtype=float)

        for i in range(1, len(group)):
            growth = (
                1.2e-8
                * np.exp(0.035 * (temp_true[i] - 50.0))
                * (tau_w[i] / 0.08) ** 0.8
            )
            removal = (0.0012 + 0.006 * (tau_w[i] / 0.08)) * rf[i - 1]
            rf[i] = max(0.0, rf[i - 1] + growth - removal)
            rf[i] = 0.7 * rf[i] + 0.3 * base_rf[i]

        rf[0] = 0.0
        rf = np.clip(rf, 0.0, 5.0e-4)
        df.loc[idx, "Rf_m2K_W"] = rf

    burst_prob = 1.0 / (24 * 60)
    burst_magnitude = 0.12e-5
    burst_decay = 0.004
    for scenario_id, group in df.groupby("scenario_id", sort=False):
        idx = group.index.to_numpy()
        t_arr = group["time_h"].to_numpy(dtype=float)
        rf_base = df.loc[idx, "Rf_m2K_W"].to_numpy(dtype=float)
        burst_draw = rng.random(len(t_arr)) < burst_prob
        for j in np.where(burst_draw)[0]:
            delta_t = t_arr - t_arr[j]
            spike = burst_magnitude * np.exp(-burst_decay * np.clip(delta_t, 0.0, None))
            spike[delta_t < 0.0] = 0.0
            rf_base = rf_base + spike
        df.loc[idx, "Rf_m2K_W"] = np.clip(rf_base, 0.0, 5.0e-4)

    return df


def apply_cip_events(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    df = df.copy()
    df["cip_event"] = 0
    df["cip_effectiveness"] = 0.0

    cip_times = [2190, 4380, 6570]
    for scenario_id, group in df.groupby("scenario_id", sort=False):
        local_index = group.index.to_numpy()
        t_local = group["time_h"].to_numpy(dtype=float)
        for cip_time in cip_times:
            pos = int(np.argmin(np.abs(t_local - cip_time)))
            event_index = local_index[pos]
            removal = float(rng.uniform(0.60, 0.95))
            df.at[event_index, "cip_event"] = 1
            df.at[event_index, "cip_effectiveness"] = removal

            post_positions = np.arange(pos + 1, len(local_index))
            if len(post_positions) > 0:
                post_indices = local_index[post_positions]
                df.loc[post_indices, "Rf_m2K_W"] = (
                    df.loc[post_indices, "Rf_m2K_W"].to_numpy(dtype=float) * (1.0 - removal)
                )

    return df


def build_corrosion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    r_wall_0 = 1.0e-4
    a_corr = 3.5e-13
    e_corr = 35_000.0
    r_gas = 8.314
    t_seconds = df["time_h"].to_numpy(dtype=float) * 3600.0

    k_corr = a_corr * np.exp(-e_corr / (r_gas * df["T_true_K"].to_numpy(dtype=float)))
    df["R_wall_m2K_W"] = r_wall_0 + k_corr * t_seconds

    return df


def recompute_heat_transfer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["U_overall_W_m2K"] = 1.0 / (1.0 / df["U_clean_W_m2K"] + df["Rf_m2K_W"])
    df["Q_W"] = df["U_overall_W_m2K"] * 29.1 * 40.0
    df["thermal_efficiency"] = df["Q_W"] / df["Q_clean_W"]

    df["U_total_W_m2K"] = 1.0 / (1.0 / df["U_clean_W_m2K"] + df["Rf_m2K_W"] + df["R_wall_m2K_W"])
    df["Q_total_W"] = df["U_total_W_m2K"] * 29.1 * 40.0
    df["efficiency_total"] = df["Q_total_W"] / df["Q_clean_W"]

    df["degradation_source"] = np.where(
        (df["Rf_m2K_W"] > 1e-5) & (df["R_wall_m2K_W"] > 1.01 * 1.0e-4),
        "combined",
        np.where(df["Rf_m2K_W"] > 1e-5, "fouling", "corrosion"),
    )

    return df


def add_sensor_measurements(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    df = df.copy()
    df["T_in_measured_C"] = df["T_true_C"] + rng.normal(0.0, 0.25, size=len(df))
    df["m_dot_measured_kg_s"] = df["m_dot_true_kg_s"] * (1.0 + rng.normal(0.0, 0.015, size=len(df)))
    df["m_dot_measured_kg_s"] = np.clip(df["m_dot_measured_kg_s"], 0.82 * df["m_dot_nominal_kg_s"], 1.18 * df["m_dot_nominal_kg_s"])
    return df


def finalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        "scenario_id",
        "time_h",
        "T_true_C",
        "T_true_K",
        "T_in_C",
        "T_in_K",
        "T_in_measured_C",
        "m_dot_nominal_kg_s",
        "m_dot_true_kg_s",
        "m_dot_measured_kg_s",
        "rho_kg_m3",
        "mu_Pa_s",
        "Re",
        "u_m_s",
        "tau_w_Pa",
        "dP_Pa",
        "Rf_m2K_W",
        "U_clean_W_m2K",
        "U_overall_W_m2K",
        "U_total_W_m2K",
        "Q_W",
        "Q_clean_W",
        "Q_total_W",
        "thermal_efficiency",
        "efficiency_total",
        "fouling_factor_TEMA",
        "cip_event",
        "cip_effectiveness",
        "R_wall_m2K_W",
        "degradation_source",
    ]
    for col in ordered:
        if col not in df.columns:
            df[col] = np.nan
    return df[ordered]


def write_metadata(df: pd.DataFrame) -> None:
    meta = {
        "version": "2.0",
        "source_dataset": "Data/version 1/shell_tube_fouling_final.csv",
        "source_rows": int(len(df)),
        "source_scenarios": int(df["scenario_id"].nunique()),
        "output_rows": int(len(df)),
        "description": "Physics-aware V2 transformation built from Version 1 synthetic shell-and-tube fouling data with explicit true and measured operating states.",
        "updates": [
            "operating-condition reconstruction",
            "temperature-and-flow-coupled hydraulics",
            "dynamic deposition/removal fouling model",
            "event-based CIP and variable effectiveness",
            "temperature-dependent corrosion layer",
            "true-vs-measured sensor realism",
            "physical consistency re-calculation of U, Q, and efficiency",
        ],
        "random_seed": 42,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    df = load_v1()
    df = build_operating_states(df)
    df = recalculate_hydraulics(df)
    df = simulate_dynamic_fouling(df)
    df = apply_cip_events(df)
    df = build_corrosion(df)
    df = recompute_heat_transfer(df)
    df = add_sensor_measurements(df)
    df = finalise_columns(df)
    df.to_csv(OUTPUT_PATH, index=False)
    write_metadata(df)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Scenarios: {df['scenario_id'].nunique()}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
