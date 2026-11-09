"""
Configuration module for CHO Heat Exchanger Monitoring Platform.
Loads settings from environment variables with sensible defaults.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

# Project root: two levels up from webapp/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    """Central configuration – every tunable knob lives here."""

    # ── Paths ───────────────────────────────────────────────────────────
    database_path: str = os.getenv(
        "DATABASE_PATH",
        str(PROJECT_ROOT / "webapp" / "data" / "cho_monitoring.db"),
    )
    v1_data_path: str = os.getenv(
        "V1_DATA_PATH",
        str(PROJECT_ROOT / "data" / "version_1" / "shell_tube_fouling_final.csv"),
    )
    model_path: str = os.getenv(
        "MODEL_PATH",
        str(PROJECT_ROOT / "models" / "stage1_v2" / "xgb_raw_3_CIPAware.joblib"),
    )
    thresholds_path: str = os.getenv(
        "THRESHOLDS_PATH",
        str(PROJECT_ROOT / "config" / "thresholds.json"),
    )

    # ── Replay knobs ────────────────────────────────────────────────────
    scenario_id: str = os.getenv("SCENARIO_ID", "T90_Q5.0")
    replay_interval_seconds: float = float(
        os.getenv("REPLAY_INTERVAL_SECONDS", "1.0")
    )
    replay_loop: bool = os.getenv("REPLAY_LOOP", "true").lower() in (
        "true", "1", "yes",
    )

    # ── Feature engineering constants ───────────────────────────────────
    cip_hours: list = field(default_factory=lambda: [2190, 4380, 6570])

    # ── Model feature order (must match training pipeline exactly) ──────
    feature_names: list = field(default_factory=lambda: [
        "T_in_measured_C", "m_dot_measured_kg_s", "rho_kg_m3", "mu_Pa_s",
        "Re", "u_m_s", "tau_w_Pa", "dP_Pa",
        "hours_since_cip", "cip_cycle", "cum_thermal_cip",
        "T_roll_24h", "T_roll_168h", "m_roll_24h", "m_roll_168h",
        "dP_roll_168h",
    ])

    # ── Thresholds (loaded lazily from JSON or defaults) ────────────────
    di_warning: float = 0.000136
    di_critical: float = 0.000648

    # ── Alert dedup window (hours) ──────────────────────────────────────
    alert_dedup_hours: float = 24.0

    # ── Cleaning persistence filter ─────────────────────────────────────
    cleaning_persistence_count: int = 3

    # ── Health scoring weights ──────────────────────────────────────────
    health_w_u: float = 0.40
    health_w_q: float = 0.30
    health_w_di: float = 0.30

    # ── Risk weights ────────────────────────────────────────────────────
    risk_w_fouling: float = 0.60
    risk_w_perf: float = 0.25
    risk_w_rate: float = 0.15

    def load_thresholds(self) -> None:
        """Override di_warning / di_critical from thresholds.json if it exists."""
        p = Path(self.thresholds_path)
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                ft = data.get("fouling_thresholds", {})
                self.di_warning = float(ft.get("warning_di_fouling", data.get("warning", self.di_warning)))
                self.di_critical = float(ft.get("critical_di_fouling", data.get("critical", self.di_critical)))
            except Exception:
                pass  # keep defaults

    def ensure_dirs(self) -> None:
        """Create parent directories for the database if needed."""
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, Any]:
        """Serialisable snapshot for /api/health."""
        return {
            "database_path": self.database_path,
            "v1_data_path": self.v1_data_path,
            "model_path": self.model_path,
            "scenario_id": self.scenario_id,
            "replay_interval_seconds": self.replay_interval_seconds,
            "replay_loop": self.replay_loop,
            "di_warning": self.di_warning,
            "di_critical": self.di_critical,
        }


# Module-level singleton – import and use everywhere
settings = AppConfig()
settings.load_thresholds()
settings.ensure_dirs()
