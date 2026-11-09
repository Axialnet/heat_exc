"""
Core Calculation & Decision Engine for CHO Heat Exchanger Monitoring.

Single source of truth for all physical calculations, model inference,
performance scoring, risk assessment, alert generation, and cleaning
recommendations.  The API and frontend must call *this* module —
never re-implementing formulas.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from webapp.config import settings

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# 1.  Water Properties
# ════════════════════════════════════════════════════════════════════════

class WaterProperties:
    """Thermophysical properties of water at atmospheric pressure."""

    @staticmethod
    def density(T_C: float) -> float:
        """
        Approximate liquid-water density [kg/m³] for 0 – 150 °C.

        Uses the simplified parabolic correlation:
            ρ = 1000 · (1 − (T_K − 277.15)² / (2 · 503 000))
        """
        T_K = T_C + 273.15
        return 1000.0 * (1.0 - (T_K - 277.15) ** 2 / (2.0 * 503000.0))

    @staticmethod
    def viscosity(T_C: float) -> float:
        """
        Approximate dynamic viscosity [Pa·s] for liquid water.

        Uses the correlation:
            μ = 2.414 × 10⁻⁵ · 10^(247.8 / (T_K − 140))
        """
        T_K = T_C + 273.15
        return 2.414e-5 * 10.0 ** (247.8 / (T_K - 140.0))


# ════════════════════════════════════════════════════════════════════════
# 2.  Feature Engineering (CIP-Aware Pipeline)
# ════════════════════════════════════════════════════════════════════════

class FeatureEngineering:
    """
    Maintains per-exchanger rolling state for the CIP-aware XGBoost
    feature pipeline.  Produces exactly 16 features in training order.

    CIP events are detected at hours 2190, 4380, 6570 within each
    scenario or whenever a ``cip_event`` flag is set in the record.
    """

    def __init__(self, cip_hours: Optional[List[int]] = None):
        self._cip_hours = set(cip_hours or settings.cip_hours)

        # CIP state
        self.hours_since_cip: float = 0.0
        self.cip_cycle: int = 0
        self.cum_thermal_cip: float = 0.0

        # Rolling windows (store raw values)
        self._T_history: Deque[float] = deque(maxlen=168)
        self._m_history: Deque[float] = deque(maxlen=168)
        self._dP_history: Deque[float] = deque(maxlen=168)

        self._prev_time_h: Optional[float] = None
        self._records_processed: int = 0

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _rolling_mean(buf: Deque[float], window: int) -> float:
        """Mean of the last *window* values, min_periods=1 semantics."""
        if not buf:
            return 0.0
        n = min(window, len(buf))
        return sum(list(buf)[-n:]) / n

    def _is_cip(self, record: Dict[str, Any]) -> bool:
        """Detect CIP from explicit flag or known hour markers."""
        if record.get("cip_event", 0) == 1:
            return True
        time_h = record.get("time_h")
        if time_h is not None and int(round(time_h)) in self._cip_hours:
            return True
        return False

    # ── main entry point ────────────────────────────────────────────────

    def process_record(self, record: Dict[str, Any]) -> Dict[str, float]:
        """
        Accept one raw reading dict (from V1 CSV row + computed rho/mu)
        and return a dict of 16 feature values keyed by feature name.
        """
        T_in = float(record["T_in_measured_C"])
        m_dot = float(record["m_dot_measured_kg_s"])
        rho = float(record["rho_kg_m3"])
        mu = float(record["mu_Pa_s"])
        Re = float(record["Re"])
        u = float(record["u_m_s"])
        tau = float(record["tau_w_Pa"])
        dP = float(record["dP_Pa"])

        # Thermal throughput increment (Q × Δt; approximate as m_dot * cp * ΔT * 1h)
        # We use a simple proxy: m_dot * T_in for cumulative tracking
        thermal_increment = abs(m_dot * T_in)

        # CIP detection
        is_cip = self._is_cip(record)
        if is_cip:
            self.cip_cycle += 1
            self.hours_since_cip = 0.0
            self.cum_thermal_cip = 0.0
        else:
            # time step size
            time_h = record.get("time_h")
            if time_h is not None and self._prev_time_h is not None:
                dt = float(time_h) - self._prev_time_h
                self.hours_since_cip += max(dt, 0)
            elif self._records_processed > 0:
                self.hours_since_cip += 1.0  # assume 1-hour step
            self.cum_thermal_cip += thermal_increment

        if record.get("time_h") is not None:
            self._prev_time_h = float(record["time_h"])

        # Update rolling buffers
        self._T_history.append(T_in)
        self._m_history.append(m_dot)
        self._dP_history.append(dP)

        self._records_processed += 1

        return {
            "T_in_measured_C": T_in,
            "m_dot_measured_kg_s": m_dot,
            "rho_kg_m3": rho,
            "mu_Pa_s": mu,
            "Re": Re,
            "u_m_s": u,
            "tau_w_Pa": tau,
            "dP_Pa": dP,
            "hours_since_cip": self.hours_since_cip,
            "cip_cycle": float(self.cip_cycle),
            "cum_thermal_cip": self.cum_thermal_cip,
            "T_roll_24h": self._rolling_mean(self._T_history, 24),
            "T_roll_168h": self._rolling_mean(self._T_history, 168),
            "m_roll_24h": self._rolling_mean(self._m_history, 24),
            "m_roll_168h": self._rolling_mean(self._m_history, 168),
            "dP_roll_168h": self._rolling_mean(self._dP_history, 168),
        }


# ════════════════════════════════════════════════════════════════════════
# 3.  Physical Calculations
# ════════════════════════════════════════════════════════════════════════

class PhysicalCalculations:
    """Pure-function helpers for fouling / performance indicators."""

    @staticmethod
    def di_fouling(U_overall: float, U_clean: float) -> float:
        """Degradation Index (fouling): 1 − U_overall / U_clean."""
        if U_clean == 0:
            return 0.0
        return 1.0 - U_overall / U_clean

    @staticmethod
    def di_total(U_total: float, U_clean: float) -> float:
        """Degradation Index (total): 1 − U_total / U_clean."""
        if U_clean == 0:
            return 0.0
        return 1.0 - U_total / U_clean

    @staticmethod
    def u_retention(U_overall: float, U_clean: float) -> float:
        """U-retention fraction: U_overall / U_clean."""
        if U_clean == 0:
            return 1.0
        return U_overall / U_clean

    @staticmethod
    def q_retention(Q: float, Q_clean: float) -> float:
        """Heat-duty retention fraction: Q / Q_clean."""
        if Q_clean == 0:
            return 1.0
        return Q / Q_clean

    @staticmethod
    def fouling_resistance(U_dirty: float, U_clean: float) -> float:
        """
        Fouling resistance R_f [m²·K/W]:
            R_f = 1/U_dirty − 1/U_clean
        """
        if U_dirty == 0 or U_clean == 0:
            return 0.0
        return 1.0 / U_dirty - 1.0 / U_clean


# ════════════════════════════════════════════════════════════════════════
# 4.  Health Scoring  (Phase 8)
# ════════════════════════════════════════════════════════════════════════

class HealthScoring:
    """
    Composite performance score  [0 – 100, 100 = healthy].

    score = 0.40 × U_retention_%  +  0.30 × Q_retention_%
          + 0.30 × (100 − min(100, DI_fouling / 0.002 × 100))
    """

    @staticmethod
    def performance_score(
        U_retention: float,
        Q_retention: float,
        DI_fouling: float,
    ) -> float:
        U_pct = U_retention * 100.0
        Q_pct = Q_retention * 100.0
        DI_pct = min(100.0, DI_fouling / 0.002 * 100.0)
        score = (
            settings.health_w_u * U_pct
            + settings.health_w_q * Q_pct
            + settings.health_w_di * (100.0 - DI_pct)
        )
        return float(np.clip(score, 0.0, 100.0))


# ════════════════════════════════════════════════════════════════════════
# 5.  Risk Engine  (Phase 9)
# ════════════════════════════════════════════════════════════════════════

class MaintenanceState(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RiskResult:
    fouling_risk: float = 0.0
    degradation_rate: float = 0.0
    ttt_warning_h: Optional[float] = None
    ttt_critical_h: Optional[float] = None
    maintenance_risk: float = 0.0
    maintenance_state: MaintenanceState = MaintenanceState.NORMAL


class RiskEngine:
    """
    Fouling-risk estimation, degradation-rate tracking, and
    time-to-threshold (TTT) projection.
    """

    def __init__(self):
        self._di_history: Deque[Tuple[float, float]] = deque(maxlen=168)
        # (time_h, DI_fouling)  — kept for slope estimation
        self._median_rate: float = 1e-9  # running median of positive rates

    def reset_history(self) -> None:
        """Clear historical points (e.g. on simulation loop reset)."""
        self._di_history.clear()
        self._median_rate = 1e-9

    # ── fouling risk (piecewise 0-100) ──────────────────────────────────

    @staticmethod
    def fouling_risk(DI: float) -> float:
        DI_warn = settings.di_warning
        DI_crit = settings.di_critical
        if DI <= DI_warn:
            return (DI / DI_warn) * 50.0 if DI_warn > 0 else 0.0
        return 50.0 + ((DI - DI_warn) / (DI_crit - DI_warn)) * 50.0

    # ── degradation rate (24-h least-squares slope) ─────────────────────

    def _update_history(self, time_h: float, DI: float) -> None:
        self._di_history.append((time_h, DI))

    def degradation_rate(self) -> float:
        """24-hour slope of DI history (least-squares)."""
        # filter to last 24 hours
        if len(self._di_history) < 2:
            return 0.0
        t_latest = self._di_history[-1][0]
        pts = [(t, d) for t, d in self._di_history if t >= t_latest - 24.0]
        if len(pts) < 2:
            return 0.0
        ts = np.array([p[0] for p in pts])
        ds = np.array([p[1] for p in pts])
        # least-squares slope
        n = len(ts)
        slope = (n * np.dot(ts, ds) - ts.sum() * ds.sum()) / max(
            n * np.dot(ts, ts) - ts.sum() ** 2, 1e-30
        )
        return float(slope)

    # ── TTT ──────────────────────────────────────────────────────────────

    @staticmethod
    def ttt(DI_current: float, rate: float, DI_target: float) -> Optional[float]:
        if rate <= 1e-9:
            return None
        remaining = DI_target - DI_current
        if remaining <= 0:
            return 0.0
        return remaining / rate

    # ── composite maintenance risk ──────────────────────────────────────

    def evaluate(
        self,
        time_h: float,
        DI_fouling: float,
        performance_score: float,
    ) -> RiskResult:
        self._update_history(time_h, DI_fouling)

        fr = self.fouling_risk(DI_fouling)
        rate = self.degradation_rate()

        # Update running median of positive rates
        if rate > 1e-9:
            self._median_rate = max(rate, 1e-9)

        perf_loss = 100.0 - performance_score  # 0-100

        rate_factor = float(np.clip(rate / (3.0 * self._median_rate), 0.0, 2.0))

        maintenance_risk = (
            settings.risk_w_fouling * fr
            + settings.risk_w_perf * perf_loss
            + settings.risk_w_rate * rate_factor * 50.0
        )
        maintenance_risk = float(np.clip(maintenance_risk, 0.0, 100.0))

        # State classification
        if maintenance_risk < 25:
            state = MaintenanceState.NORMAL
        elif maintenance_risk < 50:
            state = MaintenanceState.WATCH
        elif maintenance_risk < 75:
            state = MaintenanceState.WARNING
        else:
            state = MaintenanceState.CRITICAL

        ttt_w = self.ttt(DI_fouling, rate, settings.di_warning)
        ttt_c = self.ttt(DI_fouling, rate, settings.di_critical)

        return RiskResult(
            fouling_risk=round(fr, 4),
            degradation_rate=rate,
            ttt_warning_h=ttt_w,
            ttt_critical_h=ttt_c,
            maintenance_risk=round(maintenance_risk, 4),
            maintenance_state=state,
        )


# ════════════════════════════════════════════════════════════════════════
# 6.  Alert Engine  (Phase 10)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Alert:
    id: str
    exchanger_id: str
    timestamp: str
    severity: str  # WARNING | CRITICAL
    current_DI: float
    threshold: float
    ttt: Optional[float]
    risk_score: float
    reason: str
    recommended_action: str


class AlertEngine:
    """24-hour deduplication window per exchanger × trigger combination."""

    def __init__(self):
        # { (exchanger_id, trigger_key): last_alert_timestamp_utc }
        self._last_alert: Dict[Tuple[str, str], datetime] = {}

    def _dedup_ok(self, exchanger_id: str, trigger_key: str, now: datetime) -> bool:
        key = (exchanger_id, trigger_key)
        prev = self._last_alert.get(key)
        if prev is None:
            return True
        delta_h = (now - prev).total_seconds() / 3600.0
        return delta_h >= settings.alert_dedup_hours

    def _mark(self, exchanger_id: str, trigger_key: str, now: datetime) -> None:
        self._last_alert[(exchanger_id, trigger_key)] = now

    def evaluate(
        self,
        exchanger_id: str,
        DI: float,
        ttt_critical: Optional[float],
        risk_score: float,
        now: Optional[datetime] = None,
    ) -> List[Alert]:
        now = now or datetime.now(timezone.utc)
        alerts: List[Alert] = []

        # 1. DI >= critical
        if DI >= settings.di_critical:
            if self._dedup_ok(exchanger_id, "DI_CRIT", now):
                alerts.append(Alert(
                    id=uuid.uuid4().hex[:12],
                    exchanger_id=exchanger_id,
                    timestamp=now.isoformat(),
                    severity="CRITICAL",
                    current_DI=DI,
                    threshold=settings.di_critical,
                    ttt=ttt_critical,
                    risk_score=risk_score,
                    reason=f"DI fouling ({DI:.6f}) ≥ critical threshold ({settings.di_critical:.6f})",
                    recommended_action="Schedule immediate CIP cleaning. Inspect tube bundle.",
                ))
                self._mark(exchanger_id, "DI_CRIT", now)

        # 2. DI >= warning (strictly in [warning, critical) envelope)
        elif DI >= settings.di_warning:
            if self._dedup_ok(exchanger_id, "DI_WARN", now):
                alerts.append(Alert(
                    id=uuid.uuid4().hex[:12],
                    exchanger_id=exchanger_id,
                    timestamp=now.isoformat(),
                    severity="WARNING",
                    current_DI=DI,
                    threshold=settings.di_warning,
                    ttt=ttt_critical,
                    risk_score=risk_score,
                    reason=f"DI fouling ({DI:.6f}) ≥ warning threshold ({settings.di_warning:.6f})",
                    recommended_action="Increase monitoring frequency. Plan CIP within next maintenance window.",
                ))
                self._mark(exchanger_id, "DI_WARN", now)

        # 3. TTT <= 168 h
        if (
            ttt_critical is not None
            and ttt_critical <= 168.0
            and self._dedup_ok(exchanger_id, "TTT_168", now)
        ):
            alerts.append(Alert(
                id=uuid.uuid4().hex[:12],
                exchanger_id=exchanger_id,
                timestamp=now.isoformat(),
                severity="WARNING",
                current_DI=DI,
                threshold=settings.di_critical,
                ttt=ttt_critical,
                risk_score=risk_score,
                reason=f"Time-to-critical ≤ 168 h ({ttt_critical:.1f} h remaining)",
                recommended_action="Begin CIP planning. Critical threshold expected within 7 days.",
            ))
            self._mark(exchanger_id, "TTT_168", now)

        # 4. Risk score >= 75
        if risk_score >= 75.0 and self._dedup_ok(exchanger_id, "RISK_75", now):
            alerts.append(Alert(
                id=uuid.uuid4().hex[:12],
                exchanger_id=exchanger_id,
                timestamp=now.isoformat(),
                severity="CRITICAL",
                current_DI=DI,
                threshold=settings.di_critical,
                ttt=ttt_critical,
                risk_score=risk_score,
                reason=f"Maintenance risk score ({risk_score:.1f}) ≥ 75",
                recommended_action="Escalate to maintenance team. Immediate intervention recommended.",
            ))
            self._mark(exchanger_id, "RISK_75", now)

        return alerts


# ════════════════════════════════════════════════════════════════════════
# 7.  Cleaning Recommendation  (Phase 11)
# ════════════════════════════════════════════════════════════════════════

class RecommendationState(str, Enum):
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    PLAN_CIP = "PLAN_CIP"
    CIP_DUE_SOON = "CIP_DUE_SOON"
    URGENT_CIP = "URGENT_CIP"
    CRITICAL = "CRITICAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    DATA_STALE = "DATA_STALE"


@dataclass
class CleaningResult:
    state: RecommendationState
    text: str
    consecutive_above_warn: int = 0
    consecutive_above_crit: int = 0


class CleaningRecommendation:
    """
    Persistence filter: 3 consecutive readings above a threshold before
    escalating state.
    """

    def __init__(self, persistence_count: int = 3):
        self._n = persistence_count
        self._above_warn: int = 0
        self._above_crit: int = 0
        self._readings: int = 0

    def evaluate(
        self,
        DI: float,
        ttt_critical: Optional[float],
        risk_score: float,
    ) -> CleaningResult:
        self._readings += 1

        # Track persistence on every observation
        if DI >= settings.di_critical:
            self._above_crit += 1
            self._above_warn += 1
        elif DI >= settings.di_warning:
            self._above_crit = 0
            self._above_warn += 1
        else:
            self._above_warn = 0
            self._above_crit = 0

        if self._readings < self._n:
            return CleaningResult(
                state=RecommendationState.INSUFFICIENT_DATA,
                text="Insufficient data — collecting baseline readings.",
                consecutive_above_warn=self._above_warn,
                consecutive_above_crit=self._above_crit,
            )

        # State machine
        if self._above_crit >= self._n:
            state = RecommendationState.CRITICAL
            text = (
                "CRITICAL — DI has exceeded the critical threshold for "
                f"{self._above_crit} consecutive readings. "
                "Immediate CIP cleaning required."
            )
        elif risk_score >= 75.0:
            state = RecommendationState.URGENT_CIP
            text = (
                f"Urgent CIP recommended — maintenance risk {risk_score:.0f}/100. "
                "Schedule cleaning at earliest opportunity."
            )
        elif ttt_critical is not None and ttt_critical <= 168.0:
            state = RecommendationState.CIP_DUE_SOON
            text = (
                f"CIP due soon — projected {ttt_critical:.0f} h to critical. "
                "Plan cleaning within the next maintenance window."
            )
        elif self._above_warn >= self._n:
            state = RecommendationState.PLAN_CIP
            text = (
                f"Plan CIP — DI above warning for {self._above_warn} readings. "
                "Schedule preventive cleaning."
            )
        elif DI >= settings.di_warning:
            state = RecommendationState.MONITOR
            text = (
                "Monitor — DI approaching warning threshold. "
                "Continue monitoring and review trend."
            )
        else:
            state = RecommendationState.NORMAL
            text = "Normal operation — no cleaning action required."

        return CleaningResult(
            state=state,
            text=text,
            consecutive_above_warn=self._above_warn,
            consecutive_above_crit=self._above_crit,
        )


# ════════════════════════════════════════════════════════════════════════
# 8.  Model Inference
# ════════════════════════════════════════════════════════════════════════

class ModelInference:
    """
    Load the saved XGBoost model and run single-record inference.
    Returns ``None`` on any failure — never fabricates a value.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._path = model_path or settings.model_path
        self._load()

    def _load(self) -> None:
        p = Path(self._path)
        if not p.exists():
            logger.warning("Model file not found: %s", self._path)
            return
        try:
            import joblib
            self._model = joblib.load(p)
            logger.info("Loaded model from %s", self._path)
        except Exception:
            logger.exception("Failed to load model from %s", self._path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, features: Dict[str, float]) -> Optional[float]:
        """
        Accept a dict of 16 features (keyed by name), order them
        correctly, and return clipped DI_fouling prediction.
        """
        if self._model is None:
            return None
        try:
            row = [features[f] for f in settings.feature_names]
            X = np.array([row], dtype=np.float64)
            pred = float(self._model.predict(X)[0])
            return max(pred, 0.0)
        except Exception:
            logger.exception("Model inference failed")
            return None
