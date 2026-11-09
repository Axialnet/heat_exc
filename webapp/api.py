"""
FastAPI Application and REST/WebSocket API for CHO Heat Exchanger Monitoring.
Provides endpoints for equipment status, sensors, historical trends, alerts,
maintenance recommendations, model validation metrics, and live WebSocket streaming.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.config import settings
from webapp.database import Database
from webapp.ingestion import IngestionWorker

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("webapp.api")

# Global singleton handles
db = Database()
worker = IngestionWorker(db)
_worker_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown management."""
    global _worker_task
    logger.info("Starting up CHO Monitoring Platform...")
    # 1. Initialize SQLite schema
    await db.init_db()

    # 2. Load dataset and restore cursor
    try:
        worker.load_dataset()
        await worker.restore_cursor()
    except Exception as e:
        logger.error("Failed to initialize dataset: %s", e)

    # 3. Start autonomous background ingestion worker
    _worker_task = asyncio.create_task(worker.run())
    logger.info("Ingestion worker task launched successfully.")

    yield

    # Shutdown
    logger.info("Shutting down CHO Monitoring Platform...")
    worker.is_running = False
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutdown complete.")


app = FastAPI(
    title="CHO Heat Exchanger 24/7 Monitoring & Predictive Maintenance",
    version="1.0.0",
    description="Industrial condition assessment, fouling prognostics, and automated CIP decision engine.",
    lifespan=lifespan
)

# Enable CORS for local development and dashboard embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Root UI Route ───────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the single-page application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({
        "status": "ONLINE",
        "message": "CHO Monitoring API is active. Frontend is under /static/index.html."
    })


# ── 1. System Health & Freshness Endpoint ──────────────────────────────

@app.get("/api/health")
async def get_health():
    """
    Expose service connectivity, data feed freshness, model readiness,
    and background worker telemetry.
    """
    now = datetime.now(timezone.utc)
    freshness_s = None
    is_stale = False

    if worker.last_reading_time_utc:
        last_dt = datetime.fromisoformat(worker.last_reading_time_utc)
        freshness_s = max(0.0, (now - last_dt).total_seconds())
        # If no new reading for >30s while running and not paused, mark as stale
        if freshness_s > 30.0 and not worker.is_paused:
            is_stale = True

    db_stats = await db.get_database_stats()

    return {
        "status": "DEGRADED" if is_stale else "HEALTHY",
        "service": "CHO Predictive Maintenance Platform",
        "timestamp_utc": now.isoformat(),
        "exchanger_id": worker.exchanger_id,
        "scenario_id": worker.scenario_id,
        "data_feed": {
            "source": "Version 1 Physics Dataset (Replay)",
            "last_reading_time_utc": worker.last_reading_time_utc,
            "freshness_seconds": round(freshness_s, 1) if freshness_s is not None else None,
            "is_stale": is_stale,
            "worker_running": worker.is_running,
            "worker_paused": worker.is_paused,
            "cursor_row": worker.cursor_index,
            "total_rows": worker._total_rows,
            "progress_percent": round(worker.cursor_index / max(1, worker._total_rows) * 100.0, 1),
            "replay_interval_seconds": worker.replay_interval,
            "records_processed": worker.records_processed,
            "error_count": worker.error_count,
            "last_error": worker.last_error
        },
        "model": {
            "track_a_loaded": worker.model.is_loaded,
            "model_path": settings.model_path,
            "architecture": "CIP-Aware XGBoost (16 Features)"
        },
        "database": {
            "path": settings.database_path,
            "counts": db_stats
        }
    }


# ── 2. Current State Snapshot ──────────────────────────────────────────

@app.get("/api/current")
async def get_current_state():
    """
    Return authoritative current operating state, KPIs, sensor readings,
    risk parameters, and cleaning recommendations.
    """
    # Prefer in-memory hot cache from worker, fallback to DB
    if worker.last_state_packet:
        return worker.last_state_packet

    row = await db.get_latest_state(worker.exchanger_id)
    if not row:
        raise HTTPException(status_code=503, detail="No readings processed yet. Service initializing.")
    return row


# ── 3. Sensor Inventory & Quality Metadata ─────────────────────────────

@app.get("/api/sensors")
async def get_sensors():
    """
    Structured dictionary of every available measurement in Version 1,
    with unit, category, value, and engineering source classification.
    """
    latest = await db.get_latest_state(worker.exchanger_id)
    if not latest:
        raise HTTPException(status_code=503, detail="Sensor telemetry unavailable.")

    sensors = [
        # Temperatures
        {
            "id": "T_in_measured_C",
            "name": "Hot-Side Inlet Temperature (Measured)",
            "category": "Temperatures",
            "value": round(float(latest.get("T_in_measured_C", 0)), 2),
            "unit": "°C",
            "source_type": "DIRECT_MEASUREMENT",
            "quality": "VALID",
            "limits": {"min": 40.0, "max": 140.0}
        },
        # Flow
        {
            "id": "m_dot_measured_kg_s",
            "name": "Hot-Side Mass Flow Rate (Measured)",
            "category": "Flows",
            "value": round(float(latest.get("m_dot_measured_kg_s", 0)), 3),
            "unit": "kg/s",
            "source_type": "DIRECT_MEASUREMENT",
            "quality": "VALID",
            "limits": {"min": 1.0, "max": 10.0}
        },
        # Hydraulics
        {
            "id": "Re",
            "name": "Reynolds Number",
            "category": "Hydraulics",
            "value": round(float(latest.get("Re", 0)), 1),
            "unit": "dimensionless",
            "source_type": "CALCULATED_PHYSICS",
            "quality": "VALID",
            "limits": {"min": 2000, "max": 20000}
        },
        {
            "id": "u_m_s",
            "name": "Tube Flow Velocity",
            "category": "Hydraulics",
            "value": round(float(latest.get("u_m_s", 0)), 4),
            "unit": "m/s",
            "source_type": "CALCULATED_PHYSICS",
            "quality": "VALID",
            "limits": {"min": 0.05, "max": 0.50}
        },
        {
            "id": "tau_w_Pa",
            "name": "Wall Shear Stress",
            "category": "Hydraulics",
            "value": round(float(latest.get("tau_w_Pa", 0)), 4),
            "unit": "Pa",
            "source_type": "CALCULATED_PHYSICS",
            "quality": "VALID",
            "limits": {"min": 0.01, "max": 0.50}
        },
        {
            "id": "dP_Pa",
            "name": "Differential Pressure Drop",
            "category": "Hydraulics",
            "value": round(float(latest.get("dP_Pa", 0)), 2),
            "unit": "Pa",
            "source_type": "DIRECT_MEASUREMENT",
            "quality": "VALID",
            "limits": {"min": 20.0, "max": 300.0}
        },
        # Thermal Performance
        {
            "id": "U_overall_W_m2K",
            "name": "Overall Heat Transfer Coefficient",
            "category": "Thermal Performance",
            "value": round(float(latest.get("U_overall_W_m2K", 0)), 2),
            "unit": "W/(m²·K)",
            "source_type": "SIMULATED_STATE",
            "quality": "VALID",
            "limits": {"min": 500.0, "max": 2500.0}
        },
        {
            "id": "U_clean_W_m2K",
            "name": "Clean Reference Heat Transfer Coefficient",
            "category": "Thermal Performance",
            "value": round(float(latest.get("U_clean_W_m2K", 0)), 2),
            "unit": "W/(m²·K)",
            "source_type": "CALCULATED_PHYSICS",
            "quality": "VALID",
            "limits": {"min": 500.0, "max": 2500.0}
        },
        {
            "id": "Q_W",
            "name": "Transferred Thermal Duty",
            "category": "Thermal Performance",
            "value": round(float(latest.get("Q_W", 0)), 1),
            "unit": "W",
            "source_type": "SIMULATED_STATE",
            "quality": "VALID",
            "limits": {"min": 500000.0, "max": 3000000.0}
        },
        {
            "id": "thermal_efficiency",
            "name": "Thermal Efficiency",
            "category": "Thermal Performance",
            "value": round(float(latest.get("thermal_efficiency", 0)), 4),
            "unit": "ratio",
            "source_type": "SIMULATED_STATE",
            "quality": "VALID",
            "limits": {"min": 0.70, "max": 1.0}
        },
        # Fouling Resistance & Targets
        {
            "id": "Rf_m2K_W",
            "name": "Fouling Thermal Resistance",
            "category": "Fouling State",
            "value": float(latest.get("Rf_m2K_W", 0)),
            "unit": "m²·K/W",
            "source_type": "SIMULATED_STATE",
            "quality": "VALID",
            "limits": {"min": 0.0, "max": 0.001}
        },
        {
            "id": "di_fouling_actual",
            "name": "Degradation Index (Actual Target)",
            "category": "Fouling State",
            "value": round(float(latest.get("di_fouling_actual", 0)), 6),
            "unit": "dimensionless",
            "source_type": "CALCULATED_PHYSICS",
            "quality": "VALID",
            "limits": {"min": 0.0, "max": 0.01}
        },
        {
            "id": "di_fouling_predicted",
            "name": "Degradation Index (Model Predicted)",
            "category": "Fouling State",
            "value": round(float(latest.get("di_fouling_predicted", 0)), 6) if latest.get("di_fouling_predicted") is not None else None,
            "unit": "dimensionless",
            "source_type": "MODEL_INFERENCE",
            "quality": "VALID" if latest.get("di_fouling_predicted") is not None else "UNAVAILABLE",
            "limits": {"min": 0.0, "max": 0.01}
        },
        # Derived Fluid Properties
        {
            "id": "rho_kg_m3",
            "name": "Water Density (Calculated at T_in)",
            "category": "Derived Fluid Properties",
            "value": round(float(latest.get("rho_kg_m3", 0)), 2),
            "unit": "kg/m³",
            "source_type": "CALCULATED_PHYSICS",
            "quality": "VALID",
            "limits": {"min": 900.0, "max": 1000.0}
        },
        {
            "id": "mu_Pa_s",
            "name": "Water Dynamic Viscosity (Calculated at T_in)",
            "category": "Derived Fluid Properties",
            "value": float(f"{float(latest.get('mu_Pa_s', 0)):.6e}"),
            "unit": "Pa·s",
            "source_type": "CALCULATED_PHYSICS",
            "quality": "VALID",
            "limits": {"min": 0.0001, "max": 0.002}
        }
    ]

    return {
        "timestamp_utc": latest.get("timestamp_utc"),
        "time_h": latest.get("time_h"),
        "sensor_count": len(sensors),
        "sensors": sensors
    }


# ── 4. Historical Time-Series Trends ───────────────────────────────────

@app.get("/api/history")
async def get_history(
    hours: float = Query(default=168.0, ge=1.0, le=8760.0, description="Window size in hours"),
    limit: int = Query(default=300, ge=10, le=1000, description="Max points to return")
):
    """
    Retrieve time-ordered observations for multi-chart dashboard views.
    Includes warning and critical threshold references.
    """
    history = await db.get_history(worker.exchanger_id, hours=hours, limit=limit)
    return {
        "exchanger_id": worker.exchanger_id,
        "window_hours": hours,
        "count": len(history),
        "thresholds": {
            "warning": settings.di_warning,
            "critical": settings.di_critical
        },
        "series": history
    }


# ── 5. Alerts & Deduplication Event Log ────────────────────────────────

@app.get("/api/alerts")
async def get_alerts(
    severity: str = Query(default="ALL", description="Filter by ALL, WARNING, or CRITICAL"),
    limit: int = Query(default=50, ge=1, le=200)
):
    """Fetch event-driven alerts with deduplication tracking."""
    alerts = await db.get_alerts(exchanger_id=worker.exchanger_id, severity=severity, limit=limit)
    return {
        "count": len(alerts),
        "dedup_window_hours": settings.alert_dedup_hours,
        "alerts": alerts
    }


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an active operator alert."""
    success = await db.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert ID not found")
    return {"status": "ACKNOWLEDGED", "alert_id": alert_id}


# ── 6. Maintenance & Cleaning Recommendation Trace ─────────────────────

@app.get("/api/recommendation")
async def get_recommendation():
    """
    Provide full decision-tree trace for current cleaning recommendation:
    current condition, DI trends, persistence verification, and action required.
    """
    latest = await db.get_latest_state(worker.exchanger_id)
    if not latest:
        raise HTTPException(status_code=503, detail="Operational data unavailable.")

    di_curr = float(latest.get("di_fouling_actual", 0))
    rate = float(latest.get("degradation_rate", 0))
    ttt_crit = latest.get("ttt_critical_h")
    state = latest.get("recommendation_state", "NORMAL")
    text = latest.get("recommendation_text", "Normal operation.")

    return {
        "exchanger_id": worker.exchanger_id,
        "timestamp_utc": latest.get("timestamp_utc"),
        "time_h": latest.get("time_h"),
        "recommendation": {
            "state": state,
            "text": text,
            "cleaning_required": state in ["PLAN_CIP", "CIP_DUE_SOON", "URGENT_CIP", "CRITICAL"],
            "urgency": (
                "IMMEDIATE" if state in ["URGENT_CIP", "CRITICAL"]
                else "SCHEDULED" if state in ["PLAN_CIP", "CIP_DUE_SOON"]
                else "MONITOR"
            ),
            "recommended_window": (
                "Within 24-48 Hours" if state in ["URGENT_CIP", "CRITICAL"]
                else "Within 72 Hours" if state == "PLAN_CIP"
                else "Next Turnaround"
            )
        },
        "decision_trace": {
            "current_di": di_curr,
            "warning_threshold": settings.di_warning,
            "critical_threshold": settings.di_critical,
            "degradation_rate_per_hour": rate,
            "time_to_critical_hours": ttt_crit,
            "consecutive_above_warn": latest.get("consecutive_above_warn", 0),
            "consecutive_above_crit": latest.get("consecutive_above_crit", 0),
            "persistence_requirement_steps": settings.cleaning_persistence_count,
            "rule_applied": (
                "Rule 1: Sustained DI >= Critical for >=3 steps" if state == "CRITICAL"
                else "Rule 2: Maintenance Risk Score >= 75" if state == "URGENT_CIP"
                else "Rule 3: Projected TTT to critical <= 168h" if state == "CIP_DUE_SOON"
                else "Rule 4: Sustained DI >= Warning for >=3 steps" if state == "PLAN_CIP"
                else "Rule 5: Instantaneous DI >= Warning (awaiting persistence)" if state == "MONITOR"
                else "Rule 6: Normal envelope"
            )
        }
    }


# ── 7. Model Performance & Scientific Transparency ─────────────────────

@app.get("/api/model-performance")
async def get_model_performance():
    """
    Scientifically transparent model performance presentation:
    Separates offline validation benchmarks from live inference telemetry.
    Explicitly discloses Track B target-proxy leakage.
    """
    # 1. Verified Offline Validation Benchmarks from STAGE_1_FINAL_REPORT.md
    offline = {
        "track_a": {
            "name": "Track A — Production Model (Leakage-Free Standard)",
            "algorithm": "CIP-Aware XGBoost (16 Features)",
            "validation_r2": 0.2686,
            "test_r2": 0.2457,
            "test_mae": 0.000124,
            "feature_count": 16,
            "data_source": "Version 2 Physics-Corrected Dataset (80 Scenarios, 700,800 rows)",
            "split": "Scenario-Level Generalization (60 Train / 10 Val / 10 Test)",
            "scientific_status": "VALID_PRODUCTION_STANDARD",
            "disclosure": (
                "Strictly causal non-invasive sensor measurements and operational logs only. "
                "No target-derived variables. The R² ≈ 0.246 represents the authentic physical bound "
                "due to unobservable stochastic Poisson bursts (magnitude 12.3× mean Rf) and 15mK thermal SNR."
            )
        },
        "track_b": [
            {
                "name": "DEMO-A: Lagged State Proxy",
                "test_r2": 0.9914,
                "test_mae": 0.0000035,
                "leakage_mechanism": "Autoregressive target leakage (lag-1 Rf and rolling DI)"
            },
            {
                "name": "DEMO-B: Simulator State Variables",
                "test_r2": 0.9992,
                "test_mae": 0.0000026,
                "leakage_mechanism": "Direct internal simulation variables (Rf, U_overall, Q, thermal efficiency)"
            },
            {
                "name": "DEMO-C: Direct Algebraic Proxy",
                "test_r2": 0.9980,
                "test_mae": 0.0000048,
                "leakage_mechanism": "Direct algebraic identity (1 - U_overall / U_clean)"
            }
        ],
        "track_b_warning": (
            "SHOWCASE ONLY: Track B models use internal simulation variables not available "
            "on physical instrumentation. Near-perfect R² ≈ 0.999 is tautological target reconstruction "
            "and MUST NOT be claimed as real-world predictive accuracy."
        )
    }

    # 2. Live evaluation against Version 1 reference ground truth
    # Compute rolling MAE over the last 100 observations in DB
    live_eval = {
        "status": "EVALUATING_AGAINST_V1_REFERENCE",
        "sample_count": 0,
        "rolling_mae": None,
        "note": "Evaluated against Version 1 synthetic reference target (DI_actual)."
    }

    async with aiosqlite.connect(settings.database_path) as c:
        c.row_factory = aiosqlite.Row
        async with c.execute(
            """
            SELECT di_fouling_actual, di_fouling_predicted
            FROM derived_metrics
            WHERE di_fouling_predicted IS NOT NULL
            ORDER BY id DESC LIMIT 100
            """
        ) as cur:
            rows = await cur.fetchall()
            if len(rows) >= 10:
                diffs = [abs(r["di_fouling_actual"] - r["di_fouling_predicted"]) for r in rows]
                live_eval["sample_count"] = len(rows)
                live_eval["rolling_mae"] = round(sum(diffs) / len(diffs), 6)

    return {
        "offline_validation": offline,
        "live_evaluation": live_eval,
        "inference_engine": {
            "model_loaded": worker.model.is_loaded,
            "model_path": settings.model_path,
            "features_required": settings.feature_names
        }
    }


# ── 8. CIP Events Log ──────────────────────────────────────────────────

@app.get("/api/cip-events")
async def get_cip_events():
    """Retrieve historical Clean-in-Place cleaning events."""
    events = await db.get_cip_events(worker.exchanger_id)
    return {
        "exchanger_id": worker.exchanger_id,
        "count": len(events),
        "events": events
    }


# ── 9. Replay Control (Industrial Simulation Demo) ──────────────────────

class ReplayControlRequest(BaseModel):
    action: str  # pause | resume | restart | speed
    speed_seconds: Optional[float] = None


@app.post("/api/replay/control")
async def control_replay(req: ReplayControlRequest):
    """
    Interactive simulation control: pause, resume, accelerate, or reset replay.
    """
    if req.action == "pause":
        worker.pause()
        return {"status": "PAUSED"}
    elif req.action == "resume":
        worker.resume()
        return {"status": "RESUMED"}
    elif req.action == "restart":
        worker.reset_cursor()
        return {"status": "RESTARTED", "cursor_row": 0}
    elif req.action == "speed":
        if req.speed_seconds is not None:
            worker.set_speed(req.speed_seconds)
            return {"status": "SPEED_UPDATED", "interval_seconds": worker.replay_interval}
        raise HTTPException(status_code=400, detail="speed_seconds required")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")


# ── 10. WebSocket Telemetry Stream ─────────────────────────────────────

@app.websocket("/ws")
async def websocket_telemetry(websocket: WebSocket):
    """
    High-frequency live WebSocket channel. Pushes complete state packet
    to connected browser dashboards upon every 1-minute ingested cycle.
    """
    await worker.connect_client(websocket)
    try:
        while True:
            # Keep-alive ping/pong listener
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        worker.disconnect_client(websocket)
    except Exception as e:
        logger.warning("WebSocket connection exception: %s", e)
        worker.disconnect_client(websocket)
