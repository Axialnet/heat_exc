/**
 * CHO Heat Exchanger 24/7 Monitoring & Predictive Maintenance Platform
 * Frontend Application Controller — Live WebSocket, Chart.js, Navigation, and Analytics.
 */

// If opened directly from file system (file:///), point to localhost:8000
const API_BASE = (window.location.protocol === "file:" || !window.location.host) ? "http://localhost:8000" : "";

class DashboardApp {
  constructor() {
    this.ws = null;
    this.reconnectTimer = null;
    this.pollTimer = null;
    this.latestState = null;
    this.activePage = "overview";
    this.historyHours = 24;
    this.charts = {};
    this.selectedSensor = null;

    // Thresholds cache
    this.thresholds = {
      warning: 0.000136,
      critical: 0.000648
    };

    this.init();
  }

  async init() {
    this.setupNavigation();
    this.setupReplayControls();
    await this.fetchInitialData();
    this.initWebSocket();
    this.setupVisibilityHandling();
  }

  // ── Navigation ─────────────────────────────────────────────────────────

  setupNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
      item.addEventListener("click", () => {
        const pageId = item.dataset.page;
        this.navigateTo(pageId);
      });
    });

    // History range selector buttons on Trends page
    document.querySelectorAll(".btn-range").forEach(btn => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".btn-range").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        this.historyHours = parseFloat(btn.dataset.hours);
        this.loadTrendsData();
      });
    });

    // Sensor modal close
    document.querySelector(".modal-close")?.addEventListener("click", () => {
      this.closeModal();
    });
    document.querySelector(".modal-backdrop")?.addEventListener("click", (e) => {
      if (e.target.classList.contains("modal-backdrop")) {
        this.closeModal();
      }
    });
  }

  navigateTo(pageId) {
    this.activePage = pageId;

    // Update nav links
    document.querySelectorAll(".nav-item").forEach(item => {
      item.classList.toggle("active", item.dataset.page === pageId);
    });

    // Update page views
    document.querySelectorAll(".page-view").forEach(view => {
      view.classList.toggle("active", view.id === `page-${pageId}`);
    });

    // Trigger lazy page-specific loads
    if (pageId === "sensors") {
      this.loadSensorsData();
    } else if (pageId === "trends") {
      this.loadTrendsData();
    } else if (pageId === "maintenance") {
      this.loadMaintenanceData();
    } else if (pageId === "alerts") {
      this.loadAlertsData();
    } else if (pageId === "system") {
      this.loadSystemData();
    }
  }

  // ── Replay Controls ────────────────────────────────────────────────────

  setupReplayControls() {
    document.getElementById("btn-pause")?.addEventListener("click", async () => {
      await fetch(`${API_BASE}/api/replay/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "pause" })
      });
      document.getElementById("btn-pause").classList.add("active");
      document.getElementById("btn-resume")?.classList.remove("active");
    });

    document.getElementById("btn-resume")?.addEventListener("click", async () => {
      await fetch(`${API_BASE}/api/replay/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "resume" })
      });
      document.getElementById("btn-resume").classList.add("active");
      document.getElementById("btn-pause")?.classList.remove("active");
    });

    document.getElementById("speed-select")?.addEventListener("change", async (e) => {
      const val = parseFloat(e.target.value);
      await fetch(`${API_BASE}/api/replay/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "speed", speed_seconds: val })
      });
    });
  }

  // ── Initial State Fetch ────────────────────────────────────────────────

  async fetchInitialData() {
    try {
      const res = await fetch(`${API_BASE}/api/current`);
      if (res.ok) {
        const data = await res.json();
        this.updateState(data);
      }
    } catch (err) {
      console.warn("Initial state fetch failed:", err);
    }
  }

  // ── WebSocket Live Telemetry ───────────────────────────────────────────

  initWebSocket() {
    const isFile = window.location.protocol === "file:" || !window.location.host;
    const host = isFile ? "localhost:8000" : window.location.host;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${host}/ws`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log("WebSocket telemetry connected.");
        this.setConnectionStatus(true);
        clearInterval(this.pollTimer);
      };

      this.ws.onmessage = (event) => {
        try {
          const packet = JSON.parse(event.data);
          this.updateState(packet);
        } catch (e) {
          console.error("Failed to parse telemetry packet:", e);
        }
      };

      this.ws.onclose = () => {
        console.warn("WebSocket closed. Attempting reconnect...");
        this.setConnectionStatus(false);
        this.startFallbackPolling();
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = setTimeout(() => this.initWebSocket(), 3000);
      };

      this.ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        this.setConnectionStatus(false);
      };
    } catch (e) {
      console.error("Failed to initialize WebSocket:", e);
      this.startFallbackPolling();
    }
  }

  startFallbackPolling() {
    if (this.pollTimer) return;
    this.pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/current`);
        if (res.ok) {
          const data = await res.json();
          this.updateState(data);
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 2000);
  }

  setConnectionStatus(connected) {
    const dot = document.getElementById("connection-indicator");
    const label = document.getElementById("connection-label");
    if (dot) {
      dot.classList.toggle("disconnected", !connected);
    }
    if (label) {
      label.textContent = connected ? "LIVE STREAM" : "POLLING / RECONNECTING";
    }
  }

  setupVisibilityHandling() {
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && (!this.ws || this.ws.readyState !== WebSocket.OPEN)) {
        this.initWebSocket();
      }
    });
  }

  // ── State Updating & Rendering ─────────────────────────────────────────

  updateState(packet) {
    this.latestState = packet;
    if (packet.thresholds) {
      this.thresholds = packet.thresholds;
    }

    // 1. Update Header Badges
    document.getElementById("header-exchanger-id").textContent = packet.exchanger_id || "HX-ONLINE";
    document.getElementById("header-time-h").textContent = `T = ${Number(packet.time_h || 0).toFixed(1)} h`;
    document.getElementById("freshness-display").textContent = `Updated ${new Date().toLocaleTimeString()}`;

    // 2. Overview KPIs
    const perfScore = packet.physical?.performance_score ?? 100.0;
    const diActual = packet.physical?.di_fouling_actual ?? 0.0;
    const diPred = packet.physical?.di_fouling_predicted;
    const riskScore = packet.risk?.maintenance_risk ?? 0.0;
    const riskState = packet.risk?.maintenance_state ?? "NORMAL";
    const tttCrit = packet.risk?.ttt_critical_h;
    const rec = packet.recommendation;

    // Health Score KPI
    const healthElem = document.getElementById("kpi-health-score");
    if (healthElem) {
      healthElem.textContent = Math.round(perfScore);
      healthElem.className = `kpi-value ${perfScore > 75 ? "state-normal" : perfScore > 50 ? "state-watch" : "state-critical"}`;
    }

    // Fouling DI KPI
    const diElem = document.getElementById("kpi-di-fouling");
    if (diElem) {
      diElem.textContent = diActual.toFixed(6);
      diElem.className = `kpi-value ${diActual >= this.thresholds.critical ? "state-critical" : diActual >= this.thresholds.warning ? "state-warning" : "state-normal"}`;
    }
    const diPredSub = document.getElementById("kpi-di-pred-sub");
    if (diPredSub) {
      diPredSub.textContent = diPred !== null && diPred !== undefined ? `Predicted: ${diPred.toFixed(6)}` : "Model: Offline";
    }

    // Risk Score KPI
    const riskElem = document.getElementById("kpi-risk-score");
    if (riskElem) {
      riskElem.textContent = `${Math.round(riskScore)} / 100`;
    }
    const riskChip = document.getElementById("kpi-risk-chip");
    if (riskChip) {
      riskChip.textContent = riskState;
      riskChip.className = `chip chip-${riskState.toLowerCase()}`;
    }

    // TTT KPI
    const tttElem = document.getElementById("kpi-ttt");
    if (tttElem) {
      if (tttCrit !== null && tttCrit !== undefined && tttCrit > 0 && tttCrit < 999) {
        tttElem.textContent = `${tttCrit.toFixed(1)} h`;
        tttElem.className = `kpi-value ${tttCrit <= 48 ? "state-critical" : tttCrit <= 168 ? "state-warning" : "state-normal"}`;
      } else {
        tttElem.textContent = "Stable";
        tttElem.className = "kpi-value state-normal";
      }
    }

    // Recommendation Banner
    if (rec) {
      const banner = document.getElementById("rec-banner");
      const title = document.getElementById("rec-title");
      const body = document.getElementById("rec-body");
      const meta = document.getElementById("rec-meta");

      if (banner) {
        banner.className = `rec-banner ${rec.state}`;
      }
      if (title) {
        title.innerHTML = `<span>RECOMMENDATION: ${rec.state.replace("_", " ")}</span>`;
      }
      if (body) {
        body.textContent = rec.text;
      }
      if (meta) {
        meta.textContent = `Persistence: ${rec.consecutive_above_warn || 0} / 3 readings above warning • Critical persistence: ${rec.consecutive_above_crit || 0} / 3`;
      }
    }

    // CIP KPI
    const cipElem = document.getElementById("kpi-cip-status");
    if (cipElem) {
      cipElem.textContent = packet.cip_event ? "CIP EXECUTING" : "Online Clean";
    }

    // Update active chart in overview
    this.appendOverviewPoint(packet.time_h, diActual, diPred, perfScore);

    // If currently viewing sensors page, update live values
    if (this.activePage === "sensors") {
      this.updateSensorCardValues(packet);
    }
  }

  // ── Overview Mini Trend Chart ──────────────────────────────────────────

  appendOverviewPoint(time_h, diActual, diPred, perfScore) {
    if (!this.charts.overview) {
      this.initOverviewChart();
      return;
    }
    const chart = this.charts.overview;
    const timeLabel = `${Number(time_h).toFixed(1)}h`;

    chart.data.labels.push(timeLabel);
    chart.data.datasets[0].data.push(diActual);
    chart.data.datasets[1].data.push(diPred);

    // Limit buffer length
    if (chart.data.labels.length > 50) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
      chart.data.datasets[1].data.shift();
    }
    chart.update("none");
  }

  initOverviewChart() {
    const ctx = document.getElementById("overview-chart")?.getContext("2d");
    if (!ctx) return;

    this.charts.overview = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Actual Fouling DI",
            data: [],
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56, 189, 248, 0.1)",
            borderWidth: 2,
            tension: 0.1,
            fill: true
          },
          {
            label: "Predicted Fouling DI",
            data: [],
            borderColor: "#f97316",
            borderDash: [4, 4],
            borderWidth: 1.5,
            tension: 0.1,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#94a3b8", font: { size: 11 } } },
          tooltip: { mode: "index", intersect: false }
        },
        scales: {
          x: {
            grid: { color: "#1e293b" },
            ticks: { color: "#64748b", font: { size: 10 } }
          },
          y: {
            grid: { color: "#1e293b" },
            ticks: { color: "#64748b", font: { size: 10 } }
          }
        }
      }
    });
  }

  // ── Sensors Page View ──────────────────────────────────────────────────

  async loadSensorsData() {
    try {
      const res = await fetch(`${API_BASE}/api/sensors`);
      if (!res.ok) return;
      const data = await res.json();
      this.renderSensorCards(data.sensors);
    } catch (e) {
      console.error("Failed to load sensors:", e);
    }
  }

  renderSensorCards(sensors) {
    const container = document.getElementById("sensors-container");
    if (!container) return;

    // Group by category
    const groups = {};
    sensors.forEach(s => {
      groups[s.category] = groups[s.category] || [];
      groups[s.category].push(s);
    });

    let html = "";
    for (const [category, list] of Object.entries(groups)) {
      html += `
        <div class="sensor-category-section">
          <div class="category-title">${category}</div>
          <div class="sensor-grid">
      `;
      list.forEach(sensor => {
        html += `
          <div class="sensor-card" data-sensor-id="${sensor.id}" onclick="app.openSensorModal('${sensor.id}', '${sensor.name}', '${sensor.unit}')">
            <div class="sensor-name">${sensor.name}</div>
            <div>
              <span class="sensor-val" id="sensor-val-${sensor.id}">${sensor.value}</span>
              <span class="sensor-unit">${sensor.unit}</span>
            </div>
            <div class="sensor-meta">
              <span class="source-badge">${sensor.source_type}</span>
              <span class="chip chip-normal">${sensor.quality}</span>
            </div>
          </div>
        `;
      });
      html += `</div></div>`;
    }
    container.innerHTML = html;
  }

  updateSensorCardValues(packet) {
    if (!packet.sensors) return;
    for (const [key, val] of Object.entries(packet.sensors)) {
      const el = document.getElementById(`sensor-val-${key}`);
      if (el) el.textContent = val;
    }
    if (packet.physical) {
      for (const [key, val] of Object.entries(packet.physical)) {
        const el = document.getElementById(`sensor-val-${key}`);
        if (el) el.textContent = val;
      }
    }
  }

  openSensorModal(sensorId, name, unit) {
    this.selectedSensor = { id: sensorId, name, unit };
    document.getElementById("modal-sensor-title").textContent = name;
    document.getElementById("modal-sensor-unit").textContent = `Engineering Unit: ${unit}`;
    document.getElementById("sensor-modal-backdrop").classList.add("open");

    // Fetch history and plot
    this.renderModalChart(sensorId, name, unit);
  }

  closeModal() {
    document.getElementById("sensor-modal-backdrop").classList.remove("open");
  }

  async renderModalChart(sensorId, name, unit) {
    const res = await fetch(`${API_BASE}/api/history?hours=168&limit=300`);
    if (!res.ok) return;
    const historyData = await res.json();
    const series = historyData.series || [];

    const labels = series.map(d => `${d.time_h.toFixed(0)}h`);
    const values = series.map(d => d[sensorId] !== undefined ? d[sensorId] : null);

    const ctx = document.getElementById("modal-chart")?.getContext("2d");
    if (!ctx) return;

    if (this.charts.modal) {
      this.charts.modal.destroy();
    }

    this.charts.modal = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: name,
          data: values,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.1)",
          borderWidth: 2,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: "#1e293b" }, ticks: { color: "#64748b" } },
          y: { grid: { color: "#1e293b" }, ticks: { color: "#64748b" } }
        }
      }
    });
  }

  // ── Trends Page View ───────────────────────────────────────────────────

  async loadTrendsData() {
    const res = await fetch(`${API_BASE}/api/history?hours=${this.historyHours}&limit=400`);
    if (!res.ok) return;
    const data = await res.json();
    const series = data.series || [];

    const labels = series.map(d => `${d.time_h.toFixed(0)}h`);
    const diActual = series.map(d => d.di_fouling_actual);
    const diPred = series.map(d => d.di_fouling_predicted);
    const health = series.map(d => d.performance_score);
    const risk = series.map(d => d.maintenance_risk);
    const uRet = series.map(d => d.u_retention ? d.u_retention * 100 : null);
    const qRet = series.map(d => d.q_retention ? d.q_retention * 100 : null);
    const Tin = series.map(d => d.T_in_measured_C);
    const mdot = series.map(d => d.m_dot_measured_kg_s);
    const dP = series.map(d => d.dP_Pa);

    // 1. DI Chart with thresholds
    this.renderLineChart("chart-di", labels, [
      { label: "Actual DI", data: diActual, borderColor: "#38bdf8", borderWidth: 2 },
      { label: "Predicted DI", data: diPred, borderColor: "#f97316", borderDash: [4, 4], borderWidth: 1.5 },
      { label: `Warning Threshold (${this.thresholds.warning.toFixed(6)})`, data: Array(labels.length).fill(this.thresholds.warning), borderColor: "#facc15", borderDash: [6, 2], pointRadius: 0, borderWidth: 1.5 },
      { label: `Critical Threshold (${this.thresholds.critical.toFixed(6)})`, data: Array(labels.length).fill(this.thresholds.critical), borderColor: "#ef4444", borderDash: [6, 2], pointRadius: 0, borderWidth: 1.5 }
    ], "Fouling Degradation Index (DI)");

    // 2. Health & Retention Chart
    this.renderLineChart("chart-health", labels, [
      { label: "Health Score (0-100)", data: health, borderColor: "#10b981", borderWidth: 2 },
      { label: "U Retention %", data: uRet, borderColor: "#3b82f6", borderWidth: 1.5 },
      { label: "Q Retention %", data: qRet, borderColor: "#a855f7", borderWidth: 1.5 }
    ], "Score / Retention (%)");

    // 3. Risk Chart
    this.renderLineChart("chart-risk", labels, [
      { label: "Maintenance Risk (0-100)", data: risk, borderColor: "#f97316", backgroundColor: "rgba(249, 115, 22, 0.1)", fill: true, borderWidth: 2 },
      { label: "Critical Risk Line (75)", data: Array(labels.length).fill(75), borderColor: "#ef4444", borderDash: [4, 4], pointRadius: 0 }
    ], "Risk Score");

    // 4. Operating Conditions (Dual Axis: Temp & Flow)
    this.renderLineChart("chart-operating", labels, [
      { label: "T_in Measured (°C)", data: Tin, borderColor: "#ef4444", borderWidth: 1.5, yAxisID: "y" },
      { label: "Flow Rate (kg/s)", data: mdot, borderColor: "#06b6d4", borderWidth: 1.5, yAxisID: "y1" }
    ], "Operating Conditions", true);

    // 5. Differential Pressure Chart
    this.renderLineChart("chart-dp", labels, [
      { label: "dP (Pa)", data: dP, borderColor: "#eab308", borderWidth: 2 }
    ], "Differential Pressure (Pa)");
  }

  renderLineChart(canvasId, labels, datasets, yTitle, dualAxis = false) {
    const ctx = document.getElementById(canvasId)?.getContext("2d");
    if (!ctx) return;

    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
    }

    const scalesConfig = {
      x: { grid: { color: "#1e293b" }, ticks: { color: "#64748b", font: { size: 10 } } },
      y: {
        grid: { color: "#1e293b" },
        ticks: { color: "#64748b", font: { size: 10 } },
        title: { display: true, text: yTitle, color: "#94a3b8", font: { size: 11 } }
      }
    };

    if (dualAxis) {
      scalesConfig.y1 = {
        position: "right",
        grid: { drawOnChartArea: false },
        ticks: { color: "#06b6d4", font: { size: 10 } },
        title: { display: true, text: "Flow Rate (kg/s)", color: "#06b6d4" }
      };
    }

    this.charts[canvasId] = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#94a3b8", font: { size: 11 } } },
          tooltip: { mode: "index", intersect: false }
        },
        scales: scalesConfig
      }
    });
  }

  // ── Maintenance Page View ──────────────────────────────────────────────

  async loadMaintenanceData() {
    try {
      const res = await fetch(`${API_BASE}/api/recommendation`);
      if (!res.ok) return;
      const data = await res.json();
      const rec = data.recommendation;
      const trace = data.decision_trace;

      document.getElementById("maint-rec-state").textContent = rec.state.replace("_", " ");
      document.getElementById("maint-rec-badge").className = `chip chip-${rec.state.includes("CRITICAL") ? "critical" : rec.state.includes("PLAN") ? "warning" : "normal"}`;
      document.getElementById("maint-rec-text").textContent = rec.text;
      document.getElementById("maint-rec-window").textContent = rec.recommended_window;

      // Decision Trace
      document.getElementById("trace-current-di").textContent = Number(trace.current_di).toFixed(6);
      document.getElementById("trace-warning-threshold").textContent = Number(trace.warning_threshold).toFixed(6);
      document.getElementById("trace-critical-threshold").textContent = Number(trace.critical_threshold).toFixed(6);
      document.getElementById("trace-rate").textContent = `${(Number(trace.degradation_rate_per_hour) * 1e6).toFixed(4)} × 10⁻⁶ / h`;
      document.getElementById("trace-ttt").textContent = trace.time_to_critical_hours ? `${Number(trace.time_to_critical_hours).toFixed(1)} hours` : "N/A (stable)";
      document.getElementById("trace-persistence").textContent = `${trace.consecutive_above_warn} warn / ${trace.consecutive_above_crit} crit (req: ${trace.persistence_requirement_steps})`;
      document.getElementById("trace-rule").textContent = trace.rule_applied;

      // CIP Events Table
      const cipRes = await fetch(`${API_BASE}/api/cip-events`);
      if (cipRes.ok) {
        const cipData = await cipRes.json();
        const tbody = document.getElementById("cip-events-tbody");
        if (tbody) {
          if (!cipData.events || cipData.events.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#64748b;">No CIP events logged yet. (Scheduled at T=2190h, 4380h, 6570h)</td></tr>`;
          } else {
            tbody.innerHTML = cipData.events.map(ev => `
              <tr>
                <td><b>${ev.time_h.toFixed(0)}h</b></td>
                <td><span class="chip chip-normal">${ev.event_type}</span></td>
                <td>${(Number(ev.cip_effectiveness) * 100).toFixed(1)}%</td>
                <td>${Number(ev.di_before || 0).toFixed(6)}</td>
                <td>${ev.details}</td>
              </tr>
            `).join("");
          }
        }
      }
    } catch (e) {
      console.error("Failed to load maintenance data:", e);
    }
  }

  // ── Alerts Page View ───────────────────────────────────────────────────

  async loadAlertsData() {
    try {
      const res = await fetch(`${API_BASE}/api/alerts`);
      if (!res.ok) return;
      const data = await res.json();
      const tbody = document.getElementById("alerts-table-tbody");
      if (!tbody) return;

      if (!data.alerts || data.alerts.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:#64748b; padding:24px;">No active alerts. Equipment operating within normal parameters.</td></tr>`;
        return;
      }

      tbody.innerHTML = data.alerts.map(alt => {
        const chipClass = alt.severity === "CRITICAL" ? "chip-critical" : "chip-warning";
        return `
          <tr>
            <td><code>${alt.id}</code></td>
            <td>${new Date(alt.timestamp_utc).toLocaleTimeString()}</td>
            <td><span class="chip ${chipClass}">${alt.severity}</span></td>
            <td><b>${alt.trigger_reason}</b></td>
            <td>${Number(alt.di_current).toFixed(6)} (thresh: ${Number(alt.threshold).toFixed(6)})</td>
            <td>${alt.recommended_action}</td>
            <td>
              ${alt.acknowledged ? '<span style="color:#10b981;">✓ Acked</span>' : `<button class="btn-ctrl" onclick="app.ackAlert('${alt.id}')">Acknowledge</button>`}
            </td>
          </tr>
        `;
      }).join("");
    } catch (e) {
      console.error("Failed to load alerts:", e);
    }
  }

  async ackAlert(alertId) {
    try {
      const res = await fetch(`${API_BASE}/api/alerts/${alertId}/acknowledge`, { method: "POST" });
      if (res.ok) {
        this.loadAlertsData();
      }
    } catch (e) {
      console.error("Failed to acknowledge alert:", e);
    }
  }

  // ── System & Model Diagnostics ─────────────────────────────────────────

  async loadSystemData() {
    try {
      // 1. Model performance
      const perfRes = await fetch(`${API_BASE}/api/model-performance`);
      if (perfRes.ok) {
        const pData = await perfRes.json();
        const tA = pData.offline_validation.track_a;
        document.getElementById("track-a-r2").textContent = `R² = ${tA.test_r2}`;
        document.getElementById("track-a-mae").textContent = `MAE = ${tA.test_mae}`;
        document.getElementById("track-a-disc").textContent = tA.disclosure;

        const tB = pData.offline_validation.track_b;
        const bContainer = document.getElementById("track-b-cards");
        if (bContainer) {
          bContainer.innerHTML = tB.map(m => `
            <div style="background:#0f172a; padding:12px; border-radius:6px; margin-bottom:8px; border:1px solid #334155;">
              <div style="font-weight:700; color:#f97316;">${m.name}</div>
              <div style="font-size:16px; font-weight:800; color:#fff;">Test R² = ${m.test_r2} | MAE = ${m.test_mae}</div>
              <div style="font-size:11px; color:#94a3b8;">Leakage: ${m.leakage_mechanism}</div>
            </div>
          `).join("");
        }

        // Live eval
        const live = pData.live_evaluation;
        document.getElementById("live-eval-count").textContent = `${live.sample_count} observations`;
        document.getElementById("live-eval-mae").textContent = live.rolling_mae !== null ? `Rolling MAE: ${live.rolling_mae}` : "Evaluating...";
      }

      // 2. Health & Ingestion Telemetry
      const healthRes = await fetch(`${API_BASE}/api/health`);
      if (healthRes.ok) {
        const hData = await healthRes.json();
        const feed = hData.data_feed;
        document.getElementById("sys-records-processed").textContent = `${feed.records_processed.toLocaleString()} records`;
        document.getElementById("sys-cursor-progress").textContent = `Row ${feed.cursor_row.toLocaleString()} / ${feed.total_rows.toLocaleString()} (${feed.progress_percent}%)`;
        document.getElementById("sys-freshness").textContent = feed.freshness_seconds !== null ? `${feed.freshness_seconds} seconds` : "N/A";
        document.getElementById("sys-db-readings").textContent = `${hData.database.counts.sensor_readings_count || 0} rows`;
        document.getElementById("sys-db-alerts").textContent = `${hData.database.counts.alerts_count || 0} events`;
      }
    } catch (e) {
      console.error("Failed to load system data:", e);
    }
  }
}

// Instantiate global app controller
let app;
document.addEventListener("DOMContentLoaded", () => {
  app = new DashboardApp();
});
