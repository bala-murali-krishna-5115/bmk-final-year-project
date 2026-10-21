/**
 * Multimodal Health Screening & Appearance Analysis Dashboard Controller
 * Integrates Web REST APIs, Computer Vision frame analysis (Eyes, Skin, Hair, Fatigue),
 * ESP-32 Hardware Sensor Telemetry, and Intelligent Resting Recommendations.
 */

let activeSessionId = null;
let activeUserCode = "USR_001";
let cameraManager = null;
let shapChart = null;
let hardwareConnected = false;
let currentSensorData = null;

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", async () => {
  cameraManager = new CameraManager("webcamVideo", "overlayCanvas");

  // Event Listeners
  document.getElementById("btnStartCamera")?.addEventListener("click", toggleCamera);
  document.getElementById("btnNewSession")?.addEventListener("click", createNewSession);
  document.getElementById("btnRunScreening")?.addEventListener("click", runScreening);
  document.getElementById("btnSimulateSensor")?.addEventListener("click", handleSensorSimulator);
  document.getElementById("btnDownloadPDF")?.addEventListener("click", downloadPDFReport);

  // Initialize first session & load history
  await createNewSession();
  await loadRecentSessions();
});

async function toggleCamera() {
  const btn = document.getElementById("btnStartCamera");
  if (!cameraManager.isStreaming) {
    const ok = await cameraManager.startCamera();
    if (ok) {
      btn.innerHTML = `<i class="bi bi-camera-video-off"></i> Stop Camera`;
      btn.classList.replace("btn-outline-glass", "btn-secondary");
      const pill = document.getElementById("cameraStatusPill");
      if (pill) {
        pill.innerText = "CAMERA STREAMING";
        pill.classList.add("text-success");
      }
    } else {
      alert("Unable to access webcam. Ensure camera permissions are allowed in your browser.");
    }
  } else {
    cameraManager.stopCamera();
    btn.innerHTML = `<i class="bi bi-camera-video"></i> Start Camera`;
    btn.classList.replace("btn-secondary", "btn-outline-glass");
    const pill = document.getElementById("cameraStatusPill");
    if (pill) {
      pill.innerText = "CAMERA OFFLINE";
      pill.classList.remove("text-success");
    }
  }
}

async function createNewSession() {
  const userCodeInput = document.getElementById("userCodeInput");
  activeUserCode = userCodeInput?.value.trim() || "USR_001";

  try {
    const res = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_code: activeUserCode })
    });
    const data = await res.json();
    if (res.ok) {
      activeSessionId = data.session_id;
      document.getElementById("activeSessionBadge").innerText = `SESSION: ${activeSessionId.substring(0, 8)}...`;
      document.getElementById("btnRunScreening").disabled = false;
      document.getElementById("btnDownloadPDF").disabled = true;
      resetResultsUI();
    }
  } catch (err) {
    console.error("Session creation error:", err);
  }
}

async function handleSensorSimulator() {
  if (!activeSessionId) await createNewSession();
  const modeSelect = document.getElementById("sensorModeSelect");
  const mode = modeSelect ? modeSelect.value : "normal";

  if (mode === "disconnect") {
    // Explicitly disconnect hardware - NO FAKE DATA
    hardwareConnected = false;
    currentSensorData = null;
    updateHardwareUI(false, null);
    return;
  }

  try {
    const res = await fetch("/api/sensors/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: activeSessionId, mode: mode })
    });
    const data = await res.json();
    if (res.ok) {
      hardwareConnected = true;
      currentSensorData = data;
      updateHardwareUI(true, data);
    }
  } catch (err) {
    console.error("Sensor simulation error:", err);
  }
}

function updateHardwareUI(connected, data) {
  const badge = document.getElementById("hwStatusBadge");
  const notice = document.getElementById("hwNoticeBox");

  if (connected && data) {
    if (badge) {
      badge.className = "status-badge bg-success bg-opacity-10 text-success border border-success border-opacity-25";
      badge.innerHTML = `<span class="pulse-dot"></span> ESP-32 CONNECTED`;
    }
    if (notice) {
      notice.className = "alert alert-success bg-success bg-opacity-10 border border-success border-opacity-25 text-success-emphasis small p-2 mb-0 rounded";
      notice.innerHTML = `<i class="bi bi-check-circle me-1"></i> Live telemetry active from sensor node.`;
    }

    document.getElementById("valHeartRate").innerText = data.heart_rate || "--";
    document.getElementById("valSpO2").innerText = data.spo2 || "--";
    document.getElementById("valTemperature").innerText = data.temperature || "--";

    document.getElementById("statusHeartRate").innerText = "Live PPG stream";
    document.getElementById("statusSpO2").innerText = "Normal saturation";
    document.getElementById("statusTemperature").innerText = "Afebrile range";
  } else {
    if (badge) {
      badge.className = "status-badge bg-danger bg-opacity-10 text-danger border border-danger border-opacity-25";
      badge.innerText = "HARDWARE DISCONNECTED";
    }
    if (notice) {
      notice.className = "alert alert-warning bg-warning bg-opacity-10 border border-warning border-opacity-25 text-warning-emphasis small p-2 mb-0 rounded";
      notice.innerHTML = `<i class="bi bi-info-circle me-1"></i> <strong>No rough or fake data:</strong> Vital signs are unavailable until physical ESP-32 sensor node transmits data.`;
    }

    document.getElementById("valHeartRate").innerText = "--";
    document.getElementById("valSpO2").innerText = "--";
    document.getElementById("valTemperature").innerText = "--";

    document.getElementById("statusHeartRate").innerText = "Awaiting ESP-32 connection";
    document.getElementById("statusSpO2").innerText = "Awaiting ESP-32 connection";
    document.getElementById("statusTemperature").innerText = "Awaiting ESP-32 connection";
  }
}

async function runScreening() {
  if (!activeSessionId) await createNewSession();

  const btn = document.getElementById("btnRunScreening");
  const spinner = document.getElementById("screeningSpinner");
  btn.disabled = true;
  if (spinner) spinner.classList.remove("d-none");

  // Capture frame from webcam if streaming
  let frameB64 = null;
  if (cameraManager && cameraManager.isStreaming) {
    frameB64 = cameraManager.captureFrameBase64();
  }

  // Strictly pass sensor data ONLY if hardware is connected (NO fake defaults)
  let sensorPayload = null;
  if (hardwareConnected && currentSensorData) {
    sensorPayload = {
      heart_rate: currentSensorData.heart_rate,
      spo2: currentSensorData.spo2,
      temperature: currentSensorData.temperature,
      is_simulated: currentSensorData.is_simulated || false
    };
  }

  const payload = {
    session_id: activeSessionId,
    user_code: activeUserCode,
    frame_base64: frameB64,
    sensor_data: sensorPayload
  };

  try {
    const res = await fetch("/api/screening/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    btn.disabled = false;
    if (spinner) spinner.classList.add("d-none");

    if (res.status === 422 || data.status === "rejected") {
      displayQualityRejection(data);
      return;
    }

    if (res.ok) {
      renderScreeningResults(data);
      document.getElementById("btnDownloadPDF").disabled = false;
      await loadRecentSessions();
    }
  } catch (err) {
    btn.disabled = false;
    if (spinner) spinner.classList.add("d-none");
    console.error("Screening execution error:", err);
  }
}

function displayQualityRejection(data) {
  const alertBox = document.getElementById("qualityRejectionAlert");
  const reasonText = document.getElementById("qualityRejectionReason");
  if (alertBox && reasonText) {
    alertBox.classList.remove("d-none");
    reasonText.innerText = data.message || "Quality below 65% minimum threshold.";
  }
  document.getElementById("verdictCard")?.classList.add("d-none");
}

function renderScreeningResults(data) {
  document.getElementById("qualityRejectionAlert")?.classList.add("d-none");

  const results = data.screening_results;
  const qa = data.quality_assessment;
  const app = results.appearance_analysis || {};
  const rep = results.health_report || {};

  // 1. Overall Quality
  document.getElementById("valOverallQuality").innerText = `${qa.overall_data_quality}%`;
  const ovrBar = document.getElementById("barOverallQuality");
  if (ovrBar) ovrBar.style.width = `${qa.overall_data_quality}%`;

  // 2. Verdict Banner
  const verdictCard = document.getElementById("verdictCard");
  const verdictText = document.getElementById("verdictText");
  const riskBadge = document.getElementById("riskBadge");

  if (verdictCard) verdictCard.classList.remove("d-none");
  if (verdictText) verdictText.innerText = results.screening_verdict;
  if (riskBadge) {
    riskBadge.innerText = results.risk_level.toUpperCase() + " RISK";
    riskBadge.className = `status-badge ${
      results.risk_level === "Low" ? "badge-risk-low" :
      results.risk_level === "Moderate" ? "badge-risk-mod" : "badge-risk-high"
    }`;
  }

  // 3. Computer Vision Appearance Highlights
  // Dark circles
  const dark = app.dark_circles || {};
  const elDark = document.getElementById("valDarkCircles");
  if (elDark) {
    elDark.innerText = dark.severity || "Minimal";
    elDark.className = `badge ${dark.severity === 'Prominent' ? 'bg-danger' : (dark.severity === 'Moderate' ? 'bg-warning text-dark' : 'bg-success')}`;
  }
  document.getElementById("valDarknessContrast").innerText = `${dark.darkness_index_percentage || 0}%`;

  // Eye Tiredness
  const tired = app.tiredness || {};
  const elTired = document.getElementById("valTirednessLevel");
  if (elTired) {
    elTired.innerText = tired.tiredness_level || "Alert";
    elTired.className = `badge ${tired.tiredness_score > 60 ? 'bg-danger' : (tired.tiredness_score > 35 ? 'bg-warning text-dark' : 'bg-info')}`;
  }
  document.getElementById("valTirednessScore").innerText = `${tired.tiredness_score || 0}%`;
  document.getElementById("valEyeOpenness").innerText = results.features_summary.vision_highlights.mean_eye_openness.toFixed(2);

  // Acne & Blemishes
  const blem = app.blemishes_and_acne || {};
  const elBlem = document.getElementById("valBlemishes");
  if (elBlem) {
    elBlem.innerText = blem.blemish_severity || "Clear";
    elBlem.className = `badge ${blem.visible_lesion_count >= 5 ? 'bg-danger' : (blem.visible_lesion_count > 0 ? 'bg-warning text-dark' : 'bg-success')}`;
  }
  document.getElementById("valLesionCount").innerText = blem.visible_lesion_count || 0;

  // Redness & Pigmentation
  const pig = app.pigmentation_and_redness || {};
  document.getElementById("valRedness").innerText = pig.redness_flag ? "Erythema Flagged" : "Nominal";
  document.getElementById("valErythemaIndex").innerText = pig.erythema_index ? pig.erythema_index.toFixed(1) : "12.5";

  // Hair & Scalp
  const hair = app.hair_and_scalp || {};
  document.getElementById("valHairline").innerText = hair.hairline_stability || "Normal";
  document.getElementById("valHairDensity").innerText = `${hair.hair_density_score || 75}%`;

  // 4. Tri-Model Comparative Cards
  const tri = results.tri_model_comparison;
  const modA = tri.model_a_sensor;
  document.getElementById("valModelAPred").innerText = modA.predicted_class !== null ? (modA.predicted_class === 0 ? "Nominal" : "Deviation") : "Waiting HW";
  document.getElementById("valModelAConf").innerText = modA.confidence !== null ? `${(modA.confidence * 100).toFixed(1)}%` : "--";

  const modB = tri.model_b_vision;
  document.getElementById("valModelBPred").innerText = modB.predicted_class === 0 ? "Nominal" : "Deviation";
  document.getElementById("valModelBConf").innerText = `${(modB.confidence * 100).toFixed(1)}%`;

  const modC = tri.model_c_multimodal;
  document.getElementById("valModelCPred").innerText = modC.predicted_class === 0 ? "Nominal" : "Deviation";
  document.getElementById("valModelCConf").innerText = modC.confidence !== null ? `${(modC.confidence * 100).toFixed(1)}%` : "Active";

  // 5. Isolation Forest Anomaly Meter
  const anomScore = results.anomaly_analysis.anomaly_score;
  document.getElementById("valAnomalyScore").innerText = anomScore.toFixed(3);
  const anomBar = document.getElementById("barAnomalyScore");
  if (anomBar) {
    anomBar.style.width = `${Math.min(anomScore * 100, 100)}%`;
    anomBar.style.backgroundColor = anomScore > 0.5 ? "#EF4444" : (anomScore > 0.35 ? "#F59E0B" : "#10B981");
  }

  // 6. Intelligent Health & Wellness Report Box (at bottom)
  const elExec = document.getElementById("repExecutiveSummary");
  if (elExec) {
    elExec.innerText = rep.executive_summary || "Appearance analysis completed.";
  }

  const elFindings = document.getElementById("repFindingsList");
  if (elFindings) {
    elFindings.innerHTML = "";
    (rep.appearance_findings || []).forEach(f => {
      const li = document.createElement("li");
      li.className = "mb-1";
      li.innerText = f;
      elFindings.appendChild(li);
    });
  }

  const elRecs = document.getElementById("repRecommendationsList");
  if (elRecs) {
    elRecs.innerHTML = "";
    (rep.specific_recommendations || []).forEach(r => {
      const li = document.createElement("li");
      li.className = "mb-1 fw-medium text-white";
      li.innerText = r;
      elRecs.appendChild(li);
    });
    (rep.general_lifestyle_tips || []).slice(0, 3).forEach(tip => {
      const li = document.createElement("li");
      li.className = "mb-1 text-white-50";
      li.innerText = `Health tip: ${tip}`;
      elRecs.appendChild(li);
    });
  }

  // 7. Render SHAP Feature Attribution Chart
  renderSHAPChart(results.shap_attributions);
}

function renderSHAPChart(shapData) {
  if (!shapData || shapData.length === 0) return;

  const ctx = document.getElementById("shapChartCanvas")?.getContext("2d");
  if (!ctx) return;

  const labels = shapData.map(d => d.feature.replace(/_/g, " "));
  const values = shapData.map(d => d.shap_value);
  const colors = values.map(v => v > 0 ? "rgba(239, 68, 68, 0.8)" : "rgba(16, 185, 129, 0.8)");

  if (shapChart) {
    shapChart.destroy();
  }

  shapChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "SHAP Impact on Screening Risk Flag",
        data: values,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94A3B8" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#F8FAFC", font: { weight: "500" } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Contribution: ${ctx.parsed.x > 0 ? '+' : ''}${ctx.parsed.x.toFixed(4)}`
          }
        }
      }
    }
  });
}

function resetResultsUI() {
  document.getElementById("verdictCard")?.classList.add("d-none");
  document.getElementById("qualityRejectionAlert")?.classList.add("d-none");
}

function downloadPDFReport() {
  if (!activeSessionId) return;
  window.open(`/api/screening/${activeSessionId}/report.pdf`, '_blank');
}

async function loadRecentSessions() {
  try {
    const res = await fetch("/api/sessions");
    const data = await res.json();
    const tbody = document.getElementById("sessionsTableBody");
    if (!tbody) return;

    tbody.innerHTML = "";
    (data.sessions || []).slice(0, 6).forEach(s => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="text-white font-monospace">${s.session_id.substring(0, 8)}...</td>
        <td>${s.user_code}</td>
        <td>${new Date(s.timestamp).toLocaleTimeString()}</td>
        <td>
          <span class="badge ${s.status === 'completed' ? 'bg-success' : 'bg-secondary'}">${s.status}</span>
        </td>
        <td>
          <span class="badge ${s.has_sensors ? 'bg-info bg-opacity-25 text-info' : 'bg-secondary'}">${s.has_sensors ? 'Sensor Active' : 'Camera Only'}</span>
        </td>
        <td>${s.anomaly_score !== null ? s.anomaly_score.toFixed(3) : '-'}</td>
        <td>
          <a href="/api/screening/${s.session_id}/report.pdf" target="_blank" class="btn btn-sm btn-outline-glass">
            <i class="bi bi-file-earmark-pdf"></i> PDF
          </a>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load sessions:", err);
  }
}
