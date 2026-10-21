# Phase 6: Web Dashboard, REST API & Deployment Architecture

## 1. Overview
Phase 6 establishes the unified interactive interface, REST API orchestration, automated clinical-research PDF reporting, and deployment architecture for the **AI-Based Multimodal Health Screening System**.

---

## 2. Web Dashboard Architecture
The frontend web application provides a dark glassmorphic interface:
- **Webcam Interface**: Real-time video preview using HTML5 `navigator.mediaDevices.getUserMedia` with Canvas 2D facial alignment guides and bounding box overlays.
- **Sensor Telemetry Displays**: Live vital sign cards for Heart Rate (with heartbeat CSS pulse animation), SpO2 saturation, and Body Temperature.
- **65% Quality Gating Alert**: Explicit notification banner explaining why an analysis is refused if camera or sensor quality is inadequate.
- **Tri-Model Comparative Matrix**: Real-time display of Model A (Sensors), Model B (Vision), and Model C (Multimodal Fusion).
- **Explainable AI (SHAP) Chart**: Interactive horizontal bar chart rendering dynamic feature contributions using Chart.js.
- **1-Click PDF Report Exporter**: Direct trigger for `/api/screening/<session_id>/report.pdf`.

---

## 3. REST API Specification Table

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :---: |
| `GET` | `/` or `/dashboard` | Serve interactive Web Dashboard | 200 |
| `GET` | `/api/health` | Service and SQLite database health check | 200 |
| `POST` | `/api/sessions` | Create a new screening session | 201 |
| `GET` | `/api/sessions` | List recent screening sessions | 200 |
| `POST` | `/api/sensors/measurement` | Ingest physiological sensor reading (ESP32) | 201 |
| `POST` | `/api/sensors/simulate` | Generate simulated sensor telemetry | 201 |
| `GET` | `/api/sensors/<id>/latest` | Get latest sensor reading for session | 200 |
| `POST` | `/api/vision/analyze_frame` | Decode base64 image and extract visual features | 200 |
| `GET` | `/api/vision/<id>/latest` | Get latest visual features for session | 200 |
| `POST` | `/api/screening/run` | Execute multimodal ML inference & anomaly analysis | 200 / 422 |
| `GET` | `/api/screening/<id>/results` | Retrieve screening predictions and anomalies | 200 |
| `GET` | `/api/screening/<id>/report.pdf` | Download formal PDF screening report | 200 |
| `GET` | `/api/baselines/<user>` | Retrieve personalized baseline profile | 200 |
| `POST` | `/api/baselines/<user>/update` | Recalculate baseline from session history | 200 |

---

## 4. ReportLab PDF Clinical-Research Report Exporter
`backend/services/pdf_report_service.py` dynamically compiles a publication-grade PDF report containing:
1. Institutional Header & Audit Metadata
2. Prominent Red Disclaimer Box (Non-diagnostic academic prototype)
3. Overall Screening Status & Isolation Forest Anomaly Score
4. Tri-Model Comparative Evaluation Table
5. Physiological Measurements & Computer Vision Metrics Table
6. Data Quality & 65% Rule Verification
7. SHAP Feature Attribution Table
8. Personalized Baseline Deviation (Mahalanobis / Z-score distance)
9. Institutional Research Verification Sign-off Block

---

## 5. Running the System
```bash
# 1. Run full test suite
python -m pytest

# 2. Run Flask backend server
python -m backend.app.main

# 3. Access in browser
http://localhost:5000
```
