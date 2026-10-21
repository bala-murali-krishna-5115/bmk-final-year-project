# AI-Based Multimodal Health Screening System Using Computer Vision and Physiological Sensors
### Final Year Engineering Project & Academic Research Prototype

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange.svg)](https://developers.google.com/mediapipe)
[![Tests Passing](https://img.shields.io/badge/Tests-26%20Passed-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Project%20Status-100%25%20Completed-success.svg)]()

> **IMPORTANT DISCLAIMER**:
> This system is an **academic research prototype** designed for preliminary health pattern screening, baseline modeling, and pattern anomaly research. It is **NOT** a medical diagnostic system and does **NOT** provide medical diagnoses or replace professional medical evaluation.

---

## System Overview
This project presents an end-to-end, full-stack multimodal screening architecture that fuses two independent, complementary data streams:
1. **Webcam Computer Vision**: Facial landmarks, bilateral symmetry index, eye openness dynamics (EAR), and skin region color (RGB, HSV, Lab) & texture statistics (LBP) extracted via MediaPipe Face Mesh and OpenCV.
2. **External Hardware Sensors**: Physiological vital measurements (Heart Rate, SpO2, and Skin Temperature) sampled via an ESP32 microcontroller with MAX30102 and MLX90614 sensors over I2C and WiFi.

The system performs **feature-level early fusion** (19 continuous features), personal baseline deviation tracking via Z-scores, unsupervised anomaly detection via **Isolation Forest**, explainable AI breakdowns via **SHAP (Shapley Additive exPlanations)**, and interactive visualization on a modern dark glassmorphic web dashboard with automated publication-grade PDF report exports.

---

## Technical Architecture

```
                                 [ Screening Session ]
                                          |
                +-------------------------+-------------------------+
                |                                                   |
      [ Webcam Video Feed ]                               [ ESP32 Hardware Node ]
                |                                                   |
     MediaPipe Face Mesh (478)                             MAX30102 & MLX90614
                |                                                   |
     Camera Quality Engine (>=65%)                        Sensor Quality Engine (>=65%)
                |                                                   |
     16 Visual Features (Eyes,                            3 Vital Signs (Heart Rate,
     Symmetry, Skin Color/Texture)                        SpO2, Body Temperature)
                |                                                   |
                +-------------------------+-------------------------+
                                          |
                         [ Early Feature-Level Fusion ]
                               (19-D Feature Vector)
                                          |
                 +------------------------+------------------------+
                 |                        |                        |
        [ Tri-Model Matrix ]     [ Isolation Forest ]     [ SHAP Explainer ]
          Model A: Sensors         Anomaly Score [0, 1]     Feature Attributions
          Model B: Vision
          Model C: Multimodal
                 |                        |                        |
                 +------------------------+------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
              [ SQLite Database ]                 [ ReportLab Service ]
            SQLAlchemy ORM Models               Publication-Grade PDF
                        |                                   |
                        +-----------------+-----------------+
                                          |
                          [ Interactive Web Dashboard ]
                            HTML5, CSS3, JavaScript,
                             Bootstrap 5 & Chart.js
```

---

## Project Directory Structure

```
├── backend/
│   ├── app/
│   │   ├── config.py             # System configuration & physiological thresholds
│   │   ├── main.py               # Flask Application Factory & server runner
│   │   └── schemas/              # Pydantic schemas (Session, Sensor, Vision)
│   ├── database/
│   │   ├── connection.py         # SQLAlchemy engine & session factory
│   │   ├── init_db.py            # Database initialization script
│   │   └── models.py             # Legacy database models
│   ├── models/
│   │   └── database_models.py    # Primary SQLAlchemy ORM models (8 relational tables)
│   ├── routes/
│   │   ├── baseline_routes.py    # Personal baseline profile endpoints
│   │   ├── screening_routes.py   # Unified multimodal screening & PDF exporter
│   │   ├── sensor_routes.py      # ESP32 sensor ingestion & interactive simulator
│   │   ├── session_routes.py     # Screening session lifecycle management
│   │   └── vision_routes.py      # Frame decoding & visual feature extraction
│   └── services/
│       ├── baseline_service.py   # Historical baseline aggregation & Z-score distance
│       ├── multimodal_service.py # Tri-model comparative inference & SHAP engine
│       ├── pdf_report_service.py # ReportLab publication-grade PDF report exporter
│       └── quality_service.py    # Camera & sensor quality evaluator (65% rule)
├── computer_vision/
│   ├── camera/camera_manager.py  # OpenCV webcam capture manager
│   ├── eyes/eye_analyzer.py      # Eye Aspect Ratio (EAR) & blink analyzer
│   ├── face/landmarks.py         # MediaPipe Face Mesh landmark extractor
│   ├── quality/camera_quality.py # Brightness, blur & stability quality engine
│   ├── skin/skin_analyzer.py     # Forehead/cheek skin color & LBP texture
│   ├── symmetry/symmetry_analyzer.py # Bilateral facial symmetry scoring
│   └── feature_vector.py         # Unified CV feature extractor
├── frontend/
│   ├── index.html                # Dark glassmorphic web dashboard
│   └── static/
│       ├── css/dashboard.css     # Custom glassmorphism design system
│       └── js/
│           ├── camera.js         # Webcam getUserMedia & canvas mesh overlay
│           └── dashboard.js      # Dashboard controller, simulator & Chart.js
├── hardware/
│   ├── circuit_diagram.md        # Wiring specifications, pinouts & pull-up details
│   ├── esp32_firmware.ino        # Production-ready Arduino C++ ESP32 firmware
│   └── simulator.py              # Standalone CLI telemetry simulator
├── models/
│   ├── anomaly/                  # Visual Anomaly Detector (.joblib)
│   ├── eye/                      # Eye state classifier (.joblib)
│   ├── face_landmarker.task      # MediaPipe Face Landmarker model binary
│   ├── multimodal/               # Multimodal Classifier, Anomaly & SHAP Explainer
│   ├── sensors/                  # Sensor-only Classifier (.joblib)
│   └── visual/                   # Vision-only Classifier (.joblib)
├── src/
│   ├── data/                     # Dataset cleaning, inspection & splitters
│   ├── features/                 # Preprocessors & baseline managers
│   └── models/                   # Training modules & inference engines
├── tests/                        # 26 automated Pytest test cases (100% pass)
└── docs/                         # Comprehensive Phase reports & Viva Voce guide
```

---

## Tri-Model Experimental Benchmark

All models were evaluated on identical subject-wise test sets with zero data leakage:

| Model Architecture | Input Modality | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A** | Physiological Sensors Only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Model B** | Computer Vision Facial Only | 0.990 | 1.000 | 0.972 | 0.986 | 0.999 |
| **Model C** | **Multimodal Early Fusion** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

---

## Quick Start Guide

### 1. Environment Installation
Ensure Python 3.11+ is installed. Install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Database Initialization
Initialize the SQLite relational schema:
```bash
python -m backend.database.init_db
```

### 3. Run Automated Pytest Suite
Verify all 26 unit and integration tests:
```bash
python -m pytest
```

### 4. Start Backend Server & Web Dashboard
Start the Flask application:
```bash
python -m backend.app.main
```
Open your browser and navigate to:
```
http://localhost:5000
```

### 5. (Optional) Run Hardware Telemetry Simulator
If testing without physical ESP32 hardware, either use the dashboard's built-in simulator toggle or run:
```bash
python -m hardware.simulator --session-id <SESSION_ID> --mode normal
```

---

## Documentation Index
- [System Architecture](file:///c:/Users/balam/OneDrive/Desktop/Sarvagna%20Innovation%20Internship/Final%20Year%20Project/docs/architecture.md)
- [Phase 3: Dataset & Visual ML Pipeline](file:///c:/Users/balam/OneDrive/Desktop/Sarvagna%20Innovation%20Internship/Final%20Year%20Project/docs/PHASE_3.md)
- [Phase 4: Multimodal Fusion & Benchmark](file:///c:/Users/balam/OneDrive/Desktop/Sarvagna%20Innovation%20Internship/Final%20Year%20Project/docs/PHASE_4_MULTIMODAL_FUSION.md)
- [Phase 5: ESP32 Hardware Integration](file:///c:/Users/balam/OneDrive/Desktop/Sarvagna%20Innovation%20Internship/Final%20Year%20Project/docs/PHASE_5_HARDWARE_INTEGRATION.md)
- [Phase 6: Web Dashboard & Deployment](file:///c:/Users/balam/OneDrive/Desktop/Sarvagna%20Innovation%20Internship/Final%20Year%20Project/docs/PHASE_6_DASHBOARD_AND_DEPLOYMENT.md)
- [Final Year Project Comprehensive Report & Viva Voce Guide](file:///c:/Users/balam/OneDrive/Desktop/Sarvagna%20Innovation%20Internship/Final%20Year%20Project/docs/PROJECT_REPORT_SUMMARY.md)

---

## License & Research Attribution
Academic Final Year Engineering Project - Developed for preliminary screening pattern research.
Licensed for academic research and educational evaluation.
