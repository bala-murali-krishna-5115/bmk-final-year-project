# System Architecture

## Overview
The **AI-Based Multimodal Health Screening System** is an academic research prototype designed to fuse webcam-derived computer vision features with external physiological hardware measurements (ESP32).

## Core Modules & Design Principles
1. **Separation of Modalities**:
   - Camera: Computer vision ONLY (Eye openness, Facial symmetry, Forehead/Cheek skin stats, Lighting/Camera quality). Camera DOES NOT estimate HR, SpO2, or temperature.
   - Hardware: Sensor readings ONLY (ESP32 via JSON REST payload).
2. **Data Quality Layer**:
   - Both camera quality and sensor quality are evaluated independently.
   - Analysis is refused if `overall_data_quality < 65%`.
3. **Database Architecture**:
   - SQLite database with SQLAlchemy ORM abstraction (`users`, `screening_sessions`, `sensor_readings`, `visual_features`, `data_quality`, `model_results`, `anomaly_results`, `baseline_profiles`).
4. **Machine Learning & Fusion**:
   - Classical ML (Random Forest, XGBoost, Logistic Regression, SVM).
   - Comparative evaluation across Model A (Hardware-only), Model B (Vision-only), and Model C (Multimodal Fusion).
   - Anomaly detection via Isolation Forest and baseline deviation.
   - Explainable AI via SHAP feature importance.
