# Phase 4: Multimodal Feature Fusion & Machine Learning Pipeline

## 1. Executive Summary
Phase 4 implements **early feature-level multimodal fusion**, combining webcam-derived computer vision facial dynamics with external hardware physiological sensor measurements (ESP32).

> **ACADEMIC RESEARCH DISCLAIMER**:
> This software is an academic engineering research prototype designed for preliminary health pattern screening, baseline modeling, and anomaly research. It does **NOT** provide medical diagnoses and must not replace professional clinical evaluation.

---

## 2. Multimodal Feature Vector Architecture
The multimodal feature vector fuses **19 continuous numerical features** across two non-overlapping modalities:

### A. Computer Vision Features (16 Dimensions)
- **Eye Dynamics**: `left_eye_openness`, `right_eye_openness`, `mean_openness`, `eye_asymmetry`
- **Facial Symmetry**: `facial_symmetry_score`
- **Skin Color Statistics**: `skin_r_mean`, `skin_g_mean`, `skin_b_mean`, `skin_h_mean`, `skin_s_mean`, `skin_v_mean`, `skin_l_mean`, `skin_a_mean`, `skin_b_lab_mean`
- **Skin Texture**: `skin_texture_lbp` (Local Binary Pattern variance)
- **Camera Data Quality**: `overall_quality`

### B. Hardware Sensor Features (3 Dimensions)
- `heart_rate`: Photoplethysmography pulse measurement (bpm)
- `spo2`: Blood oxygen saturation percentage (%)
- `temperature`: Surface/skin infrared temperature (°C)

---

## 3. Data Leakage Prevention (Subject-Wise Partitioning)
- Splitting is performed strictly using **GroupShuffleSplit** on `subject_id`.
- The test suite validates zero subject overlap:
  $$\text{Train Subjects} \cap \text{Test Subjects} = \emptyset$$
  $$\text{Train Subjects} \cap \text{Validation Subjects} = \emptyset$$
- Ratios: **70% Train** (288 samples), **15% Validation** (110 samples), **15% Test** (102 samples).

---

## 4. Preprocessing & Scaler Persistence
- `StandardScaler` is fitted **exclusively** on training set partitions:
  - `artifacts/scalers/sensor_scaler.joblib`
  - `artifacts/scalers/visual_scaler.joblib`
  - `artifacts/scalers/multimodal_scaler.joblib`
- Testing and validation partitions are scaled strictly using fitted training parameters without refitting.

---

## 5. Tri-Model Comparative Benchmark
The system implements a rigorous tri-model comparative evaluation across identical test splits:

| Architecture | Input Modality | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A** | Sensors Only (`HR`, `SpO2`, `Temp`) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Model B** | Vision Only (16 Facial Features) | 0.990 | 1.000 | 0.972 | 0.986 | 0.999 |
| **Model C** | **Multimodal Early Fusion (All 19)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

*Benchmark metadata saved in `artifacts/experiments/multimodal_benchmark_report.json`.*

---

## 6. Multimodal Anomaly Detection (Isolation Forest)
- Unsupervised `IsolationForest` trained strictly on nominal baseline training vectors.
- Computes normalized anomaly score $[0.0, 1.0]$:
  $$\text{Score}_{\text{norm}} = \text{clip}\left(\frac{0.5 - s(\mathbf{x})}{0.8}, 0.0, 1.0\right)$$
- Anomaly flag triggered when $\text{Score}_{\text{norm}} \ge 0.50$.
- Saved model: `models/multimodal/multimodal_anomaly_detector.joblib`.

---

## 7. Explainable AI: SHAP Feature Attribution
- Fitted `shap.TreeExplainer` on Multimodal Model C (Random Forest).
- Produces local additive feature importance values for each screening prediction, providing explainable breakdowns of which physiological or visual features elevated or lowered pattern deviation risk.
- Saved explainer: `models/multimodal/multimodal_shap_explainer.joblib`.
