# Phase 3: Dataset & Machine Learning Pipeline Documentation

## Executive Summary & System Scope
Phase 3 implements the **Dataset Preparation, Feature Engineering, and Machine Learning Pipeline** for the AI-Based Multimodal Health Screening System prototype.

> **IMPORTANT RESEARCH DISCLAIMER**:
> This ML pipeline evaluates statistical pattern deviations and visual feature metrics. It does **NOT** estimate physiological sensor values (HR, SpO2, Temperature) from camera feeds, nor does it provide clinical medical diagnoses.

---

## 1. Feature Schema (`visual_feature_schema_v1.json`)
The feature vector is defined in `artifacts/feature_schema/visual_feature_schema_v1.json` across 16 core numerical features:
- **Eye Openness**: `left_eye_openness`, `right_eye_openness`, `mean_openness`, `eye_asymmetry`
- **Facial Symmetry**: `facial_symmetry_score`
- **Skin Color Statistics**: `skin_r_mean`, `skin_g_mean`, `skin_b_mean`, `skin_h_mean`, `skin_s_mean`, `skin_v_mean`, `skin_l_mean`, `skin_a_mean`, `skin_b_lab_mean`
- **Skin Texture**: `skin_texture_lbp` (Local Binary Pattern variance)
- **Data Quality**: `overall_quality`

---

## 2. Data Leakage Prevention & Subject-Wise Splitting
To prevent data leakage when multiple samples originate from the same individual:
- Dataset splitting uses **GroupShuffleSplit** on `subject_id`.
- The test suite explicitly asserts:
  $$\text{TRAIN SUBJECTS} \cap \text{TEST SUBJECTS} = \emptyset$$
- Ratios: **70% Train**, **15% Validation**, **15% Test**.
- Saved split files: `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`.

---

## 3. Preprocessing & Scaler Artifacts
- `StandardScaler` is fitted **exclusively on training data**.
- Fitted parameters are saved to `artifacts/scalers/visual_scaler.joblib`.
- Validation and test sets are transformed using the fitted training scaler without refitting.

---

## 4. Machine Learning Models Trained
1. **Visual Classifier (`models/visual/visual_classifier.joblib`)**:
   - Evaluated Logistic Regression, Random Forest, SVM, and XGBoost on validation F1 score.
   - Saves experiment metadata to `artifacts/experiments/experiment_visual_model.json`.
2. **Visual Anomaly Detector (`models/anomaly/visual_anomaly_detector.joblib`)**:
   - Unsupervised `IsolationForest` trained on normal baseline feature vectors.
   - Computes normalized anomaly score ($0.0$ to $1.0$) and `anomaly_flag`.
3. **Eye State Classifier (`models/eye/eye_classifier.joblib`)**:
   - Random Forest classifier trained on Eye Aspect Ratio (EAR) metrics.

---

## 5. Personal Baseline & Anomaly Engine
`PersonalBaselineManager` maintains user baseline profiles (mean $\mu$, std $\sigma$):
$$d_{\text{baseline}} = \sqrt{\sum_{i=1}^D \left(\frac{x_i - \mu_i}{\sigma_i + \epsilon}\right)^2}$$
- Flagged if baseline distance $d_{\text{baseline}} \ge 2.0$ standard deviations.

---

## 6. Inference Engine (`src/models/inference.py`)
`VisualInferenceEngine.predict(feature_dict)`:
- Rejects frames where `face_detected = False` (`prediction_status = "insufficient_data"`).
- Rejects frames where `overall_quality < 65%` (`prediction_status = "low_quality"`).
- Outputs predictions, confidence scores, anomaly scores, baseline deviation analysis, and non-diagnostic disclaimers.

---

## 7. Verification & Test Suite
All 15 system tests passed in `python -m pytest`:
- Schema validation
- Preprocessor fitting & artifact persistence
- Subject-wise leakage checks
- Data quality rejection rules
- Inference pipeline execution
