# AI-Based Multimodal Health Screening System Using Computer Vision and Physiological Sensors
## Comprehensive Final Year Engineering Project Report & Viva Voce Guide

---

## 1. Project Abstract
Preliminary health screening and fatigue/stress pattern analysis in non-clinical settings often suffer from high diagnostic costs or unimodal inaccuracies. This project presents an **AI-Based Multimodal Health Screening System** that fuses non-invasive computer vision features (facial landmarks, bilateral symmetry, eye openness dynamics, and skin color/texture statistics) with external physiological hardware sensor measurements (heart rate, SpO2, and skin temperature from an ESP32 microcontroller).

The system executes early feature-level multimodal fusion across 19 continuous features. Rigorous data leakage prevention is enforced via subject-wise GroupShuffleSplit, ensuring zero train-test identity overlap. Classical machine learning classifiers (Random Forest, Logistic Regression, SVM, XGBoost) are trained and benchmarked across three paradigms: Model A (Hardware Sensors only), Model B (Vision only), and Model C (Multimodal Fusion). The system incorporates an unsupervised **Isolation Forest** for pattern anomaly detection, **SHAP (Shapley Additive exPlanations)** for explainable AI feature attribution, and a **Personal Baseline Manager** evaluating personalized Mahalanobis/Z-score deviations over time. A strict data quality gating rule refuses analysis if overall data quality falls below 65%. The complete system is deployed with a modern dark glassmorphic web dashboard and automated clinical-research PDF report generation via ReportLab.

---

## 2. Key Architectural Innovations & Principles

### A. Strict Modality Separation
- **Camera (Webcam)**: Computer Vision **ONLY** (MediaPipe Face Mesh, Eye Aspect Ratio, Bilateral Symmetry, Skin RGB/HSV/Lab & LBP texture). The camera **never** estimates Heart Rate, SpO2, or Temperature (preventing pseudo-scientific rPPG estimation pitfalls).
- **Hardware (ESP32)**: Physiological measurements **ONLY** (MAX30102 for PPG Heart Rate and SpO2; MLX90614 for infrared surface temperature).

### B. Mandatory 65% Quality Gating Rule
Reliable machine learning inference requires clean input signals. The system independently computes:
$$\text{Overall Quality} = 0.5 \times Q_{\text{camera}} + 0.5 \times Q_{\text{sensor}}$$
If $\text{Overall Quality} < 65\%$, analysis is strictly **refused** (`HTTP 422 Unprocessable Entity`), with clear diagnostic explanations displayed to the user.

### C. Tri-Model Comparative Evaluation
To scientifically evaluate the multimodal advantage, the system benchmarks three distinct models on identical test partitions:
- **Model A (Hardware-only)**: 3 physiological features.
- **Model B (Vision-only)**: 16 visual features.
- **Model C (Multimodal Fusion)**: 19 fused features.

---

## 3. Machine Learning & Anomaly Detection Pipeline

```
[Webcam Frame]       --> MediaPipe Face Mesh --> 16 Visual Features    ---+
                                                                          +--> [StandardScaler] --> [Multimodal Classifier (Model C)] --> Risk Prediction
[ESP32 Hardware]     --> MAX30102 / MLX90614 --> 3 Sensor Features     ---+         |
                                                                                    +--> [Isolation Forest Anomaly Detector]  --> Anomaly Score [0.0, 1.0]
                                                                                    +--> [SHAP TreeExplainer]                 --> Feature Attributions
                                                                                    +--> [Personal Baseline Manager]          --> Z-score Distance
```

### Anomaly Detection & Baseline Modeling
1. **Unsupervised Isolation Forest**: Trained exclusively on normal baseline vectors to compute a normalized anomaly score:
   $$\text{Score}_{\text{norm}} = \text{clip}\left(\frac{0.5 - s(\mathbf{x})}{0.8}, 0.0, 1.0\right)$$
2. **Personal Baseline Manager**: Maintains user-specific distribution $(\mu_i, \sigma_i)$:
   $$d_{\text{baseline}} = \sqrt{\frac{1}{D}\sum_{i=1}^D \left(\frac{x_i - \mu_i}{\sigma_i + \epsilon}\right)^2}$$
   Flagged if $d_{\text{baseline}} \ge 2.0$ standard deviations.

### Explainable AI (SHAP)
Shapley values allocate additive contributions to each feature for an individual prediction:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$
This allows the dashboard to explain exactly *why* a risk flag was generated (e.g., $+0.24$ from elevated Heart Rate, $+0.15$ from reduced Eye Openness).

---

## 4. Verification & Experimental Results
- **Pytest Suite**: 26 passed test cases covering computer vision landmark extraction, camera quality, database schema, multimodal training, subject isolation, REST API routes, and PDF report generation.
- **Test Set Accuracy**:
  - Model A (Sensors Only): Accuracy 100%, F1 1.000
  - Model B (Vision Only): Accuracy 99.0%, F1 0.986
  - Model C (Multimodal Fusion): Accuracy 100%, F1 1.000

---

## 5. Viva Voce Examination Questions & Answers (Student Preparation)

#### Q1: What is the primary novelty of your project?
**Answer**: Our system implements early feature-level multimodal fusion combining independent webcam facial computer vision metrics (symmetry, blink dynamics, skin color statistics) with physical physiological sensor readings from an ESP32 (heart rate, SpO2, temperature). Unlike traditional systems that attempt to infer vital signs from video feeds (rPPG)—which is often noisy and prone to lighting errors—we maintain strict separation of modalities, pair them at the feature level, apply unsupervised Isolation Forest anomaly detection, and provide local SHAP feature attributions with a strict 65% quality gating threshold.

#### Q2: How did you ensure there is no data leakage during ML training?
**Answer**: In health and biometric applications, samples from the same person across time are highly correlated. If samples from Person A appear in both training and test sets, the model simply memorizes the person's identity rather than generalizable health patterns. We strictly enforced subject-wise partitioning using `GroupShuffleSplit` on `subject_id`, mathematically asserting that the intersection of training subjects and test subjects is strictly empty ($\text{Train} \cap \text{Test} = \emptyset$). Furthermore, the `StandardScaler` was fitted strictly on the training partition and saved to disk.

#### Q3: Why did you implement the 65% data quality threshold?
**Answer**: In machine learning, the principle of "Garbage In, Garbage Out" is critical. If a camera frame is blurred or poorly illuminated, or if a sensor probe is improperly attached, the extracted features represent noise rather than biological signals. Rather than outputting misleading risk predictions, our system computes an independent quality score. If overall quality is below 65%, the system refuses analysis with `HTTP 422` and directs the user to adjust the camera or sensor.

#### Q4: How does your Anomaly Detection work alongside classification?
**Answer**: Classification requires labeled training data (e.g. normal vs fatigued/stressed). However, unexpected health conditions or unusual patterns may not belong to predefined classes. We deployed an unsupervised `IsolationForest` trained strictly on nominal baseline vectors. It calculates how easily a new feature vector can be isolated in feature space, outputting a continuous normalized anomaly score between 0.0 and 1.0.

#### Q5: What is SHAP and why is it essential for your system?
**Answer**: SHAP (Shapley Additive exPlanations) is a game-theoretic approach that explains the output of any machine learning model. Complex ensemble models like Random Forests are often viewed as "black boxes." SHAP calculates the exact positive or negative contribution of each individual feature (e.g. $+0.22$ from low SpO2, $+0.12$ from facial asymmetry) toward the final risk probability, providing clinicians and researchers with transparent, auditable rationales.

#### Q6: How does the ESP32 communicate with the backend?
**Answer**: The ESP32 interfaces with the MAX30102 pulse oximeter and MLX90614 infrared temperature sensor over an I2C two-wire bus (GPIO 21 SDA, GPIO 22 SCL). Over local WiFi, it formats an HTTP 1.1 JSON payload containing `session_id`, `heart_rate`, `spo2`, `temperature`, and `is_simulated` and transmits it via `POST /api/sensors/measurement` to the Flask backend.

#### Q7: What happens if physical ESP32 hardware is not plugged in?
**Answer**: The system provides an interactive simulation engine (`POST /api/sensors/simulate` and `hardware/simulator.py`) with preset physiological profiles (Normal, Elevated, Fatigue, Low Quality). This allows examiners, evaluators, and developers to test the entire multimodal fusion pipeline seamlessly without hardware dependencies.

#### Q8: What database architecture did you use?
**Answer**: We utilized SQLite with SQLAlchemy ORM abstraction. The schema includes relational models for `users`, `baseline_profiles`, `screening_sessions`, `sensor_readings`, `visual_features`, `data_quality`, `model_results`, and `anomaly_results` with foreign keys and cascade deletions.

#### Q9: What is the medical/legal disclaimer policy of your project?
**Answer**: We prominently state on the dashboard, in the database models, in API responses, and on the generated PDF reports that the system is an **academic research prototype** for pattern screening and baseline research, and is **NOT** a medical diagnostic system and must never replace clinical healthcare evaluation.

#### Q10: How are PDF reports generated?
**Answer**: Using Python's `reportlab` library, the backend dynamically builds a multi-page, publication-grade document in memory (`SimpleDocTemplate`). It compiles executive verdicts, tri-model comparison tables, physiological measurements, computer vision metrics, data quality audit tables, SHAP feature attribution rankings, baseline deviation distances, and institutional sign-offs.
