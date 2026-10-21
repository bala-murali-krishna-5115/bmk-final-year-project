import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional

from src.features.preprocessing import FeaturePreprocessor, DEFAULT_FEATURE_COLS
from src.models.train_visual_model import VisualModelTrainer
from src.models.train_anomaly_detector import VisualAnomalyDetector
from src.features.baseline_manager import PersonalBaselineManager

logger = logging.getLogger(__name__)

class VisualInferenceEngine:
    """
    Unified Inference Engine for Phase 3.
    Validates input quality and schema, applies fitted scaling, and computes
    model predictions, confidence, and IsolationForest anomaly scores.
    """

    def __init__(
        self,
        scaler_path: str = "artifacts/scalers/visual_scaler.joblib",
        model_path: str = "models/visual/visual_classifier.joblib",
        anomaly_path: str = "models/anomaly/visual_anomaly_detector.joblib"
    ):
        self.preprocessor = FeaturePreprocessor.load(scaler_path) if os.path.exists(scaler_path) else None
        
        if os.path.exists(model_path):
            artifact = joblib.load(model_path)
            self.model = artifact["model"]
            self.model_name = artifact.get("model_name", "visual_classifier")
        else:
            self.model = None
            self.model_name = "not_loaded"

        self.anomaly_detector = VisualAnomalyDetector.load(anomaly_path) if os.path.exists(anomaly_path) else None

    def predict(self, feature_dict: Dict[str, Any], baseline_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes a raw/nested visual feature vector dictionary and computes inference.
        """
        # 1. Quality & Face Detection Checks
        quality_data = feature_dict.get("quality", {})
        overall_quality = quality_data.get("overall_quality", 0.0)

        face_data = feature_dict.get("face", {})
        if not face_data.get("face_detected", False):
            return {
                "prediction_status": "insufficient_data",
                "message": "No face detected in camera frame.",
                "data_quality_score": overall_quality,
                "disclaimer": "Academic research prototype - NOT a medical diagnosis."
            }

        if overall_quality < 65.0:
            return {
                "prediction_status": "low_quality",
                "message": f"Camera quality score ({overall_quality}%) below 65% threshold.",
                "data_quality_score": overall_quality,
                "disclaimer": "Academic research prototype - NOT a medical diagnosis."
            }

        # 2. Extract flat feature values
        flat_features = {}
        # Eyes
        eyes = feature_dict.get("eyes", {})
        flat_features["left_eye_openness"] = eyes.get("left_eye_openness", 0.0)
        flat_features["right_eye_openness"] = eyes.get("right_eye_openness", 0.0)
        flat_features["mean_openness"] = eyes.get("mean_openness", 0.0)
        flat_features["eye_asymmetry"] = eyes.get("eye_asymmetry", 0.0)

        # Symmetry
        symmetry = feature_dict.get("symmetry", {})
        flat_features["facial_symmetry_score"] = symmetry.get("facial_symmetry_score", 0.0)

        # Skin
        skin = feature_dict.get("skin", {})
        flat_features["skin_r_mean"] = skin.get("skin_r_mean", 0.0)
        flat_features["skin_g_mean"] = skin.get("skin_g_mean", 0.0)
        flat_features["skin_b_mean"] = skin.get("skin_b_mean", 0.0)
        flat_features["skin_h_mean"] = skin.get("skin_h_mean", 0.0)
        flat_features["skin_s_mean"] = skin.get("skin_s_mean", 0.0)
        flat_features["skin_v_mean"] = skin.get("skin_v_mean", 0.0)
        flat_features["skin_l_mean"] = skin.get("skin_l_mean", 0.0)
        flat_features["skin_a_mean"] = skin.get("skin_a_mean", 0.0)
        flat_features["skin_b_lab_mean"] = skin.get("skin_b_lab_mean", 0.0)
        flat_features["skin_texture_lbp"] = skin.get("skin_texture_lbp", 0.0)
        flat_features["overall_quality"] = overall_quality

        # Check missing features
        missing_keys = [k for k in DEFAULT_FEATURE_COLS if k not in flat_features]
        if missing_keys:
            return {
                "prediction_status": "missing_features",
                "message": f"Missing required feature keys: {missing_keys}",
                "data_quality_score": overall_quality,
                "disclaimer": "Academic research prototype - NOT a medical diagnosis."
            }

        # 3. Scaling & Preprocessing
        if self.preprocessor is None or not self.preprocessor.is_fitted:
            # Fallback if artifacts not yet saved
            X_scaled = np.array([[flat_features[k] for k in DEFAULT_FEATURE_COLS]])
        else:
            df_single = pd.DataFrame([flat_features])
            X_scaled = self.preprocessor.transform(df_single)

        # 4. Model Prediction
        prediction = 0
        confidence = 0.5
        if self.model is not None:
            prediction = int(self.model.predict(X_scaled)[0])
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_scaled)[0]
                confidence = float(np.max(probs))

        # 5. Anomaly Detection
        anomaly_score = 0.0
        anomaly_flag = False
        if self.anomaly_detector is not None and self.anomaly_detector.is_fitted:
            flags, scores = self.anomaly_detector.predict_anomaly(X_scaled)
            anomaly_flag = bool(flags[0])
            anomaly_score = float(scores[0])

        # 6. Baseline Deviation Analysis
        baseline_analysis = PersonalBaselineManager.calculate_baseline_deviation(flat_features, baseline_profile or {})

        return {
            "prediction_status": "success",
            "feature_schema_version": "1.0",
            "model_algorithm": self.model_name,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "anomaly_flag": anomaly_flag,
            "data_quality_score": overall_quality,
            "baseline_analysis": baseline_analysis,
            "disclaimer": "Academic research prototype - NOT a medical diagnostic device."
        }

import joblib
