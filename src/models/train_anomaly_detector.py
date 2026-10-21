import os
import json
import joblib
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class VisualAnomalyDetector:
    """
    Unsupervised Anomaly Detector using Isolation Forest.
    Learns normal baseline distribution and flags statistical pattern deviations.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray):
        """Fits Isolation Forest on normal baseline training vectors."""
        self.model.fit(X_train)
        self.is_fitted = True
        logger.info(f"IsolationForest anomaly detector fitted on {len(X_train)} samples.")

    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates anomaly flags (-1 for anomaly, 1 for normal) and normalized anomaly scores [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before prediction.")

        raw_scores = self.model.score_samples(X)  # Higher is normal, lower is anomalous
        preds = self.model.predict(X)  # 1 normal, -1 anomaly

        # Normalize score_samples into range 0.0 (normal) to 1.0 (extreme anomaly)
        # Empirical min/max bounds for IsolationForest score_samples [-0.8, -0.2]
        norm_scores = np.clip((0.5 - raw_scores) / 0.8, 0.0, 1.0)
        anomaly_flags = (preds == -1)

        return anomaly_flags, norm_scores

    def save(self, output_path: str = "models/anomaly/visual_anomaly_detector.joblib"):
        """Saves fitted IsolationForest model artifact."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        artifact = {
            "model": self.model,
            "contamination": self.contamination,
            "is_fitted": self.is_fitted
        }
        joblib.dump(artifact, output_path)

        meta_path = "artifacts/experiments/experiment_anomaly_detector.json"
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({
                "model_name": "visual_anomaly_detector",
                "algorithm": "IsolationForest",
                "contamination": self.contamination,
                "feature_schema_version": "1.0",
                "disclaimer": "Statistical pattern anomaly detector - NOT a medical diagnostic model."
            }, f, indent=2)

        logger.info(f"Saved anomaly detector artifact to {output_path}")

    @classmethod
    def load(cls, model_path: str = "models/anomaly/visual_anomaly_detector.joblib") -> "VisualAnomalyDetector":
        """Loads fitted anomaly detector artifact."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Anomaly detector artifact not found at {model_path}")

        artifact = joblib.load(model_path)
        instance = cls(contamination=artifact["contamination"])
        instance.model = artifact["model"]
        instance.is_fitted = artifact["is_fitted"]
        return instance
