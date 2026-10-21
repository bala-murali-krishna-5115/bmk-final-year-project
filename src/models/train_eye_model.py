import os
import json
import joblib
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

class EyeStateModelTrainer:
    """
    Trains eye-state classifier (Open / Closed) on landmark Eye Aspect Ratio (EAR) features.
    """

    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=50, random_state=random_state)

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Trains Eye state classifier and evaluates test metrics."""
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()

        metrics = {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm
        }
        logger.info(f"Eye state classifier trained (Accuracy: {acc:.4f}, F1: {f1:.4f})")
        return metrics

    def save(self, output_path: str = "models/eye/eye_classifier.joblib"):
        """Saves Eye state classifier artifact."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({"model": self.model}, output_path)
        logger.info(f"Saved eye state model artifact to {output_path}")

    @classmethod
    def load(cls, model_path: str = "models/eye/eye_classifier.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Eye model artifact not found at {model_path}")
        artifact = joblib.load(model_path)
        return artifact["model"]
