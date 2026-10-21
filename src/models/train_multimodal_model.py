import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import shap

from src.features.multimodal_preprocessing import (
    SENSOR_FEATURE_COLS,
    VISION_FEATURE_COLS,
    MULTIMODAL_FEATURE_COLS
)

logger = logging.getLogger(__name__)

class MultimodalModelTrainer:
    """
    Trains and compares:
    - Model A: Physiological Sensor Model (Hardware-only)
    - Model B: Computer Vision Facial Feature Model (Vision-only)
    - Model C: Multimodal Fusion Model (Vision + Sensor Early Fusion)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Fits model and evaluates performance metrics."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = 0.5
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm
        }

    def train_all_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sensor_scaler: Any,
        vision_scaler: Any,
        multimodal_scaler: Any,
        target_col: str = "label"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Trains Model A, Model B, Model C on identical subject-split partitions.
        """
        y_train = train_df[target_col].to_numpy()
        y_val = val_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()

        # 1. Model A: Sensors-only
        X_train_sensor = sensor_scaler.transform(train_df)
        X_test_sensor = sensor_scaler.transform(test_df)
        model_a = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight="balanced")
        metrics_a = self.evaluate_model(model_a, X_train_sensor, y_train, X_test_sensor, y_test)

        # 2. Model B: Vision-only
        X_train_vis = vision_scaler.transform(train_df)
        X_test_vis = vision_scaler.transform(test_df)
        model_b = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight="balanced")
        metrics_b = self.evaluate_model(model_b, X_train_vis, y_train, X_test_vis, y_test)

        # 3. Model C: Multimodal Fusion
        X_train_multi = multimodal_scaler.transform(train_df)
        X_test_multi = multimodal_scaler.transform(test_df)
        model_c = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight="balanced")
        metrics_c = self.evaluate_model(model_c, X_train_multi, y_train, X_test_multi, y_test)

        models = {
            "model_a_sensor": model_a,
            "model_b_vision": model_b,
            "model_c_multimodal": model_c
        }

        benchmark = {
            "Model A (Sensors Only)": {
                "features": SENSOR_FEATURE_COLS,
                "metrics": metrics_a
            },
            "Model B (Vision Only)": {
                "features": VISION_FEATURE_COLS,
                "metrics": metrics_b
            },
            "Model C (Multimodal Fusion)": {
                "features": MULTIMODAL_FEATURE_COLS,
                "metrics": metrics_c,
                "f1_improvement_over_sensor": round(metrics_c["f1_score"] - metrics_a["f1_score"], 4),
                "f1_improvement_over_vision": round(metrics_c["f1_score"] - metrics_b["f1_score"], 4)
            }
        }

        return models, benchmark


class MultimodalAnomalyDetector:
    """
    Unsupervised Anomaly Detector using Isolation Forest on fused multimodal vectors.
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
        self.model.fit(X_train)
        self.is_fitted = True

    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before prediction.")
        raw_scores = self.model.score_samples(X)
        preds = self.model.predict(X)
        norm_scores = np.clip((0.5 - raw_scores) / 0.8, 0.0, 1.0)
        anomaly_flags = (preds == -1)
        return anomaly_flags, norm_scores

    def save(self, filepath: str = "models/multimodal/multimodal_anomaly_detector.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "contamination": self.contamination,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: str = "models/multimodal/multimodal_anomaly_detector.joblib") -> "MultimodalAnomalyDetector":
        artifact = joblib.load(filepath)
        instance = cls(contamination=artifact["contamination"])
        instance.model = artifact["model"]
        instance.is_fitted = artifact["is_fitted"]
        return instance


class MultimodalSHAPExplainer:
    """
    SHAP Explainable AI integration for Multimodal Health Screening.
    Computes feature attributions explaining which physiological or visual features
    contributed to screening risk flags.
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_sample(self, X_single: np.ndarray) -> List[Dict[str, Any]]:
        """
        Computes SHAP value contributions for a single sample (shape: [1, n_features]).
        Returns sorted list of feature contributions.
        """
        shap_values = self.explainer.shap_values(X_single)
        # For binary classifier, shap_values can be list of 2 arrays [class_0, class_1] or 3D array
        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]

        contributions = []
        for name, val in zip(self.feature_names, vals):
            contributions.append({
                "feature": name,
                "shap_value": round(float(val), 4),
                "impact": "elevates_risk" if val > 0 else "lowers_risk"
            })

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions

    def save(self, filepath: str = "models/multimodal/multimodal_shap_explainer.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names
        }, filepath)

    @classmethod
    def load(cls, filepath: str = "models/multimodal/multimodal_shap_explainer.joblib") -> "MultimodalSHAPExplainer":
        artifact = joblib.load(filepath)
        return cls(model=artifact["model"], feature_names=artifact["feature_names"])
