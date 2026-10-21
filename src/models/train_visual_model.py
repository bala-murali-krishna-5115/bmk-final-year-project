import os
import json
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.features.preprocessing import FeaturePreprocessor

logger = logging.getLogger(__name__)

class VisualModelTrainer:
    """
    Trains and compares classical ML models (Logistic Regression, Random Forest, SVM, XGBoost)
    on visual feature vectors using fitted preprocessing scaling.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.candidates = {
            "LogisticRegression": LogisticRegression(random_state=random_state, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced"),
            "SVM": SVC(probability=True, random_state=random_state, class_weight="balanced"),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=random_state, eval_metric="logloss")
        }

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, str, Dict[str, Any]]:
        """
        Trains candidate models on training set, selects the best model based on validation F1-score,
        and evaluates final performance on test set.
        """
        best_name = None
        best_model = None
        best_val_f1 = -1.0
        results = {}

        for name, model in self.candidates.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)

            # Evaluate on Validation
            val_preds = model.predict(X_val)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
            
            results[name] = {"val_f1": float(val_f1)}
            logger.info(f"{name} Validation F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_name = name
                best_model = model

        logger.info(f"Best performing model selected on validation set: {best_name} (F1: {best_val_f1:.4f})")

        # Evaluate best model on test set
        test_preds = best_model.predict(X_test)
        test_probs = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        test_cm = confusion_matrix(y_test, test_preds).tolist()
        test_auc = float(roc_auc_score(y_test, test_probs)) if test_probs is not None else 0.0

        metrics = {
            "best_model_name": best_name,
            "validation_results": results,
            "test_metrics": {
                "accuracy": round(float(test_acc), 4),
                "precision": round(float(test_prec), 4),
                "recall": round(float(test_rec), 4),
                "f1_score": round(float(test_f1), 4),
                "roc_auc": round(float(test_auc), 4),
                "confusion_matrix": test_cm
            }
        }
        return best_model, best_name, metrics

    def save_model(self, model: Any, model_name: str, metrics: Dict[str, Any], output_dir: str = "models/visual"):
        """Saves trained binary model artifact and experiment metadata JSON."""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "visual_classifier.joblib")
        
        artifact = {
            "model": model,
            "model_name": model_name,
            "metrics": metrics
        }
        joblib.dump(artifact, model_path)

        meta_dir = "artifacts/experiments"
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, "experiment_visual_model.json")
        
        with open(meta_path, "w") as f:
            json.dump({
                "model_name": "visual_classifier",
                "algorithm": model_name,
                "feature_schema_version": "1.0",
                "metrics": metrics,
                "disclaimer": "Experimental test set evaluation - NOT a medical diagnostic model."
            }, f, indent=2)

        logger.info(f"Saved visual classifier artifact to {model_path} and metadata to {meta_path}")
