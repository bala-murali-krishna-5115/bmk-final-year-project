import os
import json
import joblib
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default feature column list from visual_feature_schema_v1.json
DEFAULT_FEATURE_COLS = [
    "left_eye_openness",
    "right_eye_openness",
    "mean_openness",
    "eye_asymmetry",
    "facial_symmetry_score",
    "skin_r_mean",
    "skin_g_mean",
    "skin_b_mean",
    "skin_h_mean",
    "skin_s_mean",
    "skin_v_mean",
    "skin_l_mean",
    "skin_a_mean",
    "skin_b_lab_mean",
    "skin_texture_lbp",
    "overall_quality"
]

class FeaturePreprocessor:
    """
    Fits feature scalers on training data ONLY and transforms input feature matrices.
    Saves and loads scaling artifacts to prevent train/test data leakage.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fits StandardScaler exclusively on training DataFrame and returns scaled array.
        """
        X_raw = train_df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X_raw)
        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {len(train_df)} training samples across {len(self.feature_cols)} features.")
        return X_scaled, self.feature_cols

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame using previously fitted scaler parameters.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform().")

        X_raw = df[self.feature_cols].values
        return self.scaler.transform(X_raw)

    def save(self, scaler_path: str = "artifacts/scalers/visual_scaler.joblib"):
        """Saves fitted scaler artifact."""
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        artifact = {
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "is_fitted": self.is_fitted
        }
        joblib.dump(artifact, scaler_path)
        logger.info(f"Saved feature scaler artifact to {scaler_path}")

    @classmethod
    def load(cls, scaler_path: str = "artifacts/scalers/visual_scaler.joblib") -> "FeaturePreprocessor":
        """Loads fitted scaler artifact."""
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}")

        artifact = joblib.load(scaler_path)
        instance = cls(feature_cols=artifact["feature_cols"])
        instance.scaler = artifact["scaler"]
        instance.is_fitted = artifact["is_fitted"]
        logger.info(f"Loaded feature scaler artifact from {scaler_path}")
        return instance
