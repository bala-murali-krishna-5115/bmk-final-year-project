import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

from src.features.preprocessing import DEFAULT_FEATURE_COLS

SENSOR_FEATURE_COLS = ["heart_rate", "spo2", "temperature"]
VISION_FEATURE_COLS = DEFAULT_FEATURE_COLS
MULTIMODAL_FEATURE_COLS = VISION_FEATURE_COLS + SENSOR_FEATURE_COLS

class MultimodalPreprocessor:
    """
    StandardScaler preprocessor for multimodal feature vectors.
    Fits exclusively on training data to prevent data leakage.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols or MULTIMODAL_FEATURE_COLS
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "MultimodalPreprocessor":
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        self.fit(df)
        return self.transform(df), self.feature_cols

    def save(self, filepath: str = "artifacts/scalers/multimodal_scaler.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: str = "artifacts/scalers/multimodal_scaler.joblib") -> "MultimodalPreprocessor":
        artifact = joblib.load(filepath)
        instance = cls(feature_cols=artifact["feature_cols"])
        instance.scaler = artifact["scaler"]
        instance.is_fitted = artifact["is_fitted"]
        return instance
