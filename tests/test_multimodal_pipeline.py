import os
import json
import pytest
import numpy as np
import pandas as pd

from src.features.multimodal_preprocessing import (
    MultimodalPreprocessor,
    SENSOR_FEATURE_COLS,
    VISION_FEATURE_COLS,
    MULTIMODAL_FEATURE_COLS
)
from src.models.train_multimodal_model import (
    MultimodalModelTrainer,
    MultimodalAnomalyDetector,
    MultimodalSHAPExplainer
)
from backend.services.multimodal_service import MultimodalService

def test_multimodal_feature_schema():
    """Verify multimodal feature schema file exists and defines all 19 features."""
    schema_path = "artifacts/feature_schema/multimodal_feature_schema_v1.json"
    assert os.path.exists(schema_path)
    with open(schema_path, "r") as f:
        schema = json.load(f)
    assert "multimodal_feature_list" in schema
    assert len(schema["multimodal_feature_list"]) == 19
    assert len(schema["sensor_feature_list"]) == 3
    assert len(schema["visual_feature_list"]) == 16

def test_multimodal_preprocessor():
    """Verify MultimodalPreprocessor fits strictly and scales correctly."""
    records = [{col: float(i * 1.5) for col in MULTIMODAL_FEATURE_COLS} for i in range(25)]
    df = pd.DataFrame(records)

    preprocessor = MultimodalPreprocessor(feature_cols=MULTIMODAL_FEATURE_COLS)
    X_scaled, cols = preprocessor.fit_transform(df)

    assert X_scaled.shape == (25, 19)
    assert abs(np.mean(X_scaled)) < 1e-4

def test_multimodal_anomaly_detector():
    """Verify MultimodalAnomalyDetector produces normalized anomaly scores [0, 1]."""
    detector = MultimodalAnomalyDetector(contamination=0.1, random_state=42)
    X_train = np.random.normal(0, 1, size=(50, 19))
    detector.fit(X_train)

    X_test = np.random.normal(0, 1, size=(5, 19))
    flags, scores = detector.predict_anomaly(X_test)

    assert len(flags) == 5
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

def test_multimodal_service_singleton():
    """Verify MultimodalService loads models and executes tri-model prediction."""
    service = MultimodalService.get_instance()
    
    vis = {
        "eyes": {"left_eye_openness": 0.28, "right_eye_openness": 0.28, "mean_openness": 0.28},
        "symmetry": {"facial_symmetry_score": 0.92},
        "skin": {"skin_l_mean": 125.0},
        "quality": {"overall_quality": 88.0}
    }
    sensors = {"heart_rate": 72.0, "spo2": 98.5, "temperature": 36.6}

    result = service.predict_multimodal(vis, sensors)
    assert result["status"] == "success"
    assert "tri_model_comparison" in result
    assert "model_a_sensor" in result["tri_model_comparison"]
    assert "model_b_vision" in result["tri_model_comparison"]
    assert "model_c_multimodal" in result["tri_model_comparison"]
    assert "anomaly_analysis" in result
    assert "shap_attributions" in result
    assert "disclaimer" in result
