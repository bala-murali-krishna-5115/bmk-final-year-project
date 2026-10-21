import os
import json
import pytest
import numpy as np
import pandas as pd

from src.data.clean_data import DataCleaner
from src.data.split_data import DatasetSplitter
from src.features.preprocessing import FeaturePreprocessor, DEFAULT_FEATURE_COLS
from src.models.inference import VisualInferenceEngine
from src.features.baseline_manager import PersonalBaselineManager

def test_feature_schema_exists():
    """Verify visual feature schema file exists and contains default feature columns."""
    schema_path = "artifacts/feature_schema/visual_feature_schema_v1.json"
    assert os.path.exists(schema_path)
    with open(schema_path, "r") as f:
        schema = json.load(f)
    assert "flat_feature_list" in schema
    assert len(schema["flat_feature_list"]) == len(DEFAULT_FEATURE_COLS)

def test_subject_wise_split_no_leakage():
    """Verify subject-wise splitting ensures train_subjects ∩ test_subjects = empty."""
    records = []
    for i in range(100):
        records.append({
            "subject_id": f"sub_{i % 5}",  # 5 subjects
            "label": i % 2,
            **{col: float(i) for col in DEFAULT_FEATURE_COLS}
        })
    df = pd.DataFrame(records)

    train_df, val_df, test_df, meta = DatasetSplitter.split_dataset(df, subject_col="subject_id", target_col="label")
    
    train_subs = set(train_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())
    val_subs = set(val_df["subject_id"].unique())

    assert len(train_subs.intersection(test_subs)) == 0
    assert len(train_subs.intersection(val_subs)) == 0

def test_preprocessor_fit_transform():
    """Verify FeaturePreprocessor fits ONLY on training data and scales correctly."""
    records = [{col: float(i) for col in DEFAULT_FEATURE_COLS} for i in range(20)]
    df = pd.DataFrame(records)
    
    preprocessor = FeaturePreprocessor()
    X_scaled, cols = preprocessor.fit_transform(df)
    
    assert X_scaled.shape == (20, len(DEFAULT_FEATURE_COLS))
    assert abs(np.mean(X_scaled)) < 1e-4

def test_inference_engine_quality_rejection():
    """Verify VisualInferenceEngine rejects low quality frames or frames without face."""
    engine = VisualInferenceEngine()

    # Case 1: No face detected
    res_no_face = engine.predict({"face": {"face_detected": False}, "quality": {"overall_quality": 80.0}})
    assert res_no_face["prediction_status"] == "insufficient_data"

    # Case 2: Low quality score
    res_low_q = engine.predict({"face": {"face_detected": True}, "quality": {"overall_quality": 40.0}})
    assert res_low_q["prediction_status"] == "low_quality"

def test_baseline_deviation_calculation():
    """Test PersonalBaselineManager Z-score deviation calculation."""
    baseline = {
        "means": {"mean_openness": 0.30, "facial_symmetry_score": 0.95},
        "stds": {"mean_openness": 0.02, "facial_symmetry_score": 0.02}
    }
    current = {"mean_openness": 0.15, "facial_symmetry_score": 0.70}  # Large deviation
    
    res = PersonalBaselineManager.calculate_baseline_deviation(current, baseline)
    assert res["deviation_flag"] is True
    assert res["baseline_distance"] > 2.0
