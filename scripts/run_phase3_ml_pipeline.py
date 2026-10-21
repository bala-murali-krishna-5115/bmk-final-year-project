import os
import json
import logging
import numpy as np
import pandas as pd

from src.data.clean_data import DataCleaner
from src.data.split_data import DatasetSplitter
from src.features.preprocessing import FeaturePreprocessor, DEFAULT_FEATURE_COLS
from src.models.train_visual_model import VisualModelTrainer
from src.models.train_anomaly_detector import VisualAnomalyDetector
from src.models.train_eye_model import EyeStateModelTrainer
from src.features.baseline_manager import PersonalBaselineManager
from src.models.inference import VisualInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_visual_dataset(num_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic visual feature dataset across 10 subjects
    for pipeline verification without fabricated medical claims.
    """
    np.random.seed(random_state)
    subjects = [f"subject_{i:02d}" for i in range(1, 11)]

    records = []
    for i in range(num_samples):
        subj = np.random.choice(subjects)
        # Class 0: Normal / Nominal baseline, Class 1: Visual fatigue / pattern deviation
        label = np.random.choice([0, 1], p=[0.7, 0.3])

        if label == 0:
            ear_left = np.random.normal(0.28, 0.03)
            ear_right = np.random.normal(0.28, 0.03)
            symmetry = np.random.normal(0.92, 0.04)
            skin_r = np.random.normal(140, 10)
            skin_g = np.random.normal(110, 10)
            skin_b = np.random.normal(95, 10)
            skin_l = np.random.normal(130, 8)
            skin_a = np.random.normal(145, 5)
            skin_b_lab = np.random.normal(138, 5)
            lbp = np.random.normal(450, 50)
            quality = np.random.uniform(70, 98)
        else:
            ear_left = np.random.normal(0.18, 0.04)  # Low eye openness
            ear_right = np.random.normal(0.18, 0.04)
            symmetry = np.random.normal(0.81, 0.08)  # Lower symmetry
            skin_r = np.random.normal(155, 15)
            skin_g = np.random.normal(95, 15)
            skin_b = np.random.normal(85, 15)
            skin_l = np.random.normal(115, 12)
            skin_a = np.random.normal(155, 8)
            skin_b_lab = np.random.normal(145, 8)
            lbp = np.random.normal(650, 80)
            quality = np.random.uniform(65, 95)

        ear_left = float(np.clip(ear_left, 0.05, 0.45))
        ear_right = float(np.clip(ear_right, 0.05, 0.45))
        mean_ear = (ear_left + ear_right) / 2.0
        asym = abs(ear_left - ear_right)
        sym_score = float(np.clip(symmetry, 0.50, 1.0))

        records.append({
            "session_id": f"S_{i:04d}",
            "subject_id": subj,
            "label": label,
            "left_eye_openness": round(ear_left, 4),
            "right_eye_openness": round(ear_right, 4),
            "mean_openness": round(mean_ear, 4),
            "eye_asymmetry": round(asym, 4),
            "facial_symmetry_score": round(sym_score, 4),
            "skin_r_mean": round(float(skin_r), 2),
            "skin_g_mean": round(float(skin_g), 2),
            "skin_b_mean": round(float(skin_b), 2),
            "skin_h_mean": round(float(np.random.normal(15, 3)), 2),
            "skin_s_mean": round(float(np.random.normal(80, 10)), 2),
            "skin_v_mean": round(float(np.random.normal(140, 15)), 2),
            "skin_l_mean": round(float(skin_l), 2),
            "skin_a_mean": round(float(skin_a), 2),
            "skin_b_lab_mean": round(float(skin_b_lab), 2),
            "skin_texture_lbp": round(float(lbp), 2),
            "overall_quality": round(float(quality), 2)
        })

    df = pd.DataFrame(records)
    out_dir = "data/processed/visual_features"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "visual_features.csv"), index=False)
    logger.info(f"Generated synthetic visual feature dataset ({num_samples} rows) at {out_dir}")
    return df

def run_pipeline():
    logger.info("Starting Phase 3 ML Pipeline Execution...")

    # 1. Generate / Load dataset
    df_raw = generate_synthetic_visual_dataset(num_samples=500)

    # 2. Clean dataset
    df_clean, clean_report = DataCleaner.clean_visual_features_df(df_raw, min_quality=65.0)

    # 3. Subject-wise dataset splitting (Subject IDs isolated across splits)
    train_df, val_df, test_df, split_meta = DatasetSplitter.split_dataset(
        df_clean, subject_col="subject_id", target_col="label", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    DatasetSplitter.save_splits(train_df, val_df, test_df)

    # 4. Feature Preprocessing & Scaling (Fitted ON TRAIN DATA ONLY)
    preprocessor = FeaturePreprocessor(feature_cols=DEFAULT_FEATURE_COLS)
    X_train, _ = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    preprocessor.save("artifacts/scalers/visual_scaler.joblib")

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # 5. Train Visual Classifier Models
    trainer = VisualModelTrainer()
    best_model, best_name, metrics = trainer.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
    trainer.save_model(best_model, best_name, metrics)

    # 6. Fit Anomaly Detector (Isolation Forest on Normal Train Samples)
    normal_X_train = X_train[y_train == 0]
    anomaly_detector = VisualAnomalyDetector(contamination=0.05)
    anomaly_detector.fit(normal_X_train)
    anomaly_detector.save("models/anomaly/visual_anomaly_detector.joblib")

    # 7. Train Eye State Classifier
    eye_cols = ["left_eye_openness", "right_eye_openness", "mean_openness", "eye_asymmetry"]
    X_train_eye = train_df[eye_cols].values
    X_test_eye = test_df[eye_cols].values
    y_train_eye = (train_df["mean_openness"] >= 0.20).astype(int)
    y_test_eye = (test_df["mean_openness"] >= 0.20).astype(int)

    eye_trainer = EyeStateModelTrainer()
    eye_metrics = eye_trainer.train_and_evaluate(X_train_eye, y_train_eye, X_test_eye, y_test_eye)
    eye_trainer.save("models/eye/eye_classifier.joblib")

    # 8. Test Inference Engine
    inference_engine = VisualInferenceEngine()
    test_sample = test_df.iloc[0].to_dict()
    sample_feature_dict = {
        "quality": {"overall_quality": test_sample["overall_quality"]},
        "face": {"face_detected": True, "face_count": 1},
        "eyes": {
            "left_eye_openness": test_sample["left_eye_openness"],
            "right_eye_openness": test_sample["right_eye_openness"],
            "mean_openness": test_sample["mean_openness"],
            "eye_asymmetry": test_sample["eye_asymmetry"]
        },
        "symmetry": {"facial_symmetry_score": test_sample["facial_symmetry_score"]},
        "skin": {k: test_sample[k] for k in ["skin_r_mean", "skin_g_mean", "skin_b_mean", "skin_h_mean", "skin_s_mean", "skin_v_mean", "skin_l_mean", "skin_a_mean", "skin_b_lab_mean", "skin_texture_lbp"]}
    }

    result = inference_engine.predict(sample_feature_dict)
    logger.info(f"Sample Inference Result: {json.dumps(result, indent=2)}")

    print("\nPhase 3 ML Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    run_pipeline()
