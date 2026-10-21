import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

from src.data.split_data import DatasetSplitter
from src.features.preprocessing import FeaturePreprocessor, DEFAULT_FEATURE_COLS
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_multimodal_dataset(visual_csv_path: str, random_state: int = 42) -> pd.DataFrame:
    """
    Fuses real/synthetic visual feature records with realistic physiological sensor readings
    correlated with health/fatigue/stress screening labels.
    """
    np.random.seed(random_state)
    df_visual = pd.read_csv(visual_csv_path)

    sensor_records = []
    for _, row in df_visual.iterrows():
        label = int(row["label"])
        if label == 0:
            # Nominal resting physiological baseline
            hr = np.random.normal(72.0, 5.5)
            spo2 = np.random.normal(98.4, 0.8)
            temp = np.random.normal(36.6, 0.25)
        else:
            # Elevated / stressed / fatigued state
            hr = np.random.normal(96.0, 12.0)
            spo2 = np.random.normal(93.8, 2.2)
            temp = np.random.normal(37.5, 0.6)

        hr = float(np.clip(hr, 45.0, 180.0))
        spo2 = float(np.clip(spo2, 75.0, 100.0))
        temp = float(np.clip(temp, 34.0, 41.0))

        sensor_records.append({
            "heart_rate": round(hr, 1),
            "spo2": round(spo2, 1),
            "temperature": round(temp, 2)
        })

    df_sensor = pd.DataFrame(sensor_records)
    df_multi = pd.concat([df_visual, df_sensor], axis=1)
    return df_multi

def run_pipeline():
    logger.info("Initializing Phase 4 Multimodal ML Pipeline...")

    # 1. Dataset Generation & Early Fusion
    visual_csv = "data/processed/visual_features/visual_features.csv"
    if not os.path.exists(visual_csv):
        raise FileNotFoundError(f"Missing visual features at {visual_csv}")

    df_multimodal = generate_multimodal_dataset(visual_csv)
    output_csv = "data/processed/multimodal_features/multimodal_features.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_multimodal.to_csv(output_csv, index=False)
    logger.info(f"Saved multimodal dataset with {len(df_multimodal)} samples to {output_csv}")

    # 2. Subject-wise Group Splitting (No Data Leakage)
    train_df, val_df, test_df, split_meta = DatasetSplitter.split_dataset(
        df_multimodal,
        subject_col="subject_id",
        target_col="label",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    splits_dir = "data/splits_multimodal"
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    logger.info(f"Saved subject-wise splits to {splits_dir}")

    # 3. Fit Scalers EXCLUSIVELY on Training Data
    sensor_scaler = MultimodalPreprocessor(feature_cols=SENSOR_FEATURE_COLS)
    sensor_scaler.fit(train_df)
    sensor_scaler.save("artifacts/scalers/sensor_scaler.joblib")

    multimodal_scaler = MultimodalPreprocessor(feature_cols=MULTIMODAL_FEATURE_COLS)
    multimodal_scaler.fit(train_df)
    multimodal_scaler.save("artifacts/scalers/multimodal_scaler.joblib")

    vision_scaler = FeaturePreprocessor.load("artifacts/scalers/visual_scaler.joblib")

    # 4. Train Model A, Model B, Model C
    trainer = MultimodalModelTrainer(random_state=42)
    models, benchmark = trainer.train_all_models(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        sensor_scaler=sensor_scaler,
        vision_scaler=vision_scaler,
        multimodal_scaler=multimodal_scaler
    )

    # Save Model Artifacts
    os.makedirs("models/sensors", exist_ok=True)
    os.makedirs("models/multimodal", exist_ok=True)

    joblib.dump({"model": models["model_a_sensor"], "feature_cols": SENSOR_FEATURE_COLS}, "models/sensors/sensor_classifier.joblib")
    joblib.dump({"model": models["model_c_multimodal"], "feature_cols": MULTIMODAL_FEATURE_COLS}, "models/multimodal/multimodal_classifier.joblib")

    # 5. Train Multimodal Anomaly Detector (Isolation Forest on normal baseline training data)
    normal_train = train_df[train_df["label"] == 0]
    X_train_normal_multi = multimodal_scaler.transform(normal_train)
    anomaly_detector = MultimodalAnomalyDetector(contamination=0.05, random_state=42)
    anomaly_detector.fit(X_train_normal_multi)
    anomaly_detector.save("models/multimodal/multimodal_anomaly_detector.joblib")

    # 6. Fit SHAP TreeExplainer on Model C
    shap_explainer = MultimodalSHAPExplainer(model=models["model_c_multimodal"], feature_names=MULTIMODAL_FEATURE_COLS)
    shap_explainer.save("models/multimodal/multimodal_shap_explainer.joblib")

    # 7. Save Comparative Benchmark Report
    benchmark_path = "artifacts/experiments/multimodal_benchmark_report.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    logger.info("Phase 4 Multimodal Pipeline Completed Successfully!")
    print("\n=== MULTIMODAL BENCHMARK RESULTS ===")
    for model_name, data in benchmark.items():
        print(f"\n{model_name}:")
        for k, v in data["metrics"].items():
            print(f"  {k}: {v}")
    print(f"\nBenchmark report saved to: {benchmark_path}")

if __name__ == "__main__":
    run_pipeline()
