import os
import json
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetInspector:
    """
    Inspects tabular and image datasets, checking sample counts, class distributions,
    missing values, duplicates, and subject ID distributions.
    """

    @staticmethod
    def inspect_csv(file_path: str, label_col: str = "label", subject_col: str = "subject_id") -> Dict[str, Any]:
        """
        Inspects a tabular CSV dataset.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return {"exists": False, "file_path": file_path}

        df = pd.read_csv(file_path)
        num_rows, num_cols = df.shape

        null_counts = df.isnull().sum().to_dict()
        duplicate_rows = int(df.duplicated().sum())

        class_dist = {}
        if label_col in df.columns:
            class_dist = df[label_col].value_counts().to_dict()

        subject_dist = {}
        if subject_col in df.columns:
            subject_dist = {
                "unique_subjects": int(df[subject_col].nunique()),
                "samples_per_subject": df[subject_col].value_counts().to_dict()
            }

        report = {
            "exists": True,
            "file_path": str(path),
            "num_samples": num_rows,
            "num_features": num_cols,
            "columns": list(df.columns),
            "missing_values": null_counts,
            "duplicate_rows": duplicate_rows,
            "class_distribution": class_dist,
            "subject_distribution": subject_dist
        }
        return report

    @staticmethod
    def generate_inspection_report(datasets_to_check: Dict[str, str], output_report_path: str = "reports/dataset/inspection_report.json"):
        """
        Generates and saves a consolidated inspection report for all specified datasets.
        """
        out_path = Path(output_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        consolidated = {}
        for name, path in datasets_to_check.items():
            logger.info(f"Inspecting dataset: {name} ({path})...")
            consolidated[name] = DatasetInspector.inspect_csv(path)

        with open(out_path, "w") as f:
            json.dump(consolidated, f, indent=2)

        logger.info(f"Dataset inspection report written to {out_path}")
        return consolidated

if __name__ == "__main__":
    datasets = {
        "visual_features": "data/processed/visual_features/visual_features.csv",
        "eye_dataset": "data/processed/eye/eye_state.csv"
    }
    DatasetInspector.generate_inspection_report(datasets)
