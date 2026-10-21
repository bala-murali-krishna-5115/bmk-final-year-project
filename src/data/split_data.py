import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, GroupShuffleSplit

logger = logging.getLogger(__name__)

class DatasetSplitter:
    """
    Handles reproducible dataset splitting into Train (70%), Validation (15%), and Test (15%).
    Supports GroupShuffleSplit for subject-wise splitting to prevent data leakage.
    """

    @staticmethod
    def split_dataset(
        df: pd.DataFrame,
        subject_col: Optional[str] = "subject_id",
        target_col: Optional[str] = "label",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Splits dataset into train, validation, and test sets.
        If subject_col exists in df, uses subject-wise group splitting to avoid data leakage.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

        # Case 1: Subject-wise Group Splitting
        if subject_col and subject_col in df.columns and df[subject_col].nunique() > 1:
            logger.info(f"Performing subject-wise splitting on column '{subject_col}' ({df[subject_col].nunique()} subjects)...")
            
            # First split into Train+Val and Test based on subjects
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
            train_val_idx, test_idx = next(gss_test.split(df, groups=df[subject_col]))

            train_val_df = df.iloc[train_val_idx].copy()
            test_df = df.iloc[test_idx].copy()

            # Second split Train+Val into Train and Validation based on subjects
            relative_val_ratio = val_ratio / (train_ratio + val_ratio)
            gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=random_state)
            train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[subject_col]))

            train_df = train_val_df.iloc[train_idx].copy()
            val_df = train_val_df.iloc[val_idx].copy()

            # Verify no subject leakage
            train_sub = set(train_df[subject_col].unique())
            val_sub = set(val_df[subject_col].unique())
            test_sub = set(test_df[subject_col].unique())

            assert len(train_sub.intersection(test_sub)) == 0, "Subject leakage detected between Train and Test!"
            assert len(train_sub.intersection(val_sub)) == 0, "Subject leakage detected between Train and Validation!"
            assert len(val_sub.intersection(test_sub)) == 0, "Subject leakage detected between Validation and Test!"

            split_type = "subject_wise"

        # Case 2: Standard Stratified / Random Splitting
        else:
            logger.info("Performing standard random splitting...")
            stratify = df[target_col] if (target_col and target_col in df.columns and df[target_col].nunique() > 1) else None

            train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_state, stratify=stratify)
            
            val_relative = val_ratio / (train_ratio + val_ratio)
            stratify_val = train_val_df[target_col] if (target_col and target_col in train_val_df.columns and train_val_df[target_col].nunique() > 1) else None

            train_df, val_df = train_test_split(train_val_df, test_size=val_relative, random_state=random_state, stratify=stratify_val)
            split_type = "random_stratified"

        meta = {
            "split_type": split_type,
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "random_state": random_state
        }

        logger.info(f"Split complete ({split_type}): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df, meta

    @staticmethod
    def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, split_dir: str = "data/splits"):
        """Saves train, val, and test splits as CSV files."""
        os.makedirs(split_dir, exist_ok=True)
        train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)
        logger.info(f"Saved dataset splits to {split_dir}")
