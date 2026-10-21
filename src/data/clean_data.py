import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Reusable tabular dataset cleaner.
    Handles duplicates, missing values (NaN/Inf), low-quality filtering, and produces clean reports.
    """

    @staticmethod
    def clean_visual_features_df(df: pd.DataFrame, min_quality: float = 65.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans a visual feature pandas DataFrame.

        Args:
            df (pd.DataFrame): Raw visual feature dataset.
            min_quality (float): Minimum quality threshold.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (cleaned_dataframe, cleaning_summary_dict)
        """
        initial_count = len(df)
        report = {"initial_count": initial_count, "removed": {}}

        # 1. Remove duplicate rows
        df_clean = df.drop_duplicates().copy()
        dups_removed = initial_count - len(df_clean)
        report["removed"]["duplicates"] = dups_removed

        # 2. Filter low camera quality rows if quality column exists
        if "overall_quality" in df_clean.columns:
            quality_before = len(df_clean)
            df_clean = df_clean[df_clean["overall_quality"] >= min_quality].copy()
            report["removed"]["low_quality"] = quality_before - len(df_clean)

        # 3. Replace inf / -inf with NaN
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 4. Impute numeric columns with median, drop rows with missing target labels
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        nan_before = df_clean[numeric_cols].isnull().sum().sum()
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        report["imputed_nan_count"] = int(nan_before)

        final_count = len(df_clean)
        report["final_count"] = final_count
        report["total_removed"] = initial_count - final_count

        logger.info(f"Cleaned dataset: initial={initial_count}, final={final_count}, removed={report['total_removed']}")
        return df_clean, report
