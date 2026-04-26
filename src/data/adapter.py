"""
Real Data Adapter
==================
Maps real Twitter CSV data (from data/ folder) to the internal schema
expected by the feature extraction and scoring pipeline.

The CSVs in data/ use Twitter API naming conventions with ~80 pre-computed
features for bot detection. This adapter loads, cleans, and exposes these
features for the SVM classifier.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project root data directory
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Columns to exclude from feature set (identifiers, targets, non-numeric, always-NaN)
_EXCLUDE_COLS = {
    "user_id", "label", "is_bot", "split",
    "top_source", "first_tweet_ts", "last_tweet_ts",
    # 100% NaN in the dataset
    "favourites_count", "favorites_to_statuses_ratio",
    "listed_to_followers_ratio", "following_to_follower_ratio",
}

# Column rename mapping: real CSV name → internal pipeline name
_COLUMN_MAP = {
    "label": "is_bot",
}


class RealDataAdapter:
    """Adapts real Twitter CSV data for the bot detection pipeline.

    Loads CSVs from the data/ folder, renames the label column, fills NaN
    values, and exposes the list of numeric feature columns for the SVM.

    Example:
        >>> df = RealDataAdapter.load_table("train")
        >>> features = RealDataAdapter.get_feature_columns(df)
        >>> X = df[features].values
    """

    AVAILABLE_SPLITS = ("train", "test", "val", "master")

    @staticmethod
    def is_real_schema(df: pd.DataFrame) -> bool:
        """Check if a DataFrame uses the real Twitter CSV column names."""
        real_indicators = {"followers_count", "friends_count"}
        return real_indicators.issubset(set(df.columns))

    @classmethod
    def adapt(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Adapt a real Twitter DataFrame: rename label, fill NaN, clean types.

        Args:
            df: Raw DataFrame from a real Twitter CSV.

        Returns:
            Cleaned DataFrame with 'is_bot' column and NaN-free numeric features.
        """
        if not cls.is_real_schema(df):
            logger.info("DataFrame does not use real Twitter schema; returning as-is")
            return df

        adapted = df.copy()

        # Rename label → is_bot
        adapted = adapted.rename(columns=_COLUMN_MAP)

        # Convert is_bot to boolean
        if "is_bot" in adapted.columns:
            adapted["is_bot"] = adapted["is_bot"].astype(bool)

        # Feature Engineering: Add derived behavioral signals
        if "followers_count" in adapted.columns and "account_age_days" in adapted.columns:
            adapted["velocity_metric"] = adapted["followers_count"] / (adapted["account_age_days"] + 1)
        
        if "statuses_count" in adapted.columns and "account_age_days" in adapted.columns:
            adapted["tweet_frequency"] = adapted["statuses_count"] / (adapted["account_age_days"] + 1)

        # Drop columns that are always NaN or non-useful
        drop_cols = [c for c in _EXCLUDE_COLS if c in adapted.columns and c != "is_bot" and c != "user_id"]
        adapted = adapted.drop(columns=[c for c in drop_cols if c in adapted.columns], errors="ignore")

        # Drop non-numeric columns (timestamps, source names, split)
        non_numeric = adapted.select_dtypes(exclude="number").columns.tolist()
        keep_non_numeric = {"user_id", "is_bot"}  # keep these
        drop_non_numeric = [c for c in non_numeric if c not in keep_non_numeric]
        adapted = adapted.drop(columns=drop_non_numeric, errors="ignore")

        # Fill remaining NaN with 0 (conservative for users with no tweet data)
        numeric_cols = adapted.select_dtypes(include="number").columns
        adapted[numeric_cols] = adapted[numeric_cols].fillna(0)

        # Replace any inf values
        adapted = adapted.replace([np.inf, -np.inf], 0)

        logger.info(
            "Adapted real Twitter data: %d rows, %d columns",
            len(adapted), len(adapted.columns),
        )
        return adapted

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        """Get the list of numeric feature columns (excludes user_id and is_bot).

        Args:
            df: An adapted DataFrame.

        Returns:
            List of feature column names suitable for model training.
        """
        exclude = {"user_id", "is_bot"}
        return [c for c in df.select_dtypes(include="number").columns if c not in exclude]

    @classmethod
    def load_table(
        cls,
        split: str = "test",
        data_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Load and adapt a data table CSV by split name.

        Args:
            split: One of 'train', 'test', 'val', 'master'.
            data_dir: Override path to data directory.

        Returns:
            Adapted DataFrame ready for the pipeline.
        """
        if split not in cls.AVAILABLE_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Choose from: {cls.AVAILABLE_SPLITS}"
            )

        data_path = (data_dir or _DATA_DIR) / f"{split}_table.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data table not found: {data_path}")

        logger.info("Loading data table: %s", data_path)
        df = pd.read_csv(data_path)
        return cls.adapt(df)

    @classmethod
    def list_available_tables(cls, data_dir: Optional[Path] = None) -> list:
        """List which split tables are available on disk."""
        directory = data_dir or _DATA_DIR
        return [
            split for split in cls.AVAILABLE_SPLITS
            if (directory / f"{split}_table.csv").exists()
        ]
