"""
Account-Level Feature Engineering
===================================
Extracts structural and behavioral features from user-profile metadata.
These features capture asymmetries in follower/following ratios, amplification
behavior, and posting cadence — all strong bot-network indicators.

No NLP or text analysis is used; all features are purely structural/behavioral.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AccountFeatureExtractor:
    """Extracts account-level features for coordination detection.

    Computes three engineered features from raw user-profile statistics:
        1. follow_ratio — Follower-to-following asymmetry.
        2. amplification_ratio — Retweet-to-post amplification signal.
        3. posting_velocity — Content production rate (posts/day).

    Design Notes:
        - All divisions use epsilon smoothing (ε=1) to avoid div-by-zero.
        - The transform method is idempotent — safe to call multiple times.
        - Operates on the full DataFrame in a vectorized fashion (no row-level loops).

    Example:
        >>> extractor = AccountFeatureExtractor()
        >>> enriched_df = extractor.transform(users_df)
        >>> print(enriched_df[['follow_ratio', 'amplification_ratio', 'posting_velocity']].describe())
    """

    # Epsilon to prevent division by zero in ratio calculations
    _EPSILON: float = 1.0

    # Feature column names (single source of truth)
    FEATURE_COLUMNS: List[str] = [
        "follow_ratio",
        "amplification_ratio",
        "posting_velocity",
    ]

    # Required input columns
    _REQUIRED_COLUMNS: List[str] = [
        "followers",
        "following",
        "total_retweets",
        "total_posts",
        "account_age_days",
    ]

    def transform(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Compute and append account-level features to the users DataFrame.

        This method is non-destructive — it returns a copy of the input
        DataFrame with three new feature columns appended.

        Args:
            users_df: Raw users DataFrame with columns: followers, following,
                      total_retweets, total_posts, account_age_days.

        Returns:
            pd.DataFrame: A copy of users_df with added feature columns:
                - follow_ratio: followers / (following + 1)
                - amplification_ratio: total_retweets / (total_posts + 1)
                - posting_velocity: total_posts / (account_age_days + 1)

        Raises:
            ValueError: If users_df is missing required columns.
        """
        self._validate_input(users_df)

        df = users_df.copy()

        df["follow_ratio"] = self._compute_follow_ratio(df)
        df["amplification_ratio"] = self._compute_amplification_ratio(df)
        df["posting_velocity"] = self._compute_posting_velocity(df)

        logger.info(
            "Extracted %d account features for %d users",
            len(self.FEATURE_COLUMNS),
            len(df),
        )
        return df

    def get_feature_matrix(self, users_df: pd.DataFrame) -> np.ndarray:
        """Extract only the feature columns as a NumPy matrix.

        Useful for direct input into scikit-learn estimators.

        Args:
            users_df: DataFrame that has already been transformed (must contain
                      feature columns).

        Returns:
            np.ndarray of shape (n_users, 3).

        Raises:
            ValueError: If feature columns are missing (transform not called).
        """
        missing = set(self.FEATURE_COLUMNS) - set(users_df.columns)
        if missing:
            raise ValueError(
                f"Feature columns {missing} not found. "
                "Call transform() first."
            )
        return users_df[self.FEATURE_COLUMNS].values

    # ------------------------------------------------------------------
    # Private: Individual Feature Computations
    # ------------------------------------------------------------------

    def _compute_follow_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Follower-to-following ratio.

        High values indicate influential accounts; near-zero values indicate
        sockpuppet/amplification bots that follow many but attract no followers.
        """
        return df["followers"] / (df["following"] + self._EPSILON)

    def _compute_amplification_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Retweet-to-post amplification ratio.

        Values >> 1.0 indicate accounts whose primary activity is amplifying
        others' content rather than producing original posts — a key bot signal.
        """
        return df["total_retweets"] / (df["total_posts"] + self._EPSILON)

    def _compute_posting_velocity(self, df: pd.DataFrame) -> pd.Series:
        """Posts-per-day production rate.

        Abnormally high velocity may indicate automated posting behavior.
        Combined with high amplification_ratio, this is a strong bot indicator.
        """
        return df["total_posts"] / (df["account_age_days"] + self._EPSILON)

    # ------------------------------------------------------------------
    # Private: Validation
    # ------------------------------------------------------------------

    def _validate_input(self, users_df: pd.DataFrame) -> None:
        """Validate that all required columns exist in the input DataFrame.

        Raises:
            ValueError: If users_df is empty or missing required columns.
        """
        if users_df.empty:
            raise ValueError("Input DataFrame is empty")

        missing = set(self._REQUIRED_COLUMNS) - set(users_df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame missing required columns: {missing}. "
                f"Expected: {self._REQUIRED_COLUMNS}"
            )
