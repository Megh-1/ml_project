"""
Behavioral Cluster Analyzer
==============================
Unsupervised discovery of anomalous user-behavior clusters using K-Means.
Identifies groups of accounts with structurally similar behavioral fingerprints
that may indicate coordinated inauthentic networks.

Pipeline:
    1. StandardScaler normalization (critical for K-Means distance metrics).
    2. K-Means clustering on account-level features.
    3. Per-cluster profile summarization for human interpretation.

No NLP — purely behavioral/structural signal clustering.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BehavioralClusterAnalyzer:
    """K-Means-based unsupervised behavioral clustering pipeline.

    Clusters users by their structural account features (follow_ratio,
    amplification_ratio, posting_velocity) to surface anomalous groups
    that deviate from organic behavior.

    Args:
        n_clusters: Number of clusters for K-Means. Defaults to 4.
        random_state: Random seed for reproducibility.
        feature_columns: Feature columns to use. Defaults to the standard
            account-level features.

    Example:
        >>> analyzer = BehavioralClusterAnalyzer(n_clusters=4)
        >>> labels = analyzer.fit_predict(enriched_users_df)
        >>> summary = analyzer.get_cluster_summary(enriched_users_df, labels)
        >>> print(summary)
    """

    _DEFAULT_FEATURES: List[str] = [
        "follow_ratio",
        "amplification_ratio",
        "posting_velocity",
    ]

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

        self._n_clusters = n_clusters
        self._random_state = random_state
        self._feature_columns = feature_columns or self._DEFAULT_FEATURES

        self._scaler = StandardScaler()
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        self._is_fitted = False

        logger.info(
            "BehavioralClusterAnalyzer initialized | k=%d | features=%s",
            n_clusters,
            self._feature_columns,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Fit the clustering pipeline and return cluster assignments.

        Steps:
            1. Extract feature columns from the DataFrame.
            2. Normalize features via StandardScaler.
            3. Run K-Means and return cluster labels.

        Args:
            features_df: DataFrame containing the feature columns (must have
                been transformed by AccountFeatureExtractor).

        Returns:
            np.ndarray of shape (n_users,) with integer cluster labels.

        Raises:
            ValueError: If required feature columns are missing.
        """
        self._validate_features(features_df)

        X = features_df[self._feature_columns].values
        X_scaled = self._scaler.fit_transform(X)
        labels = self._kmeans.fit_predict(X_scaled)
        self._is_fitted = True

        logger.info(
            "Clustering complete | %d users → %d clusters | inertia=%.2f",
            len(features_df),
            self._n_clusters,
            self._kmeans.inertia_,
        )
        return labels

    def get_cluster_summary(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        """Generate a human-readable summary of each cluster's profile.

        For each cluster, computes:
            - Mean of each feature (in original scale).
            - Cluster size (number of members).
            - Bot ratio (if is_bot column exists).

        Args:
            features_df: Original DataFrame with feature columns.
            labels: Cluster assignments from fit_predict.

        Returns:
            pd.DataFrame with one row per cluster and aggregated statistics.

        Raises:
            ValueError: If labels length doesn't match DataFrame length.
        """
        if len(labels) != len(features_df):
            raise ValueError(
                f"labels length ({len(labels)}) does not match "
                f"features_df length ({len(features_df)})"
            )

        df = features_df.copy()
        df["cluster_id"] = labels

        # Aggregate feature means per cluster
        agg_dict = {col: "mean" for col in self._feature_columns}
        agg_dict["cluster_id"] = "count"

        summary = df.groupby("cluster_id", as_index=False).agg(
            **{col: (col, "mean") for col in self._feature_columns},
            cluster_size=("cluster_id", "count"),
        )

        # Add bot ratio if ground-truth labels exist
        if "is_bot" in features_df.columns:
            bot_ratios = df.groupby("cluster_id")["is_bot"].mean().reset_index()
            bot_ratios.columns = ["cluster_id", "bot_ratio"]
            summary = summary.merge(bot_ratios, on="cluster_id")

        logger.info("Cluster summary:\n%s", summary.to_string(index=False))
        return summary

    @property
    def cluster_centers(self) -> Optional[np.ndarray]:
        """Cluster centroids in scaled feature space (after fitting).

        Returns:
            np.ndarray of shape (n_clusters, n_features) or None if not fitted.
        """
        if not self._is_fitted:
            return None
        return self._kmeans.cluster_centers_

    @property
    def inertia(self) -> Optional[float]:
        """Within-cluster sum of squares (after fitting).

        Lower inertia = tighter clusters. Useful for elbow-method evaluation.
        """
        if not self._is_fitted:
            return None
        return self._kmeans.inertia_

    # ------------------------------------------------------------------
    # Private: Validation
    # ------------------------------------------------------------------

    def _validate_features(self, features_df: pd.DataFrame) -> None:
        """Validate that required feature columns are present.

        Raises:
            ValueError: If features_df is empty or missing feature columns.
        """
        if features_df.empty:
            raise ValueError("features_df cannot be empty")

        missing = set(self._feature_columns) - set(features_df.columns)
        if missing:
            raise ValueError(
                f"features_df missing required feature columns: {missing}. "
                f"Run AccountFeatureExtractor.transform() first."
            )
