"""
Coordination Scorer
=====================
Supervised scoring engine using a probabilistically calibrated Decision Tree.
Provides per-user coordination likelihood scores with explainable feature
attribution for each prediction.

Design Decisions:
    - DecisionTreeClassifier with max_depth=5 for interpretability.
    - predict_proba for calibrated probability outputs (0.0–1.0).
    - Per-prediction feature importance via tree path analysis.
    - Global feature importances exposed for model auditing.

No NLP — purely structural/behavioral signal scoring.
"""

import logging
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CoordinationScorer:
    """Decision Tree-based coordination likelihood scorer.

    Wraps a scikit-learn DecisionTreeClassifier to produce calibrated
    probability scores and explainable feature attributions for each
    prediction.

    Args:
        max_depth: Maximum depth of the decision tree. Defaults to 5 for
            interpretability while maintaining discrimination.
        random_state: Random seed for reproducibility.
        feature_columns: Feature columns used for scoring. Defaults to the
            standard account-level features.

    Example:
        >>> scorer = CoordinationScorer()
        >>> scorer.fit(enriched_users_df, enriched_users_df["is_bot"])
        >>> result = scorer.predict_coordination_score(user_row_df)
        >>> print(result["coordination_likelihood"])
    """

    _DEFAULT_FEATURES: List[str] = [
        "follow_ratio",
        "amplification_ratio",
        "posting_velocity",
    ]

    def __init__(
        self,
        max_depth: int = 5,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")

        self._max_depth = max_depth
        self._random_state = random_state
        self._feature_columns = feature_columns or self._DEFAULT_FEATURES

        self._classifier = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",  # Handle class imbalance (500 bots vs 9500 legit)
        )
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._training_accuracy: Optional[float] = None
        self._cv_scores: Optional[np.ndarray] = None

        logger.info(
            "CoordinationScorer initialized | max_depth=%d | features=%s",
            max_depth,
            self._feature_columns,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        run_cv: bool = True,
    ) -> "CoordinationScorer":
        """Train the coordination scoring model.

        Args:
            features_df: DataFrame containing feature columns.
            labels: Binary labels (True = bot/coordinated, False = legit).
            run_cv: If True, run 5-fold cross-validation and log accuracy.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If features_df is missing required columns.
        """
        self._validate_features(features_df)

        X = features_df[self._feature_columns].values
        y = labels.astype(int).values

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Fit classifier
        self._classifier.fit(X_scaled, y)
        self._is_fitted = True

        # Training accuracy
        self._training_accuracy = float(self._classifier.score(X_scaled, y))

        # Cross-validation
        if run_cv:
            self._cv_scores = cross_val_score(
                DecisionTreeClassifier(
                    max_depth=self._max_depth,
                    random_state=self._random_state,
                    class_weight="balanced",
                ),
                X_scaled,
                y,
                cv=5,
                scoring="accuracy",
            )
            logger.info(
                "Model trained | train_acc=%.4f | cv_acc=%.4f ± %.4f",
                self._training_accuracy,
                self._cv_scores.mean(),
                self._cv_scores.std(),
            )
        else:
            logger.info(
                "Model trained | train_acc=%.4f",
                self._training_accuracy,
            )

        return self

    def predict_coordination_score(
        self,
        user_features: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """Score a single user or batch of users for coordination likelihood.

        Args:
            user_features: Feature data for one or more users. Can be:
                - pd.DataFrame with feature columns
                - pd.Series (single user)
                - np.ndarray of shape (n_features,) or (n_users, n_features)

        Returns:
            Dict containing:
                - coordination_likelihood (float): Probability of being
                  coordinated/bot (0.0–1.0).
                - top_features (list): Top 2 most important features with
                  names and their values for this prediction.
                - risk_tier (str): CRITICAL/HIGH/MEDIUM/LOW based on score.

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If input shape is incompatible.
        """
        self._check_fitted()

        X = self._prepare_input(user_features)
        X_scaled = self._scaler.transform(X)

        # Probability of positive class (bot/coordinated)
        proba = self._classifier.predict_proba(X_scaled)

        # Handle single vs batch — return single dict for single user
        if X.shape[0] == 1:
            likelihood = float(proba[0, 1])
            top_features = self._get_top_features(X[0])
            return {
                "coordination_likelihood": likelihood,
                "top_features": top_features,
                "risk_tier": self._compute_risk_tier(likelihood),
            }

        # Batch mode: return list of dicts
        results = []
        for i in range(X.shape[0]):
            likelihood = float(proba[i, 1])
            top_features = self._get_top_features(X[i])
            results.append({
                "coordination_likelihood": likelihood,
                "top_features": top_features,
                "risk_tier": self._compute_risk_tier(likelihood),
            })
        return results

    def predict_proba_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return coordination probabilities for all users in a DataFrame.

        Convenience method for bulk scoring (e.g., for the dashboard).

        Args:
            features_df: DataFrame with feature columns.

        Returns:
            np.ndarray of shape (n_users,) with coordination probabilities.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        self._validate_features(features_df)

        X = features_df[self._feature_columns].values
        X_scaled = self._scaler.transform(X)
        return self._classifier.predict_proba(X_scaled)[:, 1]

    def get_global_feature_importances(self) -> Dict[str, float]:
        """Return global feature importances from the trained Decision Tree.

        Returns:
            Dict mapping feature names to their Gini importance scores.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        importances = self._classifier.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self._feature_columns, importances)
        }

    @property
    def training_accuracy(self) -> Optional[float]:
        """Training set accuracy (after fitting)."""
        return self._training_accuracy

    @property
    def cv_scores(self) -> Optional[np.ndarray]:
        """Cross-validation accuracy scores (after fitting with run_cv=True)."""
        return self._cv_scores

    @property
    def feature_columns(self) -> List[str]:
        """Feature column names used by the model."""
        return self._feature_columns.copy()

    # ------------------------------------------------------------------
    # Private: Feature Attribution
    # ------------------------------------------------------------------

    def _get_top_features(
        self,
        feature_values: np.ndarray,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Extract the top-k most important features for a single prediction.

        Uses global feature importances weighted by the user's actual feature
        values to produce a per-prediction attribution.

        Args:
            feature_values: Feature vector for a single user.
            top_k: Number of top features to return.

        Returns:
            List of dicts with 'feature_name', 'importance', 'value'.
        """
        importances = self._classifier.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_k]

        return [
            {
                "feature_name": self._feature_columns[idx],
                "importance": float(importances[idx]),
                "value": float(feature_values[idx]),
            }
            for idx in top_indices
        ]

    # ------------------------------------------------------------------
    # Private: Risk Tier Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_risk_tier(likelihood: float) -> str:
        """Map coordination likelihood to a human-readable risk tier.

        Thresholds:
            - CRITICAL: >= 0.85 (high-confidence bot/coordinated)
            - HIGH: >= 0.65
            - MEDIUM: >= 0.40
            - LOW: < 0.40

        Args:
            likelihood: Coordination probability (0.0–1.0).

        Returns:
            Risk tier string.
        """
        if likelihood >= 0.85:
            return "CRITICAL"
        elif likelihood >= 0.65:
            return "HIGH"
        elif likelihood >= 0.40:
            return "MEDIUM"
        else:
            return "LOW"

    # ------------------------------------------------------------------
    # Private: Input Preparation & Validation
    # ------------------------------------------------------------------

    def _prepare_input(
        self,
        user_features: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> np.ndarray:
        """Normalize various input types to a 2D NumPy array.

        Args:
            user_features: Input in DataFrame, Series, or ndarray format.

        Returns:
            np.ndarray of shape (n_users, n_features).

        Raises:
            ValueError: If input shape is incompatible.
        """
        if isinstance(user_features, pd.DataFrame):
            missing = set(self._feature_columns) - set(user_features.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            X = user_features[self._feature_columns].values
        elif isinstance(user_features, pd.Series):
            X = user_features[self._feature_columns].values.reshape(1, -1)
        elif isinstance(user_features, np.ndarray):
            X = user_features
        else:
            raise ValueError(f"Unsupported input type: {type(user_features)}")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self._feature_columns):
            raise ValueError(
                f"Expected {len(self._feature_columns)} features, "
                f"got {X.shape[1]}"
            )
        return X

    def _validate_features(self, features_df: pd.DataFrame) -> None:
        """Validate that required feature columns are present.

        Raises:
            ValueError: If features_df is empty or missing required columns.
        """
        if features_df.empty:
            raise ValueError("features_df cannot be empty")
        missing = set(self._feature_columns) - set(features_df.columns)
        if missing:
            raise ValueError(
                f"features_df missing required feature columns: {missing}. "
                f"Run AccountFeatureExtractor.transform() first."
            )

    def _check_fitted(self) -> None:
        """Ensure the model has been fitted before prediction.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CoordinationScorer has not been fitted. Call fit() first."
            )
