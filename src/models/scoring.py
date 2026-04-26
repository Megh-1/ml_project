"""
Coordination Scorer
=====================
Supervised bot scoring engine using a probabilistic Support Vector Machine.
Provides per-user coordination likelihood scores plus lightweight feature
attribution for each prediction.

Design Decisions:
    - Linear SVM by default so feature weights remain auditable.
    - StandardScaler normalization before model fitting.
    - class_weight="balanced" for imbalanced bot-vs-legit populations.
    - predict_proba via SVC probability calibration for dashboard scores.

No NLP - purely structural/behavioral signal scoring.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class CoordinationScorer:
    """SVM-based coordination likelihood scorer.

    Wraps a scikit-learn SVC to produce probability scores and interpretable
    feature attributions for bot/coordination prediction.

    Args:
        C: SVM regularization strength. Lower values regularize more.
        kernel: SVM kernel. Defaults to "linear" for explainability.
        gamma: Kernel coefficient for non-linear kernels.
        random_state: Random seed for reproducibility.
        feature_columns: Feature columns used for scoring.
        max_depth: Deprecated compatibility argument from the old tree model.

    Example:
        >>> scorer = CoordinationScorer(C=1.0, kernel="linear")
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
        C: float = 1.0,
        kernel: str = "linear",
        gamma: Union[str, float] = "scale",
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be > 0, got {C}")
        if max_depth is not None:
            logger.warning(
                "CoordinationScorer no longer uses max_depth; ignoring legacy value %s",
                max_depth,
            )

        self._C = C
        self._kernel = kernel
        self._gamma = gamma
        self._random_state = random_state
        self._feature_columns = feature_columns or self._DEFAULT_FEATURES

        self._classifier = self._build_classifier(probability=True)
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._training_accuracy: Optional[float] = None
        self._cv_scores: Optional[np.ndarray] = None

        logger.info(
            "CoordinationScorer initialized | svm_kernel=%s | C=%.4f | features=%s",
            kernel,
            C,
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
        """Train the SVM scoring model.

        Args:
            features_df: DataFrame containing feature columns.
            labels: Binary labels (True/1 = bot/coordinated, False/0 = legit).
            run_cv: If True, run stratified cross-validation and log accuracy.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If features or labels are invalid.
        """
        self._validate_features(features_df)
        y = self._prepare_labels(labels, expected_length=len(features_df))

        X = features_df[self._feature_columns].to_numpy(dtype=float)
        X_scaled = self._scaler.fit_transform(X)

        self._classifier.fit(X_scaled, y)
        self._is_fitted = True
        self._training_accuracy = float(self._classifier.score(X_scaled, y))

        if run_cv:
            n_splits = self._compute_cv_splits(y)
            if n_splits >= 2:
                cv = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self._random_state,
                )
                self._cv_scores = cross_val_score(
                    make_pipeline(
                        StandardScaler(),
                        self._build_classifier(probability=False),
                    ),
                    X,
                    y,
                    cv=cv,
                    scoring="accuracy",
                )
                logger.info(
                    "SVM trained | train_acc=%.4f | cv_acc=%.4f +/- %.4f | folds=%d",
                    self._training_accuracy,
                    self._cv_scores.mean(),
                    self._cv_scores.std(),
                    n_splits,
                )
            else:
                self._cv_scores = None
                logger.warning(
                    "SVM trained | train_acc=%.4f | cross-validation skipped "
                    "because one class has fewer than 2 examples",
                    self._training_accuracy,
                )
        else:
            self._cv_scores = None
            logger.info("SVM trained | train_acc=%.4f", self._training_accuracy)

        return self

    def predict_coordination_score(
        self,
        user_features: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Score one or more users for coordination likelihood.

        Args:
            user_features: Feature data for one or more users. Can be:
                - pd.DataFrame with feature columns
                - pd.Series for one user
                - np.ndarray of shape (n_features,) or (n_users, n_features)

        Returns:
            A dict for one user, or a list of dicts for a batch. Each dict
            contains coordination_likelihood, top_features, and risk_tier.

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If input shape is incompatible.
        """
        self._check_fitted()

        X = self._prepare_input(user_features)
        X_scaled = self._scaler.transform(X)
        probabilities = self._classifier.predict_proba(X_scaled)
        positive_idx = self._positive_class_index()
        bot_probabilities = probabilities[:, positive_idx]

        if X.shape[0] == 1:
            likelihood = float(bot_probabilities[0])
            return {
                "coordination_likelihood": likelihood,
                "top_features": self._get_top_features(X[0]),
                "risk_tier": self._compute_risk_tier(likelihood),
            }

        results = []
        for i in range(X.shape[0]):
            likelihood = float(bot_probabilities[i])
            results.append(
                {
                    "coordination_likelihood": likelihood,
                    "top_features": self._get_top_features(X[i]),
                    "risk_tier": self._compute_risk_tier(likelihood),
                }
            )
        return results

    def predict_proba_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return bot probabilities for all users in a DataFrame.

        Args:
            features_df: DataFrame with feature columns.

        Returns:
            np.ndarray of shape (n_users,) with bot probabilities.

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If feature columns are missing.
        """
        self._check_fitted()
        self._validate_features(features_df)

        X = features_df[self._feature_columns].to_numpy(dtype=float)
        X_scaled = self._scaler.transform(X)
        probabilities = self._classifier.predict_proba(X_scaled)
        return probabilities[:, self._positive_class_index()]

    def get_global_feature_importances(self) -> Dict[str, float]:
        """Return normalized feature weights from the trained SVM.

        For the default linear kernel this uses absolute SVM coefficients.
        Non-linear kernels do not expose direct feature weights, so equal
        weights are returned as a conservative fallback.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        importances = self._get_global_importance_vector()
        return {
            name: float(importance)
            for name, importance in zip(self._feature_columns, importances)
        }

    @property
    def training_accuracy(self) -> Optional[float]:
        """Training set accuracy after fitting."""
        return self._training_accuracy

    @property
    def cv_scores(self) -> Optional[np.ndarray]:
        """Cross-validation accuracy scores after fitting with run_cv=True."""
        return self._cv_scores

    @property
    def feature_columns(self) -> List[str]:
        """Feature column names used by the model."""
        return self._feature_columns.copy()

    @property
    def svm_params(self) -> Dict[str, Any]:
        """SVM configuration used by this scorer."""
        return {
            "C": self._C,
            "kernel": self._kernel,
            "gamma": self._gamma,
            "class_weight": "balanced",
        }

    # ------------------------------------------------------------------
    # Private: Classifier Construction
    # ------------------------------------------------------------------

    def _build_classifier(self, probability: bool) -> SVC:
        """Create an SVC instance with the scorer's current configuration."""
        return SVC(
            C=self._C,
            kernel=self._kernel,
            gamma=self._gamma,
            probability=probability,
            class_weight="balanced",
            random_state=self._random_state,
        )

    # ------------------------------------------------------------------
    # Private: Feature Attribution
    # ------------------------------------------------------------------

    def _get_top_features(
        self,
        feature_values: np.ndarray,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Extract the top-k local SVM feature contributions.

        For a linear SVM, local contribution is coefficient * scaled feature
        value. This keeps explanations tied to the actual prediction. For
        non-linear kernels, falls back to global importance weights.
        """
        contributions = self._get_local_contributions(feature_values)
        magnitudes = np.abs(contributions)

        if np.isclose(magnitudes.sum(), 0.0):
            magnitudes = self._get_global_importance_vector()

        normalized = magnitudes / magnitudes.sum()
        top_indices = np.argsort(normalized)[::-1][:top_k]

        return [
            {
                "feature_name": self._feature_columns[idx],
                "importance": float(normalized[idx]),
                "value": float(feature_values[idx]),
                "direction": (
                    "raises_bot_risk" if contributions[idx] >= 0 else "lowers_bot_risk"
                ),
            }
            for idx in top_indices
        ]

    def _get_local_contributions(self, feature_values: np.ndarray) -> np.ndarray:
        """Return signed local contributions for linear SVM predictions."""
        if self._kernel == "linear" and hasattr(self._classifier, "coef_"):
            scaled = self._scaler.transform(feature_values.reshape(1, -1))[0]
            coefficients = self._classifier.coef_.reshape(-1)
            return coefficients * scaled

        return self._get_global_importance_vector()

    def _get_global_importance_vector(self) -> np.ndarray:
        """Return normalized global feature importances."""
        n_features = len(self._feature_columns)
        if self._kernel == "linear" and hasattr(self._classifier, "coef_"):
            importances = np.abs(self._classifier.coef_.reshape(-1))
            total = importances.sum()
            if total > 0:
                return importances / total

        return np.full(n_features, 1.0 / n_features)

    # ------------------------------------------------------------------
    # Private: Risk Tier Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_risk_tier(likelihood: float) -> str:
        """Map coordination likelihood to a human-readable risk tier."""
        if likelihood >= 0.85:
            return "CRITICAL"
        if likelihood >= 0.65:
            return "HIGH"
        if likelihood >= 0.40:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Private: Input Preparation & Validation
    # ------------------------------------------------------------------

    def _prepare_input(
        self,
        user_features: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> np.ndarray:
        """Normalize supported input types to a 2D NumPy array."""
        if isinstance(user_features, pd.DataFrame):
            missing = set(self._feature_columns) - set(user_features.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            X = user_features[self._feature_columns].to_numpy(dtype=float)
        elif isinstance(user_features, pd.Series):
            missing = set(self._feature_columns) - set(user_features.index)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            X = user_features[self._feature_columns].to_numpy(dtype=float).reshape(1, -1)
        elif isinstance(user_features, np.ndarray):
            X = user_features.astype(float, copy=False)
        else:
            raise ValueError(f"Unsupported input type: {type(user_features)}")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self._feature_columns):
            raise ValueError(
                f"Expected {len(self._feature_columns)} features, got {X.shape[1]}"
            )

        if not np.isfinite(X).all():
            raise ValueError("Input features contain NaN or infinite values")

        return X

    def _validate_features(self, features_df: pd.DataFrame) -> None:
        """Validate that required feature columns are present and numeric."""
        if features_df.empty:
            raise ValueError("features_df cannot be empty")

        missing = set(self._feature_columns) - set(features_df.columns)
        if missing:
            raise ValueError(
                f"features_df missing required feature columns: {missing}. "
                "Run AccountFeatureExtractor.transform() first."
            )

        values = features_df[self._feature_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("features_df contains NaN or infinite feature values")

    @staticmethod
    def _prepare_labels(labels: pd.Series, expected_length: int) -> np.ndarray:
        """Validate and normalize binary labels to 0/1 integers."""
        if len(labels) != expected_length:
            raise ValueError(
                f"labels length ({len(labels)}) does not match "
                f"features_df length ({expected_length})"
            )

        label_series = pd.Series(labels)
        if (
            pd.api.types.is_bool_dtype(label_series)
            or pd.api.types.is_integer_dtype(label_series)
            or pd.api.types.is_float_dtype(label_series)
        ):
            numeric = label_series.astype(float).to_numpy()
            if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
                raise ValueError("labels must be binary values")
            y = numeric.astype(int)
        else:
            mapping = {
                "true": 1,
                "1": 1,
                "bot": 1,
                "yes": 1,
                "false": 0,
                "0": 0,
                "legit": 0,
                "normal": 0,
                "human": 0,
                "no": 0,
            }
            normalized = label_series.astype(str).str.strip().str.lower().map(mapping)
            if normalized.isna().any():
                raise ValueError(
                    "labels must be binary values such as True/False, 1/0, "
                    "bot/legit, or yes/no"
                )
            y = normalized.astype(int).to_numpy()

        unique = np.unique(y)
        if len(unique) != 2 or set(unique) != {0, 1}:
            raise ValueError(
                "labels must contain both binary classes: 0/False for legit and "
                "1/True for bot"
            )
        return y

    @staticmethod
    def _compute_cv_splits(y: np.ndarray, desired_splits: int = 5) -> int:
        """Choose a safe stratified CV split count for the class balance."""
        class_counts = np.bincount(y.astype(int), minlength=2)
        return int(min(desired_splits, class_counts.min()))

    def _positive_class_index(self) -> int:
        """Return the predict_proba column index for the bot class."""
        classes = list(self._classifier.classes_)
        return classes.index(1)

    def _check_fitted(self) -> None:
        """Ensure the model has been fitted before prediction."""
        if not self._is_fitted:
            raise RuntimeError(
                "CoordinationScorer has not been fitted. Call fit() first."
            )
