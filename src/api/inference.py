"""
Inference Pipeline — Explanatory API Layer
=============================================
Orchestrates data retrieval, feature extraction, and model scoring into a
single callable endpoint. Designed as the integration point that a Tier-1
platform's backend (e.g., Meta's integrity systems) would hypothetically call.

Returns structured, human-readable JSON with:
    - Coordination likelihood score (0.0–1.0)
    - Primary flags explaining why the entity was flagged
    - Risk tier classification (CRITICAL/HIGH/MEDIUM/LOW)
    - Cluster membership from unsupervised analysis
"""

import logging
from typing import Dict, Any, Optional, List

import pandas as pd

from src.features.account_features import AccountFeatureExtractor
from src.features.cascade_features import CascadeFeatureExtractor
from src.models.scoring import CoordinationScorer
from src.models.clustering import BehavioralClusterAnalyzer

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference pipeline for entity analysis.

    Encapsulates the full chain: data lookup → feature extraction → model
    scoring → risk classification → human-readable explanation.

    Args:
        scorer: A fitted CoordinationScorer instance.
        cluster_analyzer: An optional fitted BehavioralClusterAnalyzer.
        account_extractor: AccountFeatureExtractor (uses default if not provided).
        cascade_extractor: CascadeFeatureExtractor (uses default if not provided).

    Example:
        >>> pipeline = InferencePipeline(scorer=fitted_scorer)
        >>> result = pipeline.analyze_entity("user_bot_00042", "account", users_df, interactions_df)
        >>> print(result["coordination_likelihood"], result["primary_flags"])
    """

    # Human-readable flag templates
    _FLAG_TEMPLATES = {
        "follow_ratio": "Near-zero follow_ratio ({value:.4f}) — asymmetric follower/following network",
        "amplification_ratio": "Extreme amplification_ratio ({value:.2f}) — retweet-heavy, minimal original content",
        "posting_velocity": "Abnormal posting_velocity ({value:.2f} posts/day) — automated cadence detected",
    }

    def __init__(
        self,
        scorer: CoordinationScorer,
        cluster_analyzer: Optional[BehavioralClusterAnalyzer] = None,
        account_extractor: Optional[AccountFeatureExtractor] = None,
        cascade_extractor: Optional[CascadeFeatureExtractor] = None,
    ) -> None:
        self._scorer = scorer
        self._cluster_analyzer = cluster_analyzer
        self._account_extractor = account_extractor or AccountFeatureExtractor()
        self._cascade_extractor = cascade_extractor or CascadeFeatureExtractor()

        logger.info("InferencePipeline initialized")

    def analyze_entity(
        self,
        entity_id: str,
        entity_type: str,
        users_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        cluster_labels: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Analyze a single entity and return a structured risk assessment.

        Orchestrates:
            1. Entity lookup in the users DataFrame.
            2. Account-level feature extraction.
            3. Coordination scoring via the Decision Tree model.
            4. Flag generation with human-readable explanations.
            5. Optional cluster assignment lookup.

        Args:
            entity_id: The user_id to analyze.
            entity_type: Type of entity ('account' or 'post').
            users_df: Full users DataFrame.
            interactions_df: Full interactions DataFrame.
            cluster_labels: Optional dict mapping user_id → cluster_id.

        Returns:
            Structured dict:
            {
                "entity_id": str,
                "entity_type": str,
                "coordination_likelihood": float,
                "risk_tier": str,
                "primary_flags": [str, ...],
                "feature_values": {str: float, ...},
                "cluster_id": int or None
            }

        Raises:
            ValueError: If entity_id not found or entity_type is invalid.
        """
        if entity_type == "account":
            return self._analyze_account(
                entity_id, users_df, interactions_df, cluster_labels
            )
        elif entity_type == "post":
            return self._analyze_post(entity_id, interactions_df)
        else:
            raise ValueError(
                f"Unsupported entity_type: '{entity_type}'. "
                "Must be 'account' or 'post'."
            )

    # ------------------------------------------------------------------
    # Private: Account Analysis
    # ------------------------------------------------------------------

    def _analyze_account(
        self,
        user_id: str,
        users_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        cluster_labels: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Full account-level analysis pipeline."""
        # 1. Lookup entity
        user_row = users_df[users_df["user_id"] == user_id]
        if user_row.empty:
            raise ValueError(f"Entity '{user_id}' not found in users_df")

        # 2. Extract features
        enriched = self._account_extractor.transform(user_row)

        # 3. Score
        score_result = self._scorer.predict_coordination_score(enriched)

        # 4. Generate flags
        primary_flags = self._generate_flags(score_result, enriched.iloc[0])

        # 5. Build feature values dict
        feature_values = {
            col: float(enriched.iloc[0][col])
            for col in self._account_extractor.FEATURE_COLUMNS
        }

        # 6. Cluster lookup
        cluster_id = None
        if cluster_labels and user_id in cluster_labels:
            cluster_id = int(cluster_labels[user_id])

        result = {
            "entity_id": user_id,
            "entity_type": "account",
            "coordination_likelihood": score_result["coordination_likelihood"],
            "risk_tier": score_result["risk_tier"],
            "primary_flags": primary_flags,
            "feature_values": feature_values,
            "cluster_id": cluster_id,
        }

        logger.info(
            "Entity analysis complete | user=%s | likelihood=%.4f | tier=%s",
            user_id,
            result["coordination_likelihood"],
            result["risk_tier"],
        )
        return result

    # ------------------------------------------------------------------
    # Private: Post/Cascade Analysis
    # ------------------------------------------------------------------

    def _analyze_post(
        self,
        post_id: str,
        interactions_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Full cascade-level analysis for a post."""
        cascade_features = self._cascade_extractor.extract_features(
            interactions_df, post_id
        )

        # Generate cascade-specific flags
        primary_flags = []
        if cascade_features["is_synchronized"]:
            primary_flags.append(
                f"Synchronized resharing detected — temporal_density "
                f"{cascade_features['temporal_density_ms']:.2f}ms (< 100ms threshold)"
            )
        if cascade_features["cascade_velocity"] > 100:
            primary_flags.append(
                f"High cascade_velocity ({cascade_features['cascade_velocity']} retweets "
                f"in first 60s) — exceeds organic virality baseline"
            )

        # Risk tier based on synchronization + velocity
        if cascade_features["is_synchronized"] and cascade_features["cascade_velocity"] > 100:
            risk_tier = "CRITICAL"
        elif cascade_features["is_synchronized"] or cascade_features["cascade_velocity"] > 100:
            risk_tier = "HIGH"
        else:
            risk_tier = "LOW"

        result = {
            "entity_id": post_id,
            "entity_type": "post",
            "coordination_likelihood": None,  # Not applicable for post-level
            "risk_tier": risk_tier,
            "primary_flags": primary_flags,
            "cascade_features": cascade_features,
            "cluster_id": None,
        }

        logger.info(
            "Post analysis complete | post=%s | tier=%s | flags=%d",
            post_id,
            risk_tier,
            len(primary_flags),
        )
        return result

    # ------------------------------------------------------------------
    # Private: Flag Generation
    # ------------------------------------------------------------------

    def _generate_flags(
        self,
        score_result: Dict[str, Any],
        user_row: pd.Series,
    ) -> List[str]:
        """Generate human-readable explanation flags for the score.

        Uses the top contributing features from the model to create
        actionable, interpretable flag strings.

        Args:
            score_result: Output from CoordinationScorer.predict_coordination_score().
            user_row: The enriched user row with feature values.

        Returns:
            List of human-readable flag strings.
        """
        flags = []
        for feat_info in score_result["top_features"]:
            feature_name = feat_info["feature_name"]
            feature_value = float(user_row[feature_name])

            if feature_name in self._FLAG_TEMPLATES:
                flag = self._FLAG_TEMPLATES[feature_name].format(value=feature_value)
            else:
                flag = f"{feature_name} = {feature_value:.4f} (importance: {feat_info['importance']:.4f})"

            flags.append(flag)
        return flags


# ======================================================================
# Convenience Function (Backward Compatibility)
# ======================================================================

def analyze_entity(
    entity_id: str,
    entity_type: str,
    users_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    scorer: CoordinationScorer,
    cluster_analyzer: Optional[BehavioralClusterAnalyzer] = None,
    cluster_labels: Optional[dict] = None,
) -> Dict[str, Any]:
    """Convenience function wrapping InferencePipeline.analyze_entity().

    This is the integration point that a platform's backend would call.

    Args:
        entity_id: The user_id or post_id to analyze.
        entity_type: 'account' or 'post'.
        users_df: Full users DataFrame.
        interactions_df: Full interactions DataFrame.
        scorer: A fitted CoordinationScorer instance.
        cluster_analyzer: Optional fitted BehavioralClusterAnalyzer.
        cluster_labels: Optional dict mapping user_id → cluster_id.

    Returns:
        Structured risk-assessment dict (see InferencePipeline.analyze_entity).
    """
    pipeline = InferencePipeline(
        scorer=scorer,
        cluster_analyzer=cluster_analyzer,
    )
    return pipeline.analyze_entity(
        entity_id=entity_id,
        entity_type=entity_type,
        users_df=users_df,
        interactions_df=interactions_df,
        cluster_labels=cluster_labels,
    )
