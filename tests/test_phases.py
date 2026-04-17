"""
Test Suite for Phases 1 & 2
==============================
Validates data generation invariants and feature engineering correctness.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.config import SimulationConfig
from src.data.simulator import SocialDataSimulator
from src.features.account_features import AccountFeatureExtractor
from src.features.cascade_features import CascadeFeatureExtractor


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def config():
    """Shared simulation config for all tests."""
    return SimulationConfig(seed=42)


@pytest.fixture(scope="module")
def simulator(config):
    """Shared simulator instance."""
    return SocialDataSimulator(config)


@pytest.fixture(scope="module")
def users_df(simulator):
    """Generated users DataFrame (computed once per module)."""
    return simulator.generate_users()


@pytest.fixture(scope="module")
def interactions_df(simulator, users_df):
    """Generated interactions DataFrame (computed once per module)."""
    return simulator.generate_interactions(users_df)


@pytest.fixture(scope="module")
def enriched_users_df(users_df):
    """Users DataFrame with account-level features."""
    extractor = AccountFeatureExtractor()
    return extractor.transform(users_df)


# ======================================================================
# Phase 1: Data Generation Tests
# ======================================================================

class TestSimulationConfig:
    """Tests for SimulationConfig validation."""

    def test_default_config_valid(self):
        cfg = SimulationConfig()
        cfg.validate()
        assert cfg.total_users == 10_000

    def test_invalid_attack_bots(self):
        with pytest.raises(ValueError, match="n_attack_bots"):
            SimulationConfig(n_bot_users=100, n_attack_bots=500).validate()

    def test_frozen_config(self):
        cfg = SimulationConfig()
        with pytest.raises(Exception):
            cfg.seed = 999


class TestSocialDataSimulator:
    """Tests for the synthetic data generator."""

    def test_users_shape(self, users_df, config):
        assert len(users_df) == config.total_users
        assert len(users_df) == config.n_legit_users + config.n_bot_users

    def test_users_columns(self, users_df):
        expected = {"user_id", "followers", "following", "account_age_days",
                    "total_posts", "total_retweets", "is_bot"}
        assert expected.issubset(set(users_df.columns))

    def test_bot_count(self, users_df, config):
        bot_count = users_df["is_bot"].sum()
        assert bot_count == config.n_bot_users

    def test_bot_fingerprint(self, users_df):
        bots = users_df[users_df["is_bot"]]
        # Bots should have very few followers
        assert (bots["followers"] <= 5).all(), "Bot followers should be <= 5"
        # Bots should have high following
        assert (bots["following"] >= 500).all(), "Bot following should be >= 500"

    def test_bot_amplification_ratio(self, users_df):
        bots = users_df[users_df["is_bot"]]
        ratios = bots["total_retweets"] / (bots["total_posts"] + 1)
        assert (ratios > 5).all(), "Bot amplification ratio should be > 5"

    def test_interactions_columns(self, interactions_df):
        expected = {"event_id", "user_id", "target_post_id", "timestamp_ms", "action_type"}
        assert expected.issubset(set(interactions_df.columns))

    def test_interactions_not_empty(self, interactions_df):
        assert len(interactions_df) > 0

    def test_coordinated_attack_present(self, interactions_df, config):
        attack_events = interactions_df[
            (interactions_df["target_post_id"] == config.attack_post_id)
            & (interactions_df["action_type"] == "retweet")
        ]
        # All 500 attack bots should have retweeted
        assert len(attack_events) >= config.n_attack_bots

    def test_attack_within_window(self, interactions_df, config):
        attack_events = interactions_df[
            (interactions_df["target_post_id"] == config.attack_post_id)
            & (interactions_df["action_type"] == "retweet")
        ]
        timestamps = attack_events["timestamp_ms"].values
        window = timestamps.max() - timestamps.min()
        assert window <= config.attack_window_ms, (
            f"Attack window {window}ms exceeds configured {config.attack_window_ms}ms"
        )

    def test_reproducibility(self, config):
        sim1 = SocialDataSimulator(config)
        sim2 = SocialDataSimulator(config)
        users1 = sim1.generate_users()
        users2 = sim2.generate_users()
        pd.testing.assert_frame_equal(users1, users2)


# ======================================================================
# Phase 2: Feature Engineering Tests
# ======================================================================

class TestAccountFeatureExtractor:
    """Tests for account-level feature extraction."""

    def test_feature_columns_exist(self, enriched_users_df):
        for col in AccountFeatureExtractor.FEATURE_COLUMNS:
            assert col in enriched_users_df.columns, f"Missing feature column: {col}"

    def test_features_non_negative(self, enriched_users_df):
        for col in AccountFeatureExtractor.FEATURE_COLUMNS:
            assert (enriched_users_df[col] >= 0).all(), f"{col} contains negative values"

    def test_follow_ratio_bots_low(self, enriched_users_df):
        bots = enriched_users_df[enriched_users_df["is_bot"]]
        # Bots: ~0 followers, high following → follow_ratio should be near 0
        assert (bots["follow_ratio"] < 0.02).all(), "Bot follow_ratio should be < 0.02"

    def test_amplification_ratio_bots_high(self, enriched_users_df):
        bots = enriched_users_df[enriched_users_df["is_bot"]]
        assert (bots["amplification_ratio"] > 3).all(), (
            "Bot amplification_ratio should be > 3"
        )

    def test_feature_matrix_shape(self, enriched_users_df):
        extractor = AccountFeatureExtractor()
        matrix = extractor.get_feature_matrix(enriched_users_df)
        assert matrix.shape == (len(enriched_users_df), 3)

    def test_empty_df_raises(self):
        extractor = AccountFeatureExtractor()
        with pytest.raises(ValueError, match="empty"):
            extractor.transform(pd.DataFrame())

    def test_missing_columns_raises(self):
        extractor = AccountFeatureExtractor()
        with pytest.raises(ValueError, match="missing required columns"):
            extractor.transform(pd.DataFrame({"user_id": [1]}))


class TestCascadeFeatureExtractor:
    """Tests for cascade-level feature extraction."""

    def test_cascade_graph_built(self, interactions_df, config):
        extractor = CascadeFeatureExtractor()
        graph = extractor.build_cascade_graph(interactions_df, config.attack_post_id)
        assert isinstance(graph, __import__("networkx").DiGraph)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_cascade_velocity_high_for_attack(self, interactions_df, config):
        extractor = CascadeFeatureExtractor()
        features = extractor.extract_features(interactions_df, config.attack_post_id)
        # 500 bots within 3s → within 60s window, velocity should be >= 400
        assert features["cascade_velocity"] >= 400, (
            f"Expected cascade_velocity >= 400, got {features['cascade_velocity']}"
        )

    def test_temporal_density_low_for_attack(self, interactions_df, config):
        extractor = CascadeFeatureExtractor()
        features = extractor.extract_features(interactions_df, config.attack_post_id)
        # 500 events in 3s → avg delta < 10ms
        assert features["temporal_density_ms"] < 10.0, (
            f"Expected temporal_density < 10ms, got {features['temporal_density_ms']}"
        )

    def test_synchronization_detected(self, interactions_df, config):
        extractor = CascadeFeatureExtractor()
        features = extractor.extract_features(interactions_df, config.attack_post_id)
        assert features["is_synchronized"] is True

    def test_invalid_post_raises(self, interactions_df):
        extractor = CascadeFeatureExtractor()
        with pytest.raises(ValueError, match="No retweet events"):
            extractor.extract_features(interactions_df, "nonexistent_post_999")

    def test_features_dict_keys(self, interactions_df, config):
        extractor = CascadeFeatureExtractor()
        features = extractor.extract_features(interactions_df, config.attack_post_id)
        expected_keys = {
            "post_id", "cascade_velocity", "temporal_density_ms",
            "total_nodes", "total_edges", "total_retweets",
            "cascade_duration_ms", "is_synchronized",
        }
        assert expected_keys == set(features.keys())


# ======================================================================
# Phase 3: Machine Learning Pipeline Tests
# ======================================================================

@pytest.fixture(scope="module")
def cluster_labels(enriched_users_df):
    """Cluster assignments from BehavioralClusterAnalyzer."""
    from src.models.clustering import BehavioralClusterAnalyzer
    analyzer = BehavioralClusterAnalyzer(n_clusters=4)
    return analyzer.fit_predict(enriched_users_df), analyzer


@pytest.fixture(scope="module")
def scorer(enriched_users_df):
    """Trained CoordinationScorer."""
    from src.models.scoring import CoordinationScorer
    scorer = CoordinationScorer(max_depth=5)
    scorer.fit(enriched_users_df, enriched_users_df["is_bot"])
    return scorer


class TestBehavioralClusterAnalyzer:
    """Tests for K-Means behavioral clustering."""

    def test_labels_shape(self, cluster_labels, enriched_users_df):
        labels, _ = cluster_labels
        assert len(labels) == len(enriched_users_df)

    def test_labels_range(self, cluster_labels):
        labels, _ = cluster_labels
        assert set(labels).issubset({0, 1, 2, 3})

    def test_cluster_summary_shape(self, cluster_labels, enriched_users_df):
        labels, analyzer = cluster_labels
        summary = analyzer.get_cluster_summary(enriched_users_df, labels)
        assert len(summary) == 4  # 4 clusters
        assert "cluster_size" in summary.columns
        assert "bot_ratio" in summary.columns

    def test_cluster_centers_exist(self, cluster_labels):
        _, analyzer = cluster_labels
        centers = analyzer.cluster_centers
        assert centers is not None
        assert centers.shape == (4, 3)  # 4 clusters x 3 features

    def test_invalid_n_clusters(self):
        from src.models.clustering import BehavioralClusterAnalyzer
        with pytest.raises(ValueError, match="n_clusters"):
            BehavioralClusterAnalyzer(n_clusters=1)

    def test_empty_df_raises(self):
        from src.models.clustering import BehavioralClusterAnalyzer
        analyzer = BehavioralClusterAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.fit_predict(pd.DataFrame())


class TestCoordinationScorer:
    """Tests for Decision Tree coordination scoring."""

    def test_training_accuracy_high(self, scorer):
        assert scorer.training_accuracy is not None
        assert scorer.training_accuracy > 0.85, (
            f"Training accuracy {scorer.training_accuracy} should be > 0.85"
        )

    def test_cv_scores_exist(self, scorer):
        assert scorer.cv_scores is not None
        assert len(scorer.cv_scores) == 5
        assert scorer.cv_scores.mean() > 0.80

    def test_predict_single_user(self, scorer, enriched_users_df):
        # Score a single bot user
        bot_user = enriched_users_df[enriched_users_df["is_bot"]].iloc[0]
        result = scorer.predict_coordination_score(bot_user)

        assert "coordination_likelihood" in result
        assert "top_features" in result
        assert "risk_tier" in result
        assert 0.0 <= result["coordination_likelihood"] <= 1.0
        assert len(result["top_features"]) == 2

    def test_bot_scores_high(self, scorer, enriched_users_df):
        bots = enriched_users_df[enriched_users_df["is_bot"]]
        scores = scorer.predict_proba_batch(bots)
        # Most bots should score > 0.5
        assert (scores > 0.5).mean() > 0.80

    def test_legit_scores_low(self, scorer, enriched_users_df):
        legit = enriched_users_df[~enriched_users_df["is_bot"]]
        scores = scorer.predict_proba_batch(legit)
        # Most legit users should score < 0.5
        assert (scores < 0.5).mean() > 0.80

    def test_global_feature_importances(self, scorer):
        importances = scorer.get_global_feature_importances()
        assert len(importances) == 3
        for name in ["follow_ratio", "amplification_ratio", "posting_velocity"]:
            assert name in importances
            assert 0.0 <= importances[name] <= 1.0

    def test_risk_tiers(self, scorer):
        from src.models.scoring import CoordinationScorer
        assert CoordinationScorer._compute_risk_tier(0.90) == "CRITICAL"
        assert CoordinationScorer._compute_risk_tier(0.70) == "HIGH"
        assert CoordinationScorer._compute_risk_tier(0.50) == "MEDIUM"
        assert CoordinationScorer._compute_risk_tier(0.20) == "LOW"

    def test_unfitted_raises(self):
        from src.models.scoring import CoordinationScorer
        scorer = CoordinationScorer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scorer.predict_proba_batch(pd.DataFrame({"follow_ratio": [1.0], "amplification_ratio": [1.0], "posting_velocity": [1.0]}))

# ======================================================================
# Phase 4: Explanatory API Layer Tests
# ======================================================================

@pytest.fixture(scope="module")
def inference_pipeline(enriched_users_df, scorer):
    """Trained InferencePipeline."""
    from src.api.inference import InferencePipeline
    return InferencePipeline(scorer=scorer)


class TestInferencePipeline:
    """Tests for the inference API layer."""

    def test_analyze_bot_account(self, inference_pipeline, enriched_users_df, interactions_df):
        bot_id = enriched_users_df[enriched_users_df["is_bot"]].iloc[0]["user_id"]
        result = inference_pipeline.analyze_entity(
            bot_id, "account", enriched_users_df, interactions_df
        )
        assert result["entity_id"] == bot_id
        assert result["entity_type"] == "account"
        assert 0.0 <= result["coordination_likelihood"] <= 1.0
        assert result["risk_tier"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        assert isinstance(result["primary_flags"], list)
        assert len(result["primary_flags"]) > 0
        assert "feature_values" in result

    def test_analyze_legit_account(self, inference_pipeline, enriched_users_df, interactions_df):
        legit_id = enriched_users_df[~enriched_users_df["is_bot"]].iloc[0]["user_id"]
        result = inference_pipeline.analyze_entity(
            legit_id, "account", enriched_users_df, interactions_df
        )
        assert result["entity_type"] == "account"
        assert result["coordination_likelihood"] < 0.85  # Legit users shouldn't be CRITICAL

    def test_analyze_post(self, inference_pipeline, enriched_users_df, interactions_df, config):
        result = inference_pipeline.analyze_entity(
            config.attack_post_id, "post", enriched_users_df, interactions_df
        )
        assert result["entity_type"] == "post"
        assert result["risk_tier"] == "CRITICAL"
        assert len(result["primary_flags"]) > 0
        assert "cascade_features" in result

    def test_invalid_entity_type(self, inference_pipeline, enriched_users_df, interactions_df):
        with pytest.raises(ValueError, match="Unsupported entity_type"):
            inference_pipeline.analyze_entity(
                "x", "invalid_type", enriched_users_df, interactions_df
            )

    def test_nonexistent_user(self, inference_pipeline, enriched_users_df, interactions_df):
        with pytest.raises(ValueError, match="not found"):
            inference_pipeline.analyze_entity(
                "user_nonexistent", "account", enriched_users_df, interactions_df
            )

    def test_convenience_function(self, enriched_users_df, interactions_df, scorer, config):
        from src.api.inference import analyze_entity
        result = analyze_entity(
            config.attack_post_id, "post",
            enriched_users_df, interactions_df, scorer
        )
        assert result["entity_type"] == "post"
        assert result["risk_tier"] == "CRITICAL"


# ======================================================================
# Phase 5: Dashboard Smoke Tests
# ======================================================================

class TestDashboard:
    """Smoke tests for the Streamlit dashboard module."""

    def test_dashboard_importable(self):
        """Verify the dashboard module can be imported without errors."""
        from app.main import main, render_tab_monitoring, render_tab_cascade
        assert callable(main)
        assert callable(render_tab_monitoring)
        assert callable(render_tab_cascade)

    def test_load_data_function(self):
        """Verify the data loading function works."""
        from app.main import load_data
        users_df, interactions_df, config = load_data()
        assert len(users_df) == 10_000
        assert len(interactions_df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

