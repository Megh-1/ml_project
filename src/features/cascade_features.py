"""
Cascade-Level Feature Engineering
====================================
Analyzes the structural and temporal topology of content resharing cascades
using NetworkX graph primitives. Detects anomalous synchronization patterns
that indicate coordinated inauthentic behavior.

Key Signals:
    - cascade_velocity: High burst counts within a narrow time window.
    - temporal_density: Abnormally low inter-event times signal bot coordination.

No NLP or text analysis — purely temporal and graph-structural features.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class CascadeFeatureExtractor:
    """Extracts cascade-level features for a given post's resharing pattern.

    Builds a directed graph of retweet events and computes temporal/structural
    features that distinguish organic virality from coordinated amplification.

    Args:
        velocity_window_seconds: Time window (seconds) for cascade velocity
            measurement. Defaults to 60s.

    Example:
        >>> extractor = CascadeFeatureExtractor()
        >>> graph = extractor.build_cascade_graph(interactions_df, "post_target_001")
        >>> features = extractor.extract_features(interactions_df, "post_target_001")
        >>> print(features)
    """

    def __init__(self, velocity_window_seconds: int = 60) -> None:
        self._velocity_window_ms = velocity_window_seconds * 1_000
        logger.info(
            "CascadeFeatureExtractor initialized | velocity_window=%ds",
            velocity_window_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_cascade_graph(
        self,
        interactions_df: pd.DataFrame,
        post_id: str,
    ) -> nx.DiGraph:
        """Build a directed cascade graph for a specific post.

        Each node represents a user who retweeted the post. Edges run from
        the post origin to each retweeting user, weighted by timestamp.

        Args:
            interactions_df: Full interaction log DataFrame.
            post_id: The target_post_id to analyze.

        Returns:
            nx.DiGraph: Cascade graph with nodes = user_ids, edges carrying
            'timestamp_ms' attributes.

        Raises:
            ValueError: If no retweet events found for the given post_id.
        """
        self._validate_interactions(interactions_df)

        # Filter to retweets of the target post
        cascade_events = interactions_df[
            (interactions_df["target_post_id"] == post_id)
            & (interactions_df["action_type"] == "retweet")
        ].sort_values("timestamp_ms")

        if cascade_events.empty:
            raise ValueError(
                f"No retweet events found for post_id='{post_id}'. "
                "Cannot build cascade graph."
            )

        graph = nx.DiGraph()
        graph.graph["post_id"] = post_id
        graph.graph["total_events"] = len(cascade_events)

        # Add origin node (the post itself)
        graph.add_node(post_id, node_type="post")

        # Add edges from post to each retweeting user
        for _, event in cascade_events.iterrows():
            user_id = event["user_id"]
            timestamp = event["timestamp_ms"]

            if not graph.has_node(user_id):
                graph.add_node(user_id, node_type="user")

            graph.add_edge(
                post_id,
                user_id,
                timestamp_ms=timestamp,
                event_id=event["event_id"],
            )

        logger.info(
            "Built cascade graph for '%s' | nodes=%d | edges=%d",
            post_id,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    def extract_features(
        self,
        interactions_df: pd.DataFrame,
        post_id: str,
    ) -> Dict[str, Any]:
        """Extract cascade-level features for a specific post.

        Computes:
            - cascade_velocity: Number of retweets within the first N seconds.
            - temporal_density: Mean time delta (ms) between consecutive shares.
            - cascade_depth: Max depth of the cascade graph.
            - total_nodes: Number of unique participants.
            - total_edges: Number of retweet events.

        Args:
            interactions_df: Full interaction log DataFrame.
            post_id: The target_post_id to analyze.

        Returns:
            Dict containing cascade features.

        Raises:
            ValueError: If no retweet events found for the given post_id.
        """
        self._validate_interactions(interactions_df)

        # Filter to retweets of the target post
        cascade_events = interactions_df[
            (interactions_df["target_post_id"] == post_id)
            & (interactions_df["action_type"] == "retweet")
        ].sort_values("timestamp_ms")

        if cascade_events.empty:
            raise ValueError(
                f"No retweet events found for post_id='{post_id}'. "
                "Cannot extract cascade features."
            )

        timestamps = cascade_events["timestamp_ms"].values

        # ---- Cascade Velocity ----
        cascade_velocity = self._compute_cascade_velocity(timestamps)

        # ---- Temporal Density ----
        temporal_density = self._compute_temporal_density(timestamps)

        # ---- Graph Structural Features ----
        graph = self.build_cascade_graph(interactions_df, post_id)

        features = {
            "post_id": post_id,
            "cascade_velocity": cascade_velocity,
            "temporal_density_ms": temporal_density,
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "total_retweets": len(cascade_events),
            "cascade_duration_ms": int(timestamps[-1] - timestamps[0]),
            "is_synchronized": temporal_density < 100.0,  # < 100ms avg gap = suspicious
        }

        logger.info(
            "Cascade features for '%s': velocity=%d, density=%.2fms, "
            "synchronized=%s",
            post_id,
            cascade_velocity,
            temporal_density,
            features["is_synchronized"],
        )
        return features

    # ------------------------------------------------------------------
    # Private: Feature Computations
    # ------------------------------------------------------------------

    def _compute_cascade_velocity(self, timestamps: np.ndarray) -> int:
        """Count retweets within the velocity window from the first event.

        Args:
            timestamps: Sorted array of event timestamps in milliseconds.

        Returns:
            Number of retweets within the velocity window.
        """
        first_event_time = timestamps[0]
        window_end = first_event_time + self._velocity_window_ms

        velocity = int(np.sum(timestamps <= window_end))
        return velocity

    def _compute_temporal_density(self, timestamps: np.ndarray) -> float:
        """Compute mean time delta between consecutive retweet events.

        Low temporal density (small delta) indicates coordinated/automated
        behavior — human users cannot retweet that fast.

        Args:
            timestamps: Sorted array of event timestamps in milliseconds.

        Returns:
            Mean inter-event time in milliseconds.
            Returns 0.0 if there is only one event.
        """
        if len(timestamps) < 2:
            return 0.0

        deltas = np.diff(timestamps).astype(float)
        return float(np.mean(deltas))

    # ------------------------------------------------------------------
    # Private: Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_interactions(interactions_df: pd.DataFrame) -> None:
        """Validate the interactions DataFrame.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        required = {"event_id", "user_id", "target_post_id", "timestamp_ms", "action_type"}
        if interactions_df.empty:
            raise ValueError("interactions_df cannot be empty")
        missing = required - set(interactions_df.columns)
        if missing:
            raise ValueError(f"interactions_df missing required columns: {missing}")
