"""
Social Data Simulator
======================
Generates realistic synthetic social-media datasets with injected anomalies
for training and evaluating coordination-detection models.

Architecture:
    - Factory pattern for user/interaction generation.
    - Deterministic seeding for full experiment reproducibility.
    - Clear separation between organic activity and injected attacks.

Output DataFrames:
    users_df:
        user_id | followers | following | account_age_days | total_posts |
        total_retweets | is_bot

    interactions_df:
        event_id | user_id | target_post_id | timestamp_ms | action_type
"""

import uuid
import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from .config import SimulationConfig

logger = logging.getLogger(__name__)


class SocialDataSimulator:
    """Enterprise-grade synthetic social-media data generator.

    Produces two DataFrames simulating a social platform's user base and
    interaction firehose. Supports injection of bot accounts and coordinated
    attack patterns for anomaly-detection model development.

    Args:
        config: Simulation configuration object. Uses defaults if not provided.

    Example:
        >>> from src.data import SocialDataSimulator, SimulationConfig
        >>> sim = SocialDataSimulator(SimulationConfig(seed=123))
        >>> users_df, interactions_df = sim.run()
        >>> print(users_df.shape, interactions_df.shape)
    """

    # Supported interaction types for organic activity
    _ORGANIC_ACTION_TYPES = ["retweet", "like", "reply", "quote"]

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self._config = config or SimulationConfig()
        self._config.validate()
        self._rng = np.random.default_rng(self._config.seed)
        logger.info(
            "SocialDataSimulator initialized | legit=%d | bots=%d | seed=%d",
            self._config.n_legit_users,
            self._config.n_bot_users,
            self._config.seed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_users(self) -> pd.DataFrame:
        """Generate the user population DataFrame.

        Creates two sub-populations:
            1. Legitimate users with organic behavioral distributions.
            2. Bot users with anomalous fingerprints (high following, ~0 followers,
               extreme retweet ratios, young accounts).

        Returns:
            pd.DataFrame with columns: user_id, followers, following,
            account_age_days, total_posts, total_retweets, is_bot.
        """
        legit_df = self._generate_legit_users()
        bot_df = self._generate_bot_users()

        users_df = pd.concat([legit_df, bot_df], ignore_index=True)
        users_df = users_df.sample(frac=1, random_state=self._config.seed).reset_index(drop=True)

        logger.info(
            "Generated %d users | %d legit | %d bots",
            len(users_df),
            len(legit_df),
            len(bot_df),
        )
        return users_df

    def generate_interactions(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Generate the interaction event log DataFrame.

        Produces two layers of activity:
            1. Organic interactions spread over the full time range.
            2. Coordinated attack: n_attack_bots retweet the target post
               within the attack_window_ms.

        Args:
            users_df: The user population DataFrame (from generate_users).

        Returns:
            pd.DataFrame with columns: event_id, user_id, target_post_id,
            timestamp_ms, action_type.

        Raises:
            ValueError: If users_df is empty or missing required columns.
        """
        self._validate_users_df(users_df)

        organic_df = self._generate_organic_interactions(users_df)
        attack_df = self._generate_coordinated_attack(users_df)

        interactions_df = pd.concat([organic_df, attack_df], ignore_index=True)
        interactions_df = interactions_df.sort_values("timestamp_ms").reset_index(drop=True)

        logger.info(
            "Generated %d interactions | %d organic | %d attack",
            len(interactions_df),
            len(organic_df),
            len(attack_df),
        )
        return interactions_df

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full data generation pipeline.

        Convenience method that generates users first, then interactions.

        Returns:
            Tuple of (users_df, interactions_df).
        """
        users_df = self.generate_users()
        interactions_df = self.generate_interactions(users_df)
        return users_df, interactions_df

    # ------------------------------------------------------------------
    # Private: User Generation
    # ------------------------------------------------------------------

    def _generate_legit_users(self) -> pd.DataFrame:
        """Generate legitimate user accounts with organic distributions."""
        n = self._config.n_legit_users
        cfg = self._config

        followers = self._rng.integers(
            cfg.legit_followers_range[0], cfg.legit_followers_range[1], size=n
        )
        following = self._rng.integers(
            cfg.legit_following_range[0], cfg.legit_following_range[1], size=n
        )
        account_age = self._rng.integers(
            cfg.legit_account_age_range[0], cfg.legit_account_age_range[1], size=n
        )
        total_posts = self._rng.integers(1, 500, size=n)
        # Legitimate users: retweets are a fraction of their posts
        total_retweets = (total_posts * self._rng.uniform(0.05, 0.6, size=n)).astype(int)

        return pd.DataFrame(
            {
                "user_id": [f"user_legit_{i:05d}" for i in range(n)],
                "followers": followers,
                "following": following,
                "account_age_days": account_age,
                "total_posts": total_posts,
                "total_retweets": total_retweets,
                "is_bot": False,
            }
        )

    def _generate_bot_users(self) -> pd.DataFrame:
        """Generate bot accounts with anomalous behavioral fingerprints.

        Bot characteristics:
            - Near-zero followers (sockpuppet accounts).
            - Very high following count (mass-follow to appear legit).
            - Very young accounts.
            - Extremely high retweet-to-post ratio (amplification bots).
        """
        n = self._config.n_bot_users
        cfg = self._config

        followers = self._rng.integers(
            cfg.bot_followers_range[0], cfg.bot_followers_range[1] + 1, size=n
        )
        following = self._rng.integers(
            cfg.bot_following_range[0], cfg.bot_following_range[1], size=n
        )
        account_age = self._rng.integers(
            cfg.bot_account_age_range[0], cfg.bot_account_age_range[1] + 1, size=n
        )
        total_posts = self._rng.integers(1, 20, size=n)
        # Bots: retweet ratio >> 10x posts (amplification fingerprint)
        total_retweets = (
            total_posts * self._rng.uniform(cfg.bot_min_retweet_ratio, 50.0, size=n)
        ).astype(int)

        return pd.DataFrame(
            {
                "user_id": [f"user_bot_{i:05d}" for i in range(n)],
                "followers": followers,
                "following": following,
                "account_age_days": account_age,
                "total_posts": total_posts,
                "total_retweets": total_retweets,
                "is_bot": True,
            }
        )

    # ------------------------------------------------------------------
    # Private: Interaction Generation
    # ------------------------------------------------------------------

    def _generate_organic_interactions(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Generate organic interaction events spread across the time range."""
        n = self._config.n_organic_interactions
        cfg = self._config

        # Time range in milliseconds
        time_range_ms = cfg.time_range_days * 24 * 60 * 60 * 1_000
        base_timestamp = 1_700_000_000_000  # Anchor epoch (Nov 2023)

        user_ids = self._rng.choice(users_df["user_id"].values, size=n)
        post_ids = [f"post_{self._rng.integers(0, cfg.n_posts):05d}" for _ in range(n)]
        timestamps = base_timestamp + self._rng.integers(0, time_range_ms, size=n)
        actions = self._rng.choice(self._ORGANIC_ACTION_TYPES, size=n, p=[0.4, 0.35, 0.15, 0.1])

        return pd.DataFrame(
            {
                "event_id": [f"evt_{uuid.uuid4().hex[:12]}" for _ in range(n)],
                "user_id": user_ids,
                "target_post_id": post_ids,
                "timestamp_ms": timestamps,
                "action_type": actions,
            }
        )

    def _generate_coordinated_attack(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Inject a coordinated retweet attack by bot users.

        All attack bots retweet the same target post within the configured
        attack window (default: 3 seconds), creating a detectable temporal
        density anomaly.
        """
        cfg = self._config

        # Select bot users for the attack
        bot_users = users_df[users_df["is_bot"]]["user_id"].values
        attack_bots = self._rng.choice(bot_users, size=cfg.n_attack_bots, replace=False)

        # Attack occurs at a random point within the time range
        time_range_ms = cfg.time_range_days * 24 * 60 * 60 * 1_000
        base_timestamp = 1_700_000_000_000
        attack_start = base_timestamp + self._rng.integers(
            time_range_ms // 4, 3 * time_range_ms // 4
        )

        # All attack retweets land within the attack window
        timestamps = attack_start + self._rng.integers(0, cfg.attack_window_ms, size=cfg.n_attack_bots)

        return pd.DataFrame(
            {
                "event_id": [f"evt_atk_{uuid.uuid4().hex[:12]}" for _ in range(cfg.n_attack_bots)],
                "user_id": attack_bots,
                "target_post_id": cfg.attack_post_id,
                "timestamp_ms": timestamps,
                "action_type": "retweet",
            }
        )

    # ------------------------------------------------------------------
    # Private: Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_users_df(users_df: pd.DataFrame) -> None:
        """Validate the users DataFrame before generating interactions.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        required_columns = {"user_id", "is_bot"}
        if users_df.empty:
            raise ValueError("users_df cannot be empty")
        missing = required_columns - set(users_df.columns)
        if missing:
            raise ValueError(f"users_df missing required columns: {missing}")
