"""
Simulation Configuration Module
================================
Centralizes all tuneable parameters for the synthetic data generation pipeline.
Uses Python dataclasses for type-safe, immutable configuration objects that can
be serialized, validated, and injected across the system.

Enterprise Design Pattern: Configuration-as-Code with single source of truth.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable configuration for the SocialDataSimulator.

    All parameters governing synthetic user populations, interaction patterns,
    and coordinated attack injection are centralized here to enable reproducible
    experiments and A/B testing of detection thresholds.

    Attributes:
        n_legit_users: Number of legitimate (organic) user accounts to generate.
        n_bot_users: Number of bot accounts to inject into the population.
        n_posts: Total number of unique posts in the content universe.
        n_organic_interactions: Number of organic interactions to generate.
        attack_post_id: The specific post_id targeted by the coordinated attack.
        attack_window_ms: Time window (ms) within which bots execute the attack.
        n_attack_bots: Number of bots participating in the coordinated attack.
        time_range_days: Span of the simulation timeline in days.
        seed: Random seed for full reproducibility.
        bot_followers_range: (min, max) follower count for bot accounts.
        bot_following_range: (min, max) following count for bot accounts.
        bot_account_age_range: (min, max) account age in days for bots.
        legit_followers_range: (min, max) follower count for legit accounts.
        legit_following_range: (min, max) following count for legit accounts.
        legit_account_age_range: (min, max) account age in days for legit users.
    """

    # Population sizing
    n_legit_users: int = 9_500
    n_bot_users: int = 500
    n_posts: int = 10_000
    n_organic_interactions: int = 50_000

    # Coordinated attack parameters
    attack_post_id: str = "post_target_001"
    attack_window_ms: int = 3_000  # 3-second burst
    n_attack_bots: int = 500
    time_range_days: int = 30

    # Reproducibility
    seed: int = 42

    # Bot behavioral fingerprint
    bot_followers_range: tuple = (0, 5)
    bot_following_range: tuple = (500, 5_000)
    bot_account_age_range: tuple = (1, 30)
    bot_min_retweet_ratio: float = 10.0

    # Legitimate user behavioral ranges
    legit_followers_range: tuple = (10, 50_000)
    legit_following_range: tuple = (10, 2_000)
    legit_account_age_range: tuple = (30, 3_650)  # 1 month to 10 years

    @property
    def total_users(self) -> int:
        """Total user population size."""
        return self.n_legit_users + self.n_bot_users

    def validate(self) -> None:
        """Validate configuration invariants.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.n_legit_users <= 0:
            raise ValueError(f"n_legit_users must be positive, got {self.n_legit_users}")
        if self.n_bot_users <= 0:
            raise ValueError(f"n_bot_users must be positive, got {self.n_bot_users}")
        if self.n_attack_bots > self.n_bot_users:
            raise ValueError(
                f"n_attack_bots ({self.n_attack_bots}) cannot exceed "
                f"n_bot_users ({self.n_bot_users})"
            )
        if self.attack_window_ms <= 0:
            raise ValueError(f"attack_window_ms must be positive, got {self.attack_window_ms}")
        if self.time_range_days <= 0:
            raise ValueError(f"time_range_days must be positive, got {self.time_range_days}")
