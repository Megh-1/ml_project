"""
Unified Data Loader
=====================
Maps all 5 real-world training datasets to the standard schema expected
by AccountFeatureExtractor and CoordinationScorer:

    user_id, followers, following, account_age_days,
    total_posts, total_retweets, is_bot

Missing columns are imputed with sensible defaults or medians.
"""

import os
import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Standard schema that the existing pipeline expects
STANDARD_COLUMNS = [
    "user_id", "followers", "following",
    "account_age_days", "total_posts", "total_retweets", "is_bot",
]


class UnifiedDataLoader:
    """Loads and maps all 5 training datasets to the standard schema.

    Each dataset has different column names and available features.
    This loader maps what's available and imputes the rest so that
    the existing AccountFeatureExtractor / CoordinationScorer pipeline
    works without modification.

    Args:
        data_dir: Path to the training_data directory.
    """

    def __init__(self, data_dir: str = "training_data") -> None:
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_merge(self) -> pd.DataFrame:
        """Load all 5 datasets, map to standard schema, and merge.

        Returns:
            pd.DataFrame with columns matching STANDARD_COLUMNS.
        """
        # Only datasets with REAL feature data (followers, following, posts).
        # Excluded:
        #   - bot_detection_data.csv (50k): missing 'following' — corrupts follow_ratio
        #   - twitter_human_bots_dataset.csv (37k): ALL features imputed — pure noise
        loaders = [
            self._load_fake_social_media,
            self._load_instafake,
            self._load_instagram_fake_profile,
        ]

        dfs = []
        for loader_fn in loaders:
            try:
                df = loader_fn()
                dfs.append(df)
                logger.info(
                    "Loaded %s: %d rows", loader_fn.__name__, len(df)
                )
            except Exception as e:
                logger.warning("Failed %s: %s", loader_fn.__name__, e)

        if not dfs:
            raise ValueError("No datasets could be loaded.")

        merged = pd.concat(dfs, ignore_index=True)

        # Drop rows without a label
        merged = merged.dropna(subset=["is_bot"])
        merged["is_bot"] = merged["is_bot"].astype(int)

        # Impute remaining NaN numerics with the merged median
        numeric_cols = ["followers", "following", "account_age_days",
                        "total_posts", "total_retweets"]
        for col in numeric_cols:
            median_val = merged[col].median()
            merged[col] = merged[col].fillna(median_val)

        # Ensure no negatives
        for col in numeric_cols:
            merged[col] = merged[col].clip(lower=0)

        logger.info(
            "Merged dataset: %d rows | bot=%d | human=%d",
            len(merged),
            merged["is_bot"].sum(),
            (merged["is_bot"] == 0).sum(),
        )
        return merged[STANDARD_COLUMNS]

    def get_train_val_test_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the merged DataFrame 70/15/15.

        Returns:
            (train_df, val_df, test_df) — each with all STANDARD_COLUMNS.
        """
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=42, stratify=df["is_bot"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=42,
            stratify=temp_df["is_bot"]
        )
        logger.info(
            "Split: train=%d | val=%d | test=%d",
            len(train_df), len(val_df), len(test_df),
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Private: Per-dataset loaders
    # ------------------------------------------------------------------

    def _load_bot_detection_data(self) -> pd.DataFrame:
        """bot_detection_data.csv — 50k Twitter rows.

        Has: User ID, Follower Count, Retweet Count, Created At, Bot Label.
        Missing: following, total_posts.
        """
        path = os.path.join(self.data_dir, "bot_detection_data.csv")
        raw = pd.read_csv(path)

        # Compute account_age_days from Created At
        now = datetime.now()
        created = pd.to_datetime(raw["Created At"], errors="coerce")
        age_days = (now - created).dt.days.clip(lower=1)

        return pd.DataFrame({
            "user_id": raw["User ID"].astype(str),
            "followers": raw["Follower Count"],
            "following": np.nan,              # will be imputed
            "account_age_days": age_days,
            "total_posts": 1.0,               # one tweet per row
            "total_retweets": raw["Retweet Count"],
            "is_bot": raw["Bot Label"],
        })

    def _load_fake_social_media(self) -> pd.DataFrame:
        """fake_social_media.csv — 3k mixed-platform rows.

        Has: followers, following, account_age_days, posts, is_fake.
        Missing: total_retweets.
        """
        path = os.path.join(self.data_dir, "fake_social_media.csv")
        raw = pd.read_csv(path)

        return pd.DataFrame({
            "user_id": [f"fsm_{i}" for i in range(len(raw))],
            "followers": raw["followers"],
            "following": raw["following"],
            "account_age_days": raw["account_age_days"],
            "total_posts": raw["posts"],
            "total_retweets": (raw["posts"] * 0.3).round(),  # estimate
            "is_bot": raw["is_fake"],
        })

    def _load_instafake(self) -> pd.DataFrame:
        """instafake_training_data.csv — ~2.6k Instagram rows.

        Has: userFollowerCount, userFollowingCount, userMediaCount, isFake.
        Missing: account_age_days, total_retweets.
        """
        path = os.path.join(self.data_dir, "instafake_training_data.csv")
        raw = pd.read_csv(path)

        return pd.DataFrame({
            "user_id": [f"insta_{i}" for i in range(len(raw))],
            "followers": raw["userFollowerCount"],
            "following": raw["userFollowingCount"],
            "account_age_days": np.nan,       # will be imputed
            "total_posts": raw["userMediaCount"],
            "total_retweets": 0.0,            # Instagram — no retweets
            "is_bot": raw["isFake"],
        })

    def _load_instagram_fake_profile(self) -> pd.DataFrame:
        """Instagram_fake_profile_dataset.csv — 5k Instagram rows.

        Has: #followers, #follows, #posts, fake.
        Missing: account_age_days, total_retweets.
        """
        path = os.path.join(
            self.data_dir, "Instagram_fake_profile_dataset.csv"
        )
        raw = pd.read_csv(path)

        return pd.DataFrame({
            "user_id": [f"igfp_{i}" for i in range(len(raw))],
            "followers": raw["#followers"],
            "following": raw["#follows"],
            "account_age_days": np.nan,       # will be imputed
            "total_posts": raw["#posts"],
            "total_retweets": 0.0,            # Instagram — no retweets
            "is_bot": raw["fake"],
        })

    def _load_twitter_human_bots(self) -> pd.DataFrame:
        """twitter_human_bots_dataset.csv — 37k Twitter rows.

        Has: id, account_type (bot/human).
        Missing: ALL feature columns — will be imputed with median.
        """
        path = os.path.join(
            self.data_dir, "twitter_human_bots_dataset.csv"
        )
        raw = pd.read_csv(path)

        is_bot = raw["account_type"].apply(
            lambda x: 1 if str(x).strip().lower() == "bot" else 0
        )

        return pd.DataFrame({
            "user_id": raw["id"].astype(str),
            "followers": np.nan,
            "following": np.nan,
            "account_age_days": np.nan,
            "total_posts": np.nan,
            "total_retweets": np.nan,
            "is_bot": is_bot,
        })


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = UnifiedDataLoader("training_data")
    df = loader.load_and_merge()
    print(f"\nTotal merged records: {len(df)}")
    print(f"Distribution:\n{df['is_bot'].value_counts()}")
    print(f"\nSample:\n{df.head()}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
