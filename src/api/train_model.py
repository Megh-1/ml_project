"""
Model Training Script
=======================
Trains the existing CoordinationScorer (DecisionTree) on the unified
real-world dataset instead of synthetic data.

Usage:
    python src/api/train_model.py
"""

import sys
import os
import logging
import joblib

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.data_loader import UnifiedDataLoader
from src.features.account_features import AccountFeatureExtractor
from src.models.scoring import CoordinationScorer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "trained_scorer.pkl"
)


def train_and_evaluate():
    """Full training pipeline using the existing architecture."""

    # ---- 1. Load & merge real datasets ----
    logger.info("=" * 60)
    logger.info("STEP 1: Loading all 5 datasets...")
    logger.info("=" * 60)

    loader = UnifiedDataLoader("training_data")
    merged = loader.load_and_merge()
    train_df, val_df, test_df = loader.get_train_val_test_split(merged)

    logger.info("Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))

    # ---- 2. Feature extraction using existing AccountFeatureExtractor ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Extracting features (follow_ratio, amplification_ratio, posting_velocity)...")
    logger.info("=" * 60)

    extractor = AccountFeatureExtractor()
    train_enriched = extractor.transform(train_df)
    val_enriched = extractor.transform(val_df)
    test_enriched = extractor.transform(test_df)

    # ---- 3. Train CoordinationScorer (DecisionTree) ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Training CoordinationScorer (DecisionTree, max_depth=5)...")
    logger.info("=" * 60)

    scorer = CoordinationScorer(max_depth=5, random_state=42)
    scorer.fit(train_enriched, train_enriched["is_bot"], run_cv=True)

    # ---- 4. Evaluate on all splits ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Evaluating on Train / Val / Test...")
    logger.info("=" * 60)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    results = {}
    for name, enriched_df in [("Train", train_enriched), ("Val", val_enriched), ("Test", test_enriched)]:
        y_true = enriched_df["is_bot"].astype(int).values
        scores = scorer.predict_proba_batch(enriched_df)
        y_pred = (scores >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        logger.info(
            "  %5s → Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
            name, acc, prec, rec, f1,
        )

    # ---- 5. Save trained scorer ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Saving trained model...")
    logger.info("=" * 60)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(scorer, MODEL_SAVE_PATH)
    logger.info("Model saved to: %s", MODEL_SAVE_PATH)

    # Also save metrics for the app to display
    metrics_path = MODEL_SAVE_PATH.replace(".pkl", "_metrics.pkl")
    joblib.dump(results, metrics_path)
    logger.info("Metrics saved to: %s", metrics_path)

    logger.info("\n✅ Training complete!")
    return scorer, results


if __name__ == "__main__":
    train_and_evaluate()
