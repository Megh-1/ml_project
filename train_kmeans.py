import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Helper: load a dataset by name
# ─────────────────────────────────────────────
def load_dataset(name: str):
    """Returns (X_train, y_train, X_val, y_val, X_test, y_test)."""
    if name == "dataset1":
        # skew_fixed CSVs use 'user_id' as the ID column and may have a 'split' column
        DROP_COLS = [c for c in ['user_id', 'id', 'split', 'label'] if True]  # resolved below
        train = pd.read_csv('skew_fixed/train_split.csv')
        val   = pd.read_csv('skew_fixed/val_split.csv')
        test  = pd.read_csv('skew_fixed/test_split.csv')
        # Only drop columns that actually exist
        DROP_COLS = [c for c in ['user_id', 'id', 'split'] if c in train.columns]
        DROP_COLS_LABEL = DROP_COLS + ['label']
    elif name == "dataset2":
        train = pd.read_csv('dataset2/d2_train.csv')
        val   = pd.read_csv('dataset2/d2_val.csv')
        test  = pd.read_csv('dataset2/d2_test.csv')
        DROP_COLS = [c for c in ['id', 'user_id', 'split'] if c in train.columns]
        DROP_COLS_LABEL = DROP_COLS + ['label']
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X_train = train.drop(columns=DROP_COLS_LABEL)
    y_train = train['label'].values
    X_val   = val.drop(columns=DROP_COLS_LABEL)
    y_val   = val['label'].values
    X_test  = test.drop(columns=DROP_COLS_LABEL)
    y_test  = test['label'].values

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────
# Helper: evaluate cluster predictions
# ─────────────────────────────────────────────
def evaluate(split_name: str, y_true, y_pred, scores):
    print(f"\n--- {split_name} Results ---")
    print(f"F1 Score          : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    try:
        print(f"ROC-AUC           : {roc_auc_score(y_true, scores):.4f}")
        print(f"PR-AUC            : {average_precision_score(y_true, scores):.4f}")
    except Exception:
        print("ROC-AUC / PR-AUC  : N/A (single class in truth)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Human (0)", "Bot (1)"],
                                zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"            Pred Human  Pred Bot")
    print(f"Actual Human  {cm[0,0]:>8}  {cm[0,1]:>8}")
    print(f"Actual Bot    {cm[1,0]:>8}  {cm[1,1]:>8}")


# ─────────────────────────────────────────────
# Core K-Means pipeline
# ─────────────────────────────────────────────
def run_kmeans(dataset_name: str, k: int = 2):
    print(f"\n{'='*60}")
    print(f" K-MEANS (k={k}) on {dataset_name.upper()}")
    print(f"{'='*60}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name)

    print(f"Train : {X_train.shape}  Bots={y_train.sum()} / Humans={len(y_train)-y_train.sum()}")
    print(f"Val   : {X_val.shape}    Bots={y_val.sum()} / Humans={len(y_val)-y_val.sum()}")
    print(f"Test  : {X_test.shape}   Bots={y_test.sum()} / Humans={len(y_test)-y_test.sum()}")

    # ── 1. Impute & scale ──────────────────────────────────────
    # Keep only numeric columns (Dataset 1 has timestamp/string columns)
    X_train = X_train.select_dtypes(include=[np.number])
    X_val   = X_val[X_train.columns]
    X_test  = X_test[X_train.columns]
    print(f"Using {X_train.shape[1]} numeric features.")

    imp = SimpleImputer(strategy='median')
    X_train_imp = imp.fit_transform(X_train)
    X_val_imp   = imp.transform(X_val)
    X_test_imp  = imp.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_val_sc   = scaler.transform(X_val_imp)
    X_test_sc  = scaler.transform(X_test_imp)

    # ── 2. Fit K-Means on training data (unsupervised) ─────────
    print(f"\nFitting K-Means (k={k}) on training set (unsupervised)...")
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    km.fit(X_train_sc)

    train_clusters = km.labels_  # 0 or 1 (arbitrary)

    # ── 3. Assign cluster → bot/human via majority vote ────────
    #       Whichever cluster has more actual bots is the "bot cluster"
    bot_rate = {}
    for c in range(k):
        mask = train_clusters == c
        if mask.sum() == 0:
            bot_rate[c] = 0.0
        else:
            bot_rate[c] = y_train[mask].mean()

    # The cluster with the highest bot fraction gets label=1
    cluster_to_label = {c: int(bot_rate[c] >= 0.5) for c in range(k)}

    print("\nCluster assignment (by majority vote on training labels):")
    for c in range(k):
        assigned = cluster_to_label[c]
        n = (train_clusters == c).sum()
        print(f"  Cluster {c}: {n:>6} samples | bot_rate={bot_rate[c]:.3f} → assigned as {'BOT' if assigned==1 else 'HUMAN'}")

    # ── 4. Predict: assign clusters on val/test, map to labels ─
    def predict(X_sc):
        clusters = km.predict(X_sc)
        labels = np.array([cluster_to_label[c] for c in clusters])

        # Use negative distance to bot cluster centroid as "bot score"
        # (closer to bot centroid = higher probability of being a bot)
        bot_cluster = [c for c, l in cluster_to_label.items() if l == 1]
        if bot_cluster:
            bc = bot_cluster[0]
            # Euclidean distance to bot centroid
            dists = np.linalg.norm(X_sc - km.cluster_centers_[bc], axis=1)
            scores = -dists  # negative distance: higher = more bot-like
            # Normalise to [0,1] for interpretability
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = labels.astype(float)

        return labels, scores

    train_preds, train_scores = predict(X_train_sc)
    val_preds,   val_scores   = predict(X_val_sc)
    test_preds,  test_scores  = predict(X_test_sc)

    # ── 5. Evaluate ────────────────────────────────────────────
    evaluate("Training Set", y_train, train_preds, train_scores)
    evaluate("Validation Set", y_val,   val_preds,   val_scores)
    evaluate("Test Set",       y_test,  test_preds,  test_scores)

    # ── 6. Inertia / cluster quality ──────────────────────────
    print(f"\nK-Means Inertia (within-cluster sum of squares): {km.inertia_:.2f}")
    print("(Lower = tighter, more distinct clusters)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Run on Dataset 1 (skew_fixed — the hard, realistic dataset)
    run_kmeans("dataset1", k=2)

    # Run on Dataset 2 (raw profiles — the simpler, bot-heavy dataset)
    run_kmeans("dataset2", k=2)

    print("\n\nDone.")
