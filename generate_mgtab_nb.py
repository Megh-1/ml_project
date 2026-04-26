import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell("# MGStBot-large Bot Detection\\n**Architecture**: LightGBM + XGBoost + Random Forest Ensemble with Threshold Tuning\\n**Data**: PyTorch Tensor Node Features (788-dim)"),
    
    new_markdown_cell("## Cell 1 — Install Dependencies"),
    new_code_cell("!pip install torch lightgbm xgboost scikit-learn pandas numpy matplotlib seaborn -q"),
    
    new_markdown_cell("## Cell 2 — Imports & Seed"),
    new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
print('All imports OK')"""),

    new_markdown_cell("## Cell 3 — Load MGStBot-large Data"),
    new_code_cell("""# Load PyTorch tensors
DATA_DIR = 'data/MGStBot-large/'

features_tensor = torch.load(os.path.join(DATA_DIR, 'large_features.pt'))
labels_tensor = torch.load(os.path.join(DATA_DIR, 'labels_bot.pt'))

print('Features shape:', features_tensor.shape)
print('Labels shape:', labels_tensor.shape)

# Convert to NumPy for Scikit-Learn / LightGBM / XGBoost
X = features_tensor.numpy()
y = labels_tensor.numpy()

# The dataset contains 410,199 nodes, but only the first 10,199 are labelled users.
# We slice X to match the labels length.
X = X[:len(y)]

# Check label distribution
unique, counts = np.unique(y, return_counts=True)
print('\\nLabel distribution:')
for u, c in zip(unique, counts):
    print(f'Label {u}: {c}')
"""),

    new_markdown_cell("## Cell 4 — Train/Val/Test Split"),
    new_code_cell("""# Using a 70/15/15 split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp)

print(f'Train samples: {X_train.shape[0]}')
print(f'Val samples:   {X_val.shape[0]}')
print(f'Test samples:  {X_test.shape[0]}')"""),

    new_markdown_cell("## Cell 5 — Feature Scaling"),
    new_code_cell("""scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

num_bots   = (y_train == 1).sum()
num_humans = (y_train == 0).sum()
scale_w    = num_humans / num_bots

print(f'Bots in Train: {num_bots}')
print(f'Humans in Train: {num_humans}')
print(f'scale_pos_weight: {scale_w:.4f}')"""),

    new_markdown_cell("## Cell 6 — Train LightGBM"),
    new_code_cell("""lgb_model = lgb.LGBMClassifier(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    num_leaves        = 31,
    max_depth         = -1,
    scale_pos_weight  = scale_w,
    random_state      = SEED,
    n_jobs            = -1,
    verbose           = -1
)

lgb_model.fit(
    X_train, y_train,
    eval_set  = [(X_val, y_val)],
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)
print(f'Best iteration: {lgb_model.best_iteration_}')"""),

    new_markdown_cell("## Cell 7 — Train XGBoost"),
    new_code_cell("""xgb_model = XGBClassifier(
    n_estimators          = 1000,
    learning_rate         = 0.05,
    max_depth             = 6,
    scale_pos_weight      = scale_w,
    eval_metric           = 'logloss',
    early_stopping_rounds = 50,
    random_state          = SEED,
    n_jobs                = -1,
    verbosity             = 0
)

xgb_model.fit(
    X_train, y_train,
    eval_set = [(X_val, y_val)],
    verbose  = 100
)
print(f'Best iteration: {getattr(xgb_model, "best_iteration", getattr(xgb_model, "best_ntree_limit", "N/A"))}')"""),

    new_markdown_cell("## Cell 8 — Train Random Forest"),
    new_code_cell("""rf_model = RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = None,
    class_weight     = 'balanced_subsample',
    random_state     = SEED,
    n_jobs           = -1
)

rf_model.fit(X_train, y_train)
print('Random Forest trained')"""),

    new_markdown_cell("## Cell 9 — Ensemble + Threshold Tuning (Validation Set)"),
    new_code_cell("""# Soft vote ensemble on validation set
lgb_val = lgb_model.predict_proba(X_val)[:, 1]
xgb_val = xgb_model.predict_proba(X_val)[:, 1]
rf_val  = rf_model.predict_proba(X_val)[:, 1]

val_probs = 0.40 * lgb_val + 0.40 * xgb_val + 0.20 * rf_val

# --- Threshold tuning ---
PRECISION_FLOOR = 0.72   # lower this if recall is too low, raise to boost precision

precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, val_probs)

best_thresh = 0.5
best_f1     = 0.0

for p, r, t in zip(precision_arr[:-1], recall_arr[:-1], thresholds):
    if p >= PRECISION_FLOOR:
        f = 2 * p * r / (p + r + 1e-9)
        if f > best_f1:
            best_f1     = f
            best_thresh = t

val_preds = (val_probs >= best_thresh).astype(int)

print(f'Optimal threshold : {best_thresh:.4f}')
print(f'Val Precision     : {precision_score(y_val, val_preds):.4f}')
print(f'Val Recall        : {recall_score(y_val, val_preds):.4f}')
print(f'Val F1            : {f1_score(y_val, val_preds):.4f}')

# Plot PR curve
plt.figure(figsize=(8, 5))
plt.plot(recall_arr, precision_arr, lw=2, label='PR Curve')
plt.axhline(y=PRECISION_FLOOR, color='red', linestyle='--', label=f'Precision floor = {PRECISION_FLOOR}')
plt.scatter([recall_score(y_val, val_preds)],
            [precision_score(y_val, val_preds)],
            color='green', zorder=5, s=100, label=f'Chosen (t={best_thresh:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — Validation Set (MGStBot-large)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mgstbot_large_pr_curve.png', dpi=150)
plt.show()"""),

    new_markdown_cell("## Cell 10 — Final Evaluation on Test Set"),
    new_code_cell("""# Ensemble on test set
lgb_test  = lgb_model.predict_proba(X_test)[:, 1]
xgb_test  = xgb_model.predict_proba(X_test)[:, 1]
rf_test   = rf_model.predict_proba(X_test)[:, 1]

test_probs = 0.40 * lgb_test + 0.40 * xgb_test + 0.20 * rf_test
test_preds = (test_probs >= best_thresh).astype(int)

print('=' * 45)
print(f'  Accuracy  : {accuracy_score(y_test, test_preds):.4f}')
print(f'  Precision : {precision_score(y_test, test_preds):.4f}')
print(f'  Recall    : {recall_score(y_test, test_preds):.4f}')
print(f'  F1 Score  : {f1_score(y_test, test_preds):.4f}')
print(f'  AUC-ROC   : {roc_auc_score(y_test, test_probs):.4f}')
print('=' * 45)

# Confusion matrix
cm   = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['Human', 'Bot'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix — Test Set (MGStBot-large)')
plt.tight_layout()
plt.savefig('mgstbot_large_confusion_matrix.png', dpi=150)
plt.show()""")
])

with open('mgtab_bot_detection.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("Notebook generated successfully.")
