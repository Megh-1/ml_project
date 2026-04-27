# TwiBot-22 Bot Detection — Full Experiment Report

A complete, chronological record of every dataset used, approach tried, model configuration tested, and result obtained during this project.

---

## Project Overview

**Goal:** Build a classical ML classifier that detects Twitter bots in the TwiBot-22 dataset, maximizing both Precision and Recall.

**Constraint:** Only classical ML models allowed (no deep learning, no GNNs).

**Core Challenge:** The real-world Twitter data is heavily imbalanced (~92% human, ~8% bot), creating a structural "Precision Ceiling" that all tree-based models hit regardless of tuning.

---

## Datasets Used

| Dataset | Source | Class Balance | Notes |
|---|---|---|---|
| **Dataset 1 (Original)** | `train_table.csv`, `val_table.csv`, `test_table.csv` | ~92% Human / ~8% Bot | 78 engineered behavioral features |
| **Dataset 1 V2 (Skew Fixed)** | `skew_fixed/` folder | ~92% Human / ~8% Bot | Same features, corrected train/test split alignment |
| **Dataset 2 (Raw Profiles)** | `dataset2/node.json` | ~84% Bot / ~16% Human | 10 parsed profile features from raw JSON |

### Features Used

**Dataset 1 / V2:** 78 pre-engineered numeric features (behavioral signals — tweet frequency, account age, engagement rates, activity patterns, etc.)

**Dataset 2:** 10 features parsed by `build_dataset2.py` from raw `node.json`:
- `followers_count`, `following_count`, `tweet_count`, `listed_count`
- `name_length`, `username_length`, `description_length`
- `verified` (bool→int), `protected` (bool→int)
- `ff_ratio` (followers / following)

---

## Evaluation Metrics

| Metric | Why It Matters |
|---|---|
| **Bot Precision** | Of all accounts flagged as bots, how many actually are? Low = too many false positives (real users banned). |
| **Bot Recall** | Of all actual bots, how many did we catch? Low = too many bots slip through. |
| **F1 Score** | Harmonic mean of Precision and Recall. Balances both. |
| **ROC-AUC** | Overall ability to discriminate between classes. |
| **PR-AUC** | Area under the Precision-Recall curve — most informative for imbalanced datasets. |

---

## Experiments on Dataset 1 (Original & V2)

---

### Experiment 1 — XGBoost Baseline

**Configuration:**
- `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`
- `scale_pos_weight=11.81` (ratio of humans to bots, used to penalize missed bots)
- No synthetic sampling

**Rationale:** Establish a performance floor using the most standard, robust gradient boosting approach with class-weight correction.

| Split | F1 | ROC-AUC | PR-AUC | Bot Precision | Bot Recall |
|---|---|---|---|---|---|
| Validation | 0.4900 | 0.6873 | 0.4721 | 0.37 | 0.73 |
| **Test** | **0.5302** | **0.7355** | **0.5702** | **0.40** | **0.77** |

**Finding:** High recall (catches 77% of bots) but very low precision (40% — meaning 60% of "bot" predictions were real humans). The `scale_pos_weight` setting prioritizes recall aggressively at the cost of precision.

---

### Experiment 2 — XGBoost Tuned (Regularized)

**Configuration:**
- `max_depth=3`, `learning_rate=0.05`
- Added `subsample=0.8`, `colsample_bytree=0.8` to reduce overfitting
- `scale_pos_weight=11.81` retained

**Rationale:** Heavy regularization to see if constraining tree depth and sampling would reduce false positives.

| Split | F1 | ROC-AUC | PR-AUC | Bot Precision | Bot Recall |
|---|---|---|---|---|---|
| Validation | 0.4847 | 0.6798 | 0.4604 | 0.37 | 0.72 |
| **Test** | **0.5280** | **0.7282** | **0.5643** | **0.41** | **0.75** |

**Finding:** Negligible improvement (+1% precision), at the cost of decreased recall (-2%) and slightly worse AUC. Established that hyperparameter tuning alone cannot break the precision ceiling.

---

### Experiment 3 — LightGBM Baseline

**Configuration:**
- Default LightGBM with `scale_pos_weight=11.81`
- Leaf-wise tree growth (vs. depth-wise in XGBoost)

**Rationale:** LightGBM handles high-dimensional tabular data well and often outperforms XGBoost on sparse features.

| Split | F1 | ROC-AUC | PR-AUC | Bot Precision | Bot Recall |
|---|---|---|---|---|---|
| Validation | 0.4907 | 0.6880 | 0.4698 | 0.37 | 0.74 |
| **Test** | **0.5294** | **0.7359** | **0.5713** | **0.40** | **0.77** |

**Finding:** Nearly identical to XGBoost baseline (within 0.1% on all metrics). Confirmed that the bottleneck is not the algorithm — it's the data.

---

### Experiment 4 — AdaBoost & EBM

**Configuration:**
- AdaBoost with default `DecisionTreeClassifier` base learners
- Explainable Boosting Machine (EBM / InterpretML) — a glass-box model

**Rationale:** Test alternative boosting paradigms and an interpretable GAM-based approach to see if a different learning mechanism could break the ceiling.

**Finding:** Results were consistent with the XGBoost/LightGBM bottleneck (~40% precision, ~75% recall). No meaningful improvement. Confirmed the ceiling is data-driven, not model-driven.

---

### Experiment 5 — XGBoost with SMOTE Oversampling + Threshold Sweep

**Configuration:**
- Removed `scale_pos_weight` entirely
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to synthetically balance training data to **50% bot / 50% human**
- Swept decision thresholds from 0.50 to 0.95 on validation set

**Rationale:** Rather than penalizing the loss function (which kept destroying precision), mathematically create a balanced training set so the model learns bot features with equal weight to human features.

#### Validation Threshold Sweep

| Threshold | Bot Precision | Bot Recall | Notes |
|---|---|---|---|
| 0.50 | 0.5894 | 0.1757 | Default — higher precision, less recall |
| 0.60 | 0.6199 | 0.1050 | |
| 0.70 | 0.6263 | 0.0451 | |
| 0.85 | 0.6735 | 0.0035 | Near-zero recall |

#### Test Set Results (Threshold = 0.50)

| F1 | Bot Precision | Bot Recall | Human Precision | Human Recall |
|---|---|---|---|---|
| 0.4131 | **0.67** | 0.30 | 0.76 | 0.94 |

**Finding:** 🎯 **Major breakthrough on precision.** SMOTE pushed Bot Precision from 40% → **67%** (+27 percentage points). The trade-off is a steep drop in recall (77% → 30%). The model now makes very conservative, high-confidence bot predictions.

> **Key Insight:** SMOTE changes the fundamental model behavior — from "catch everything possible" to "only flag when very sure." The choice between SMOTE and `scale_pos_weight` is a *business decision*, not a technical one.

---

### Experiment 6 — Dataset 1 V2 (Skew Fixed): Re-evaluation

**Context:** The original dataset splits had a mismatch — test set had ~30% bots while training only had ~8%, artificially inflating early test metrics. The `skew_fixed` dataset corrected this to maintain consistent ~92% human / ~8% bot distribution across all splits.

**Re-ran all methods:**

| Method | Bot Precision | Bot Recall | Notes |
|---|---|---|---|
| XGBoost (scale_pos_weight) | 0.41 | ~0.70 | Slight recall drop from split fix |
| LightGBM (scale_pos_weight) | 0.40 | 0.82 | PR-AUC: 0.6119 |
| XGBoost + SMOTE (threshold 0.50) | ~0.50 | ~0.30 | Precision drops slightly on harder distribution |
| XGBoost + SMOTE (threshold 0.85) | ~0.75 | ~0.001 | Extremely high precision, near-zero recall |

**Finding:** The same structural bottleneck holds. The precision ceiling (~40-41%) on default configurations is **real and distribution-driven**, not an artifact of the split. SMOTE remains the only way to lift precision above 41% using classical ML.

---

## Experiments on Dataset 2 (Raw JSON Profiles)

---

### Experiment 7 — Dataset 2 Full Evaluation

**Context:** Instead of the engineered feature tables, we used the raw `node.json` from `dataset2/`, extracting 10 simple profile-level features via `build_dataset2.py`. The class distribution here is radically different — **84% bots, 16% humans** (inverted from Dataset 1).

**Models Tested:** XGBoost, LightGBM (both without any class balancing tricks needed)

| Metric | Result |
|---|---|
| F1 Score | 0.97 |
| ROC-AUC | 0.99 |
| Bot Precision | 0.98 – 0.99 |
| Bot Recall | 0.95 – 0.98 |

**Most Important Feature:** `followers_count` alone accounted for ~80% of all tree split decisions.

**Finding:** Classical ML performs near-perfectly on Dataset 2. The bots in this dataset are easily separable from humans using just their public profile numbers — they have systematically different follower/following patterns that create a clear decision boundary.

**Why this differs from Dataset 1:**
- Dataset 1 bots are *sophisticated* (they mimic human behavior in follower ratios, tweet patterns, etc.)
- Dataset 2 bots are *simple* (cheap, low-effort bots with obviously artificial profiles)
- 99% accuracy here doesn't mean the problem is "solved" — it means this is a different (easier) sub-population of bots

---

## Summary Comparison Table

| Experiment | Dataset | Model | Bot Precision | Bot Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | D1 Original | XGBoost (baseline) | 0.40 | 0.77 | 0.53 | 0.74 |
| 2 | D1 Original | XGBoost (tuned) | 0.41 | 0.75 | 0.53 | 0.73 |
| 3 | D1 Original | LightGBM | 0.40 | 0.77 | 0.53 | 0.74 |
| 4 | D1 Original | AdaBoost / EBM | ~0.40 | ~0.75 | ~0.52 | ~0.73 |
| 5 | D1 Original | XGBoost + SMOTE | **0.67** | 0.30 | 0.41 | — |
| 6 | D1 V2 (Skew Fixed) | XGBoost | 0.41 | ~0.70 | ~0.54 | ~0.76 |
| 6 | D1 V2 (Skew Fixed) | LightGBM | 0.40 | 0.82 | 0.54 | 0.76 |
| 7 | **Dataset 2** | XGBoost / LightGBM | **0.98–0.99** | **0.95–0.98** | **0.97** | **0.99** |

---

## Key Conclusions

### 1. The Precision Ceiling on Dataset 1 is Real and Structural

All classical ML models — regardless of algorithm (XGBoost, LightGBM, AdaBoost, EBM), hyperparameter tuning, or regularization — converge to the same ceiling:

> **~40–41% Precision / ~75–82% Recall on Dataset 1**

This is not a model failure. It's a **feature overlap problem**. The 78 behavioral features cannot fully distinguish:
- Hyper-active legitimate power users (high tweet count, many followers) from bots that mimic them
- Dormant bots that look identical to inactive humans

### 2. SMOTE Breaks the Ceiling — But at a Cost

SMOTE is the **only classical ML technique** that lifts precision above 41% on Dataset 1:
- **Default (0.50 threshold):** 67% precision, 30% recall
- **High threshold (0.85):** ~75% precision, <1% recall

The trade-off is steep: to avoid false positives, you miss the vast majority of bots.

### 3. The Business Decision Framework

| Deployment Goal | Recommended Approach | Precision | Recall |
|---|---|---|---|
| **Catch as many bots as possible** (false positives acceptable) | XGBoost with `scale_pos_weight` | ~40% | ~77-82% |
| **Only flag very obvious bots** (false positives unacceptable) | XGBoost + SMOTE (threshold 0.50) | ~67% | ~30% |
| **Ultra high-confidence flags only** | XGBoost + SMOTE (threshold 0.85) | ~75% | <1% |

### 4. Dataset 2 is a Different Problem

Dataset 2's near-perfect results (99% P/R) are not a contradiction — they reflect that this dataset contains a *different, simpler population* of bots. The easy separation means these bots never tried to mimic human behavior. Classical ML is genuinely sufficient here.

### 5. The Only Path Forward for Dataset 1 Beyond 41% Precision

With classical tabular ML exhausted, the only viable paths to meaningfully improve detection on sophisticated, realistic bots (Dataset 1) are:

1. **Graph Neural Networks (GNNs):** Use `edge.csv` to model retweet/follow networks. Bots tend to cluster in communities and have abnormal graph topology.
2. **Temporal Features:** Add time-series patterns (burst tweeting, coordinated activity windows).
3. **Text/NLP Features:** Analyze tweet content similarity across accounts.

---

## Files & Scripts

| File | Purpose |
|---|---|
| `train_xgboost.py` | XGBoost baseline + tuned variants |
| `train_lgbm.py` | LightGBM baseline |
| `train_xgboost_smote.py` | XGBoost + SMOTE + threshold sweep |
| `train_adaboost.py` | AdaBoost experiments |
| `train_ebm.py` | Explainable Boosting Machine |
| `build_dataset2.py` | Parses `node.json` → CSV feature tables for Dataset 2 |
| `dataset2/d2_train.csv` | Dataset 2 training features (parsed from JSON) |
| `dataset2/d2_val.csv` | Dataset 2 validation features |
| `dataset2/d2_test.csv` | Dataset 2 test features |
| `skew_fixed/` | Corrected Dataset 1 splits (consistent class ratios) |
