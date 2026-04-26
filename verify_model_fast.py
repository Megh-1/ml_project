import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from src.data.adapter import RealDataAdapter

def main():
    print("🚀 Starting Fast Terminal Verification...")
    
    # 1. Load Data
    print("📂 Loading data tables...")
    train_df = RealDataAdapter.load_table("train")
    val_df = RealDataAdapter.load_table("val")
    test_df = RealDataAdapter.load_table("test")
    
    feature_cols = RealDataAdapter.get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df["is_bot"].astype(int).values
    
    X_val = val_df[feature_cols].values
    y_val = val_df["is_bot"].astype(int).values
    
    X_test = test_df[feature_cols].values
    y_test = test_df["is_bot"].astype(int).values
    
    # 2. Build the Triple-Threat Ensemble
    print("🧠 Training Triple-Threat Ensemble (XGB + RF + HGB)...")
    spw = 3.0
    
    xgb = XGBClassifier(
        objective="binary:logistic", tree_method="hist", scale_pos_weight=spw,
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, class_weight="balanced",
        min_samples_leaf=4, random_state=42, n_jobs=-1
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=100, max_depth=5, l2_regularization=1.5, random_state=42
    )
    
    model = VotingClassifier(
        estimators=[('xgb', xgb), ('rf', rf), ('hgb', hgb)],
        voting='soft'
    )
    model.fit(X_train, y_train)
    
    # 3. Threshold Tuning on VAL set (Fast)
    print("🎯 Tuning threshold on Validation set...")
    val_probs = model.predict_proba(X_val)[:, 1]
    
    prec, rec, thresholds = precision_recall_curve(y_val, val_probs)
    p_arr, r_arr = prec[:-1], rec[:-1]
    
    valid_idx = np.where(p_arr >= 0.60)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(r_arr[valid_idx])]
        optimal_threshold = float(thresholds[best_idx])
    else:
        f1 = np.where((p_arr + r_arr) > 0, 2*p_arr*r_arr/(p_arr + r_arr), 0)
        optimal_threshold = float(thresholds[np.argmax(f1)])
        
    print(f"✅ Optimal Threshold found: {optimal_threshold:.4f}")
    
    # 4. Final Evaluation on Test Table
    print("🔬 Evaluating on Test set...")
    test_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (test_probs >= optimal_threshold).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n" + "="*40)
    print("📊 FAST TEST SET RESULTS (TERMINAL)")
    print("="*40)
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {pre:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print(f"Threshold: {optimal_threshold:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
