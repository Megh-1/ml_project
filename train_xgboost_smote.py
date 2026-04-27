import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report, precision_score, recall_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    DROP_COLS = ['id', 'label']

    train = pd.read_csv('dataset2/d2_train.csv')
    val   = pd.read_csv('dataset2/d2_val.csv')
    test  = pd.read_csv('dataset2/d2_test.csv')

    X_train = train.drop(columns=DROP_COLS).copy()
    y_train = train['label']
    
    X_val = val.drop(columns=DROP_COLS).copy()
    y_val = val['label']
    
    X_test = test.drop(columns=DROP_COLS).copy()
    y_test = test['label']

    print("Imputing missing values...")
    imp = SimpleImputer(strategy='median')
    X_train_imp = imp.fit_transform(X_train)
    X_val_imp   = imp.transform(X_val)
    X_test_imp  = imp.transform(X_test)
    
    print("Applying SMOTE oversampling to training set...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_imp, y_train)

    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"SMOTE class distribution: {np.bincount(y_train_smote)}")

    print("\nTraining XGBoost on SMOTE balanced data...")
    # Removing scale_pos_weight since data is purely balanced via SMOTE
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    # We do not use Validation Set for early stopping during SMOTE, just eval
    model.fit(
        X_train_smote, y_train_smote,
        eval_set=[(X_val_imp, y_val)],
        verbose=10
    )

    print("\n--- Validation Set Evaluation (Standard 0.5 Threshold) ---")
    val_preds = model.predict(X_val_imp)
    val_probs = model.predict_proba(X_val_imp)[:, 1]
    
    print(f"F1 Score: {f1_score(y_val, val_preds):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_val, val_probs):.4f}")
    print(f"PR AUC Score: {average_precision_score(y_val, val_probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_val, val_preds))
    
    # Phase 2: Threshold Tuning
    print("\n--- Threshold Analysis on Validation Set ---")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    for thresh in thresholds:
        t_preds = (val_probs >= thresh).astype(int)
        p = precision_score(y_val, t_preds)
        r = recall_score(y_val, t_preds)
        f1 = f1_score(y_val, t_preds)
        print(f"Threshold = {thresh:.2f}  |  Precision = {p:.4f}  |  Recall = {r:.4f}  |  F1 = {f1:.4f}")

    opt_thresh = 0.9 # We will assume 0.90 or whatever is desired for precision
    
    print("\n--- Test Set Evaluation (Standard 0.5 Threshold) ---")
    test_preds = model.predict(X_test_imp)
    test_probs = model.predict_proba(X_test_imp)[:, 1]
    
    print(f"F1 Score: {f1_score(y_test, test_preds):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, test_preds))

    print("\n--- Test Set Evaluation (High Precision Threshold = 0.90) ---")
    t_test_preds = (test_probs >= opt_thresh).astype(int)
    print("\nClassification Report:\n", classification_report(y_test, t_test_preds))

if __name__ == "__main__":
    main()
