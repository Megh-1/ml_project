import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report
import lightgbm as lgb
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
    
    pos_count = train['label'].sum()
    neg_count = len(train) - pos_count
    scale_pos_weight = neg_count / pos_count
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    print("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)]
    )

    print("\n--- Validation Set Evaluation ---")
    val_preds = model.predict(X_val_imp)
    val_probs = model.predict_proba(X_val_imp)[:, 1]
    
    print(f"F1 Score: {f1_score(y_val, val_preds):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_val, val_probs):.4f}")
    print(f"PR AUC Score: {average_precision_score(y_val, val_probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_val, val_preds))
    
    print("\n--- Test Set Evaluation ---")
    test_preds = model.predict(X_test_imp)
    test_probs = model.predict_proba(X_test_imp)[:, 1]
    
    print(f"F1 Score: {f1_score(y_test, test_preds):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, test_probs):.4f}")
    print(f"PR AUC Score: {average_precision_score(y_test, test_probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, test_preds))

if __name__ == "__main__":
    main()
