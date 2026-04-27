import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    DROP_COLS = ['user_id', 'label', 'split',
                 'first_tweet_ts', 'last_tweet_ts',
                 'top_source', 'n_source_files']

    train = pd.read_csv('drive-download-20260426T085618Z-3-001/train_table.csv')
    val   = pd.read_csv('drive-download-20260426T085618Z-3-001/val_table.csv')
    test  = pd.read_csv('drive-download-20260426T085618Z-3-001/test_table.csv')

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
    sample_weights_train = np.where(y_train == 1, scale_pos_weight, 1)

    print("Training AdaBoost (shallow trees)...")
    base_estimator = DecisionTreeClassifier(max_depth=2)
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,
        random_state=42,
        learning_rate=0.1
    )
    
    model.fit(
        X_train_imp, y_train,
        sample_weight=sample_weights_train
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
