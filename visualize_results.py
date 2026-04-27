import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import xgboost as xgb
import os

def main():
    print("Loading Dataset 1 (skew_fixed)...")
    train = pd.read_csv('skew_fixed/train_split.csv')
    test  = pd.read_csv('skew_fixed/test_split.csv')

    # Drop non-numeric and target
    X_train = train.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
    y_train = train['label']
    
    X_test = test.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
    y_test = test['label']

    print("Imputing...")
    imp = SimpleImputer(strategy='median')
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp  = imp.transform(X_test)
    
    # Calculate scale_pos_weight
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    print("Training Baseline XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_imp, y_train)

    # Predictions
    probs = model.predict_proba(X_test_imp)[:, 1]
    preds = model.predict(X_test_imp)

    # Create local visuals dir if not exists
    artifact_path = "visuals"
    os.makedirs(artifact_path, exist_ok=True)

    print("Generating Plots...")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Dataset 1 Baseline')
    plt.legend(loc="lower right")
    plt.savefig(f"{artifact_path}/roc_curve.png")
    plt.close()

    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    avg_prec = average_precision_score(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_prec:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Dataset 1 Baseline')
    plt.legend(loc="lower left")
    plt.savefig(f"{artifact_path}/pr_curve.png")
    plt.close()

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Dataset 1 Baseline')
    plt.savefig(f"{artifact_path}/confusion_matrix.png")
    plt.close()

    # 4. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    plt.figure(figsize=(10, 8))
    plt.title('Top 15 Feature Importances - Dataset 1 Baseline')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(f"{artifact_path}/feature_importance.png")
    plt.close()

    # 5. Probability Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(probs[y_test == 0], color="blue", label="Human", kde=True, stat="density", common_norm=False)
    sns.histplot(probs[y_test == 1], color="red", label="Bot", kde=True, stat="density", common_norm=False)
    plt.xlabel('Predicted Bot Probability')
    plt.title('Score Distribution by Class - Dataset 1 Baseline')
    plt.legend()
    plt.savefig(f"{artifact_path}/probability_dist.png")
    plt.close()

    print(f"All plots saved to {artifact_path}")

if __name__ == "__main__":
    main()
