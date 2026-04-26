import nbformat

with open('mgtab_bot_detection.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

new_cell_10_source = """# Ensemble on test set
lgb_test  = lgb_model.predict_proba(X_test)[:, 1]
xgb_test  = xgb_model.predict_proba(X_test)[:, 1]
rf_test   = rf_model.predict_proba(X_test)[:, 1]

test_probs = 0.40 * lgb_test + 0.40 * xgb_test + 0.20 * rf_test

# Get hard predictions for all models
lgb_preds = (lgb_test >= best_thresh).astype(int)
xgb_preds = (xgb_test >= best_thresh).astype(int)
rf_preds  = (rf_test >= best_thresh).astype(int)
ens_preds = (test_probs >= best_thresh).astype(int)

# Compile results
def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_prob)
    }

results = {
    'LightGBM': get_metrics(y_test, lgb_preds, lgb_test),
    'XGBoost': get_metrics(y_test, xgb_preds, xgb_test),
    'Random Forest': get_metrics(y_test, rf_preds, rf_test),
    'Ensemble': get_metrics(y_test, ens_preds, test_probs)
}

results_df = pd.DataFrame(results).T
print("=" * 60)
print("             MULTI-MODEL COMPARISON (TEST SET)      ")
print("=" * 60)
print(results_df.round(4))
print("=" * 60)

# Confusion matrix for Ensemble
cm   = confusion_matrix(y_test, ens_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['Human', 'Bot'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix — Ensemble (MGStBot-large)')
plt.tight_layout()
plt.savefig('mgstbot_large_confusion_matrix.png', dpi=150)
plt.show()"""

for cell in nb.cells:
    if cell.cell_type == 'code' and '# Ensemble on test set' in cell.source:
        cell.source = new_cell_10_source

with open('mgtab_bot_detection.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("Updated Cell 10 successfully.")
