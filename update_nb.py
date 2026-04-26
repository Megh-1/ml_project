import json

with open('mgtab_bot_detection.ipynb', 'r') as f:
    nb = json.load(f)

# Find index of Random Forest cell
rf_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('Random Forest trained' in line for line in cell.get('source', [])):
        rf_idx = i
        break

# Cells to add
new_cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8a — Train CatBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cb_model = CatBoostClassifier(\n",
    "    n_estimators      = 1000,\n",
    "    learning_rate     = 0.05,\n",
    "    depth             = 6,\n",
    "    scale_pos_weight  = scale_w,\n",
    "    random_state      = SEED,\n",
    "    verbose           = False,\n",
    "    thread_count      = -1\n",
    ")\n",
    "\n",
    "cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))\n",
    "print('CatBoost trained')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8b — Train Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr_model = LogisticRegression(\n",
    "    random_state = SEED,\n",
    "    class_weight = 'balanced',\n",
    "    max_iter     = 1000\n",
    ")\n",
    "\n",
    "lr_model.fit(X_train, y_train)\n",
    "print('Logistic Regression trained')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8c — Train Extra Trees"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "et_model = ExtraTreesClassifier(\n",
    "    n_estimators     = 300,\n",
    "    class_weight     = 'balanced',\n",
    "    random_state     = SEED,\n",
    "    n_jobs           = -1\n",
    ")\n",
    "\n",
    "et_model.fit(X_train, y_train)\n",
    "print('Extra Trees trained')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8d — Train MLP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "mlp_model = MLPClassifier(\n",
    "    hidden_layer_sizes = (128, 64),\n",
    "    max_iter           = 500,\n",
    "    random_state       = SEED\n",
    ")\n",
    "\n",
    "mlp_model.fit(X_train, y_train)\n",
    "print('MLP trained')"
   ]
  }
]

# Insert new cells
for i, new_cell in enumerate(new_cells):
    nb['cells'].insert(rf_idx + 1 + i, new_cell)

# Find Ensemble cell (now shifted)
ens_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('val_probs = 0.40 * lgb_val' in line for line in cell.get('source', [])):
        ens_idx = i
        break

ens_source = [
    "# Soft vote ensemble on validation set\n",
    "lgb_val = lgb_model.predict_proba(X_val)[:, 1]\n",
    "xgb_val = xgb_model.predict_proba(X_val)[:, 1]\n",
    "rf_val  = rf_model.predict_proba(X_val)[:, 1]\n",
    "cb_val  = cb_model.predict_proba(X_val)[:, 1]\n",
    "lr_val  = lr_model.predict_proba(X_val)[:, 1]\n",
    "et_val  = et_model.predict_proba(X_val)[:, 1]\n",
    "mlp_val = mlp_model.predict_proba(X_val)[:, 1]\n",
    "\n",
    "val_probs = (lgb_val + xgb_val + rf_val + cb_val + lr_val + et_val + mlp_val) / 7.0\n",
    "\n",
    "# --- Threshold tuning ---\n",
    "PRECISION_FLOOR = 0.72   # lower this if recall is too low, raise to boost precision\n",
    "\n",
    "precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, val_probs)\n",
    "\n",
    "best_thresh = 0.5\n",
    "best_f1     = 0.0\n",
    "\n",
    "for p, r, t in zip(precision_arr[:-1], recall_arr[:-1], thresholds):\n",
    "    if p >= PRECISION_FLOOR:\n",
    "        f = 2 * p * r / (p + r + 1e-9)\n",
    "        if f > best_f1:\n",
    "            best_f1     = f\n",
    "            best_thresh = t\n",
    "\n",
    "val_preds = (val_probs >= best_thresh).astype(int)\n",
    "\n",
    "print(f'Optimal threshold : {best_thresh:.4f}')\n",
    "print(f'Val Precision     : {precision_score(y_val, val_preds):.4f}')\n",
    "print(f'Val Recall        : {recall_score(y_val, val_preds):.4f}')\n",
    "print(f'Val F1            : {f1_score(y_val, val_preds):.4f}')\n",
    "\n",
    "# Plot PR curve\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(recall_arr, precision_arr, lw=2, label='PR Curve')\n",
    "plt.axhline(y=PRECISION_FLOOR, color='red', linestyle='--', label=f'Precision floor = {PRECISION_FLOOR}')\n",
    "plt.scatter([recall_score(y_val, val_preds)],\n",
    "            [precision_score(y_val, val_preds)],\n",
    "            color='green', zorder=5, s=100, label=f'Chosen (t={best_thresh:.3f})')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.title('Precision-Recall Curve — Validation Set (MGTAB)')\n",
    "plt.legend()\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig('mgtab_pr_curve.png', dpi=150)\n",
    "plt.show()"
]
if ens_idx != -1:
    nb['cells'][ens_idx]['source'] = ens_source

# Find Final evaluation cell
eval_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('MULTI-MODEL COMPARISON' in line for line in cell.get('source', [])):
        eval_idx = i
        break

eval_source = [
    "# Ensemble on test set\n",
    "lgb_test  = lgb_model.predict_proba(X_test)[:, 1]\n",
    "xgb_test  = xgb_model.predict_proba(X_test)[:, 1]\n",
    "rf_test   = rf_model.predict_proba(X_test)[:, 1]\n",
    "cb_test   = cb_model.predict_proba(X_test)[:, 1]\n",
    "lr_test   = lr_model.predict_proba(X_test)[:, 1]\n",
    "et_test   = et_model.predict_proba(X_test)[:, 1]\n",
    "mlp_test  = mlp_model.predict_proba(X_test)[:, 1]\n",
    "\n",
    "test_probs = (lgb_test + xgb_test + rf_test + cb_test + lr_test + et_test + mlp_test) / 7.0\n",
    "\n",
    "# Get hard predictions for all models\n",
    "lgb_preds = (lgb_test >= best_thresh).astype(int)\n",
    "xgb_preds = (xgb_test >= best_thresh).astype(int)\n",
    "rf_preds  = (rf_test >= best_thresh).astype(int)\n",
    "cb_preds  = (cb_test >= best_thresh).astype(int)\n",
    "lr_preds  = (lr_test >= best_thresh).astype(int)\n",
    "et_preds  = (et_test >= best_thresh).astype(int)\n",
    "mlp_preds = (mlp_test >= best_thresh).astype(int)\n",
    "ens_preds = (test_probs >= best_thresh).astype(int)\n",
    "\n",
    "# Compile results\n",
    "def get_metrics(y_true, y_pred, y_prob):\n",
    "    return {\n",
    "        'Accuracy': accuracy_score(y_true, y_pred),\n",
    "        'Precision': precision_score(y_true, y_pred),\n",
    "        'Recall': recall_score(y_true, y_pred),\n",
    "        'F1 Score': f1_score(y_true, y_pred),\n",
    "        'AUC-ROC': roc_auc_score(y_true, y_prob)\n",
    "    }\n",
    "\n",
    "results = {\n",
    "    'LightGBM': get_metrics(y_test, lgb_preds, lgb_test),\n",
    "    'XGBoost': get_metrics(y_test, xgb_preds, xgb_test),\n",
    "    'Random Forest': get_metrics(y_test, rf_preds, rf_test),\n",
    "    'CatBoost': get_metrics(y_test, cb_preds, cb_test),\n",
    "    'Logistic Regression': get_metrics(y_test, lr_preds, lr_test),\n",
    "    'Extra Trees': get_metrics(y_test, et_preds, et_test),\n",
    "    'MLP': get_metrics(y_test, mlp_preds, mlp_test),\n",
    "    'Ensemble': get_metrics(y_test, ens_preds, test_probs)\n",
    "}\n",
    "\n",
    "results_df = pd.DataFrame(results).T\n",
    "print(\"=\" * 60)\n",
    "print(\"             MULTI-MODEL COMPARISON (TEST SET)      \")\n",
    "print(\"=\" * 60)\n",
    "print(results_df.round(4))\n",
    "print(\"=\" * 60)\n",
    "\n",
    "# Confusion matrix for Ensemble\n",
    "cm   = confusion_matrix(y_test, ens_preds)\n",
    "disp = ConfusionMatrixDisplay(cm, display_labels=['Human', 'Bot'])\n",
    "fig, ax = plt.subplots(figsize=(6, 5))\n",
    "disp.plot(ax=ax, cmap='Blues', colorbar=False)\n",
    "plt.title('Confusion Matrix — Ensemble (MGTAB)')\n",
    "plt.tight_layout()\n",
    "plt.savefig('mgtab_confusion_matrix.png', dpi=150)\n",
    "plt.show()"
]
if eval_idx != -1:
    nb['cells'][eval_idx]['source'] = eval_source

with open('mgtab_bot_detection.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
