import time, numpy as np
from src.data.adapter import RealDataAdapter
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve

train_df = RealDataAdapter.load_table('train')
features = RealDataAdapter.get_feature_columns(train_df)
X_train = train_df[features].values
y_train = train_df['is_bot'].astype(int).values
spw = (y_train == 0).sum() / y_train.sum()
print(f'Train: {len(X_train):,} rows, spw={spw:.1f}')

t0 = time.time()
model = XGBClassifier(
    objective='binary:logistic', tree_method='hist', scale_pos_weight=spw,
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, eval_metric='logloss',
)
model.fit(X_train, y_train)
print(f'Train: {time.time()-t0:.1f}s')

val_df = RealDataAdapter.load_table('val')
val_probs = model.predict_proba(val_df[features].values)[:, 1]
y_val = val_df['is_bot'].astype(int).values
prec, rec, thresholds = precision_recall_curve(y_val, val_probs)
f1s = np.where((prec[:-1]+rec[:-1]) > 0, 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]), 0)
opt_t = thresholds[np.argmax(f1s)]
print(f'Threshold: {opt_t:.3f} (val F1: {f1s.max():.2%})')

test_df = RealDataAdapter.load_table('test')
probs = model.predict_proba(test_df[features].values)[:, 1]
y_test = test_df['is_bot'].astype(int).values
print(classification_report(y_test, (probs >= opt_t).astype(int), target_names=['Legit', 'Bot']))
