"""
evaluate_model.py
─────────────────
Evaluates the trained LightGBM model against the held-out test set.
Reads training_metadata.json for stored metrics and re-runs predictions
on the test split for live verification.

Run after train_model.py: python evaluate_model.py
Takes ~1-2 minutes.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH    = "datasets/training_features.parquet"
MODEL_PATH    = "models/lgbm_model.pkl"
COLUMNS_PATH  = "models/feature_columns.json"
METADATA_PATH = "models/training_metadata.json"

# ── Load model and metadata ───────────────────────────────────────────────────
print("=" * 65)
print("  StockFlow — Model Evaluation")
print("=" * 65)

print("\n[1] Loading model artifacts...")
model = joblib.load(MODEL_PATH)

with open(COLUMNS_PATH) as f:
    FEATURE_COLS = json.load(f)

with open(METADATA_PATH) as f:
    metadata = json.load(f)

print(f"  Model type   : {metadata['model_type']}")
print(f"  Trained at   : {metadata['trained_at']}")
print(f"  Features     : {len(FEATURE_COLS)}")
print(f"  Train rows   : {metadata['train_rows']:,}")
print(f"  Test rows    : {metadata['test_rows']:,}")

# ── Load test data ────────────────────────────────────────────────────────────
print("\n[2] Loading test data...")
df = pd.read_parquet(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])

train_split_date = pd.to_datetime(metadata["train_split_date"])
test_df = df[df["date"] > train_split_date].copy()

X_test = test_df[FEATURE_COLS].copy()
y_test = test_df["label"].copy()

# Clean infinities and NaN
X_test = X_test.replace([np.inf, -np.inf], np.nan)
mask   = X_test.notna().all(axis=1)
X_test = X_test[mask]
y_test = y_test[mask]

print(f"  Test samples : {len(X_test):,}")
print(f"  Date range   : {test_df['date'].min().date()} → {test_df['date'].max().date()}")
print(f"  Label split  : {y_test.mean()*100:.1f}% UP / {(1-y_test.mean())*100:.1f}% DOWN")

# ── Generate predictions ──────────────────────────────────────────────────────
print("\n[3] Generating predictions...")
y_pred_proba = model.predict_proba(X_test.values)[:, 1]
y_pred       = (y_pred_proba >= 0.5).astype(int)

# ── Stored metrics from training ──────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"STORED METRICS (from training_metadata.json)")
print(f"{'─'*65}")
m = metadata["metrics"]
print(f"  Train AUC            : {m['train_auc']:.4f}")
print(f"  Test AUC             : {m['test_auc']:.4f}")
print(f"  Test Accuracy        : {m['test_accuracy']*100:.2f}%")
print(f"  Test Balanced Acc    : {m['test_balanced_accuracy']*100:.2f}%")
print(f"  Test Precision (UP)  : {m['test_precision']*100:.2f}%")
print(f"  Test Recall (UP)     : {m['test_recall']*100:.2f}%")
print(f"  Test F1-Score        : {m['test_f1']:.4f}")

# ── Live re-computed metrics ──────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"LIVE RE-COMPUTED METRICS (on current test split)")
print(f"{'─'*65}")
live_auc  = roc_auc_score(y_test, y_pred_proba)
live_acc  = accuracy_score(y_test, y_pred)
live_bacc = balanced_accuracy_score(y_test, y_pred)
live_prec = precision_score(y_test, y_pred, zero_division=0)
live_rec  = recall_score(y_test, y_pred, zero_division=0)
live_f1   = f1_score(y_test, y_pred, zero_division=0)

print(f"  AUC                  : {live_auc:.4f}")
print(f"  Accuracy             : {live_acc*100:.2f}%")
print(f"  Balanced Accuracy    : {live_bacc*100:.2f}%")
print(f"  Precision (UP)       : {live_prec*100:.2f}%")
print(f"  Recall (UP)          : {live_rec*100:.2f}%")
print(f"  F1-Score             : {live_f1:.4f}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"CONFUSION MATRIX")
print(f"{'─'*65}")
cm = confusion_matrix(y_test, y_pred)
print(f"                  Predicted DOWN   Predicted UP")
print(f"  Actual DOWN     {cm[0,0]:12,}   {cm[0,1]:12,}")
print(f"  Actual UP       {cm[1,0]:12,}   {cm[1,1]:12,}")

tn, fp, fn, tp = cm.ravel()
print(f"\n  True Negatives  (DOWN correct): {tn:,}")
print(f"  False Positives (DOWN as UP)  : {fp:,}")
print(f"  False Negatives (UP as DOWN)  : {fn:,}")
print(f"  True Positives  (UP correct)  : {tp:,}")

# ── Prediction distribution ───────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"PREDICTION DISTRIBUTION")
print(f"{'─'*65}")
print(f"  Predicted UP   : {(y_pred == 1).sum():,} ({(y_pred == 1).mean()*100:.1f}%)")
print(f"  Predicted DOWN : {(y_pred == 0).sum():,} ({(y_pred == 0).mean()*100:.1f}%)")
print(f"  Avg confidence : {np.mean(np.abs(y_pred_proba - 0.5) + 0.5)*100:.2f}%")

# ── Feature importance ────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"TOP 15 FEATURE IMPORTANCES")
print(f"{'─'*65}")
importances = sorted(metadata["feature_importance"],
                     key=lambda x: x["importance"], reverse=True)
for i, feat in enumerate(importances[:15], 1):
    bar = "█" * min(int(feat["importance"] / 20), 30)
    print(f"  {i:2d}. {feat['feature']:25s} {feat['importance']:6d}  {bar}")

# ── Classification report ─────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"CLASSIFICATION REPORT")
print(f"{'─'*65}")
print(classification_report(y_test, y_pred,
                             target_names=["DOWN", "UP"],
                             zero_division=0))

# ── Sample predictions ────────────────────────────────────────────────────────
print(f"{'─'*65}")
print(f"SAMPLE PREDICTIONS (20 random test rows)")
print(f"{'─'*65}")
print(f"  {'Ticker':12s} {'Date':12s} {'Prob UP':8s} {'Predicted':10s} {'Actual':8s} {'OK':4s}")
print(f"  {'─'*60}")

sample_idx = np.random.choice(len(X_test), size=min(20, len(X_test)), replace=False)
test_df_clean = test_df[mask].reset_index(drop=True)

for i in sample_idx:
    row      = test_df_clean.iloc[i]
    prob     = y_pred_proba[i]
    pred_lbl = "UP" if prob >= 0.5 else "DOWN"
    act_lbl  = "UP" if row["label"] == 1 else "DOWN"
    ok       = "✓" if pred_lbl == act_lbl else "✗"
    print(f"  {row['ticker']:12s} {str(row['date'].date()):12s} "
          f"{prob*100:6.1f}%  {pred_lbl:10s} {act_lbl:8s} {ok}")

# ── Summary verdict ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"EVALUATION SUMMARY")
print(f"{'='*65}")
print(f"  Test AUC     : {live_auc:.4f}  {'✓ Production-ready' if live_auc > 0.54 else '⚠ Below threshold'}")
print(f"  Accuracy     : {live_acc*100:.2f}%")
print(f"  Both classes : {'✓ Predicting UP and DOWN' if (y_pred==0).sum() > 0 and (y_pred==1).sum() > 0 else '✗ Only predicting one class'}")

if live_auc > 0.56:
    print(f"\n  Model is performing well (AUC > 0.56)")
elif live_auc > 0.54:
    print(f"\n  Model is acceptable (AUC > 0.54) — consider adding more features")
else:
    print(f"\n  Model needs improvement (AUC < 0.54) — retrain with better features")

print(f"{'='*65}\n")
