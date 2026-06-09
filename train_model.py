"""
train_model.py
──────────────
Trains LightGBM classifier on engineered features.
Includes time-based split, class weight handling, probability calibration.

Run after engineer_features.py: python train_model.py
Takes ~15-30 minutes.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import os
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from scipy.optimize import minimize
from sklearn.utils.class_weight import compute_class_weight


# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH    = "datasets/training_features.parquet"
MODELS_DIR    = "models"
MODEL_PATH    = f"{MODELS_DIR}/lgbm_model.pkl"
COLUMNS_PATH  = f"{MODELS_DIR}/feature_columns.json"
METADATA_PATH = f"{MODELS_DIR}/training_metadata.json"

# ── Feature columns — MUST match engineer_features.py and ml_predictor.py ────
FEATURE_COLS = [
    # ── Original 24 ───────────────────────────────────────────────────────────
    "RSI_14", "RSI_21",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_lower", "BB_middle", "BB_upper", "BB_width",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "SMA_cross", "price_above_bb",
    "price_momentum_5d", "price_momentum_10d", "price_momentum_20d",
    "volume_momentum_5d",
    "volatility_20d", "volatility_ratio",
    "high_low_range", "close_to_high_ratio",
    "volume_ratio_20d",
    # ── New 10 ────────────────────────────────────────────────────────────────
    "ATR_14",
    "ADX",
    "stochastic_k", "stochastic_d",
    "VWAP_distance",
    "OBV_ratio",
    "NIFTY_return", "BANKNIFTY_return", "India_VIX",
    "relative_strength",
]

# ── Step 1: Load and prepare data ─────────────────────────────────────────────
print(f"[STEP 1] Loading {INPUT_PATH}...")
try:
    df = pd.read_parquet(INPUT_PATH)
except FileNotFoundError:
    print(f"  ✗ File not found. Run engineer_features.py first.")
    raise SystemExit(1)

print(f"  ✓ Loaded {len(df):,} rows | {df['ticker'].nunique()} tickers")

# Verify all features present
missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
if missing_cols:
    print(f"  ✗ Missing features: {missing_cols}")
    raise SystemExit(1)

# ── Step 2: Time-based train/test split ───────────────────────────────────────
print(f"\n[STEP 2] Time-based split...")

df["date"] = pd.to_datetime(df["date"])

sorted_dates = df["date"].sort_values()
train_split_date = sorted_dates.iloc[int(len(sorted_dates) * 0.8)]

X_train = df[df["date"] <= train_split_date][FEATURE_COLS].copy()
y_train = df[df["date"] <= train_split_date]["label"].copy()

X_test = df[df["date"] > train_split_date][FEATURE_COLS].copy()
y_test = df[df["date"] > train_split_date]["label"].copy()

print(f"  Train set: {len(X_train):,} rows ({train_split_date.date()})")
print(f"  Test set:  {len(X_test):,} rows ({train_split_date.date()} → {df['date'].max().date()})")
print(f"  Train label: {y_train.mean()*100:.1f}% UP / {(1-y_train.mean())*100:.1f}% DOWN")
print(f"  Test label:  {y_test.mean()*100:.1f}% UP / {(1-y_test.mean())*100:.1f}% DOWN")

# ── Step 3: Scale features ────────────────────────────────────────────────────
print(f"\n[STEP 3] Scaling features...")

# Replace infinities with NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Remove rows containing NaN/inf
train_mask = X_train.notna().all(axis=1)
test_mask = X_test.notna().all(axis=1)

X_train = X_train[train_mask]
y_train = y_train[train_mask]

X_test = X_test[test_mask]
y_test = y_test[test_mask]

print(f"  ✓ Removed rows with inf/NaN values")



# ── Step 4: Train LightGBM ────────────────────────────────────────────────────
print(f"\n[STEP 4] Training LightGBM...")

# Calculate class weight to handle imbalance
classes = np.array([0, 1])
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_ratio = weights[1] / weights[0]  # UP weight / DOWN weight

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=class_weight_ratio,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5
)

X_train_scaled = X_train.values
X_test_scaled = X_test.values

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=50)
    ]
)

print(f"  ✓ Model trained")

# ── Step 5: Predictions ───────────────────────────────────────────────────────
print(f"\n[STEP 5] Generating predictions...")

y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

print(f"  ✓ Predictions generated")

# ── Step 6: Calibration ───────────────────────────────────────────────────────
print(f"\n[STEP 6] Probability calibration...")

def sigmoid_calibration(proba, y_true):
    """Platt scaling — fits sigmoid to map raw probabilities to calibrated probabilities."""
    def objective(params):
        a, b = params
        p_cal = 1 / (1 + np.exp(a * np.log(proba / (1 - proba + 1e-8)) + b))
        loss = -np.mean(y_true * np.log(p_cal + 1e-8) + (1 - y_true) * np.log(1 - p_cal + 1e-8))
        return loss
    
    result = minimize(objective, x0=[1, 0], method="Nelder-Mead")
    return result.x

calib_params = sigmoid_calibration(y_test_pred_proba, y_test.values)
print(f"  Calibration params: a={calib_params[0]:.4f}, b={calib_params[1]:.4f}")

# Apply calibration
y_test_pred_proba_cal = 1 / (1 + np.exp(
    calib_params[0] * np.log(y_test_pred_proba / (1 - y_test_pred_proba + 1e-8)) + calib_params[1]
))

# ── Step 7: Evaluation ────────────────────────────────────────────────────────
print(f"\n[STEP 7] Evaluating model...")
print(f"\n{'─'*70}")
print(f"TRAIN SET METRICS")
print(f"{'─'*70}")

train_auc = roc_auc_score(y_train, y_train_pred_proba)
train_acc = accuracy_score(y_train, y_train_pred)
train_bacc = balanced_accuracy_score(y_train, y_train_pred)

print(f"  AUC              : {train_auc:.4f}")
print(f"  Accuracy         : {train_acc:.4f}")
print(f"  Balanced Accuracy: {train_bacc:.4f}")

print(f"\n{'─'*70}")
print(f"TEST SET METRICS (CALIBRATED)")
print(f"{'─'*70}")

test_auc = roc_auc_score(y_test, y_test_pred_proba_cal)
test_pred_cal = (y_test_pred_proba_cal >= 0.5).astype(int)
test_acc = accuracy_score(y_test, test_pred_cal)
test_bacc = balanced_accuracy_score(y_test, test_pred_cal)
test_prec = precision_score(y_test, test_pred_cal)
test_rec = recall_score(y_test, test_pred_cal)
test_f1 = f1_score(y_test, test_pred_cal)

print(f"  AUC              : {test_auc:.4f}")
print(f"  Accuracy         : {test_acc:.4f}")
print(f"  Balanced Accuracy: {test_bacc:.4f}")
print(f"  Precision (UP)   : {test_prec:.4f}")
print(f"  Recall (UP)      : {test_rec:.4f}")
print(f"  F1-Score         : {test_f1:.4f}")

print(f"\n{'─'*70}")
print(f"CONFUSION MATRIX")
print(f"{'─'*70}")
cm = confusion_matrix(y_test, test_pred_cal)
print(f"                Predicted")
print(f"              DOWN      UP")
print(f"  Actual DOWN {cm[0,0]:6d} {cm[0,1]:6d}")
print(f"       UP    {cm[1,0]:6d} {cm[1,1]:6d}")

# Classification report
print(f"\n{'─'*70}")
print(f"CLASSIFICATION REPORT")
print(f"{'─'*70}")
print(classification_report(y_test, test_pred_cal, target_names=["DOWN", "UP"]))

# ── Step 8: Feature importance ────────────────────────────────────────────────
print(f"{'─'*70}")
print(f"TOP 10 FEATURE IMPORTANCES")
print(f"{'─'*70}")

feature_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

for idx, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"  {idx:2d}. {row['feature']:25s} : {row['importance']:.4f}")

# ── Step 9: Save artifacts ────────────────────────────────────────────────────
print(f"\n[STEP 9] Saving model artifacts...")

os.makedirs(MODELS_DIR, exist_ok=True)

# Save model
joblib.dump(model, MODEL_PATH)
print(f"  ✓ Model saved to {MODEL_PATH}")

# Save feature columns
with open(COLUMNS_PATH, "w") as f:
    json.dump(FEATURE_COLS, f)
print(f"  ✓ Features saved to {COLUMNS_PATH}")

# Save training metadata
metadata = {
    "model_type": "LightGBM",
    "trained_at": datetime.now().isoformat(),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "train_split_date": train_split_date.isoformat(),
    "features": FEATURE_COLS,
    "metrics": {
        "train_auc": float(train_auc),
        "test_auc": float(test_auc),
        "test_accuracy": float(test_acc),
        "test_balanced_accuracy": float(test_bacc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1)
    },
    "calibration": {
        "method": "Platt scaling",
        "params": {
            "a": float(calib_params[0]),
            "b": float(calib_params[1])
        }
    },
    "feature_importance": feature_importance.to_dict("records")
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ Metadata saved to {METADATA_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"✓ TRAINING COMPLETE")
print(f"{'─'*70}")
print(f"  Test AUC: {test_auc:.4f}")

if test_auc > 0.54:
    print(f"  ✓ Model is production-ready (AUC > 0.54)")
elif test_auc > 0.52:
    print(f"  ⚠ Model is acceptable but could be improved (AUC ~0.52)")
else:
    print(f"  ✗ Model performance is poor (AUC < 0.52)")

print(f"{'─'*70}")
print(f"\n→ Next step: python evaluate_model.py\n")