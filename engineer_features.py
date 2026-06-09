"""
engineer_features.py
─────────────────────
Loads raw OHLCV data, computes technical indicators,
creates 14-day forward label WITHOUT lookahead bias.

v2 Changes:
  - BUG FIX: volatility_ratio now uses per-row vol60 (was collapsed scalar)
  - NEW: ATR_14       — normalized average true range
  - NEW: ADX          — trend strength (14)
  - NEW: stochastic_k — %K overbought/oversold
  - NEW: stochastic_d — %D signal line
  - NEW: VWAP_distance — price distance from 20d rolling VWAP
  - NEW: OBV_ratio    — on-balance volume vs 20d mean
  - NEW: NIFTY_return — daily NIFTY 50 return (market regime)
  - NEW: BANKNIFTY_return — daily Bank NIFTY return
  - NEW: India_VIX    — India VIX level (regime volatility)
  - NEW: relative_strength — stock 20d momentum minus NIFTY 20d return

Run after collect_data.py: python engineer_features.py
Takes ~10-15 minutes for 8,011 stocks.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import json
import os
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH    = "datasets/training_raw.parquet"
OUTPUT_PATH   = "datasets/training_features.parquet"
METADATA_PATH = "models/feature_metadata.json"

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_ROWS_PER_TICKER = 60
FORWARD_DAYS        = 14
DEAD_ZONE           = 0.03   # ±3% — wider dead zone = cleaner labels

# ── Feature list — MUST match train_model.py and ml_predictor.py exactly ──────
FEATURE_COLS = [
    # ── Existing 24 ───────────────────────────────────────────────────────────
    "RSI_14", "RSI_21",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_lower", "BB_middle", "BB_upper", "BB_width",   # normalized to price
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",            # normalized to price
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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load raw data
# ══════════════════════════════════════════════════════════════════════════════
print(f"[STEP 1] Loading {INPUT_PATH}...")
try:
    df = pd.read_parquet(INPUT_PATH)
except FileNotFoundError:
    print(f"  File not found. Run collect_data.py first.")
    raise SystemExit(1)

df["date"] = pd.to_datetime(df["date"])
df.sort_values(["ticker", "date"], inplace=True)
df.reset_index(drop=True, inplace=True)

data_start = df["date"].min().strftime("%Y-%m-%d")
data_end   = df["date"].max().strftime("%Y-%m-%d")
print(f"  Loaded {len(df):,} rows | {df['ticker'].nunique()} tickers")
print(f"  Date range: {data_start} → {data_end}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Fetch index data (NIFTY, BANKNIFTY, India VIX) — done ONCE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[STEP 2] Fetching index data from Yahoo Finance...")

# Fetch with a buffer before data_start to allow rolling calculations on early rows
FETCH_START = (pd.to_datetime(data_start) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")

def fetch_index(symbol, col_name, is_return=True):
    """Download an index, return a Series of daily % return or raw close."""
    try:
        raw = yf.download(symbol, start=FETCH_START, end=data_end,
                          auto_adjust=True, progress=False)["Close"]
        raw = raw.squeeze()   # ensure Series even if single column
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        if is_return:
            return raw.pct_change() * 100
        return raw
    except Exception as e:
        print(f"  WARNING: Could not fetch {symbol}: {e}. Filling with 0/NaN.")
        return pd.Series(dtype=float)

nifty_ret     = fetch_index("^NSEI",     "NIFTY_return",     is_return=True)
banknifty_ret = fetch_index("^NSEBANK",  "BANKNIFTY_return", is_return=True)
india_vix     = fetch_index("^INDIAVIX", "India_VIX",        is_return=False)

index_df = pd.DataFrame({
    "NIFTY_return":     nifty_ret,
    "BANKNIFTY_return": banknifty_ret,
    "India_VIX":        india_vix,
})
index_df.index.name = "date"
index_df = index_df.reset_index()
index_df["date"] = pd.to_datetime(index_df["date"])

# Pre-compute 20d cumulative NIFTY return (used in relative_strength)
# Rolling sum of daily % return ≈ cumulative % over window
nifty_cum20 = nifty_ret.rolling(20).sum()
nifty_cum20.index = pd.to_datetime(nifty_cum20.index).tz_localize(None)
nifty_cum20_df = nifty_cum20.rename("NIFTY_cum20").reset_index()
nifty_cum20_df.columns = ["date", "NIFTY_cum20"]

index_df = index_df.merge(nifty_cum20_df, on="date", how="left")

print(f"  Index data: {len(index_df)} trading days fetched")
missing_pct = index_df[["NIFTY_return", "India_VIX"]].isna().mean() * 100
print(f"  NaN %  — NIFTY: {missing_pct['NIFTY_return']:.1f}%  VIX: {missing_pct['India_VIX']:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Feature engineering per ticker
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[STEP 3] Computing features per ticker...")

all_frames  = []
skipped     = []
errors      = []
grouped     = df.groupby("ticker")
num_tickers = len(grouped)

for idx, (ticker, sub) in enumerate(grouped, 1):
    sub = sub.copy().reset_index(drop=True)

    if len(sub) < MIN_ROWS_PER_TICKER:
        skipped.append((ticker, len(sub)))
        continue

    try:
        price = sub["close"]

        # ── Existing: Momentum / RSI ──────────────────────────────────────────
        sub["RSI_14"] = ta.rsi(price, length=14)
        sub["RSI_21"] = ta.rsi(price, length=21)

        # ── Existing: MACD — named columns to avoid positional order bugs ─────
        macd_df = ta.macd(price, fast=12, slow=26, signal=9)
        sub["MACD"]        = macd_df["MACD_12_26_9"]
        sub["MACD_signal"] = macd_df["MACDs_12_26_9"]
        sub["MACD_hist"]   = macd_df["MACDh_12_26_9"]

        # ── Existing: Bollinger Bands — normalized to price ───────────────────
        bb_df = ta.bbands(price, length=20, std=2)
        bb_lower_col  = [c for c in bb_df.columns if c.startswith("BBL")][0]
        bb_middle_col = [c for c in bb_df.columns if c.startswith("BBM")][0]
        bb_upper_col  = [c for c in bb_df.columns if c.startswith("BBU")][0]

        sub["BB_lower"]  = bb_df[bb_lower_col]  / price
        sub["BB_middle"] = bb_df[bb_middle_col] / price
        sub["BB_upper"]  = bb_df[bb_upper_col]  / price
        sub["BB_width"]  = sub["BB_upper"] - sub["BB_lower"]

        # ── Existing: Moving averages — normalized to price ───────────────────
        sub["SMA_20"] = price.rolling(20).mean() / price
        sub["SMA_50"] = price.rolling(50).mean() / price
        sub["EMA_12"] = price.ewm(span=12, adjust=False).mean() / price
        sub["EMA_26"] = price.ewm(span=26, adjust=False).mean() / price

        # ── Existing: Cross signals ───────────────────────────────────────────
        sma20_raw = price.rolling(20).mean()
        sma50_raw = price.rolling(50).mean()
        sub["SMA_cross"]      = (sma20_raw > sma50_raw).astype(int)
        bb_upper_raw          = bb_df[bb_upper_col]
        bb_lower_raw          = bb_df[bb_lower_col]
        sub["price_above_bb"] = (
            (price > bb_upper_raw).astype(int) -
            (price < bb_lower_raw).astype(int)
        )

        # ── Existing: Price momentum ──────────────────────────────────────────
        sub["price_momentum_5d"]  = price.pct_change(5)  * 100
        sub["price_momentum_10d"] = price.pct_change(10) * 100
        sub["price_momentum_20d"] = price.pct_change(20) * 100
        sub["volume_momentum_5d"] = sub["volume"].pct_change(5) * 100

        # ── FIXED: Volatility — vol60 is now per-row (was collapsed scalar) ───
        daily_ret             = price.pct_change()
        vol20                 = daily_ret.rolling(20).std() * 100
        vol60                 = daily_ret.rolling(60).std() * 100   # ← FIX
        sub["volatility_20d"]   = vol20
        sub["volatility_ratio"] = vol20 / (vol60 + 1e-9)           # ← FIX

        # ── Existing: Range / volume ──────────────────────────────────────────
        sub["high_low_range"]      = (sub["high"] - sub["low"]) / price * 100
        sub["close_to_high_ratio"] = price / sub["high"]
        vol_mean20                 = sub["volume"].rolling(20).mean()
        sub["volume_ratio_20d"]    = sub["volume"] / (vol_mean20 + 1e-9)

        # ══════════════════════════════════════════════════════════════════════
        # NEW FEATURES
        # ══════════════════════════════════════════════════════════════════════

        # ── NEW: ATR_14 — normalized to price (removes cross-stock scale) ─────
        atr_raw         = ta.atr(sub["high"], sub["low"], sub["close"], length=14)
        sub["ATR_14"]   = atr_raw / price * 100

        # ── NEW: ADX — trend strength 0-100 ──────────────────────────────────
        adx_df      = ta.adx(sub["high"], sub["low"], sub["close"], length=14)
        adx_col     = [c for c in adx_df.columns if c.startswith("ADX")][0]
        sub["ADX"]  = adx_df[adx_col]

        # ── NEW: Stochastic %K and %D ─────────────────────────────────────────
        stoch_df          = ta.stoch(sub["high"], sub["low"], sub["close"],
                                     k=14, d=3, smooth_k=3)
        stoch_k_col       = [c for c in stoch_df.columns if c.startswith("STOCHk")][0]
        stoch_d_col       = [c for c in stoch_df.columns if c.startswith("STOCHd")][0]
        sub["stochastic_k"] = stoch_df[stoch_k_col]
        sub["stochastic_d"] = stoch_df[stoch_d_col]

        # ── NEW: VWAP distance — rolling 20d proxy (daily OHLCV) ─────────────
        # True intraday VWAP needs tick data; this rolling version is the standard
        # daily equivalent used by quant funds on EOD data.
        typical_price    = (sub["high"] + sub["low"] + sub["close"]) / 3
        cum_tp_vol       = (typical_price * sub["volume"]).rolling(20).sum()
        cum_vol          = sub["volume"].rolling(20).sum()
        vwap             = cum_tp_vol / (cum_vol + 1e-9)
        sub["VWAP_distance"] = (price - vwap) / (vwap + 1e-9) * 100

        # ── NEW: OBV ratio — normalized to 20d rolling mean ──────────────────
        obv_raw          = ta.obv(price, sub["volume"])
        obv_mean20       = obv_raw.rolling(20).mean().abs()
        sub["OBV_ratio"] = obv_raw / (obv_mean20 + 1e-9)

        # ── NEW: Index features — merge from pre-fetched index_df ─────────────
        sub = sub.merge(index_df[["date", "NIFTY_return",
                                   "BANKNIFTY_return", "India_VIX",
                                   "NIFTY_cum20"]],
                        on="date", how="left")

        # ── NEW: Relative strength — stock alpha vs NIFTY over 20d ────────────
        # stock 20d momentum minus NIFTY 20d cumulative return
        sub["relative_strength"] = sub["price_momentum_20d"] - sub["NIFTY_cum20"]

        # Drop helper column
        sub.drop(columns=["NIFTY_cum20"], inplace=True, errors="ignore")

        # ══════════════════════════════════════════════════════════════════════
        # Label — no lookahead bias
        # ══════════════════════════════════════════════════════════════════════
        sub["future_close"]   = price.shift(-FORWARD_DAYS)
        sub["forward_return"] = (sub["future_close"] - price) / price * 100

        sub["label"] = np.where(
            sub["forward_return"] >  (DEAD_ZONE * 100),  1,    # UP
            np.where(
            sub["forward_return"] < -(DEAD_ZONE * 100),  0,    # DOWN
            np.nan                                              # dead zone
        ))

        sub = sub.dropna(subset=["label"])
        sub = sub.drop(columns=["future_close", "forward_return"], errors="ignore")

        all_frames.append(sub)

        if idx % 100 == 0 or idx == num_tickers:
            dist = sub["label"].value_counts().to_dict()
            print(f"  [{idx:4d}/{num_tickers}] {ticker:12s} "
                  f"({len(sub)} rows) label: {dist}")

    except Exception as e:
        errors.append((ticker, str(e)))
        print(f"  [{idx:4d}/{num_tickers}] {ticker:12s} ERROR: {str(e)[:80]}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Combine
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[STEP 4] Combining all tickers...")
if not all_frames:
    print("  No data processed. Exiting.")
    raise SystemExit(1)

df_features = pd.concat(all_frames, ignore_index=True)
df_features.sort_values(["ticker", "date"], inplace=True)
df_features.reset_index(drop=True, inplace=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Data quality checks
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[STEP 5] Data quality checks...")
rows_before = len(df_features)
df_features = df_features.dropna(subset=FEATURE_COLS + ["label"])
print(f"  Dropped {rows_before - len(df_features):,} rows with NaN features")

# Per-feature NaN audit (helps catch index merge misses)
nan_audit = df_features[FEATURE_COLS].isna().sum()
nan_features = nan_audit[nan_audit > 0]
if not nan_features.empty:
    print(f"  WARNING — Features still containing NaN after dropna:")
    for feat, cnt in nan_features.items():
        print(f"    {feat:25s}: {cnt:,}")
else:
    print(f"  All {len(FEATURE_COLS)} features clean (0 NaN)")

label_dist = df_features["label"].value_counts()
total      = len(df_features)
up_pct     = label_dist.get(1, 0) / total * 100
down_pct   = label_dist.get(0, 0) / total * 100
print(f"  Label: DOWN={label_dist.get(0,0):,} ({down_pct:.1f}%)  "
      f"UP={label_dist.get(1,0):,} ({up_pct:.1f}%)")

if abs(up_pct - down_pct) > 15:
    print(f"  ⚠  Imbalanced labels. Use scale_pos_weight in LightGBM:")
    print(f"     scale_pos_weight = {label_dist.get(0,0) / max(label_dist.get(1,0),1):.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Save
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[STEP 6] Saving outputs...")
os.makedirs("datasets", exist_ok=True)
os.makedirs("models",   exist_ok=True)

df_features.to_parquet(OUTPUT_PATH, index=False)

metadata = {
    "feature_cols":    FEATURE_COLS,
    "total_rows":      len(df_features),
    "unique_tickers":  int(df_features["ticker"].nunique()),
    "dead_zone":       DEAD_ZONE,
    "forward_days":    FORWARD_DAYS,
    "normalization":   "BB/SMA/EMA/ATR normalized to price ratio; OBV to 20d mean",
    "index_features":  ["NIFTY_return", "BANKNIFTY_return", "India_VIX", "relative_strength"],
    "bug_fixes":       ["volatility_ratio: vol60 is now per-row Series (was .mean() scalar)"],
    "date_range": {
        "start": df_features["date"].min().isoformat(),
        "end":   df_features["date"].max().isoformat()
    },
    "label_distribution": {
        "UP":   int(label_dist.get(1, 0)),
        "DOWN": int(label_dist.get(0, 0))
    },
    "skipped_tickers": skipped,
    "error_tickers":   errors,
    "created_at":      datetime.now().isoformat()
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'─'*65}")
print(f"FEATURE ENGINEERING COMPLETE")
print(f"  Rows         : {len(df_features):,}")
print(f"  Tickers      : {df_features['ticker'].nunique()}")
print(f"  Features     : {len(FEATURE_COLS)}  (24 original + 10 new)")
print(f"  Dead zone    : ±{DEAD_ZONE*100:.0f}%")
print(f"  Skipped      : {len(skipped)} tickers (< {MIN_ROWS_PER_TICKER} rows)")
print(f"  Errors       : {len(errors)} tickers")
print(f"  Output       : {OUTPUT_PATH}")
print(f"  Metadata     : {METADATA_PATH}")
print(f"{'─'*65}")
print(f"\n→ Next step: python train_model.py")
print(f"\n  ⚠  IMPORTANT — In train_model.py, use a walk-forward split:")
print(f"     train = df[df['date'] < '2023-01-01']")
print(f"     test  = df[df['date'] >= '2023-01-01']")
print(f"     (NOT random train_test_split — that leaks future data)\n")