"""
ml_predictor.py
───────────────
Production ML predictor for 14-day price direction.
Supports both Indian (.NS / .BO) and US stocks.
Always returns UP or DOWN — no NEUTRAL cop-out.

Bugs fixed:
  - high_low_range / close_to_high_ratio now use daily candle H/L (not 52w)
  - Market index fetched ONCE and cached — not re-downloaded per stock
  - Indian stocks  → NIFTY + BANKNIFTY + India VIX as index features
  - US stocks      → SPY return + QQQ return + CBOE VIX as index features
  - relative_strength computed correctly for both markets
  - Always returns UP or DOWN — NEUTRAL removed
"""

import joblib
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/lgbm_model.pkl"
COLUMNS_PATH  = "models/feature_columns.json"
METADATA_PATH = "models/training_metadata.json"

# ── Indian exchange suffixes ──────────────────────────────────────────────────
INDIAN_SUFFIXES = (".NS", ".BO")

# ── Index symbols per market ──────────────────────────────────────────────────
INDIA_INDICES = {
    "market": "^NSEI",      # NIFTY 50      → fills NIFTY_return slot
    "sector": "^NSEBANK",   # Bank NIFTY    → fills BANKNIFTY_return slot
    "vix":    "^INDIAVIX",  # India VIX     → fills India_VIX slot
}
US_INDICES = {
    "market": "SPY",        # S&P 500 ETF   → fills NIFTY_return slot
    "sector": "QQQ",        # NASDAQ ETF    → fills BANKNIFTY_return slot
    "vix":    "^VIX",       # CBOE VIX      → fills India_VIX slot
}

# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════
MODEL_AVAILABLE = False
model           = None
feature_cols    = []

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        model = joblib.load(MODEL_PATH)
        with open(COLUMNS_PATH) as f:
            feature_cols = json.load(f)
        MODEL_AVAILABLE = True
        logger.info(f"[ML] Model loaded — {len(feature_cols)} features")
    else:
        logger.warning("[ML] Model files not found — run train_model.py first")
except Exception as e:
    logger.error(f"[ML] Failed to load model: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Market index cache — one entry per market, loaded once per session
# ══════════════════════════════════════════════════════════════════════════════
_cache: dict = {}

def _load_index_cache(market: str) -> dict:
    """
    Returns dict with keys: NIFTY_return, BANKNIFTY_return, India_VIX, NIFTY_cum20.
    Uses safe fallback values on fetch failure so prediction still runs.
    For India: uses NIFTY / BankNIFTY / IndiaVIX.
    For US:    uses SPY / QQQ / CBOE VIX — same feature slots, different data.
    """
    global _cache
    if market in _cache:
        return _cache[market]

    symbols = INDIA_INDICES if market == "india" else US_INDICES
    result  = {
        "NIFTY_return":     0.0,
        "BANKNIFTY_return": 0.0,
        "India_VIX":        15.0,
        "NIFTY_cum20":      0.0,
    }

    try:
        raw = yf.download(symbols["market"], period="3mo",
                          auto_adjust=True, progress=False)["Close"].squeeze()
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        ret = raw.pct_change() * 100
        result["NIFTY_return"] = float(ret.iloc[-1]) if not ret.empty else 0.0
        result["NIFTY_cum20"]  = float(ret.rolling(20).sum().iloc[-1]) \
                                  if len(ret) >= 20 else 0.0
        logger.info(f"[ML] {market} market index loaded ({symbols['market']})")
    except Exception as e:
        logger.warning(f"[ML] {symbols['market']} fetch failed: {e}")

    try:
        raw = yf.download(symbols["sector"], period="3mo",
                          auto_adjust=True, progress=False)["Close"].squeeze()
        ret = raw.pct_change() * 100
        result["BANKNIFTY_return"] = float(ret.iloc[-1]) if not ret.empty else 0.0
        logger.info(f"[ML] {market} sector index loaded ({symbols['sector']})")
    except Exception as e:
        logger.warning(f"[ML] {symbols['sector']} fetch failed: {e}")

    try:
        raw = yf.download(symbols["vix"], period="5d",
                          auto_adjust=True, progress=False)["Close"].squeeze()
        result["India_VIX"] = float(raw.iloc[-1]) if not raw.empty else 15.0
        logger.info(f"[ML] {market} VIX loaded ({symbols['vix']})")
    except Exception as e:
        logger.warning(f"[ML] {symbols['vix']} fetch failed: {e}")

    _cache[market] = result
    return result


def invalidate_cache():
    """Call between trading sessions to force fresh index data."""
    global _cache
    _cache = {}
    logger.info("[ML] Index cache cleared")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _is_indian(ticker: str) -> bool:
    return any(ticker.upper().endswith(s) for s in INDIAN_SUFFIXES)


def _fetch_ohlcv(ticker: str) -> pd.DataFrame | None:
    """Download 6 months of daily OHLCV. Returns None if insufficient data."""
    try:
        hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
        if hist.empty or len(hist) < 60:
            logger.warning(f"[ML] {ticker} — {len(hist)} rows (need 60+)")
            return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as e:
        logger.error(f"[ML] {ticker} OHLCV fetch failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Feature computation — mirrors engineer_features.py exactly
# ══════════════════════════════════════════════════════════════════════════════
def _compute_features(ohlcv: pd.DataFrame, mkt: dict) -> dict | None:
    try:
        closes  = ohlcv["Close"].astype(float).reset_index(drop=True)
        highs   = ohlcv["High"].astype(float).reset_index(drop=True)
        lows    = ohlcv["Low"].astype(float).reset_index(drop=True)
        volumes = ohlcv["Volume"].astype(float).reset_index(drop=True)
        price   = float(closes.iloc[-1])

        if price <= 0:
            return None

        f = {}

        # ── RSI ───────────────────────────────────────────────────────────────
        f["RSI_14"] = float(ta.rsi(closes, length=14).iloc[-1])
        f["RSI_21"] = float(ta.rsi(closes, length=21).iloc[-1])

        # ── MACD ──────────────────────────────────────────────────────────────
        macd_df          = ta.macd(closes, fast=12, slow=26, signal=9)
        f["MACD"]        = float(macd_df["MACD_12_26_9"].iloc[-1])
        f["MACD_signal"] = float(macd_df["MACDs_12_26_9"].iloc[-1])
        f["MACD_hist"]   = float(macd_df["MACDh_12_26_9"].iloc[-1])

        # ── Bollinger Bands — normalized to price ─────────────────────────────
        bb_df         = ta.bbands(closes, length=20, std=2)
        bb_lower_col  = [c for c in bb_df.columns if c.startswith("BBL")][0]
        bb_middle_col = [c for c in bb_df.columns if c.startswith("BBM")][0]
        bb_upper_col  = [c for c in bb_df.columns if c.startswith("BBU")][0]
        bb_lower_val  = float(bb_df[bb_lower_col].iloc[-1])
        bb_upper_val  = float(bb_df[bb_upper_col].iloc[-1])

        f["BB_lower"]  = bb_lower_val                          / price
        f["BB_middle"] = float(bb_df[bb_middle_col].iloc[-1]) / price
        f["BB_upper"]  = bb_upper_val                          / price
        f["BB_width"]  = f["BB_upper"] - f["BB_lower"]

        # ── Moving averages — normalized to price ─────────────────────────────
        f["SMA_20"] = float(closes.rolling(20).mean().iloc[-1]) / price
        f["SMA_50"] = float(closes.rolling(50).mean().iloc[-1]) / price
        f["EMA_12"] = float(closes.ewm(span=12, adjust=False).mean().iloc[-1]) / price
        f["EMA_26"] = float(closes.ewm(span=26, adjust=False).mean().iloc[-1]) / price

        # ── Cross signals ─────────────────────────────────────────────────────
        sma20_val = float(closes.rolling(20).mean().iloc[-1])
        sma50_val = float(closes.rolling(50).mean().iloc[-1])
        f["SMA_cross"]      = 1 if sma20_val > sma50_val else 0
        f["price_above_bb"] = (
             1 if price > bb_upper_val else
            -1 if price < bb_lower_val else 0
        )

        # ── Price momentum (%) ────────────────────────────────────────────────
        f["price_momentum_5d"]  = float(closes.pct_change(5).iloc[-1])  * 100
        f["price_momentum_10d"] = float(closes.pct_change(10).iloc[-1]) * 100
        f["price_momentum_20d"] = float(closes.pct_change(20).iloc[-1]) * 100
        f["volume_momentum_5d"] = float(volumes.pct_change(5).iloc[-1]) * 100

        # ── Volatility ────────────────────────────────────────────────────────
        daily_ret = closes.pct_change()
        vol20     = float(daily_ret.rolling(20).std().iloc[-1]) * 100
        vol60     = float(daily_ret.rolling(60).std().iloc[-1]) * 100
        f["volatility_20d"]   = vol20
        f["volatility_ratio"] = vol20 / (vol60 + 1e-9)

        # ── Range — FIXED: daily candle H/L, NOT 52-week ──────────────────────
        last_high = float(highs.iloc[-1])
        last_low  = float(lows.iloc[-1])
        f["high_low_range"]      = (last_high - last_low) / price * 100
        f["close_to_high_ratio"] = price / (last_high + 1e-9)

        # ── Volume ────────────────────────────────────────────────────────────
        vol_mean20            = float(volumes.rolling(20).mean().iloc[-1])
        f["volume_ratio_20d"] = float(volumes.iloc[-1]) / (vol_mean20 + 1e-9)

        # ── ATR_14 ────────────────────────────────────────────────────────────
        atr_raw    = ta.atr(highs, lows, closes, length=14)
        f["ATR_14"] = float(atr_raw.iloc[-1]) / price * 100

        # ── ADX ───────────────────────────────────────────────────────────────
        adx_df  = ta.adx(highs, lows, closes, length=14)
        adx_col = [c for c in adx_df.columns if c.startswith("ADX")][0]
        f["ADX"] = float(adx_df[adx_col].iloc[-1])

        # ── Stochastic ────────────────────────────────────────────────────────
        stoch_df    = ta.stoch(highs, lows, closes, k=14, d=3, smooth_k=3)
        stoch_k_col = [c for c in stoch_df.columns if c.startswith("STOCHk")][0]
        stoch_d_col = [c for c in stoch_df.columns if c.startswith("STOCHd")][0]
        f["stochastic_k"] = float(stoch_df[stoch_k_col].iloc[-1])
        f["stochastic_d"] = float(stoch_df[stoch_d_col].iloc[-1])

        # ── VWAP distance ─────────────────────────────────────────────────────
        typical = (highs + lows + closes) / 3
        vwap    = (typical * volumes).rolling(20).sum() / \
                  (volumes.rolling(20).sum() + 1e-9)
        f["VWAP_distance"] = float(((closes - vwap) / (vwap + 1e-9) * 100).iloc[-1])

        # ── OBV ratio ─────────────────────────────────────────────────────────
        obv_raw    = ta.obv(closes, volumes)
        obv_mean20 = obv_raw.rolling(20).mean().abs()
        f["OBV_ratio"] = float((obv_raw / (obv_mean20 + 1e-9)).iloc[-1])

        # ── Index features (market-aware from cache) ──────────────────────────
        f["NIFTY_return"]     = mkt["NIFTY_return"]
        f["BANKNIFTY_return"] = mkt["BANKNIFTY_return"]
        f["India_VIX"]        = mkt["India_VIX"]
        f["relative_strength"] = f["price_momentum_20d"] - mkt["NIFTY_cum20"]

        # ── Fill any features the model expects but aren't computed above ─────
        for col in feature_cols:
            if col not in f:
                logger.warning(f"[ML] Feature '{col}' missing — filling 0")
                f[col] = 0.0

        return f

    except Exception as e:
        logger.error(f"[ML] Feature computation error: {e}", exc_info=True)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════
def predict_direction(metrics: dict) -> dict:
    """
    Predict 14-day price direction for any stock — Indian or US.

    Args:
        metrics: {"ticker": "RELIANCE.NS"}  or  {"ticker": "AAPL"}
                 Optionally include pre-fetched arrays to skip yfinance call:
                   price_history, high_history, low_history, volume_history

    Returns:
        {
            "direction":  "UP" | "DOWN",
            "confidence": float (0.50 – 1.00),
            "horizon":    "14d",
            "market":     "india" | "us"
        }
    """
    FALLBACK = {"direction": "DOWN", "confidence": 0.5,
                "horizon": "14d", "market": "unknown"}

    ticker = metrics.get("ticker", "").strip().upper()
    if not ticker:
        return FALLBACK

    if not MODEL_AVAILABLE:
        logger.warning("[ML] Model not loaded — run train_model.py first")
        return FALLBACK

    market = "india" if _is_indian(ticker) else "us"
    mkt    = _load_index_cache(market)

    # Use pre-supplied OHLCV arrays if all four are present and long enough
    ph = metrics.get("price_history",  [])
    hh = metrics.get("high_history",   [])
    lh = metrics.get("low_history",    [])
    vh = metrics.get("volume_history", [])

    if len(ph) >= 60 and len(hh) >= 60 and len(lh) >= 60 and len(vh) >= 60:
        ohlcv = pd.DataFrame({
            "Open":   ph,
            "High":   hh,
            "Low":    lh,
            "Close":  ph,
            "Volume": vh,
        })
    else:
        ohlcv = _fetch_ohlcv(ticker)
        if ohlcv is None:
            return FALLBACK

    features_dict = _compute_features(ohlcv, mkt)
    if features_dict is None:
        return FALLBACK

    # Build feature vector in exact training column order
    feature_vector = []
    for col in feature_cols:
        val = features_dict.get(col, 0.0)
        if pd.isna(val) or np.isinf(val):
            val = 0.0
        feature_vector.append(float(val))

    X = np.array(feature_vector).reshape(1, -1)

    try:
        prob_up = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.error(f"[ML] predict_proba failed for {ticker}: {e}")
        return FALLBACK

    # Always UP or DOWN — no NEUTRAL
    direction  = "UP" if prob_up >= 0.5 else "DOWN"
    confidence = round(max(prob_up, 1 - prob_up), 3)

    logger.info(f"[ML] {ticker} ({market}) → {direction} ({confidence:.1%})")
    return {
        "direction":  direction,
        "confidence": confidence,
        "horizon":    "14d",
        "market":     market,
    }