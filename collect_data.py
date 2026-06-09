"""
collect_data.py
───────────────
Fetches 5 years of daily OHLCV data for ALL 8,011 stocks from all_stocks.csv.
Uses checkpointing to avoid restarting from scratch if interrupted.

Run once: python collect_data.py
Takes 2-4 hours for 8,011 stocks (with 1s delay between requests).

→ Next step: python engineer_features.py
"""

import pandas as pd
import yfinance as yf
import time
import os
import json
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_PATH       = "datasets/training_raw.parquet"
CHECKPOINT_PATH   = "datasets/.collect_checkpoint.json"
BATCH_SIZE        = 100
DELAY_PER_REQUEST = 1.0   # seconds — prevents yfinance rate limiting

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    """Load checkpoint (ticker names only — no DataFrames)."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
            print(f"  → Resuming from checkpoint")
            print(f"    Completed : {len(data['completed_tickers'])} tickers")
            print(f"    Failed    : {len(data['failed_tickers'])} tickers")
            return data
    print(f"  → Starting fresh")
    return {
        "completed_tickers": [],
        "failed_tickers":    [],
        "start_time":        datetime.now().isoformat()
    }

def save_checkpoint(checkpoint: dict):
    """Save checkpoint — only stores ticker name strings, never DataFrames."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)

def append_to_parquet(df: pd.DataFrame, path: str):
    """
    Append a batch DataFrame to the Parquet file on disk.
    If the file doesn't exist yet, create it.
    Data goes straight to disk — never accumulates in RAM.
    """
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)

# ── Step 1: Load stock list ───────────────────────────────────────────────────
print("=" * 70)
print("  StockFlow — Data Collection Pipeline")
print("=" * 70)
print("\n[STEP 1] Loading stock list from all_stocks.csv...")

try:
    all_stocks  = pd.read_csv("datasets/all_stocks.csv")
    all_tickers = all_stocks["ticker"].dropna().unique().tolist()
    print(f"  ✓ Found {len(all_tickers)} tickers")
except FileNotFoundError:
    print("  ✗ datasets/all_stocks.csv not found — check your path")
    raise SystemExit(1)

# ── Step 2: Load checkpoint, compute remaining tickers ────────────────────────
print(f"\n[STEP 2] Checking for previous checkpoint...")
os.makedirs("datasets", exist_ok=True)
checkpoint = load_checkpoint()

# Key fix: filter by name, not by index position
# This is safe even if the CSV order changes between runs
already_processed = set(checkpoint["completed_tickers"]) | set(checkpoint["failed_tickers"])
remaining_tickers = [t for t in all_tickers if t not in already_processed]

print(f"  → {len(remaining_tickers)} tickers remaining to fetch")

if not remaining_tickers:
    print("  ✓ All tickers already collected — skipping to save step")

# ── Step 3: Download OHLCV data ───────────────────────────────────────────────
print(f"\n[STEP 3] Downloading OHLCV data (5 years per ticker)...")
print(f"  Delay between requests : {DELAY_PER_REQUEST}s")
print(f"  Checkpoint every       : {BATCH_SIZE} tickers\n")

batch_frames   = []   # Holds current batch only — flushed to disk every BATCH_SIZE
start_time     = datetime.fromisoformat(checkpoint["start_time"])
total          = len(all_tickers)
completed_so_far = len(checkpoint["completed_tickers"])

for i, ticker in enumerate(remaining_tickers, start=1):
    global_i = completed_so_far + i   # Position across the full run including prior sessions

    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="5y")

        if hist.empty:
            print(f"  [{global_i:5d}/{total}] {ticker:12s} ✗ EMPTY (no data returned)")
            checkpoint["failed_tickers"].append(ticker)
            time.sleep(DELAY_PER_REQUEST)
            continue

        # Clean and tag with ticker
        hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        hist.columns = ["open", "high", "low", "close", "volume"]
        hist.index.name = "date"
        hist.reset_index(inplace=True)
        hist["date"]   = pd.to_datetime(hist["date"]).dt.tz_localize(None)
        hist["ticker"] = ticker

        batch_frames.append(hist)
        checkpoint["completed_tickers"].append(ticker)

        print(f"  [{global_i:5d}/{total}] {ticker:12s} ✓  {len(hist)} rows  "
              f"({hist['date'].min().date()} → {hist['date'].max().date()})")

    except Exception as e:
        print(f"  [{global_i:5d}/{total}] {ticker:12s} ✗ ERROR: {str(e)[:60]}")
        checkpoint["failed_tickers"].append(ticker)

    # Every BATCH_SIZE tickers: flush data to disk, save checkpoint, free RAM
    if i % BATCH_SIZE == 0 and batch_frames:
        batch_df = pd.concat(batch_frames, ignore_index=True)
        append_to_parquet(batch_df, OUTPUT_PATH)
        batch_frames = []   # Free RAM — the batch is safely on disk now

        save_checkpoint(checkpoint)

        elapsed_secs = (datetime.now() - start_time).total_seconds()
        rate_per_hr  = global_i / (elapsed_secs / 3600) if elapsed_secs > 0 else 1
        eta_hrs      = (total - global_i) / rate_per_hr if rate_per_hr > 0 else 0

        print(f"\n  ── Checkpoint saved ──────────────────────────────────────────")
        print(f"     Progress  : {global_i}/{total} ({100*global_i/total:.1f}%)")
        print(f"     ETA       : {eta_hrs:.1f} hours")
        print(f"     Success   : {len(checkpoint['completed_tickers'])}")
        print(f"     Failed    : {len(checkpoint['failed_tickers'])}")
        print(f"  ─────────────────────────────────────────────────────────────\n")

    time.sleep(DELAY_PER_REQUEST)

# ── Step 4: Flush any remaining batch ────────────────────────────────────────
if batch_frames:
    print(f"\n[STEP 4] Flushing final batch ({len(batch_frames)} tickers)...")
    batch_df = pd.concat(batch_frames, ignore_index=True)
    append_to_parquet(batch_df, OUTPUT_PATH)
    print(f"  ✓ Saved to {OUTPUT_PATH}")
else:
    print(f"\n[STEP 4] No remaining batch to flush")

save_checkpoint(checkpoint)

# ── Step 5: Verify output ─────────────────────────────────────────────────────
print(f"\n[STEP 5] Verifying output file...")
if os.path.exists(OUTPUT_PATH):
    df = pd.read_parquet(OUTPUT_PATH)
    print(f"  ✓ File verified")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Unique tickers   : {df['ticker'].nunique():,}")
    print(f"  Date range       : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Success          : {len(checkpoint['completed_tickers']):,} tickers")
    print(f"  Failed           : {len(checkpoint['failed_tickers']):,} tickers")
    print(f"  Output           : {OUTPUT_PATH}")

    if checkpoint["failed_tickers"]:
        failed_preview = ", ".join(checkpoint["failed_tickers"][:10])
        more = len(checkpoint["failed_tickers"]) - 10
        print(f"  Failed tickers   : {failed_preview}" + (f" ... and {more} more" if more > 0 else ""))
        # Save full failed list for debugging
        with open("datasets/failed_tickers.json", "w") as f:
            json.dump(checkpoint["failed_tickers"], f, indent=2)
        print(f"  Full failed list : datasets/failed_tickers.json")

    # Clean up checkpoint now that everything succeeded
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"\n  ✓ Checkpoint cleaned up")

    print(f"\n{'=' * 70}")
    print(f"  → Next step: python engineer_features.py")
    print(f"{'=' * 70}\n")

else:
    print(f"  ✗ Output file not found — something went wrong")
    raise SystemExit(1)