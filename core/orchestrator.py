import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

from agents.data_fetcher import fetch_stock_data
from agents.sentiment_agent import analyze_sentiment
from agents.anomaly_detector import detect_anomalies
from agents.memo_writer import write_memo
from agents.ml_predictor import predict_direction

executor = ThreadPoolExecutor(max_workers=8)


def _get_currency(ticker: str) -> str:
    return "₹" if ticker.endswith((".NS", ".BO")) else "$"


# ── ASYNC PIPELINE ──
async def run_analysis_async(ticker: str) -> dict:
    """
    Runs the full multi-agent analysis pipeline.
    ticker is already a valid yfinance ticker from the autocomplete dropdown.
    No LLM resolver needed — ticker comes directly from all_stocks.csv.
    """
    loop = asyncio.get_event_loop()

    ticker    = ticker.strip().upper()
    is_indian = ticker.endswith((".NS", ".BO"))
    currency  = _get_currency(ticker)

    print(f"[PIPELINE] Analyzing: {ticker}")

    # Step 1 — fetch financial data (other agents depend on this)
    metrics = await loop.run_in_executor(executor, fetch_stock_data, ticker)

    # Step 2 — sentiment, anomaly, ML prediction run IN PARALLEL
    sentiment, anomalies, prediction = await asyncio.gather(
        loop.run_in_executor(executor, analyze_sentiment, ticker, metrics.get("company_name", ticker)),
        loop.run_in_executor(executor, detect_anomalies, metrics),
        loop.run_in_executor(executor, predict_direction, metrics)
    )

    # Step 3 — memo writer synthesizes all agent outputs
    memo = await loop.run_in_executor(
        executor, write_memo, ticker, metrics, sentiment, anomalies, prediction, is_indian, currency
    )

    # Step 4 — build chart data for frontend
    import yfinance as yf
    from datetime import datetime, timedelta
    chart_data = []
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
        if not hist.empty:
            for dt, close in zip(hist.index, hist["Close"]):
                chart_data.append({
                    "time":  dt.strftime("%Y-%m-%d"),
                    "value": round(float(close), 2)
                })
    except Exception:
        price_history = metrics.get("price_history", [])
        if price_history:
            base_date = datetime.utcnow() - timedelta(days=len(price_history))
            for i, price in enumerate(price_history):
                chart_data.append({
                    "time":  (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "value": round(float(price), 2)
                })

    return {
        "memo":         memo,
        "chart_data":   chart_data,
        "ticker":       ticker,
        "company_name": metrics.get("company_name", ticker),
        "currency":     currency,
        "price_change": metrics.get("price_change_6m"),
    }


# sync wrapper for main.py
def run_analysis(ticker: str) -> dict:
    return asyncio.run(run_analysis_async(ticker))
