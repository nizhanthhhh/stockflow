import yfinance as yf
from datetime import datetime

def fetch_stock_data(ticker: str) -> dict:
    """Non-LLM agent: fetches financial data safely."""

    try:
        stock = yf.Ticker(ticker)

        # safer alternative
        info = stock.get_info()

        hist = stock.history(period="6mo")

        # Calculate free cash flow with fallback
        fcf = info.get("freeCashflow")
        if fcf is None:
            # Fallback: operatingCashflow - capex
            op_cf = info.get("operatingCashflow")
            capex = info.get("capitalExpenditures")
            if op_cf and capex:
                fcf = op_cf + capex  # capex is already negative in yfinance

        metrics = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margins": info.get("profitMargins"),
            # Yahoo Finance returns debtToEquity as a percentage (e.g. 150.0 means 1.5x)
            # Divide by 100 to get the actual ratio
            "debt_to_equity": round(info.get("debtToEquity") / 100, 2) if info.get("debtToEquity") is not None else None,
            "free_cash_flow": fcf,
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "analyst_target": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "description": (info.get("longBusinessSummary") or "")[:300],
            "timestamp": datetime.utcnow().isoformat()
        }

        # 📊 Historical insights
        if not hist.empty:
            metrics["avg_volume_30d"] = int(hist["Volume"].tail(30).mean())
            metrics["latest_volume"] = int(hist["Volume"].iloc[-1])

            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]

            if start_price != 0:
                metrics["price_change_6m"] = round(
                    (end_price - start_price) / start_price * 100, 2
                )

            metrics["price_history"] = hist["Close"].tail(60).tolist()
            metrics["volume_history"] = hist["Volume"].tail(90).tolist()

        return metrics

    except Exception as e:
        return {
            "ticker": ticker.upper(),
            "error": str(e),
            "status": "failed"
        }