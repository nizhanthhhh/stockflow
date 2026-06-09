from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Formatting helpers (agent-local, no dependency on orchestrator) ──
def _fmt(val, prefix="", suffix="", is_indian=False):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        val = round(val, 2)
    if isinstance(val, (int, float)):
        if is_indian and abs(val) >= 10_000_000:
            crores = val / 10_000_000
            if crores >= 100_000:
                return f"{prefix}{crores/100_000:.2f} Lakh Cr{suffix}"
            return f"{prefix}{crores:,.2f} Cr{suffix}"
        if prefix:
            if abs(val) >= 1_000_000_000:
                return f"{prefix}{val/1_000_000_000:.2f}B{suffix}"
            elif abs(val) >= 1_000_000:
                return f"{prefix}{val/1_000_000:.2f}M{suffix}"
        return f"{prefix}{val:,}{suffix}"
    return str(val)


SECTOR_BENCHMARKS = {
    "Technology":             {"pe_fair": 30, "growth_strong": 0.15, "margin_healthy": 0.15},
    "Consumer Cyclical":      {"pe_fair": 25, "growth_strong": 0.10, "margin_healthy": 0.08},
    "Consumer Defensive":     {"pe_fair": 20, "growth_strong": 0.06, "margin_healthy": 0.07},
    "Healthcare":             {"pe_fair": 22, "growth_strong": 0.08, "margin_healthy": 0.12},
    "Financial Services":     {"pe_fair": 15, "growth_strong": 0.07, "margin_healthy": 0.20},
    "Energy":                 {"pe_fair": 14, "growth_strong": 0.05, "margin_healthy": 0.10},
    "Industrials":            {"pe_fair": 20, "growth_strong": 0.07, "margin_healthy": 0.10},
    "Basic Materials":        {"pe_fair": 16, "growth_strong": 0.05, "margin_healthy": 0.08},
    "Real Estate":            {"pe_fair": 35, "growth_strong": 0.05, "margin_healthy": 0.25},
    "Utilities":              {"pe_fair": 18, "growth_strong": 0.03, "margin_healthy": 0.12},
    "Communication Services": {"pe_fair": 22, "growth_strong": 0.10, "margin_healthy": 0.15},
}
DEFAULT_BENCHMARK  = {"pe_fair": 22, "growth_strong": 0.10, "margin_healthy": 0.10}
CONGLOMERATES      = ["Reliance","Tata","Mahindra","Bajaj","Adani","Berkshire","3M","GE","Softbank","Samsung"]


def _get_benchmark(metrics: dict) -> dict:
    name = metrics.get("company_name", "")
    if any(c in name for c in CONGLOMERATES):
        return {"pe_fair": 22, "growth_strong": 0.08, "margin_healthy": 0.10, "label": "Conglomerate"}
    bm = SECTOR_BENCHMARKS.get(metrics.get("sector", ""), DEFAULT_BENCHMARK).copy()
    bm["label"] = metrics.get("sector", "Unknown")
    return bm


def write_memo(
    ticker: str,
    metrics: dict,
    sentiment: dict,
    anomalies: dict,
    prediction: dict = None,
    is_indian: bool = False,
    currency: str = "$",
) -> str:
    """
    Memo-writer agent.
    Receives structured outputs from all data agents and produces
    a professional investment memo via Groq/Llama.
    """
    if prediction is None:
        prediction = {"direction": "UNKNOWN", "confidence": 0.0, "horizon": "14d"}
    bm     = _get_benchmark(metrics)
    sector = metrics.get("sector", "Unknown")

    # ── Pre-compute signals ──
    pe     = metrics.get("pe_ratio")
    growth = metrics.get("revenue_growth")
    margin = metrics.get("profit_margins")
    price  = metrics.get("current_price") or 0
    target = metrics.get("analyst_target") or 0
    upside = round((target - price) / price * 100, 1) if price and target else None
    fcf    = metrics.get("free_cash_flow")

    fcf_signal    = ("Strong generator"   if fcf and fcf > 0 else
                     "Negative generator" if fcf and fcf < 0 else
                     "Data unavailable")
    pe_signal     = ("Expensive" if pe and pe > bm["pe_fair"] * 1.3 else
                     "Fair"      if pe and pe > bm["pe_fair"] * 0.7 else
                     "Cheap"     if pe else "N/A")
    growth_signal = ("Strong"   if growth and growth >= bm["growth_strong"]       else
                     "Moderate" if growth and growth >= bm["growth_strong"] * 0.5 else
                     "Weak"     if growth else "N/A")
    margin_signal = ("Healthy"  if margin and margin >= bm["margin_healthy"] else
                     "Thin"     if margin and margin >= 0                    else
                     "Negative" if margin else "N/A")

    # ── Format ML prediction signal ──
    ml_dir  = prediction.get("direction", "UNKNOWN")
    ml_conf = prediction.get("confidence", 0.0)
    ml_hor  = prediction.get("horizon", "14d")
    ml_signal = (
        f"{ml_dir} ({ml_conf*100:.0f}% confidence, {ml_hor})"
        if ml_dir != "UNKNOWN"
        else "Insufficient data"
    )

    flags     = "\n".join([f"  - {f}" for f in anomalies.get("flags", [])]) or "  - None"
    headlines = "\n".join(
        [f"  - [{h['sentiment'].upper()}] {h['title']}" for h in sentiment.get("top_headlines", [])]
    ) or "  - None"

    prompt = f"""You are a conservative senior investment analyst. Write an investment memo using ONLY the data below. Never invent numbers. Use correct grammar ('1 article' not '1 articles').

STRICT RECOMMENDATION RULES:
- STRONG BUY:  risk=LOW + sentiment=Bullish + upside>20% + FCF confirmed positive — ALL must be true
- BUY:         risk=LOW or MEDIUM + sentiment=Bullish or Neutral + upside>10%
- HOLD:        mixed signals, upside 0–10%, medium risk, thin margin, or FCF unknown
- SELL:        risk=HIGH or negative margins or price above analyst target
- STRONG SELL: risk=HIGH + negative cashflow + bearish sentiment — ALL must be true
- If FCF is N/A or Data unavailable, maximum recommendation is BUY.
- Thin profit margin alone prevents STRONG BUY.
- Default to HOLD when uncertain.

===DATA===
Company: {metrics.get('company_name')} ({ticker})
Sector: {sector} | Industry: {metrics.get('industry')}
Benchmark: {bm['label']}

Price: {_fmt(price, currency, is_indian=is_indian)} | Target: {_fmt(target, currency, is_indian=is_indian)} | Upside: {_fmt(upside)}%
Market Cap: {_fmt(metrics.get('market_cap'), currency, is_indian=is_indian)}
P/E: {_fmt(pe)} (fair: ~{bm['pe_fair']}x) → {pe_signal}
Forward P/E: {_fmt(metrics.get('forward_pe'))}
Revenue Growth: {_fmt(growth)} (strong: >{bm['growth_strong']*100}%) → {growth_signal}
Profit Margin: {_fmt(margin)} (healthy: >{bm['margin_healthy']*100}%) → {margin_signal}
Debt/Equity: {_fmt(metrics.get('debt_to_equity'))}
Free Cash Flow: {_fmt(fcf, currency, is_indian=is_indian)} → {fcf_signal}
52W High: {_fmt(metrics.get('52w_high'), currency, is_indian=is_indian)} | Low: {_fmt(metrics.get('52w_low'), currency, is_indian=is_indian)}
6M Change: {_fmt(metrics.get('price_change_6m'))}%
Analyst Consensus: {metrics.get('recommendation','N/A').upper()}
Sentiment: {sentiment.get('overall')} (score: {sentiment.get('average_sentiment')})
Articles: {sentiment.get('positive_count')} positive / {sentiment.get('neutral_count')} neutral / {sentiment.get('negative_count')} negative
Headlines:
{headlines}
Risk: {anomalies.get('risk_level')} | Volatility: {_fmt(anomalies.get('annualized_volatility'))}% | Volume: {_fmt(anomalies.get('volume_ratio'))}x avg
ML Prediction: {ml_signal}
Flags:
{flags}
Business: {metrics.get('description','')}
===END DATA===

Output EXACTLY this format:

---

# {metrics.get('company_name')} ({ticker})
**{sector}  ·  {metrics.get('industry')}**

---

## Snapshot
| Metric | Value | Signal |
|--------|-------|--------|
| Price | {_fmt(price, currency, is_indian=is_indian)} | Target {_fmt(target, currency, is_indian=is_indian)} ({_fmt(upside)}% upside) |
| Market Cap | {_fmt(metrics.get('market_cap'), currency, is_indian=is_indian)} | [Large/Mid/Small cap] |
| P/E Ratio | {_fmt(pe)} | {pe_signal} vs {bm['label']} fair {bm['pe_fair']}x |
| Revenue Growth | {_fmt(growth)} | {growth_signal} for {bm['label']} |
| Profit Margin | {_fmt(margin)} | {margin_signal} for {bm['label']} |
| Debt / Equity | {_fmt(metrics.get('debt_to_equity'))} | [Low <0.5 / Moderate 0.5–1.5 / High >1.5] |
| Free Cash Flow | {_fmt(fcf, currency, is_indian=is_indian)} | {fcf_signal} |
| ML Prediction | {ml_dir} | {ml_signal} |

---

## Market Sentiment  —  [BULLISH / NEUTRAL / BEARISH]
> [1 sentence OR "Insufficient news data."]

- Positive: {sentiment.get('positive_count')} articles
- Neutral:  {sentiment.get('neutral_count')} articles
- Negative: {sentiment.get('negative_count')} articles

**Top headlines:**
[Top 3 verbatim OR "No recent headlines found."]

---

## Risk Flags  —  [{anomalies.get('risk_level')} RISK]
[Each flag as "- <flag>" OR "No anomalies detected."]

Volatility: {_fmt(anomalies.get('annualized_volatility'))}% annualized  ·  Volume: {_fmt(anomalies.get('volume_ratio'))}x avg

---

## Valuation
[2 sentences using signals + upside numbers.]

---

## Recommendation
> **[STRONG BUY / BUY / HOLD / SELL / STRONG SELL]**

[3 sentences: why this rating, what upgrades it, what to watch.]

VERDICT: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]

---

*Informational only — not financial advice. Data from Yahoo Finance, may be delayed 15 minutes.*
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content
