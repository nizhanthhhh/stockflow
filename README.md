# StockFlow — AI-Powered Investment Research Platform

> Generate professional investment memos for any listed stock in seconds using a multi-agent AI pipeline.

---

## What it does

StockFlow takes a stock name or ticker as input and runs it through a coordinated pipeline of specialized AI agents — each responsible for a distinct analytical task. The result is a structured investment memo with a BUY / HOLD / SELL verdict, market sentiment analysis, risk flags, valuation assessment, and a 6-month price chart.

---

## Architecture

```
User Input (search/autocomplete)
        │
        ▼
   /search Endpoint
        │
        ▼
 Selected Valid Ticker
        │
        ▼
    Orchestrator
        │
        ├─── data_fetcher ──────────────────► Financial Metrics (yfinance)
        │           │
        │           ├──► sentiment_agent ──► News Sentiment (VADER + RSS)
        │           ├──► anomaly_detector ─► Statistical Risk Flags (scipy)
        │           └──► ml_predictor ─────► 14-day Price Direction (LightGBM)
        │
        └─── memo_writer (Groq/Llama 3.3) ─► Investment Memo
```

All four specialist agents run in parallel after the data fetch, cutting latency roughly in half.

---

## Agents

| Agent | Type | Responsibility |
|---|---|---|
| `data_fetcher` | Non-LLM | Fetches 6-month OHLCV + fundamentals from Yahoo Finance |
| `sentiment_agent` | Non-LLM | Scrapes RSS feeds, scores headlines with VADER sentiment |
| `anomaly_detector` | Non-LLM | Z-score analysis, volatility, volume spikes, valuation flags |
| `ml_predictor` | ML Model | LightGBM classifier — predicts 14-day price direction with confidence % |
| `memo_writer` | LLM | Synthesizes all agent outputs into a structured investment memo |

---

## Tech Stack

**Backend**
- FastAPI — API server with SSE streaming
- Groq / Llama 3.3 70B — LLM for investment memo synthesis
- yfinance — financial data
- NLTK VADER — sentiment analysis
- scipy / numpy — statistical anomaly detection
- LightGBM + pandas-ta — ML price direction model
- RapidFuzz — fuzzy stock search

**Frontend**
- Vanilla HTML / CSS / JavaScript — no framework
- Lightweight Charts (TradingView) — 6-month price chart
- marked.js — markdown rendering

---

## Features

- **Hybrid stock search** — deterministic priority scoring + RapidFuzz fuzzy matching across 8,011 US and NSE-listed stocks
- **Investment memo** — structured report with Snapshot table, sentiment, risk flags, valuation, and recommendation
- **Verdict card** — prominent BUY / HOLD / SELL card as the hero of the report
- **6-month price chart** — interactive Lightweight Charts area chart with 6M % change
- **ML prediction** — LightGBM model trained on 500+ stocks predicts 14-day direction
- **Dark mode** — system-aware with manual toggle
- **Recent searches** — localStorage-based search history

---

## Project Structure

```
stockflow/
├── agents/
│   ├── data_fetcher.py       # yfinance data agent
│   ├── sentiment_agent.py    # VADER sentiment agent
│   ├── anomaly_detector.py   # Statistical anomaly agent
│   ├── ml_predictor.py       # LightGBM ML agent
│   └── memo_writer.py        # LLM memo writer agent
├── core/
│   └── orchestrator.py       # Async pipeline coordinator
├── datasets/
│   ├── all_stocks.csv              # 8,011 US + NSE stocks for search
│   ├── failed_tickers.json         # Record of tickers that failed to fetch
│   ├── training_raw.parquet        # Raw training data (gitignored)
│   └── training_features.parquet   # Engineered features for training (gitignored)
├── models/
│   ├── feature_columns.json      # ML feature schema
│   ├── feature_metadata.json     # Feature metadata and descriptions
│   ├── lgbm_model.pkl            # Trained LightGBM model
│   └── training_metadata.json    # Training configuration and metadata
├── frontend/
│   └── index.html            # Single-page UI
├── collect_data.py           # ML training data collection script
├── engineer_features.py      # Feature engineering script
├── train_model.py            # LightGBM training script
├── evaluate_model.py         # Model evaluation and testing script
└── main.py                   # FastAPI entry point
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Ankith34/stockflow.git
cd stockflow
```

**2. Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

**5. Run the server**
```bash
python main.py
```

Open `http://127.0.0.1:8000`

---

## ML Model (Optional)

To train the price direction model from scratch:

```bash
python collect_data.py       # ~70 min — fetches S&P 500 historical data
python engineer_features.py  # ~15-20 min  — computes technical indicators
python train_model.py        # ~5 min — trains, saves to models/
```

The trained model artifacts (`models/*.pkl`) are not included in the repo due to file size. The app runs without them — `ml_predictor` returns `UNKNOWN` gracefully if models are missing.

---

## Data Sources

- **Yahoo Finance** (via yfinance) — price data, fundamentals, analyst targets
- **Google News RSS + Yahoo Finance RSS** — news headlines for sentiment
- **NSE India + NASDAQ** — stock universe for autocomplete search

---

## Limitations

- Yahoo Finance data can be delayed 15 minutes and occasionally returns stale values for large-cap stocks
- NSE stocks with low global coverage may return incomplete data from yfinance
- ML model accuracy is ~55–60% AUC — used as a probabilistic signal, not a definitive prediction

---

*Not financial advice. For educational and demonstration purposes only.*
