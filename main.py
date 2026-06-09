from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import csv
import re

load_dotenv()                    # must be first

from core.orchestrator import run_analysis

app = FastAPI()

# ── Stock search data — loaded once at startup ──────────────────────────────

# Regex to strip common corporate suffixes before name matching
_STRIP_SUFFIXES = re.compile(
    r'\b(limited|ltd|inc|incorporated|corporation|corp|plc|llc|group|holdings|co)\b\.?',
    re.IGNORECASE
)

def _clean_name(name: str) -> str:
    """Lowercase + strip corporate suffixes for cleaner fuzzy matching."""
    return _STRIP_SUFFIXES.sub('', name).strip().lower()

# Load CSV into memory once — pre-compute lowercase fields so no work is done per request
_STOCKS: list[dict] = []
try:
    with open("datasets/all_stocks.csv", newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker", "").strip()
            name   = row.get("name", "").strip()
            exch   = row.get("exchange", "").strip()
            if ticker and name:
                _STOCKS.append({
                    "ticker":       ticker,
                    "name":         name,
                    "exchange":     exch,
                    "ticker_lower": ticker.lower(),   # pre-lowercased for case-insensitive match
                    "clean_name":   _clean_name(name), # pre-cleaned for better matching
                })
except FileNotFoundError:
    print("[WARN] datasets/all_stocks.csv not found — /search will return empty results")

print(f"[STARTUP] Loaded {len(_STOCKS)} stocks into memory")


# ── /search endpoint ─────────────────────────────────────────────────────────

@app.get("/search")
async def search_stocks(q: str = Query(default="")):
    """
    Hybrid stock search — 3 steps:
      Step 1: Deterministic scoring (exact ticker > ticker prefix > name prefix > name contains)
      Step 2: RapidFuzz fallback ONLY if Step 1 yields fewer than 8 results
      Step 3: Merge, sort descending, return top 8 (no score field in response)
    """
    # Empty query guard — return immediately before any processing
    q = q.strip().lower()
    if not q:
        return []

    seen_tickers: set[str] = set()
    results: list[tuple[int, dict]] = []  # (score, stock_dict)

    # ── Step 1: deterministic priority scoring ──────────────────────────────
    for stock in _STOCKS:
        tl = stock["ticker_lower"]  # pre-lowercased ticker
        cn = stock["clean_name"]    # pre-cleaned company name

        if tl == q:
            score = 100             # exact ticker match — highest priority
        elif tl.startswith(q):
            score = 85              # ticker starts with query
        elif cn.startswith(q):
            score = 70              # company name starts with query
        elif q in cn:
            score = 50              # query found anywhere in company name
        else:
            continue                # no match — skip entirely

        seen_tickers.add(tl)
        results.append((score, stock))

    # Sort Step 1 results by score descending
    results.sort(key=lambda x: x[0], reverse=True)

    # ── Step 2: RapidFuzz fallback — only triggered if fewer than 8 results ─
    if len(results) < 8:
        try:
            from rapidfuzz import process, fuzz
            # Only search stocks not already found in Step 1
            candidates = [s for s in _STOCKS if s["ticker_lower"] not in seen_tickers]
            fuzzy_hits = process.extract(
                q,
                [s["clean_name"] for s in candidates],
                scorer=fuzz.token_set_ratio,
                limit=8 - len(results),
                score_cutoff=45,    # minimum quality threshold
            )
            for _, fuzzy_score, idx in fuzzy_hits:
                stock = candidates[idx]
                # Cap fuzzy score at 40 — always ranked below any Step 1 result
                results.append((min(fuzzy_score // 3, 40), stock))
        except ImportError:
            pass  # rapidfuzz not installed — skip fallback silently

    # ── Step 3: final sort, deduplicate, slice top 8 ────────────────────────
    results.sort(key=lambda x: x[0], reverse=True)

    seen: set[str] = set()
    output = []
    for _, stock in results:
        if stock["ticker"] not in seen:
            seen.add(stock["ticker"])
            output.append({
                "ticker":   stock["ticker"],
                "name":     stock["name"],
                "exchange": stock["exchange"],
                # score intentionally omitted from response
            })
        if len(output) == 8:
            break

    return output


# ── /analyze endpoint (unchanged) ────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    ticker: str

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """Run the full analysis pipeline and stream the result."""

    def generate():
        try:
            yield f"data: 🔍 Fetching financial data for **{request.ticker.upper()}**...\n\n"
            result = run_analysis(request.ticker)
            # Strip leading horizontal rules the LLM sometimes adds
            result["memo"] = result["memo"].lstrip('-').lstrip('\n').strip()
            yield f"data: {json.dumps(result)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ❌ Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/")
async def serve_frontend():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
