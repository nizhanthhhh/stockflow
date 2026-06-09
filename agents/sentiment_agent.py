import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

def analyze_sentiment(ticker: str, company_name: str) -> dict:
    sia = SentimentIntensityAnalyzer()

    clean_name   = company_name.replace(" Limited", "").replace(" Inc.", "") \
                               .replace(" Corporation", "").replace(" Corp.", "").strip()
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")

    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={clean_name}+stock&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={clean_name}+stock&hl=en-IN&gl=IN&ceid=IN:en",
        f"https://news.google.com/rss/search?q={clean_name}+shares&hl=en-IN&gl=IN&ceid=IN:en",
        f"https://news.google.com/rss/search?q={clean_ticker}+NSE&hl=en-IN&gl=IN&ceid=IN:en",
    ]

    headlines = []
    scores    = []
    seen      = set()

    # Prepare relevant words for stricter matching
    relevant_words = [w.lower() for w in clean_name.split() if len(w) > 4]

    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:8]:
                title = entry.get("title", "").strip()
                title_lower = title.lower()
                
                # Stricter relevance check: must match at least 2 words from company name OR the ticker
                word_matches = sum(1 for w in relevant_words if w in title_lower)
                ticker_match = clean_ticker.lower() in title_lower
                
                if not (word_matches >= 2 or ticker_match):
                    continue  # skip irrelevant headlines like "Reliance Naval" for Reliance
                
                if title and title not in seen:
                    seen.add(title)
                    score = sia.polarity_scores(title)
                    headlines.append({
                        "title":     title,
                        "compound":  score["compound"],
                        "sentiment": "positive" if score["compound"] >  0.05
                                else "negative" if score["compound"] < -0.05
                                else "neutral"
                    })
                    scores.append(score["compound"])
        except Exception:
            continue

    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        "average_sentiment": round(avg_score, 3),
        "overall":           "Bullish" if avg_score >  0.05
                        else "Bearish" if avg_score < -0.05
                        else "Neutral",
        "positive_count":    sum(1 for s in scores if s >  0.05),
        "negative_count":    sum(1 for s in scores if s < -0.05),
        "neutral_count":     sum(1 for s in scores if -0.05 <= s <= 0.05),
        "total_articles":    len(headlines),
        "top_headlines":     headlines[:5],
    }