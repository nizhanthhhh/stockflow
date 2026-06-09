from scipy import stats
import numpy as np

def detect_anomalies(metrics: dict) -> dict:
    """Non-LLM agent: statistical anomaly detection."""
    
    flags = []
    anomalies = {}
    
    price_history = metrics.get("price_history", [])
    
    if len(price_history) >= 10:
        prices = np.array(price_history)
        z_scores = np.abs(stats.zscore(prices))
        
        # Flag if latest price is a statistical outlier (z > 2)
        latest_z = z_scores[-1]
        if latest_z > 2:
            anomalies["price_anomaly"] = True
            direction = "above" if prices[-1] > np.mean(prices) else "below"
            flags.append(f"Price is {latest_z:.2f} standard deviations {direction} its 30-day mean — unusual movement detected")        
        # Check volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # annualized
        anomalies["annualized_volatility"] = round(volatility * 100, 2)
        
        if volatility > 0.5:  # 50% annualized = high risk
            flags.append(f"High volatility: {volatility*100:.1f}% annualized")
    
    # Volume spike detection
    avg_vol = metrics.get("avg_volume_30d", 0)
    latest_vol = metrics.get("latest_volume", 0)
    if avg_vol > 0:
        vol_ratio = latest_vol / avg_vol
        anomalies["volume_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 2:
            flags.append(f"Volume spike: {vol_ratio:.1f}x above 30-day average")
    
    # Valuation checks
    pe = metrics.get("pe_ratio")
    if pe and pe > 50:
        flags.append(f"High P/E ratio: {pe:.1f} (potential overvaluation)")
    if pe and pe < 0:
        flags.append("Negative P/E ratio (company currently unprofitable)")
    
    debt_eq = metrics.get("debt_to_equity")
    if debt_eq and debt_eq > 2:
        flags.append(f"Very high debt-to-equity: {debt_eq:.2f}x")
    
    # Distance from 52-week high/low
    price = metrics.get("current_price", 0)
    high = metrics.get("52w_high", 0)
    low = metrics.get("52w_low", 0)
    if high > 0:
        pct_from_high = ((price - high) / high) * 100
        if pct_from_high < -30:
            flags.append(f"Trading {abs(pct_from_high):.1f}% below 52-week high")
    
    anomalies["flags"] = flags
    anomalies["risk_level"] = (
        "HIGH" if len(flags) >= 3 else
        "MEDIUM" if len(flags) >= 1 else
        "LOW"
    )
    
    return anomalies