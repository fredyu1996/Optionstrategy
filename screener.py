"""
screener.py
Core data fetching and analysis module for S&P 500 options screener.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fallback S&P 500 tickers (top 50 by market cap) if Wikipedia fetch fails
# ---------------------------------------------------------------------------
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "LLY", "UNH", "V", "XOM", "MA", "JNJ", "PG", "COST", "HD", "MRK",
    "ABBV", "CVX", "KO", "PEP", "BAC", "WMT", "ORCL", "NFLX", "CRM", "AMD",
    "MCD", "TMO", "LIN", "ABT", "DHR", "TXN", "CSCO", "NEE", "PM", "WFC",
    "RTX", "UNP", "HON", "INTU", "AMGN", "SPGI", "IBM", "GS", "ISRG", "CAT",
]


@st.cache_data(ttl=3600)
def get_sp500_tickers() -> pd.DataFrame:
    """Fetch S&P 500 tickers from Wikipedia; fallback to top-50 list."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={"id": "constituents"})
        df = tables[0]
        df = df.rename(
            columns={
                "Symbol": "ticker",
                "Security": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "sub_industry",
            }
        )
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df[["ticker", "name", "sector", "sub_industry"]].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(
            {
                "ticker": FALLBACK_TICKERS,
                "name": FALLBACK_TICKERS,
                "sector": ["Unknown"] * len(FALLBACK_TICKERS),
                "sub_industry": ["Unknown"] * len(FALLBACK_TICKERS),
            }
        )


# ---------------------------------------------------------------------------
# Macro data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900)
def get_macro_data() -> dict:
    """
    Fetch VIX, SPY, and TNX data and derive regime labels.

    Returns
    -------
    dict with keys:
        vix_current, vix_30d_avg, vix_regime, spy_price,
        spy_above_50ma, spy_above_200ma, spy_ret_1m, spy_ret_3m,
        tnx_yield, market_regime, market_bias,
        vix_hist (Series, 90d), spy_hist (Series, 90d)
    """
    result = {
        "vix_current": 20.0,
        "vix_30d_avg": 20.0,
        "vix_regime": "normal",
        "spy_price": 450.0,
        "spy_above_50ma": True,
        "spy_above_200ma": True,
        "spy_ret_1m": 0.0,
        "spy_ret_3m": 0.0,
        "tnx_yield": 4.5,
        "market_regime": "Normal Volatility — Balanced Market",
        "market_bias": "neutral",
        "vix_hist": pd.Series(dtype=float),
        "spy_hist": pd.Series(dtype=float),
    }

    try:
        raw = yf.download(
            ["^VIX", "SPY", "^TNX"],
            period="1y",
            auto_adjust=True,
            progress=False,
        )
        close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)

        # --- VIX ---
        if "^VIX" in close.columns:
            vix = close["^VIX"].dropna()
            result["vix_current"] = float(vix.iloc[-1])
            result["vix_30d_avg"] = float(vix.tail(30).mean())
            result["vix_hist"] = vix.tail(90)
            v = result["vix_current"]
            if v < 15:
                result["vix_regime"] = "low"
            elif v < 20:
                result["vix_regime"] = "normal"
            elif v < 30:
                result["vix_regime"] = "high"
            else:
                result["vix_regime"] = "extreme"

        # --- SPY ---
        if "SPY" in close.columns:
            spy = close["SPY"].dropna()
            result["spy_price"] = float(spy.iloc[-1])
            result["spy_hist"] = spy.tail(90)
            result["spy_above_50ma"] = float(spy.iloc[-1]) > float(spy.tail(50).mean())
            result["spy_above_200ma"] = float(spy.iloc[-1]) > float(spy.tail(200).mean())
            if len(spy) >= 21:
                result["spy_ret_1m"] = float((spy.iloc[-1] / spy.iloc[-21] - 1) * 100)
            if len(spy) >= 63:
                result["spy_ret_3m"] = float((spy.iloc[-1] / spy.iloc[-63] - 1) * 100)

        # --- TNX (10yr yield) ---
        if "^TNX" in close.columns:
            tnx = close["^TNX"].dropna()
            result["tnx_yield"] = float(tnx.iloc[-1])

    except Exception:
        pass

    # Derive market regime string and bias
    vr = result["vix_regime"]
    above50 = result["spy_above_50ma"]
    above200 = result["spy_above_200ma"]
    ret1m = result["spy_ret_1m"]

    if vr in ("high", "extreme"):
        if ret1m < -3:
            result["market_regime"] = "High Fear — Bearish Trend"
            result["market_bias"] = "bearish"
        else:
            result["market_regime"] = "High Volatility — Uncertain Market"
            result["market_bias"] = "neutral"
    elif vr == "low":
        if above50 and above200:
            result["market_regime"] = "Low Volatility — Strong Bull Market"
            result["market_bias"] = "bullish"
        else:
            result["market_regime"] = "Low Volatility — Weak/Choppy Market"
            result["market_bias"] = "neutral"
    else:
        if above50 and above200 and ret1m > 0:
            result["market_regime"] = "Normal Volatility — Bullish Trend"
            result["market_bias"] = "bullish"
        elif not above50 and ret1m < 0:
            result["market_regime"] = "Normal Volatility — Bearish Trend"
            result["market_bias"] = "bearish"
        else:
            result["market_regime"] = "Normal Volatility — Balanced Market"
            result["market_bias"] = "neutral"

    return result


# ---------------------------------------------------------------------------
# Volatility helpers
# ---------------------------------------------------------------------------

def compute_hv(prices: pd.Series, window: int = 30) -> float:
    """Compute annualized historical volatility over *window* trading days."""
    if prices is None or len(prices) < window + 2:
        return np.nan
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < window:
        return np.nan
    hv = float(log_returns.tail(window).std() * np.sqrt(252))
    return hv if np.isfinite(hv) else np.nan


def _days_to_expiry(expiry_str: str) -> int:
    """Return calendar days from today to expiry date string (YYYY-MM-DD)."""
    try:
        exp_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
        return max(0, (exp_dt - datetime.now()).days)
    except Exception:
        return 9999


def get_atm_iv(ticker_obj, target_dte: int = 30) -> float:
    """
    Retrieve ATM implied volatility from the options chain expiry
    nearest to *target_dte* calendar days.

    Returns float (annualised fraction, e.g. 0.30 for 30 %) or np.nan.
    """
    try:
        expirations = ticker_obj.options
        if not expirations:
            return np.nan

        # Pick the expiry whose DTE is closest to target_dte
        best_exp = min(
            expirations,
            key=lambda e: abs(_days_to_expiry(e) - target_dte),
        )

        chain = ticker_obj.option_chain(best_exp)
        calls = chain.calls
        puts = chain.puts

        # Current price
        info = ticker_obj.fast_info
        price = float(getattr(info, "last_price", 0) or 0)
        if price <= 0:
            hist = ticker_obj.history(period="2d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if price <= 0:
            return np.nan

        # ATM call: strike closest to current price
        if calls.empty or puts.empty:
            return np.nan

        calls = calls.copy()
        puts = puts.copy()
        calls["dist"] = (calls["strike"] - price).abs()
        puts["dist"] = (puts["strike"] - price).abs()

        atm_call_iv = float(calls.loc[calls["dist"].idxmin(), "impliedVolatility"])
        atm_put_iv = float(puts.loc[puts["dist"].idxmin(), "impliedVolatility"])

        avg_iv = (atm_call_iv + atm_put_iv) / 2
        return avg_iv if np.isfinite(avg_iv) and avg_iv > 0 else np.nan

    except Exception:
        return np.nan


def get_earnings_info(ticker_obj) -> tuple:
    """
    Return (next_earnings_date_str, days_to_earnings).
    If unknown, returns (None, 999).
    """
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return None, 999

        # calendar is a dict-like or DataFrame depending on yfinance version
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
            elif cal.shape[1] > 0:
                val = cal.iloc[0, 0]
            else:
                return None, 999
        elif isinstance(cal, dict):
            val = cal.get("Earnings Date", [None])
            if isinstance(val, (list, tuple)):
                val = val[0] if val else None
        else:
            return None, 999

        if val is None or pd.isna(val):
            return None, 999

        if isinstance(val, (pd.Timestamp, datetime)):
            dt = val
        else:
            dt = pd.to_datetime(val)

        days = max(0, (dt.date() - datetime.now().date()).days)
        return dt.strftime("%Y-%m-%d"), days
    except Exception:
        return None, 999


# ---------------------------------------------------------------------------
# Batch fundamentals screening
# ---------------------------------------------------------------------------

def _safe_get(info: dict, *keys, default=np.nan):
    for k in keys:
        v = info.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI(14) from a price series."""
    if prices is None or len(prices) < period + 1:
        return np.nan
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def batch_screen_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Download 1-year price history for *tickers*, compute HV30, HV252,
    momentum (1m, 3m), RSI(14), and fetch fundamentals.

    Returns DataFrame with columns:
        ticker, price, hv30, hv252, ret_1m, ret_3m, rsi, pe_ratio, market_cap, avg_volume
    """
    records = []

    # ----- Bulk price download -----
    if not tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers,
            period="1y",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception:
        raw = pd.DataFrame()

    def _get_close(ticker: str) -> pd.Series:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                return raw[ticker]["Close"].dropna()
            else:
                # Single ticker case
                return raw["Close"].dropna()
        except Exception:
            return pd.Series(dtype=float)

    # ----- Per-ticker fundamentals (lightweight info fetch) -----
    for t in tickers:
        row: dict = {"ticker": t}
        try:
            prices = _get_close(t)

            if prices.empty or len(prices) < 10:
                continue

            row["price"] = float(prices.iloc[-1])
            row["hv30"] = compute_hv(prices, 30)
            row["hv252"] = compute_hv(prices, 252)

            # Momentum
            row["ret_1m"] = float((prices.iloc[-1] / prices.iloc[-21] - 1) * 100) if len(prices) >= 21 else np.nan
            row["ret_3m"] = float((prices.iloc[-1] / prices.iloc[-63] - 1) * 100) if len(prices) >= 63 else np.nan

            # RSI
            row["rsi"] = compute_rsi(prices, 14)

            # Fundamentals via fast_info (lighter than .info)
            tk = yf.Ticker(t)
            try:
                fi = tk.fast_info
                row["market_cap"] = float(getattr(fi, "market_cap", None) or np.nan)
                row["avg_volume"] = float(getattr(fi, "three_month_average_volume", None) or np.nan)
                # P/E requires full info or ttm EPS
                info_dict = tk.info or {}
                row["pe_ratio"] = _safe_get(info_dict, "trailingPE", "forwardPE")
            except Exception:
                row["market_cap"] = np.nan
                row["avg_volume"] = np.nan
                row["pe_ratio"] = np.nan

        except Exception:
            continue

        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Ensure all columns present
    for col in ["price", "hv30", "hv252", "ret_1m", "ret_3m", "rsi", "pe_ratio", "market_cap", "avg_volume"]:
        if col not in df.columns:
            df[col] = np.nan
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# IV enrichment
# ---------------------------------------------------------------------------

def enrich_with_iv(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    For each row in *df*, fetch ATM IV and earnings info.
    Adds columns: atm_iv, iv_hv_ratio, next_earnings, days_to_earnings.
    Calls progress_callback(i, total) if provided.
    """
    atm_ivs = []
    iv_hv_ratios = []
    next_earnings_list = []
    days_to_earnings_list = []

    total = len(df)
    for i, row in enumerate(df.itertuples(), 1):
        if progress_callback:
            progress_callback(i, total)
        try:
            tk = yf.Ticker(row.ticker)
            iv = get_atm_iv(tk)
            ne, dte = get_earnings_info(tk)

            atm_ivs.append(iv)
            hv = row.hv30 if hasattr(row, "hv30") and not np.isnan(row.hv30) else np.nan
            iv_hv_ratios.append(iv / hv if (np.isfinite(iv) and np.isfinite(hv) and hv > 0) else np.nan)
            next_earnings_list.append(ne)
            days_to_earnings_list.append(dte)
        except Exception:
            atm_ivs.append(np.nan)
            iv_hv_ratios.append(np.nan)
            next_earnings_list.append(None)
            days_to_earnings_list.append(999)

    df = df.copy()
    df["atm_iv"] = atm_ivs
    df["iv_hv_ratio"] = iv_hv_ratios
    df["next_earnings"] = next_earnings_list
    df["days_to_earnings"] = days_to_earnings_list
    return df


# ---------------------------------------------------------------------------
# Strategy scoring
# ---------------------------------------------------------------------------

def _iv_hv_label(ratio: float) -> str:
    if pd.isna(ratio):
        return "Unknown"
    if ratio > 1.5:
        return "Very Expensive"
    if ratio > 1.2:
        return "Expensive"
    if ratio >= 0.8:
        return "Fair"
    return "Cheap"


def score_strategies(df: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """
    Add strategy scoring columns to *df*.
    Returns df with added columns:
        low_risk_score, high_risk_score, best_strategy, iv_label
    """
    df = df.copy()
    low_scores = []
    high_scores = []
    best_strats = []

    vix_regime = macro.get("vix_regime", "normal")
    market_bias = macro.get("market_bias", "neutral")

    for row in df.itertuples():
        iv_hv = getattr(row, "iv_hv_ratio", np.nan)
        if pd.isna(iv_hv):
            iv_hv = 1.0  # treat as neutral when unknown

        dte = getattr(row, "days_to_earnings", 999)
        if pd.isna(dte):
            dte = 999
        dte = int(dte)

        hv30 = getattr(row, "hv30", np.nan)
        if pd.isna(hv30):
            hv30 = 0.25

        ret_1m = getattr(row, "ret_1m", np.nan)
        if pd.isna(ret_1m):
            ret_1m = 0.0

        rsi = getattr(row, "rsi", np.nan)
        if pd.isna(rsi):
            rsi = 50.0

        mkt_cap = getattr(row, "market_cap", np.nan)
        avg_vol = getattr(row, "avg_volume", np.nan)

        # ---- LOW RISK (premium selling) score ----
        low = 50.0

        # IV/HV signal (primary)
        if iv_hv > 1.5:
            low += 30
        elif iv_hv > 1.2:
            low += 18
        elif iv_hv < 0.8:
            low -= 20

        # Earnings proximity (risk factor)
        if dte < 7:
            low -= 30
        elif dte < 14:
            low -= 15
        elif dte > 30:
            low += 10

        # VIX regime (high vol = better premium)
        if vix_regime == "extreme":
            low += 20
        elif vix_regime == "high":
            low += 12
        elif vix_regime == "low":
            low -= 8

        # Liquidity bonus
        if not pd.isna(mkt_cap) and mkt_cap > 1e11:  # >100B
            low += 8
        if not pd.isna(avg_vol) and avg_vol > 5e6:
            low += 5

        # RSI neutral zone preferred
        if 40 < rsi < 65:
            low += 5

        low = max(0, min(100, low))

        # ---- HIGH RISK (premium buying) score ----
        high = 50.0

        # IV/HV signal
        if iv_hv < 0.8:
            high += 25
        elif iv_hv > 1.5:
            high -= 25
        elif iv_hv > 1.2:
            high -= 12

        # Near earnings = good for long straddle / long options
        if dte < 7:
            high += 25
        elif dte < 14:
            high += 18
        elif dte < 21:
            high += 10
        elif dte > 60:
            high -= 5

        # Momentum
        if abs(ret_1m) > 10:
            high += 15
        elif abs(ret_1m) > 5:
            high += 8

        # RSI extremes
        if rsi > 75 or rsi < 25:
            high += 10

        # Market bias
        if market_bias == "bullish" and ret_1m > 3:
            high += 5
        elif market_bias == "bearish" and ret_1m < -3:
            high += 5

        high = max(0, min(100, high))

        # ---- Best strategy label ----
        if low >= high:
            if iv_hv > 1.5:
                best = "Iron Condor" if rsi > 40 and rsi < 65 else "Covered Call"
            elif iv_hv > 1.2:
                best = "Bull Put Spread" if market_bias != "bearish" else "Cash-Secured Put"
            else:
                best = "Cash-Secured Put"
        else:
            if dte < 14:
                best = "Long Straddle"
            elif market_bias == "bullish":
                best = "Bull Call Spread"
            elif market_bias == "bearish":
                best = "Long Put"
            else:
                best = "Long Call"

        low_scores.append(round(low, 1))
        high_scores.append(round(high, 1))
        best_strats.append(best)

    df["low_risk_score"] = low_scores
    df["high_risk_score"] = high_scores
    df["best_strategy"] = best_strats
    df["iv_label"] = df["iv_hv_ratio"].apply(_iv_hv_label)
    return df
