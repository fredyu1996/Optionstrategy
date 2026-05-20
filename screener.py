"""
screener.py - Core data fetching and analysis module for S&P 500 Options Screener
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import math
import streamlit as st

warnings.filterwarnings('ignore')

# Fallback S&P 500 tickers (top 50 by market cap) if Wikipedia fetch fails
FALLBACK_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'AVGO', 'JPM',
    'LLY', 'UNH', 'V', 'XOM', 'MA', 'JNJ', 'PG', 'COST', 'HD', 'MRK',
    'ABBV', 'CVX', 'KO', 'PEP', 'BAC', 'WMT', 'ORCL', 'NFLX', 'CRM', 'AMD',
    'MCD', 'TMO', 'LIN', 'ABT', 'DHR', 'TXN', 'CSCO', 'NEE', 'PM', 'WFC',
    'RTX', 'UNP', 'HON', 'INTU', 'AMGN', 'SPGI', 'IBM', 'GS', 'ISRG', 'CAT'
]


@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia, fallback to top 50."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        df = df.rename(columns={
            'Symbol': 'ticker',
            'Security': 'name',
            'GICS Sector': 'sector',
            'GICS Sub-Industry': 'sub_industry'
        })
        df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)
        return df[['ticker', 'name', 'sector', 'sub_industry']].reset_index(drop=True)
    except Exception:
        return pd.DataFrame({
            'ticker': FALLBACK_TICKERS,
            'name': FALLBACK_TICKERS,
            'sector': ['Unknown'] * len(FALLBACK_TICKERS),
            'sub_industry': ['Unknown'] * len(FALLBACK_TICKERS)
        })


@st.cache_data(ttl=900)
def get_macro_data():
    """
    Fetch macro data: VIX, SPY, TNX with trend analysis.
    Returns dict with market regime information.
    """
    result = {
        'vix_current': 20.0,
        'vix_30d_avg': 20.0,
        'vix_regime': 'normal',
        'spy_price': 450.0,
        'spy_above_50ma': True,
        'spy_above_200ma': True,
        'spy_ret_1m': 0.0,
        'spy_ret_3m': 0.0,
        'tnx_yield': 4.5,
        'market_regime': 'Normal Volatility',
        'market_bias': 'neutral',
        'vix_history': None,
    }

    try:
        end = datetime.now()
        start = end - timedelta(days=200)

        # Download VIX
        vix_data = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=True)
        if not vix_data.empty:
            vix_close = vix_data['Close'].squeeze()
            result['vix_current'] = float(vix_close.iloc[-1])
            result['vix_30d_avg'] = float(vix_close.tail(30).mean())
            result['vix_history'] = vix_close.tail(90)

            vix_val = result['vix_current']
            if vix_val < 15:
                result['vix_regime'] = 'low'
            elif vix_val < 20:
                result['vix_regime'] = 'normal'
            elif vix_val < 30:
                result['vix_regime'] = 'high'
            else:
                result['vix_regime'] = 'extreme'

        # Download SPY
        spy_data = yf.download('SPY', start=start, end=end, progress=False, auto_adjust=True)
        if not spy_data.empty:
            spy_close = spy_data['Close'].squeeze()
            result['spy_price'] = float(spy_close.iloc[-1])

            ma50 = float(spy_close.tail(50).mean()) if len(spy_close) >= 50 else float(spy_close.mean())
            ma200 = float(spy_close.tail(200).mean()) if len(spy_close) >= 200 else float(spy_close.mean())
            result['spy_above_50ma'] = result['spy_price'] > ma50
            result['spy_above_200ma'] = result['spy_price'] > ma200

            if len(spy_close) >= 21:
                result['spy_ret_1m'] = float((spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100)
            if len(spy_close) >= 63:
                result['spy_ret_3m'] = float((spy_close.iloc[-1] / spy_close.iloc[-63] - 1) * 100)

        # Download TNX (10Y Treasury yield)
        tnx_data = yf.download('^TNX', start=start, end=end, progress=False, auto_adjust=True)
        if not tnx_data.empty:
            tnx_close = tnx_data['Close'].squeeze()
            result['tnx_yield'] = float(tnx_close.iloc[-1])

    except Exception as e:
        pass  # Return defaults

    # Determine market regime description and bias
    vix = result['vix_current']
    spy_bull = result['spy_above_50ma'] and result['spy_above_200ma']
    spy_ret = result['spy_ret_1m']

    if vix < 15:
        regime = 'Low Volatility'
        bias = 'bullish' if spy_bull else 'neutral'
    elif vix < 20:
        regime = 'Normal Volatility'
        bias = 'bullish' if spy_bull and spy_ret > 0 else ('bearish' if spy_ret < -3 else 'neutral')
    elif vix < 30:
        regime = 'Elevated Volatility'
        bias = 'bearish' if not spy_bull else 'neutral'
    else:
        regime = 'High Fear / Crisis'
        bias = 'bearish'

    result['market_regime'] = regime
    result['market_bias'] = bias
    return result


def compute_hv(prices: pd.Series, window: int = 30) -> float:
    """
    Compute annualized historical volatility over given window.
    Returns HV as a decimal (e.g., 0.25 = 25%).
    """
    if len(prices) < window + 1:
        return np.nan
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < window:
        return np.nan
    hv = log_returns.tail(window).std() * np.sqrt(252)
    return float(hv)


def _norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def compute_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Compute Black-Scholes Greeks.
    S: stock price, K: strike, T: years to expiry, r: risk-free rate, sigma: IV (decimal)
    Returns dict: delta, gamma, theta (daily $/contract), vega (per 1% IV/contract)
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan}
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)
        vega = S * _norm_pdf(d1) * sqrt_T * 100 / 100  # per 1% IV change, per share
        if option_type == 'call':
            delta = _norm_cdf(d1)
            theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
                     - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365
        else:
            delta = _norm_cdf(d1) - 1
            theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
                     + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365
        return {
            'delta': round(delta, 3),
            'gamma': round(gamma, 4),
            'theta': round(theta * 100, 3),   # per contract (100 shares), daily
            'vega': round(vega * 100, 3),      # per contract (100 shares), per 1% IV
        }
    except Exception:
        return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan}


def get_atm_iv(ticker_obj, target_dte: int = 30) -> float:
    """
    Get ATM implied volatility from options chain nearest to target DTE.
    Averages ATM call and put IV.
    Returns float (decimal) or np.nan.
    """
    try:
        expirations = ticker_obj.options
        if not expirations:
            return np.nan

        today = datetime.now().date()

        # Find expiry closest to target_dte
        best_exp = None
        best_diff = float('inf')
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                dte = (exp_date - today).days
                if dte < 7:
                    continue
                diff = abs(dte - target_dte)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str
            except Exception:
                continue

        if best_exp is None:
            return np.nan

        chain = ticker_obj.option_chain(best_exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            return np.nan

        # Get current price
        hist = ticker_obj.history(period='1d')
        if hist.empty:
            return np.nan
        current_price = float(hist['Close'].iloc[-1])

        # Find ATM strike (closest to current price)
        if 'strike' not in calls.columns:
            return np.nan

        strikes = calls['strike'].values
        atm_idx = np.argmin(np.abs(strikes - current_price))
        atm_strike = strikes[atm_idx]

        # Get IV for ATM call
        call_row = calls[calls['strike'] == atm_strike]
        put_row = puts[puts['strike'] == atm_strike]

        call_iv = np.nan
        put_iv = np.nan

        if not call_row.empty and 'impliedVolatility' in call_row.columns:
            call_iv_val = call_row['impliedVolatility'].values[0]
            if call_iv_val > 0.001:
                call_iv = float(call_iv_val)

        if not put_row.empty and 'impliedVolatility' in put_row.columns:
            put_iv_val = put_row['impliedVolatility'].values[0]
            if put_iv_val > 0.001:
                put_iv = float(put_iv_val)

        ivs = [v for v in [call_iv, put_iv] if not np.isnan(v)]
        if ivs:
            return float(np.mean(ivs))
        return np.nan

    except Exception:
        return np.nan


def get_earnings_info(ticker_obj):
    """
    Get next earnings date and days until earnings.
    Returns (next_earnings_date_str, days_to_earnings_int).
    """
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return (None, None)

        # calendar can be a DataFrame or dict depending on yfinance version
        if isinstance(cal, pd.DataFrame):
            if 'Earnings Date' in cal.index:
                earnings_val = cal.loc['Earnings Date'].values[0]
            elif 'Earnings Date' in cal.columns:
                earnings_val = cal['Earnings Date'].iloc[0]
            else:
                return (None, None)
        elif isinstance(cal, dict):
            earnings_val = cal.get('Earnings Date', [None])[0] if cal.get('Earnings Date') else None
            if earnings_val is None:
                return (None, None)
        else:
            return (None, None)

        if earnings_val is None:
            return (None, None)

        if hasattr(earnings_val, 'date'):
            earnings_date = earnings_val.date()
        else:
            earnings_date = pd.to_datetime(earnings_val).date()

        today = datetime.now().date()
        days = (earnings_date - today).days

        if days < 0:
            return (None, None)

        return (str(earnings_date), int(days))

    except Exception:
        return (None, None)


def _compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI for given price series."""
    if len(prices) < period + 1:
        return np.nan
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.tail(period).mean()
    avg_loss = loss.tail(period).mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


@st.cache_data(ttl=1800)
def batch_screen_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Batch download 1y price history for list of tickers.
    Compute HV30, HV252, momentum (1m, 3m), RSI(14).
    Fetch info dict for P/E and market cap.
    Returns DataFrame with columns: ticker, price, hv30, hv252, ret_1m, ret_3m, rsi, pe_ratio, market_cap
    """
    results = []

    if not tickers:
        return pd.DataFrame()

    # Batch download price data
    try:
        raw = yf.download(
            tickers,
            period='1y',
            auto_adjust=True,
            progress=False,
            group_by='ticker',
            threads=True
        )
    except Exception:
        raw = pd.DataFrame()

    for ticker in tickers:
        try:
            row = {'ticker': ticker}

            # Extract price series
            close_series = None
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    # MultiIndex: columns are (ticker, field) or (field, ticker)
                    level0 = raw.columns.get_level_values(0).unique().tolist()
                    if ticker in level0:
                        close_series = raw[ticker]['Close'].dropna()
                    elif 'Close' in level0:
                        close_series = raw['Close'][ticker].dropna()
                else:
                    # Flat columns (legacy single-ticker without group_by)
                    if 'Close' in raw.columns:
                        close_series = raw['Close'].dropna()

            if close_series is None or len(close_series) < 5:
                continue

            # Extract High, Low, Open arrays (needed for SMC signals)
            high_series = None
            low_series = None
            open_series = None
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    level0 = raw.columns.get_level_values(0).unique().tolist()
                    if ticker in level0:
                        high_series = raw[ticker]['High'].dropna()
                        low_series = raw[ticker]['Low'].dropna()
                        open_series = raw[ticker]['Open'].dropna()
                    elif 'High' in level0:
                        high_series = raw['High'][ticker].dropna()
                        low_series = raw['Low'][ticker].dropna()
                        open_series = raw['Open'][ticker].dropna()
                else:
                    # Flat columns (legacy single-ticker without group_by)
                    if 'High' in raw.columns:
                        high_series = raw['High'].dropna()
                    if 'Low' in raw.columns:
                        low_series = raw['Low'].dropna()
                    if 'Open' in raw.columns:
                        open_series = raw['Open'].dropna()

            row['high_arr'] = high_series.values if high_series is not None else np.array([])
            row['low_arr'] = low_series.values if low_series is not None else np.array([])
            row['open_arr'] = open_series.values if open_series is not None else np.array([])

            row['price'] = float(close_series.iloc[-1])
            row['hv30'] = compute_hv(close_series, window=30)
            row['hv252'] = compute_hv(close_series, window=min(252, len(close_series) - 1))

            # Returns
            if len(close_series) >= 21:
                row['ret_1m'] = float((close_series.iloc[-1] / close_series.iloc[-21] - 1) * 100)
            else:
                row['ret_1m'] = np.nan

            if len(close_series) >= 63:
                row['ret_3m'] = float((close_series.iloc[-1] / close_series.iloc[-63] - 1) * 100)
            else:
                row['ret_3m'] = np.nan

            row['rsi'] = _compute_rsi(close_series)

            # Moving averages
            row['ma20'] = float(close_series.rolling(20).mean().iloc[-1]) if len(close_series) >= 20 else np.nan
            row['ma50'] = float(close_series.rolling(50).mean().iloc[-1]) if len(close_series) >= 50 else np.nan
            row['ma200'] = float(close_series.rolling(200).mean().iloc[-1]) if len(close_series) >= 200 else np.nan

            # 52-week high/low
            wk = min(len(close_series), 252)
            row['wk52_high'] = float(close_series.tail(wk).max())
            row['wk52_low'] = float(close_series.tail(wk).min())

            # Trend classification based on price vs MAs
            p = row['price']
            above = [p > row['ma20'] if not np.isnan(row['ma20']) else None,
                     p > row['ma50'] if not np.isnan(row['ma50']) else None,
                     p > row['ma200'] if not np.isnan(row['ma200']) else None]
            bullish = sum(1 for x in above if x is True)
            valid = sum(1 for x in above if x is not None)
            if valid == 0:
                row['trend'] = 'Unknown'
            elif bullish == valid:
                row['trend'] = 'Strong Up' if not np.isnan(row.get('ret_1m', np.nan)) and row.get('ret_1m', 0) > 5 else 'Up'
            elif bullish == 0:
                row['trend'] = 'Strong Down' if not np.isnan(row.get('ret_1m', np.nan)) and row.get('ret_1m', 0) < -5 else 'Down'
            else:
                row['trend'] = 'Sideways'

            # Fundamentals
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}
                row['pe_ratio'] = info.get('trailingPE', np.nan)
                row['market_cap'] = info.get('marketCap', np.nan)
                row['avg_volume'] = info.get('averageVolume', np.nan)
                row['beta'] = info.get('beta', np.nan)
            except Exception:
                row['pe_ratio'] = np.nan
                row['market_cap'] = np.nan
                row['avg_volume'] = np.nan
                row['beta'] = np.nan

            results.append(row)

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.reset_index(drop=True)


def enrich_with_iv(df: pd.DataFrame, progress_callback=None, risk_free_rate: float = 0.045) -> pd.DataFrame:
    """
    For each row in df, fetch ATM IV, earnings info, and compute ATM Greeks.
    Adds columns: atm_iv, iv_hv_ratio, next_earnings, days_to_earnings,
                  atm_delta, atm_gamma, atm_theta, atm_vega.
    """
    atm_ivs = []
    iv_hv_ratios = []
    next_earnings_list = []
    days_to_earnings_list = []
    atm_deltas, atm_gammas, atm_thetas, atm_vegas = [], [], [], []
    atm_call_ois = []
    atm_put_ois = []
    atm_mid_prices = []
    atm_spread_pcts = []

    total = len(df)

    for i, row in df.iterrows():
        ticker_str = row['ticker']
        try:
            t = yf.Ticker(ticker_str)
            iv = get_atm_iv(t, target_dte=30)
            earnings_date, days = get_earnings_info(t)
        except Exception:
            iv = np.nan
            earnings_date = None
            days = None

        atm_ivs.append(iv)

        hv30 = row.get('hv30', np.nan)
        if not np.isnan(iv) and not np.isnan(hv30) and hv30 > 0:
            iv_hv_ratios.append(float(iv / hv30))
        else:
            iv_hv_ratios.append(np.nan)

        next_earnings_list.append(earnings_date)
        days_to_earnings_list.append(days)

        # Compute ATM Greeks for 30 DTE call (representative for screening)
        price = row.get('price', np.nan)
        if not np.isnan(iv) and not np.isnan(price) and price > 0:
            g = compute_greeks(price, price, 30 / 365, risk_free_rate, iv, 'call')
        else:
            g = {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan}
        atm_deltas.append(g['delta'])
        atm_gammas.append(g['gamma'])
        atm_thetas.append(g['theta'])
        atm_vegas.append(g['vega'])

        # Near-ATM Open Interest (liquidity signal)
        call_oi = np.nan
        put_oi = np.nan
        try:
            price_val = row.get('price', 0)
            exps = t.options
            if exps and price_val > 0:
                today = datetime.now().date()
                best_exp = None
                best_diff = float('inf')
                for exp_str in exps:
                    try:
                        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                        dte_days = (exp_date - today).days
                        if dte_days < 7:
                            continue
                        diff = abs(dte_days - 30)
                        if diff < best_diff:
                            best_diff = diff
                            best_exp = exp_str
                    except Exception:
                        continue
                if best_exp:
                    ch = t.option_chain(best_exp)
                    near_calls = ch.calls[abs(ch.calls['strike'] - price_val) / price_val < 0.05]
                    near_puts = ch.puts[abs(ch.puts['strike'] - price_val) / price_val < 0.05]
                    if not near_calls.empty and 'openInterest' in near_calls.columns:
                        call_oi = float(near_calls['openInterest'].mean())
                    if not near_puts.empty and 'openInterest' in near_puts.columns:
                        put_oi = float(near_puts['openInterest'].mean())
        except Exception:
            pass
        atm_call_ois.append(call_oi)
        atm_put_ois.append(put_oi)

        # Compute ATM mid price and spread %
        atm_mid = np.nan
        atm_spread = np.nan
        try:
            expirations = t.options
            if expirations:
                today = datetime.now().date()
                best_exp = None
                best_diff = float('inf')
                for exp_str in expirations:
                    try:
                        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                        dte_days = (exp_date - today).days
                        if dte_days < 7:
                            continue
                        diff = abs(dte_days - 30)
                        if diff < best_diff:
                            best_diff = diff
                            best_exp = exp_str
                    except Exception:
                        continue
                if best_exp:
                    chain = t.option_chain(best_exp)
                    calls = chain.calls
                    price_val = row.get('price', 0)
                    if not calls.empty and price_val > 0:
                        strikes = calls['strike'].values
                        atm_idx = int(np.argmin(np.abs(strikes - price_val)))
                        atm_row = calls.iloc[atm_idx]
                        try:
                            bid = float(atm_row['bid'])
                            ask = float(atm_row['ask'])
                        except Exception:
                            bid = np.nan
                            ask = np.nan
                        if not np.isnan(bid) and not np.isnan(ask) and bid > 0 and ask > 0:
                            mid = (bid + ask) / 2
                            atm_mid = mid
                            atm_spread = (ask - bid) / mid if mid > 0 else np.nan
        except Exception:
            pass

        atm_mid_prices.append(atm_mid)
        atm_spread_pcts.append(atm_spread)

        if progress_callback:
            progress_callback(i + 1, total)

    df = df.copy()
    df['atm_iv'] = atm_ivs
    df['iv_hv_ratio'] = iv_hv_ratios
    df['next_earnings'] = next_earnings_list
    df['days_to_earnings'] = days_to_earnings_list
    df['atm_delta'] = atm_deltas
    df['atm_gamma'] = atm_gammas
    df['atm_theta'] = atm_thetas
    df['atm_vega'] = atm_vegas
    df['atm_call_oi'] = atm_call_ois
    df['atm_put_oi'] = atm_put_ois
    df['atm_mid_price'] = atm_mid_prices
    df['atm_spread_pct'] = atm_spread_pcts

    return df


def score_strategies(df: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """
    Score each stock for Long Call and Long Put suitability.
    Returns df with added columns: lc_score, lp_score, best_strategy.
    """
    df = df.copy()
    lc_scores = []
    lp_scores = []
    best_strategies = []

    vix = macro.get('vix_current', 20.0)
    market_bias = macro.get('market_bias', 'neutral')

    for _, row in df.iterrows():
        iv_hv = row.get('iv_hv_ratio', np.nan)
        days_earn = row.get('days_to_earnings', None)
        rsi = row.get('rsi', np.nan)
        ret_1m = row.get('ret_1m', np.nan)
        trend = row.get('trend', 'Unknown')
        atm_delta = row.get('atm_delta', np.nan)
        atm_theta = row.get('atm_theta', np.nan)  # negative per day per contract
        call_oi = row.get('atm_call_oi', np.nan)
        put_oi = row.get('atm_put_oi', np.nan)

        # --- Shared base: IV cheapness (most important for option buyers) ---
        iv_bonus = 0.0
        if not np.isnan(iv_hv):
            if iv_hv < 0.6:
                iv_bonus = 35
            elif iv_hv < 0.8:
                iv_bonus = 25
            elif iv_hv < 1.0:
                iv_bonus = 10

        # Shared: VIX low = options cheap
        vix_bonus = 0.0
        if vix < 15:
            vix_bonus = 20
        elif vix < 20:
            vix_bonus = 10

        # Shared: earnings penalty (IV crush risk for option buyers)
        earn_bonus = 0.0
        if days_earn is None or days_earn > 30:
            earn_bonus = 10
        elif days_earn < 14:
            earn_bonus = -15

        # Shared: Delta bonus (want 0.3-0.5 range for good leverage without overpaying)
        delta_bonus = 0.0
        if not np.isnan(atm_delta):
            abs_delta = abs(atm_delta)
            if 0.35 <= abs_delta <= 0.55:
                delta_bonus = 10
            elif 0.25 <= abs_delta < 0.35:
                delta_bonus = 5

        # Shared: Theta penalty (lower theta decay = less time cost)
        theta_penalty = 0.0
        if not np.isnan(atm_theta):
            # atm_theta is negative (decay per day per contract in $)
            daily_decay = abs(atm_theta)
            if daily_decay > 50:
                theta_penalty = -15
            elif daily_decay > 25:
                theta_penalty = -8
            elif daily_decay < 10:
                theta_penalty = 5  # very low decay = good

        # --- Long Call score ---
        lc = iv_bonus + vix_bonus + earn_bonus + delta_bonus + theta_penalty

        # Trend
        if trend == 'Strong Up':
            lc += 20
        elif trend == 'Up':
            lc += 12
        elif trend == 'Sideways':
            lc += 2
        elif trend == 'Down':
            lc -= 5
        elif trend == 'Strong Down':
            lc -= 10

        # Momentum
        if not np.isnan(ret_1m):
            if ret_1m > 10:
                lc += 15
            elif ret_1m > 5:
                lc += 10
            elif ret_1m > 2:
                lc += 5

        # RSI
        if not np.isnan(rsi):
            if 45 <= rsi < 70:
                lc += 8
            elif rsi >= 70:
                lc -= 8  # overbought = bad entry for calls

        # Market bias
        if market_bias == 'bullish':
            lc += 8

        # Call OI (liquidity)
        if not np.isnan(call_oi):
            if call_oi > 5000:
                lc += 10
            elif call_oi > 1000:
                lc += 5

        # --- Long Put score ---
        lp = iv_bonus + vix_bonus + earn_bonus + delta_bonus + theta_penalty

        # Trend
        if trend == 'Strong Down':
            lp += 20
        elif trend == 'Down':
            lp += 12
        elif trend == 'Sideways':
            lp += 2
        elif trend == 'Up':
            lp -= 5
        elif trend == 'Strong Up':
            lp -= 10

        # Momentum (bearish)
        if not np.isnan(ret_1m):
            if ret_1m < -10:
                lp += 15
            elif ret_1m < -5:
                lp += 10
            elif ret_1m < -2:
                lp += 5

        # RSI overbought = put entry signal
        if not np.isnan(rsi):
            if rsi > 70:
                lp += 15
            elif rsi > 60:
                lp += 5
            elif rsi < 40:
                lp -= 5  # already oversold = bad put entry

        # Market bias
        if market_bias == 'bearish':
            lp += 8

        # Put OI (liquidity)
        if not np.isnan(put_oi):
            if put_oi > 5000:
                lp += 10
            elif put_oi > 1000:
                lp += 5

        lc = max(0.0, min(100.0, lc))
        lp = max(0.0, min(100.0, lp))

        lc_scores.append(round(lc, 1))
        lp_scores.append(round(lp, 1))
        best_strategies.append('Long Call' if lc >= lp else 'Long Put')

    df['lc_score'] = lc_scores
    df['lp_score'] = lp_scores
    df['best_strategy'] = best_strategies

    return df
