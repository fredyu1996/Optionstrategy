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

from indicators import compute_ema_signals

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


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Compute Average True Range over last `period` candles."""
    n = len(closes)
    if n < 2:
        return float(np.mean(highs - lows)) if len(highs) > 0 else 1.0
    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) >= period:
        return float(np.mean(trs[-period:]))
    return float(np.mean(trs)) if trs else 1.0


def _empty_smc() -> dict:
    """Return all-False SMC signal dict (used when data is insufficient)."""
    return {k: False for k in [
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
        'discount_zone', 'premium_zone', 'near_bullish_ob', 'near_bearish_ob',
        'in_bullish_fvg', 'in_bearish_fvg',
    ]}


def _empty_recommendation() -> dict:
    return {
        'strike': None, 'expiry': None, 'dte': None,
        'delta': np.nan, 'gamma': np.nan, 'theta': np.nan,
        'cost': np.nan, 'affordable': False, 'breakeven': np.nan,
        'iv_crush_warning': False, 'delta_target_center': 0.40,
        'smc_active_count': 0, 'reason': 'No options data available.',
        'chain_df': pd.DataFrame(), 'flag': 'no_data',
    }


def _compute_delta_range(iv_hv: float, smc_count: int, is_call: bool) -> tuple:
    """Return (lo, hi, center) for the desired delta range."""
    if np.isnan(iv_hv) or iv_hv <= 0:
        lo, hi = 0.35, 0.45
    elif iv_hv < 0.8:
        lo, hi = 0.45, 0.55
    elif iv_hv <= 1.0:
        lo, hi = 0.35, 0.45
    else:
        lo, hi = 0.25, 0.35

    if smc_count >= 3:
        lo -= 0.05
        hi -= 0.05
    elif smc_count <= 1:
        lo += 0.05
        hi += 0.05

    lo = max(0.05, lo)
    hi = min(0.95, hi)
    center = (lo + hi) / 2

    if not is_call:
        lo, hi, center = -hi, -lo, -center
    return lo, hi, center


def _select_best_strike(
    strike_data: list,
    delta_lo: float,
    delta_hi: float,
    delta_center: float,
    max_risk_usd: float,
    is_call: bool,
    stock_price: float,
) -> tuple:
    """
    From list of strike dicts (keys: strike, delta, cost, affordable, gamma, theta),
    return (chosen_dict, flag). flag: None | 'outside_ideal_range' | 'over_budget'.
    """
    if not strike_data:
        return {}, 'over_budget'

    in_range = [s for s in strike_data if delta_lo <= s['delta'] <= delta_hi and s['affordable']]
    if in_range:
        return min(in_range, key=lambda x: abs(x['delta'] - delta_center)), None

    affordable = [s for s in strike_data if s['affordable']]
    if affordable:
        return min(affordable, key=lambda x: abs(x['delta'] - delta_center)), 'outside_ideal_range'

    return min(strike_data, key=lambda x: x['cost']), 'over_budget'


@st.cache_data(ttl=900)
def get_strike_recommendation(
    ticker_str: str,
    strategy: str,
    iv_hv: float,
    smc_tuple: tuple,
    max_risk_usd: float,
) -> dict:
    """
    Recommend a specific strike for ticker based on IV/HV, SMC signals, and budget.

    smc_tuple order: (bos_bull, bos_bear, choch_bull, choch_bear, disc, prem,
                      ob_bull, ob_bear, fvg_bull, fvg_bear)
    """
    smc_keys = [
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
        'discount_zone', 'premium_zone', 'near_bullish_ob', 'near_bearish_ob',
        'in_bullish_fvg', 'in_bearish_fvg',
    ]
    smc = dict(zip(smc_keys, smc_tuple))
    is_call = (strategy == 'Long Call')

    relevant_keys = (
        ['bos_bullish', 'discount_zone', 'near_bullish_ob', 'in_bullish_fvg', 'choch_bullish']
        if is_call else
        ['bos_bearish', 'premium_zone', 'near_bearish_ob', 'in_bearish_fvg', 'choch_bearish']
    )
    smc_count = sum(1 for k in relevant_keys if smc.get(k, False))

    delta_lo, delta_hi, delta_center = _compute_delta_range(iv_hv, smc_count, is_call)

    try:
        t = yf.Ticker(ticker_str)
        expiry = _find_best_expiry(t, target_dte=30)
        if expiry is None:
            return _empty_recommendation()

        today = datetime.now().date()
        exp_date = datetime.strptime(expiry, '%Y-%m-%d').date()
        dte = max((exp_date - today).days, 1)
        T = dte / 365

        hist = t.history(period='1d')
        S = float(hist['Close'].iloc[-1]) if not hist.empty else 0.0
        if S <= 0:
            return _empty_recommendation()

        chain = t.option_chain(expiry)
        options = chain.calls if is_call else chain.puts
        if options.empty:
            return _empty_recommendation()
    except Exception:
        return _empty_recommendation()

    strike_data = []
    for _, opt_row in options.iterrows():
        try:
            K = float(opt_row['strike'])
            iv_val = float(opt_row.get('impliedVolatility') or 0)
            bid = float(opt_row.get('bid') or 0)
            ask = float(opt_row.get('ask') or 0)
            if iv_val <= 0 or bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            cost = mid * 100
            g = compute_greeks(S, K, T, 0.045, iv_val, 'call' if is_call else 'put')
            if np.isnan(g['delta']):
                continue
            strike_data.append({
                'strike': K,
                'delta': g['delta'],
                'gamma': g['gamma'],
                'theta': g['theta'],
                'cost': cost,
                'affordable': cost <= max_risk_usd,
            })
        except Exception:
            continue

    if not strike_data:
        return _empty_recommendation()

    strike_data.sort(key=lambda x: x['strike'])
    chosen, flag = _select_best_strike(
        strike_data, delta_lo, delta_hi, delta_center, max_risk_usd, is_call, S
    )

    # Handle edge case where _select_best_strike returns empty dict
    if not chosen:
        return _empty_recommendation()

    # Build human-readable reason
    if np.isnan(iv_hv) or iv_hv <= 0:
        iv_desc = "fair vol → Δ 0.35–0.45 target"
    elif iv_hv < 0.8:
        iv_desc = "cheap vol → ATM target"
    elif iv_hv <= 1.0:
        iv_desc = "fair vol → Δ 0.35–0.45 target"
    else:
        iv_desc = "expensive vol → Δ 0.25–0.35 target"

    if smc_count >= 3:
        smc_adj = f"{smc_count} SMC signals → shifted OTM"
    elif smc_count <= 1:
        smc_adj = f"{smc_count} SMC signals → shifted toward ATM"
    else:
        smc_adj = f"{smc_count} SMC signals → no adjustment"

    reason = f"IV/HV {iv_hv:.2f} ({iv_desc}). {smc_adj}. Best affordable: Δ {chosen['delta']:.2f}."

    cost_per_share = chosen['cost'] / 100
    breakeven = (chosen['strike'] + cost_per_share if is_call
                 else chosen['strike'] - cost_per_share)

    # 5-strike chain: 2 below + chosen + 2 above
    strikes_list = [s['strike'] for s in strike_data]
    try:
        idx = strikes_list.index(chosen['strike'])
    except ValueError:
        idx = len(strike_data) // 2
    subset = strike_data[max(0, idx - 2): min(len(strike_data), idx + 3)]

    chain_df = pd.DataFrame([{
        'Strike': f"${s['strike']:.0f}" + (" ★" if s['strike'] == chosen['strike'] else ""),
        'Delta': round(s['delta'], 3),
        'Cost': f"${s['cost']:.0f}",
        'Θ/day': f"${s['theta']:.2f}",
        'Affordable': "✓" if s['affordable'] else "✗",
    } for s in subset])

    return {
        'strike': chosen['strike'],
        'expiry': expiry,
        'dte': dte,
        'delta': chosen['delta'],
        'gamma': chosen['gamma'],
        'theta': chosen['theta'],
        'cost': chosen['cost'],
        'affordable': chosen['affordable'],
        'breakeven': breakeven,
        'iv_crush_warning': (not (np.isnan(iv_hv) or iv_hv <= 0) and iv_hv > 1.0),
        'delta_target_center': delta_center,
        'smc_active_count': smc_count,
        'reason': reason,
        'chain_df': chain_df,
        'flag': flag,
    }


def compute_smc_signals(ohlcv_df: pd.DataFrame) -> dict:
    """
    Compute SMC signals from OHLCV DataFrame.
    Requires columns: Open, High, Low, Close. Min 20 rows.
    Returns dict of bool signals.
    """
    if len(ohlcv_df) < 20:
        return _empty_smc()

    highs = ohlcv_df['High'].values.astype(float)
    lows = ohlcv_df['Low'].values.astype(float)
    closes = ohlcv_df['Close'].values.astype(float)
    opens = ohlcv_df['Open'].values.astype(float)
    n = len(closes)
    current_price = closes[-1]

    # Swing highs/lows (5-candle lookback each side)
    swing_highs = []
    swing_lows = []
    for i in range(5, n - 5):
        window_high = highs[i - 5:i + 6]
        if np.isclose(highs[i], np.max(window_high)):
            swing_highs.append((i, highs[i]))
        window_low = lows[i - 5:i + 6]
        if np.isclose(lows[i], np.min(window_low)):
            swing_lows.append((i, lows[i]))

    # Break of Structure
    # If no swing highs/lows detected (e.g. strong monotone trend), fall back to
    # comparing current price against the prior-period high/low (excluding last 5 bars).
    if swing_highs:
        bos_bullish = bool(current_price > swing_highs[-1][1])
    else:
        prior_high = float(np.max(highs[:max(1, n - 5)])) if n > 5 else float(np.max(highs))
        bos_bullish = bool(current_price > prior_high)

    if swing_lows:
        bos_bearish = bool(current_price < swing_lows[-1][1])
    else:
        prior_low = float(np.min(lows[:max(1, n - 5)])) if n > 5 else float(np.min(lows))
        bos_bearish = bool(current_price < prior_low)

    # Change of Character
    choch_bullish = False
    choch_bearish = False
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        recent_highs = [x[1] for x in swing_highs[-3:]]
        recent_lows = [x[1] for x in swing_lows[-3:]]
        was_uptrend = len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[-2]
        was_downtrend = len(recent_highs) >= 2 and recent_highs[-1] < recent_highs[-2]
        if was_uptrend and len(recent_lows) >= 2 and recent_lows[-1] < recent_lows[-2]:
            choch_bearish = True
        if was_downtrend and len(recent_lows) >= 2 and recent_lows[-1] > recent_lows[-2]:
            choch_bullish = True

    # Discount / Premium Zone
    discount_zone = False
    premium_zone = False
    if swing_highs and swing_lows:
        range_high = swing_highs[-1][1]
        range_low = swing_lows[-1][1]
        if range_high > range_low:
            equilibrium = (range_high + range_low) / 2
            discount_zone = bool(current_price < equilibrium)
            premium_zone = bool(current_price > equilibrium)

    # Order Blocks
    atr = _compute_atr(highs, lows, closes, period=14)
    near_bullish_ob = False
    near_bearish_ob = False
    lookback_start = max(0, n - 50)

    for i in range(lookback_start, n - 3):
        is_bearish_candle = closes[i] < opens[i]
        is_bullish_candle = closes[i] > opens[i]
        ob_high = max(opens[i], closes[i])
        ob_low = min(opens[i], closes[i])

        if is_bearish_candle and not near_bullish_ob:
            up_count = sum(1 for j in range(i + 1, min(i + 4, n)) if closes[j] > closes[j - 1])
            large_up = (i + 1 < n and closes[i + 1] - opens[i + 1] > 1.5 * atr
                        and closes[i + 1] > opens[i + 1])
            if up_count >= 3 or large_up:
                if ob_low * 0.99 <= current_price <= ob_high * 1.01:
                    near_bullish_ob = True

        if is_bullish_candle and not near_bearish_ob:
            down_count = sum(1 for j in range(i + 1, min(i + 4, n)) if closes[j] < closes[j - 1])
            large_down = (i + 1 < n and opens[i + 1] - closes[i + 1] > 1.5 * atr
                          and closes[i + 1] < opens[i + 1])
            if down_count >= 3 or large_down:
                if ob_low * 0.99 <= current_price <= ob_high * 1.01:
                    near_bearish_ob = True

    # Fair Value Gaps
    in_bullish_fvg = False
    in_bearish_fvg = False
    fvg_start = max(0, n - 20)

    for i in range(fvg_start, n - 2):
        if lows[i + 2] > highs[i]:
            fvg_low, fvg_high = highs[i], lows[i + 2]
            if fvg_low <= current_price <= fvg_high:
                in_bullish_fvg = True
        if highs[i + 2] < lows[i]:
            fvg_low, fvg_high = highs[i + 2], lows[i]
            if fvg_low <= current_price <= fvg_high:
                in_bearish_fvg = True

    return {
        'bos_bullish': bos_bullish,
        'bos_bearish': bos_bearish,
        'choch_bullish': choch_bullish,
        'choch_bearish': choch_bearish,
        'discount_zone': discount_zone,
        'premium_zone': premium_zone,
        'near_bullish_ob': near_bullish_ob,
        'near_bearish_ob': near_bearish_ob,
        'in_bullish_fvg': in_bullish_fvg,
        'in_bearish_fvg': in_bearish_fvg,
    }


def get_atm_iv(ticker_obj, target_dte: int = 30) -> float:
    """
    Get ATM implied volatility from options chain nearest to target DTE.
    Averages ATM call and put IV.
    Returns float (decimal) or np.nan.
    """
    try:
        best_exp = _find_best_expiry(ticker_obj, target_dte)
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


def _find_best_expiry(ticker_obj, target_dte: int = 30) -> str | None:
    """Return the option expiry string closest to target_dte. Returns None if none found."""
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None
        today = datetime.now().date()
        best_exp = None
        best_diff = float('inf')
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                dte_days = (exp_date - today).days
                if dte_days < 7:
                    continue
                diff = abs(dte_days - target_dte)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str
            except Exception:
                continue
        return best_exp
    except Exception:
        return None


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

            # Extract OHLCV series for this ticker
            ohlcv_series = {}
            for field in ['Close', 'High', 'Low', 'Open']:
                series = None
                if not raw.empty:
                    try:
                        if isinstance(raw.columns, pd.MultiIndex):
                            level0 = raw.columns.get_level_values(0)
                            if ticker in level0:
                                # (ticker, field) orientation
                                series = raw[ticker][field].dropna()
                            elif field in level0:
                                # (field, ticker) orientation
                                series = raw[field][ticker].dropna()
                        else:
                            if field in raw.columns:
                                series = raw[field].dropna()
                    except Exception:
                        pass
                ohlcv_series[field] = series

            close_series = ohlcv_series['Close']
            if close_series is None or len(close_series) < 5:
                continue

            row['high_arr'] = ohlcv_series['High'].values if ohlcv_series['High'] is not None else np.array([])
            row['low_arr'] = ohlcv_series['Low'].values if ohlcv_series['Low'] is not None else np.array([])
            row['open_arr'] = ohlcv_series['Open'].values if ohlcv_series['Open'] is not None else np.array([])

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

            # EMA signals (daily)
            _ema = compute_ema_signals(close_series)
            row['ema_bull_stack'] = _ema['ema_bull_stack']
            row['ema_bear_stack'] = _ema['ema_bear_stack']
            row['above_ema20'] = _ema['above_ema20']
            row['above_ema50'] = _ema['above_ema50']

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

            # Compute SMC signals
            if (len(row.get('high_arr', [])) >= 20 and
                    len(row.get('low_arr', [])) >= 20 and
                    len(row.get('open_arr', [])) >= 20):
                ohlcv_for_smc = pd.DataFrame({
                    'Open': row['open_arr'],
                    'High': row['high_arr'],
                    'Low': row['low_arr'],
                    'Close': close_series.values,
                })
                smc = compute_smc_signals(ohlcv_for_smc)
            else:
                smc = _empty_smc()

            row.update({f'smc_{k}': v for k, v in smc.items()})

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
            if price_val > 0:
                best_exp = _find_best_expiry(t)
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
            best_exp = _find_best_expiry(t)
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
    Score each stock for Long Call / Long Put suitability.
    Returns df with columns: lc_score, lp_score, best_strategy.
    """
    df = df.copy()
    lc_scores, lp_scores, best_strategies = [], [], []

    market_bias = macro.get('market_bias', 'neutral')

    for _, row in df.iterrows():
        iv_hv      = row.get('iv_hv_ratio', np.nan)
        days_earn  = row.get('days_to_earnings', None)
        rsi        = row.get('rsi', np.nan)
        ret_1m     = row.get('ret_1m', np.nan)
        trend      = row.get('trend', 'Unknown')
        atm_theta  = row.get('atm_theta', np.nan)
        call_oi    = row.get('atm_call_oi', np.nan)
        put_oi     = row.get('atm_put_oi', np.nan)
        spread_pct = row.get('atm_spread_pct', np.nan)

        # ── Shared base ───────────────────────────────────────────────────
        iv_pts = 0.0
        if not (isinstance(iv_hv, float) and np.isnan(iv_hv)) and iv_hv is not None:
            if iv_hv < 0.6:
                iv_pts = 25
            elif iv_hv < 0.8:
                iv_pts = 15
            elif iv_hv < 1.0:
                iv_pts = 8

        earn_pts = 0.0
        if days_earn is None or days_earn > 30:
            earn_pts = 10
        elif days_earn < 14:
            earn_pts = -15

        theta_pts = 0.0
        if not (isinstance(atm_theta, float) and np.isnan(atm_theta)) and atm_theta is not None:
            daily_decay = abs(atm_theta)
            if daily_decay < 10:
                theta_pts = 5
            elif daily_decay > 50:
                theta_pts = -15
            elif daily_decay > 25:
                theta_pts = -8

        spread_pts = 0.0
        if not (isinstance(spread_pct, float) and np.isnan(spread_pct)) and spread_pct is not None:
            if spread_pct > 0.20:
                spread_pts = -10
            elif spread_pct > 0.10:
                spread_pts = -5

        shared = iv_pts + earn_pts + theta_pts + spread_pts

        # ── Long Call ─────────────────────────────────────────────────────
        lc = shared

        if trend in ('Up', 'Strong Up'):
            if not (isinstance(rsi, float) and np.isnan(rsi)) and rsi is not None:
                if rsi < 45:
                    lc += 15
                elif rsi <= 60:
                    lc += 20
                elif rsi <= 70:
                    lc += 10
                else:
                    lc -= 5
            else:
                lc += 10
        elif trend == 'Sideways':
            lc += 2
        elif trend == 'Down':
            lc -= 5
        elif trend == 'Strong Down':
            lc -= 10

        if not (isinstance(ret_1m, float) and np.isnan(ret_1m)) and ret_1m is not None:
            if 2 <= ret_1m <= 8:
                lc += 8
            elif ret_1m > 10:
                lc -= 5

        if market_bias == 'bullish':
            lc += 5
        if not (isinstance(call_oi, float) and np.isnan(call_oi)) and call_oi is not None:
            if call_oi > 5000:
                lc += 10
            elif call_oi > 1000:
                lc += 5

        if row.get('smc_bos_bullish'):     lc += 15
        if row.get('smc_bos_bearish'):     lc -= 10
        if row.get('smc_choch_bullish'):   lc += 5
        if row.get('smc_choch_bearish'):   lc -= 8
        if row.get('smc_discount_zone'):   lc += 10
        if row.get('smc_premium_zone'):    lc -= 5
        if row.get('smc_near_bullish_ob'): lc += 15
        if row.get('smc_near_bearish_ob'): lc -= 10
        if row.get('smc_in_bullish_fvg'):  lc += 10
        if row.get('smc_in_bearish_fvg'):  lc -= 5

        # ── Long Put ──────────────────────────────────────────────────────
        lp = shared

        if trend in ('Down', 'Strong Down'):
            if not (isinstance(rsi, float) and np.isnan(rsi)) and rsi is not None:
                if rsi > 55:
                    lp += 15
                elif rsi >= 45:
                    lp += 20
                elif rsi >= 30:
                    lp += 10
                else:
                    lp -= 5
            else:
                lp += 10
        elif trend == 'Sideways':
            lp += 2
        elif trend == 'Up':
            lp -= 5
        elif trend == 'Strong Up':
            lp -= 10

        if not (isinstance(ret_1m, float) and np.isnan(ret_1m)) and ret_1m is not None:
            if -8 <= ret_1m <= -2:
                lp += 8
            elif ret_1m < -10:
                lp -= 5

        if market_bias == 'bearish':
            lp += 5
        if not (isinstance(put_oi, float) and np.isnan(put_oi)) and put_oi is not None:
            if put_oi > 5000:
                lp += 10
            elif put_oi > 1000:
                lp += 5

        if row.get('smc_bos_bearish'):     lp += 15
        if row.get('smc_bos_bullish'):     lp -= 10
        if row.get('smc_choch_bearish'):   lp += 5
        if row.get('smc_choch_bullish'):   lp -= 8
        if row.get('smc_premium_zone'):    lp += 10
        if row.get('smc_discount_zone'):   lp -= 5
        if row.get('smc_near_bearish_ob'): lp += 15
        if row.get('smc_near_bullish_ob'): lp -= 10
        if row.get('smc_in_bearish_fvg'):  lp += 10
        if row.get('smc_in_bullish_fvg'):  lp -= 5

        lc = max(0.0, min(100.0, lc))
        lp = max(0.0, min(100.0, lp))

        lc_scores.append(round(lc, 1))
        lp_scores.append(round(lp, 1))
        best_strategies.append('Long Call' if lc >= lp else 'Long Put')

    df['lc_score'] = lc_scores
    df['lp_score'] = lp_scores
    df['best_strategy'] = best_strategies
    return df
