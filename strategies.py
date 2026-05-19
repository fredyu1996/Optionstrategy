"""
strategies.py - Options strategy recommendation engine for S&P 500 Options Screener
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from screener import compute_greeks

warnings.filterwarnings('ignore')

# Strategy definitions with metadata
STRATEGIES = {
    'Long Call': {
        'risk_level': 'High',
        'ideal_conditions': 'Low IV, strong bullish conviction, cheap options',
        'max_profit': 'Unlimited (stock price - strike - premium)',
        'max_loss': 'Premium paid (100%)',
        'timeframe': '60-90 DTE to allow time',
        'tags': ['premium_buying', 'bullish', 'leverage'],
        'description': 'Buy OTM call for leveraged upside exposure with defined risk',
    },
    'Long Put': {
        'risk_level': 'High',
        'ideal_conditions': 'Low IV, bearish conviction or portfolio hedge needed',
        'max_profit': 'Strike price - premium paid (if stock goes to zero)',
        'max_loss': 'Premium paid (100%)',
        'timeframe': '60-90 DTE to allow time',
        'tags': ['premium_buying', 'bearish', 'hedge'],
        'description': 'Buy OTM put for leveraged downside exposure or portfolio protection',
    },
}


def get_strategy_score(row: dict, macro: dict) -> dict:
    """
    Score Long Call and Long Put for a given stock row and macro context.
    Returns dict of {strategy_name: score (0-100)}.
    """
    scores = {}

    iv_hv = row.get('iv_hv_ratio', np.nan)
    days_earn = row.get('days_to_earnings', None)
    ret_1m = row.get('ret_1m', np.nan)
    rsi = row.get('rsi', np.nan)

    vix = macro.get('vix_current', 20.0)
    market_bias = macro.get('market_bias', 'neutral')
    vix_regime = macro.get('vix_regime', 'normal')

    iv_cheap = not np.isnan(iv_hv) and iv_hv < 0.8
    near_earnings = days_earn is not None and days_earn < 14
    bullish_momentum = not np.isnan(ret_1m) and ret_1m > 3
    bearish_momentum = not np.isnan(ret_1m) and ret_1m < -3
    strong_bull = not np.isnan(ret_1m) and ret_1m > 8
    strong_bear = not np.isnan(ret_1m) and ret_1m < -8
    overbought = not np.isnan(rsi) and rsi > 70

    # --- Long Call ---
    lc_score = 0
    if iv_cheap:
        lc_score += 35
    if strong_bull:
        lc_score += 25
    elif bullish_momentum:
        lc_score += 15
    if market_bias == 'bullish':
        lc_score += 15
    if vix_regime == 'low':
        lc_score += 10
    if not near_earnings:
        lc_score += 10
    if not overbought:
        lc_score += 5
    scores['Long Call'] = min(100, lc_score)

    # --- Long Put ---
    lp_score = 0
    if iv_cheap:
        lp_score += 25
    if strong_bear:
        lp_score += 30
    elif bearish_momentum:
        lp_score += 20
    if market_bias == 'bearish':
        lp_score += 15
    if vix_regime == 'low':
        lp_score += 10
    if overbought:
        lp_score += 10
    scores['Long Put'] = min(100, lp_score)

    return scores


def suggest_strategies(stock_data: dict, macro: dict = None) -> list:
    """
    Given a dict of stock metrics and macro context,
    return a list of strategy dicts sorted by score.
    Each strategy dict has:
    - strategy: name
    - risk_level: 'Low' | 'Medium' | 'High'
    - score: 0-100
    - rationale: explanation string
    - setup: specific action string
    - ideal_conditions: string
    - max_profit: string
    - max_loss: string
    - timeframe: string
    - tags: list of strings
    """
    if macro is None:
        macro = {
            'vix_current': 20.0,
            'vix_regime': 'normal',
            'market_bias': 'neutral',
        }

    scores = get_strategy_score(stock_data, macro)

    ticker = stock_data.get('ticker', 'UNKNOWN')
    price = stock_data.get('price', 0)
    iv_hv = stock_data.get('iv_hv_ratio', np.nan)
    atm_iv = stock_data.get('atm_iv', np.nan)
    hv30 = stock_data.get('hv30', np.nan)
    ret_1m = stock_data.get('ret_1m', np.nan)
    rsi = stock_data.get('rsi', np.nan)
    days_earn = stock_data.get('days_to_earnings', None)
    next_earn = stock_data.get('next_earnings', None)

    vix = macro.get('vix_current', 20.0)
    market_bias = macro.get('market_bias', 'neutral')

    result = []

    for strategy_name, score in sorted(scores.items(), key=lambda x: -x[1]):
        meta = STRATEGIES.get(strategy_name, {})
        rationale_parts = []

        # Build rationale dynamically
        if not np.isnan(iv_hv):
            iv_label = _iv_hv_label(iv_hv)
            rationale_parts.append(f"IV/HV ratio is {iv_hv:.2f} ({iv_label})")

        if not np.isnan(atm_iv):
            rationale_parts.append(f"ATM IV at {atm_iv*100:.1f}%")

        if days_earn is not None:
            rationale_parts.append(f"Earnings in {days_earn} days ({next_earn})")
        else:
            rationale_parts.append("No near-term earnings catalyst")

        if not np.isnan(ret_1m):
            trend = "bullish" if ret_1m > 2 else ("bearish" if ret_1m < -2 else "neutral")
            rationale_parts.append(f"1-month momentum: {ret_1m:+.1f}% ({trend})")

        if not np.isnan(rsi):
            rsi_label = "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral")
            rationale_parts.append(f"RSI at {rsi:.0f} ({rsi_label})")

        rationale_parts.append(f"Market: VIX {vix:.1f}, bias {market_bias}")

        rationale = ". ".join(rationale_parts)

        # Build setup string
        setup = _build_setup_string(strategy_name, ticker, price, atm_iv, iv_hv)

        strategy_dict = {
            'strategy': strategy_name,
            'risk_level': meta.get('risk_level', 'Medium'),
            'score': score,
            'rationale': rationale,
            'setup': setup,
            'ideal_conditions': meta.get('ideal_conditions', ''),
            'max_profit': meta.get('max_profit', 'N/A'),
            'max_loss': meta.get('max_loss', 'N/A'),
            'timeframe': meta.get('timeframe', '30-45 DTE'),
            'tags': meta.get('tags', []),
            'description': meta.get('description', ''),
        }
        result.append(strategy_dict)

    return result


def _iv_hv_label(iv_hv: float) -> str:
    """Return human-readable label for IV/HV ratio."""
    if iv_hv > 1.5:
        return 'Very Expensive'
    elif iv_hv > 1.2:
        return 'Expensive'
    elif iv_hv > 0.8:
        return 'Fair'
    else:
        return 'Cheap'


def _build_setup_string(strategy: str, ticker: str, price: float,
                        atm_iv: float, iv_hv: float) -> str:
    """Build a human-readable setup action string."""
    iv_str = f"{atm_iv*100:.0f}% IV" if not np.isnan(atm_iv) else "current IV"

    setups = {
        'Long Call': (
            f"Buy 1 OTM call at ~5% above ${price:.2f} with {iv_str}. "
            "Target 60-90 DTE to allow time for thesis to play out."
        ),
        'Long Put': (
            f"Buy 1 OTM put at ~5% below ${price:.2f} with {iv_str}. "
            "Target 60-90 DTE. Size position to 1-2% of portfolio."
        ),
    }
    return setups.get(strategy, f"Enter {strategy} position on {ticker} at ${price:.2f}.")


def get_specific_contracts(ticker_str: str, strategy: str, stock_data: dict) -> dict:
    """
    Fetch option chain and return specific contract recommendations.
    Returns dict with:
    - expiry: date string
    - dte: int
    - legs: list of {type, strike, bid, ask, mid, iv, delta, volume, oi}
    - estimated_premium: float
    - max_profit: float
    - max_loss: float
    - breakeven: list of floats
    """
    result = {
        'expiry': None,
        'dte': None,
        'legs': [],
        'estimated_premium': None,
        'max_profit': None,
        'max_loss': None,
        'breakeven': [],
        'error': None,
    }

    try:
        t = yf.Ticker(ticker_str)
        expirations = t.options

        if not expirations:
            result['error'] = 'No options available'
            return result

        today = datetime.now().date()
        current_price = stock_data.get('price', 0)

        target_dte = 75  # Long Call / Long Put: give time for thesis to play out

        # Find best expiry
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
            result['error'] = 'Could not find suitable expiry'
            return result

        exp_date = datetime.strptime(best_exp, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        result['expiry'] = best_exp
        result['dte'] = dte

        chain = t.option_chain(best_exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        if calls.empty or puts.empty:
            result['error'] = 'Empty option chain'
            return result

        # Sort and ensure numeric types
        calls = calls.sort_values('strike').reset_index(drop=True)
        puts = puts.sort_values('strike').reset_index(drop=True)

        # Find ATM index
        atm_call_idx = int(np.argmin(np.abs(calls['strike'].values - current_price)))
        atm_put_idx = int(np.argmin(np.abs(puts['strike'].values - current_price)))

        legs = []

        kw = dict(current_price=current_price, dte=dte)

        if strategy == 'Long Call':
            leg = _extract_leg(calls.iloc[min(atm_call_idx + 1, len(calls) - 1)], 'call', 'buy', **kw)
            legs.append(leg)
            premium = leg['mid']
            result['estimated_premium'] = round(premium * 100, 2)
            result['max_profit'] = 'Unlimited'
            result['max_loss'] = round(premium * 100, 2)
            result['breakeven'] = [round(leg['strike'] + premium, 2)]

        elif strategy == 'Long Put':
            leg = _extract_leg(puts.iloc[max(atm_put_idx - 1, 0)], 'put', 'buy', **kw)
            legs.append(leg)
            premium = leg['mid']
            result['estimated_premium'] = round(premium * 100, 2)
            result['max_profit'] = round((leg['strike'] - premium) * 100, 2)
            result['max_loss'] = round(premium * 100, 2)
            result['breakeven'] = [round(leg['strike'] - premium, 2)]

        else:
            result['error'] = f"Unknown strategy: {strategy}"
            return result

        result['legs'] = legs

    except Exception as e:
        result['error'] = str(e)

    return result


def _extract_leg(row: pd.Series, option_type: str, action: str,
                 current_price: float = None, dte: int = 45,
                 risk_free_rate: float = 0.045) -> dict:
    """Extract relevant fields from an option chain row and compute Greeks."""
    bid = float(row.get('bid', 0) or 0)
    ask = float(row.get('ask', 0) or 0)
    mid = round((bid + ask) / 2, 2) if (bid + ask) > 0 else 0.0
    iv = float(row.get('impliedVolatility', 0) or 0)
    strike = float(row.get('strike', 0))

    greeks = {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan}
    if current_price and iv > 0 and dte > 0:
        greeks = compute_greeks(current_price, strike, dte / 365, risk_free_rate, iv, option_type)

    return {
        'type': option_type,
        'action': action,
        'strike': strike,
        'bid': bid,
        'ask': ask,
        'mid': mid,
        'iv': iv,
        'delta': greeks['delta'],
        'gamma': greeks['gamma'],
        'theta': greeks['theta'],
        'vega': greeks['vega'],
        'volume': int(row.get('volume', 0) or 0),
        'oi': int(row.get('openInterest', 0) or 0),
        'last': float(row.get('lastPrice', 0) or 0),
    }


def get_option_chain_display(ticker_str: str, target_dte: int = 30) -> dict:
    """
    Fetch option chain for display purposes.
    Returns dict with: expiry, dte, calls DataFrame, puts DataFrame, current_price.
    """
    result = {
        'expiry': None,
        'dte': None,
        'calls': pd.DataFrame(),
        'puts': pd.DataFrame(),
        'current_price': None,
        'error': None,
    }

    try:
        t = yf.Ticker(ticker_str)
        expirations = t.options

        if not expirations:
            result['error'] = 'No options available'
            return result

        # Get current price
        hist = t.history(period='1d')
        if hist.empty:
            result['error'] = 'Could not fetch price'
            return result
        current_price = float(hist['Close'].iloc[-1])
        result['current_price'] = current_price

        today = datetime.now().date()
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
            result['error'] = 'No suitable expiry found'
            return result

        exp_date = datetime.strptime(best_exp, '%Y-%m-%d').date()
        result['expiry'] = best_exp
        result['dte'] = (exp_date - today).days

        chain = t.option_chain(best_exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        # Filter to reasonable strike range (±20% of current price)
        calls = calls[
            (calls['strike'] >= current_price * 0.80) &
            (calls['strike'] <= current_price * 1.20)
        ].reset_index(drop=True)
        puts = puts[
            (puts['strike'] >= current_price * 0.80) &
            (puts['strike'] <= current_price * 1.20)
        ].reset_index(drop=True)

        # Select display columns
        display_cols_call = [c for c in ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility'] if c in calls.columns]
        display_cols_put = [c for c in ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility'] if c in puts.columns]

        result['calls'] = calls[display_cols_call]
        result['puts'] = puts[display_cols_put]

    except Exception as e:
        result['error'] = str(e)

    return result
