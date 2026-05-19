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
    'Covered Call': {
        'risk_level': 'Low',
        'ideal_conditions': 'High IV, neutral-to-slightly bullish outlook, own or willing to own shares',
        'max_profit': 'Strike price - cost basis + premium received',
        'max_loss': 'Cost basis - premium received (stock goes to zero)',
        'timeframe': '30-45 DTE',
        'tags': ['premium_selling', 'income', 'stock_ownership'],
        'description': 'Sell OTM call against long stock position to collect premium',
    },
    'Cash-Secured Put': {
        'risk_level': 'Low',
        'ideal_conditions': 'High IV, bullish-to-neutral, willing to own stock at lower price',
        'max_profit': 'Premium received',
        'max_loss': 'Strike price - premium received (stock goes to zero)',
        'timeframe': '30-45 DTE',
        'tags': ['premium_selling', 'income', 'acquisition'],
        'description': 'Sell OTM put with full cash collateral to potentially acquire stock at discount',
    },
    'Bull Put Spread': {
        'risk_level': 'Low',
        'ideal_conditions': 'High IV, moderately bullish, defined risk needed',
        'max_profit': 'Net premium received',
        'max_loss': 'Width of spread - net premium received',
        'timeframe': '30-45 DTE',
        'tags': ['premium_selling', 'defined_risk', 'bullish'],
        'description': 'Sell OTM put and buy lower-strike put for defined risk credit spread',
    },
    'Iron Condor': {
        'risk_level': 'Low',
        'ideal_conditions': 'Very high IV, neutral range-bound expectation, low directional bias',
        'max_profit': 'Net premium received',
        'max_loss': 'Width of widest spread - net premium received',
        'timeframe': '30-45 DTE',
        'tags': ['premium_selling', 'neutral', 'range_bound', 'defined_risk'],
        'description': 'Combine bull put spread and bear call spread for neutral premium collection',
    },
    'Wheel Strategy': {
        'risk_level': 'Low',
        'ideal_conditions': 'High IV, long-term bullish, want income from stock holdings',
        'max_profit': 'Accumulated premium over cycles',
        'max_loss': 'Cost basis - total premium collected',
        'timeframe': '30-45 DTE per cycle',
        'tags': ['premium_selling', 'income', 'systematic', 'stock_ownership'],
        'description': 'Cycle: Sell CSP → assigned → Sell CC → called away → repeat',
    },
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
    'Bull Call Spread': {
        'risk_level': 'High',
        'ideal_conditions': 'Moderately bullish, IV not too high, defined risk/reward',
        'max_profit': 'Width of spread - net debit paid',
        'max_loss': 'Net debit paid',
        'timeframe': '30-60 DTE',
        'tags': ['premium_buying', 'bullish', 'defined_risk'],
        'description': 'Buy ATM call and sell OTM call to reduce cost of directional bet',
    },
    'Long Straddle': {
        'risk_level': 'High',
        'ideal_conditions': 'Low IV, expecting large move (earnings/event), direction uncertain',
        'max_profit': 'Unlimited (large move in either direction)',
        'max_loss': 'Total premium paid for both options',
        'timeframe': 'Expiry around catalyst event',
        'tags': ['premium_buying', 'volatility', 'earnings', 'event_driven'],
        'description': 'Buy ATM call and ATM put for earnings/event volatility play',
    },
}


def get_strategy_score(row: dict, macro: dict) -> dict:
    """
    Score each strategy type for a given stock row and macro context.
    Returns dict of {strategy_name: score (0-100)}.
    """
    scores = {}

    iv_hv = row.get('iv_hv_ratio', np.nan)
    days_earn = row.get('days_to_earnings', None)
    atm_iv = row.get('atm_iv', np.nan)
    hv30 = row.get('hv30', np.nan)
    ret_1m = row.get('ret_1m', np.nan)
    ret_3m = row.get('ret_3m', np.nan)
    rsi = row.get('rsi', np.nan)
    market_cap = row.get('market_cap', np.nan)
    avg_vol = row.get('avg_volume', np.nan)
    beta = row.get('beta', np.nan)

    vix = macro.get('vix_current', 20.0)
    market_bias = macro.get('market_bias', 'neutral')
    vix_regime = macro.get('vix_regime', 'normal')

    # Helper flags
    iv_expensive = not np.isnan(iv_hv) and iv_hv > 1.2
    iv_very_expensive = not np.isnan(iv_hv) and iv_hv > 1.5
    iv_cheap = not np.isnan(iv_hv) and iv_hv < 0.8
    iv_very_cheap = not np.isnan(iv_hv) and iv_hv < 0.6
    near_earnings = days_earn is not None and days_earn < 14
    earnings_imminent = days_earn is not None and days_earn < 7
    earnings_far = days_earn is None or days_earn > 30
    large_cap = not np.isnan(market_cap) and market_cap > 10e9
    liquid = not np.isnan(avg_vol) and avg_vol > 500000
    bullish_momentum = not np.isnan(ret_1m) and ret_1m > 3
    bearish_momentum = not np.isnan(ret_1m) and ret_1m < -3
    strong_bull = not np.isnan(ret_1m) and ret_1m > 8
    strong_bear = not np.isnan(ret_1m) and ret_1m < -8
    overbought = not np.isnan(rsi) and rsi > 70
    oversold = not np.isnan(rsi) and rsi < 30
    spy_bullish = market_bias == 'bullish'
    spy_bearish = market_bias == 'bearish'

    # --- Covered Call ---
    cc_score = 0
    if iv_expensive:
        cc_score += 35
    if iv_very_expensive:
        cc_score += 15
    if earnings_far:
        cc_score += 20
    if large_cap and liquid:
        cc_score += 15
    if vix > 20:
        cc_score += 10
    if overbought:
        cc_score += 5
    if spy_bullish:
        cc_score += 5
    scores['Covered Call'] = min(100, cc_score)

    # --- Cash-Secured Put ---
    csp_score = 0
    if iv_expensive:
        csp_score += 35
    if earnings_far:
        csp_score += 20
    if large_cap and liquid:
        csp_score += 15
    if spy_bullish or (not spy_bearish):
        csp_score += 10
    if vix > 20:
        csp_score += 10
    if oversold:
        csp_score += 10
    scores['Cash-Secured Put'] = min(100, csp_score)

    # --- Bull Put Spread ---
    bps_score = 0
    if iv_expensive:
        bps_score += 35
    if earnings_far:
        bps_score += 15
    if large_cap:
        bps_score += 10
    if spy_bullish:
        bps_score += 15
    if bullish_momentum:
        bps_score += 10
    if vix > 20:
        bps_score += 10
    if not near_earnings:
        bps_score += 5
    scores['Bull Put Spread'] = min(100, bps_score)

    # --- Iron Condor ---
    ic_score = 0
    if iv_very_expensive:
        ic_score += 45
    elif iv_expensive:
        ic_score += 25
    if earnings_far:
        ic_score += 20
    if large_cap and liquid:
        ic_score += 15
    if vix > 25:
        ic_score += 15
    # Penalize if strong directional momentum
    if strong_bull or strong_bear:
        ic_score -= 15
    if abs(ret_1m if not np.isnan(ret_1m) else 0) < 5:
        ic_score += 5  # Neutral momentum is good for IC
    scores['Iron Condor'] = min(100, max(0, ic_score))

    # --- Wheel Strategy ---
    wheel_score = 0
    if iv_expensive:
        wheel_score += 30
    if earnings_far:
        wheel_score += 20
    if large_cap and liquid:
        wheel_score += 20
    if spy_bullish:
        wheel_score += 10
    if vix > 20:
        wheel_score += 10
    if not np.isnan(beta) and 0.5 <= beta <= 1.5:
        wheel_score += 10  # Moderate beta ideal for wheel
    scores['Wheel Strategy'] = min(100, wheel_score)

    # --- Long Call ---
    lc_score = 0
    if iv_cheap:
        lc_score += 35
    if strong_bull:
        lc_score += 25
    elif bullish_momentum:
        lc_score += 15
    if spy_bullish:
        lc_score += 15
    if vix_regime == 'low':
        lc_score += 10
    if not near_earnings:
        lc_score += 10  # Avoid buying right before earnings collapse
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
    if spy_bearish:
        lp_score += 15
    if vix_regime == 'low':
        lp_score += 10
    if overbought:
        lp_score += 10
    scores['Long Put'] = min(100, lp_score)

    # --- Bull Call Spread ---
    bcs_score = 0
    if bullish_momentum:
        bcs_score += 25
    if strong_bull:
        bcs_score += 15
    if spy_bullish:
        bcs_score += 15
    if not iv_very_expensive:
        bcs_score += 15  # Spreads work better when IV not too high
    if not near_earnings:
        bcs_score += 10
    if large_cap:
        bcs_score += 10
    if not overbought:
        bcs_score += 10
    scores['Bull Call Spread'] = min(100, bcs_score)

    # --- Long Straddle ---
    ls_score = 0
    if earnings_imminent:
        ls_score += 50
    elif near_earnings:
        ls_score += 30
    if iv_cheap or iv_very_cheap:
        ls_score += 30
    if large_cap and liquid:
        ls_score += 10
    if not np.isnan(hv30) and hv30 > 0.40:
        ls_score += 10  # High mover stock
    scores['Long Straddle'] = min(100, ls_score)

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
        'Covered Call': (
            f"Own 100 shares of {ticker} at ~${price:.2f}. "
            f"Sell 1 OTM call (~5% above current price) with {iv_str}. "
            "Target 30-45 DTE expiry."
        ),
        'Cash-Secured Put': (
            f"Secure ${price*100:.0f} cash. "
            f"Sell 1 OTM put (~5% below ${price:.2f}) at {iv_str}. "
            "Target 30-45 DTE, collect premium."
        ),
        'Bull Put Spread': (
            f"Sell put at ~5% below ${price:.2f}, "
            f"buy put at ~10% below ${price:.2f}. "
            "Collect net credit. Target 30-45 DTE."
        ),
        'Iron Condor': (
            f"Sell OTM call (~5% above ${price:.2f}) + buy further OTM call. "
            f"Sell OTM put (~5% below ${price:.2f}) + buy further OTM put. "
            "Collect net credit from both spreads."
        ),
        'Wheel Strategy': (
            f"Step 1: Sell CSP at target acquisition price below ${price:.2f}. "
            "If assigned, Step 2: Sell CC at or above cost basis. "
            "Repeat cycle to accumulate premium."
        ),
        'Long Call': (
            f"Buy 1 OTM call at ~5% above ${price:.2f} with {iv_str}. "
            "Target 60-90 DTE to allow time for thesis to play out."
        ),
        'Long Put': (
            f"Buy 1 OTM put at ~5% below ${price:.2f} with {iv_str}. "
            "Target 60-90 DTE. Size position to 1-2% of portfolio."
        ),
        'Bull Call Spread': (
            f"Buy ATM call at ${price:.2f}, sell OTM call at ~5-10% above. "
            "Pay net debit for defined risk bullish exposure. Target 30-60 DTE."
        ),
        'Long Straddle': (
            f"Buy 1 ATM call AND 1 ATM put at ~${price:.2f} with {iv_str}. "
            "Expire around earnings/catalyst date. "
            "Profit from large move in either direction."
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

        # Target DTE based on strategy
        target_dte = 45
        if strategy in ['Long Call', 'Long Put']:
            target_dte = 75
        elif strategy == 'Long Straddle':
            days_earn = stock_data.get('days_to_earnings')
            if days_earn and 5 <= days_earn <= 60:
                target_dte = days_earn + 2

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

        if strategy == 'Covered Call':
            otm_idx = min(atm_call_idx + 1, len(calls) - 1)
            leg = _extract_leg(calls.iloc[otm_idx], 'call', 'sell', **kw)
            legs.append(leg)
            premium = leg['mid']
            result['estimated_premium'] = round(premium * 100, 2)
            result['max_profit'] = round(premium * 100, 2)
            result['max_loss'] = round((current_price - premium) * 100, 2)
            result['breakeven'] = [round(current_price - premium, 2)]

        elif strategy == 'Cash-Secured Put':
            otm_idx = max(atm_put_idx - 1, 0)
            leg = _extract_leg(puts.iloc[otm_idx], 'put', 'sell', **kw)
            legs.append(leg)
            premium = leg['mid']
            strike = leg['strike']
            result['estimated_premium'] = round(premium * 100, 2)
            result['max_profit'] = round(premium * 100, 2)
            result['max_loss'] = round((strike - premium) * 100, 2)
            result['breakeven'] = [round(strike - premium, 2)]

        elif strategy == 'Bull Put Spread':
            sell_idx = max(atm_put_idx - 1, 0)
            buy_idx = max(atm_put_idx - 3, 0)
            sell_leg = _extract_leg(puts.iloc[sell_idx], 'put', 'sell', **kw)
            buy_leg = _extract_leg(puts.iloc[buy_idx], 'put', 'buy', **kw)
            legs = [sell_leg, buy_leg]
            net_credit = sell_leg['mid'] - buy_leg['mid']
            spread_width = sell_leg['strike'] - buy_leg['strike']
            result['estimated_premium'] = round(net_credit * 100, 2)
            result['max_profit'] = round(net_credit * 100, 2)
            result['max_loss'] = round((spread_width - net_credit) * 100, 2)
            result['breakeven'] = [round(sell_leg['strike'] - net_credit, 2)]

        elif strategy == 'Iron Condor':
            put_sell_idx = max(atm_put_idx - 1, 0)
            put_buy_idx = max(atm_put_idx - 3, 0)
            call_sell_idx = min(atm_call_idx + 1, len(calls) - 1)
            call_buy_idx = min(atm_call_idx + 3, len(calls) - 1)

            ps = _extract_leg(puts.iloc[put_sell_idx], 'put', 'sell', **kw)
            pb = _extract_leg(puts.iloc[put_buy_idx], 'put', 'buy', **kw)
            cs = _extract_leg(calls.iloc[call_sell_idx], 'call', 'sell', **kw)
            cb = _extract_leg(calls.iloc[call_buy_idx], 'call', 'buy', **kw)
            legs = [ps, pb, cs, cb]

            net_credit = (ps['mid'] - pb['mid']) + (cs['mid'] - cb['mid'])
            put_width = ps['strike'] - pb['strike']
            call_width = cb['strike'] - cs['strike']
            max_loss_width = max(put_width, call_width)

            result['estimated_premium'] = round(net_credit * 100, 2)
            result['max_profit'] = round(net_credit * 100, 2)
            result['max_loss'] = round((max_loss_width - net_credit) * 100, 2)
            result['breakeven'] = [
                round(ps['strike'] - net_credit, 2),
                round(cs['strike'] + net_credit, 2),
            ]

        elif strategy == 'Long Call':
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

        elif strategy == 'Bull Call Spread':
            buy_leg = _extract_leg(calls.iloc[atm_call_idx], 'call', 'buy', **kw)
            sell_leg = _extract_leg(calls.iloc[min(atm_call_idx + 2, len(calls) - 1)], 'call', 'sell', **kw)
            legs = [buy_leg, sell_leg]
            net_debit = buy_leg['mid'] - sell_leg['mid']
            spread_width = sell_leg['strike'] - buy_leg['strike']
            result['estimated_premium'] = round(-net_debit * 100, 2)
            result['max_profit'] = round((spread_width - net_debit) * 100, 2)
            result['max_loss'] = round(net_debit * 100, 2)
            result['breakeven'] = [round(buy_leg['strike'] + net_debit, 2)]

        elif strategy == 'Long Straddle':
            call_leg = _extract_leg(calls.iloc[atm_call_idx], 'call', 'buy', **kw)
            put_leg = _extract_leg(puts.iloc[atm_put_idx], 'put', 'buy', **kw)
            legs = [call_leg, put_leg]
            total_premium = call_leg['mid'] + put_leg['mid']
            atm_strike = calls.iloc[atm_call_idx]['strike']
            result['estimated_premium'] = round(total_premium * 100, 2)
            result['max_profit'] = 'Unlimited'
            result['max_loss'] = round(total_premium * 100, 2)
            result['breakeven'] = [
                round(atm_strike - total_premium, 2),
                round(atm_strike + total_premium, 2),
            ]

        else:
            # Wheel Strategy - same as CSP for first leg
            otm_idx = max(atm_put_idx - 1, 0)
            leg = _extract_leg(puts.iloc[otm_idx], 'put', 'sell', **kw)
            legs.append(leg)
            result['estimated_premium'] = round(leg['mid'] * 100, 2)
            result['max_profit'] = round(leg['mid'] * 100, 2)
            result['max_loss'] = round((leg['strike'] - leg['mid']) * 100, 2)
            result['breakeven'] = [round(leg['strike'] - leg['mid'], 2)]

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
        hist = t.history(period='1d', progress=False)
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
