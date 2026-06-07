# signals.py
"""
signals.py - Entry readiness and exit rules for Long Call / Long Put positions.
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def compute_entry_readiness(row: dict, strategy: str) -> dict:
    """
    Evaluate 7 entry conditions for Long Call or Long Put.

    Returns:
        status: 'enter' | 'wait' | 'not_yet'
        met: int
        total: int  (number of checks)
        checks: list of {label, passed, value}
    """
    if strategy not in ('Long Call', 'Long Put'):
        raise ValueError(f"Unknown strategy: {strategy!r}")
    is_call = (strategy == 'Long Call')
    rsi = row.get('rsi', np.nan)
    iv_hv = row.get('iv_hv_ratio', np.nan)
    trend = row.get('trend', 'Unknown')
    days_earn = row.get('days_to_earnings', None)
    if days_earn is not None and pd.isna(days_earn):
        days_earn = None

    if is_call:
        checks = [
            {
                'label': 'Trend',
                'passed': trend in ('Up', 'Strong Up'),
                'value': trend,
            },
            {
                'label': 'RSI < 50',
                'passed': not pd.isna(rsi) and rsi < 50,
                'value': f"{rsi:.0f}" if not pd.isna(rsi) else 'N/A',
            },
            {
                'label': 'Bullish SMC signal',
                'passed': bool(
                    row.get('smc_bos_bullish')
                    or row.get('smc_near_bullish_ob')
                    or row.get('smc_in_bullish_fvg')
                ),
                'value': _active_smc_labels(
                    row, ['smc_bos_bullish', 'smc_near_bullish_ob', 'smc_in_bullish_fvg']
                ) or 'none',
            },
            {
                'label': 'Discount Zone',
                'passed': bool(row.get('smc_discount_zone')),
                'value': 'active' if row.get('smc_discount_zone') else 'not active',
            },
            {
                'label': 'IV/HV < 1.0',
                'passed': not pd.isna(iv_hv) and iv_hv < 1.0,
                'value': f"{iv_hv:.2f}" if not pd.isna(iv_hv) else 'N/A',
            },
            {
                'label': 'No near earnings',
                'passed': days_earn is None or days_earn > 14,
                'value': f"{days_earn}d" if days_earn is not None else 'none',
            },
            {
                'label': 'EMA bullish',
                'passed': bool(row.get('ema_bull_stack') and row.get('above_ema20')),
                'value': ('stack+>EMA20'
                          if (row.get('ema_bull_stack') and row.get('above_ema20'))
                          else 'no'),
            },
        ]
    else:
        checks = [
            {
                'label': 'Trend',
                'passed': trend in ('Down', 'Strong Down'),
                'value': trend,
            },
            {
                'label': 'RSI > 50',
                'passed': not pd.isna(rsi) and rsi > 50,
                'value': f"{rsi:.0f}" if not pd.isna(rsi) else 'N/A',
            },
            {
                'label': 'Bearish SMC signal',
                'passed': bool(
                    row.get('smc_bos_bearish')
                    or row.get('smc_near_bearish_ob')
                    or row.get('smc_in_bearish_fvg')
                ),
                'value': _active_smc_labels(
                    row, ['smc_bos_bearish', 'smc_near_bearish_ob', 'smc_in_bearish_fvg']
                ) or 'none',
            },
            {
                'label': 'Premium Zone',
                'passed': bool(row.get('smc_premium_zone')),
                'value': 'active' if row.get('smc_premium_zone') else 'not active',
            },
            {
                'label': 'IV/HV < 1.0',
                'passed': not pd.isna(iv_hv) and iv_hv < 1.0,
                'value': f"{iv_hv:.2f}" if not pd.isna(iv_hv) else 'N/A',
            },
            {
                'label': 'No near earnings',
                'passed': days_earn is None or days_earn > 14,
                'value': f"{days_earn}d" if days_earn is not None else 'none',
            },
            {
                'label': 'EMA bearish',
                'passed': bool(row.get('ema_bear_stack') and not row.get('above_ema20')),
                'value': ('stack+<EMA20'
                          if (row.get('ema_bear_stack') and not row.get('above_ema20'))
                          else 'no'),
            },
        ]

    met = sum(1 for c in checks if c['passed'])
    total = len(checks)

    if met >= total - 1:
        status = 'enter'
    elif met >= (total + 1) // 2:
        status = 'wait'
    else:
        status = 'not_yet'

    return {'status': status, 'met': met, 'total': total, 'checks': checks}


def compute_exit_rules(row: dict, strategy: str, rec: dict) -> dict:
    """
    Compute exit rules from screener row and strike recommendation.

    rec: result of get_strike_recommendation() — needs 'cost' and 'dte'.

    Returns:
        take_profit_usd, stop_loss_usd, take_profit_pct, stop_loss_pct,
        time_exit_dte, time_exit_date, time_exit_msg,
        tech_triggers: list of {label, triggered, current_value}
    """
    if strategy not in ('Long Call', 'Long Put'):
        raise ValueError(f"Unknown strategy: {strategy!r}")
    is_call = (strategy == 'Long Call')
    rsi = row.get('rsi', np.nan)
    cost = rec.get('cost', np.nan)
    dte = rec.get('dte', None)

    if not pd.isna(cost):
        take_profit_usd = round(cost * 2.0, 2)
        stop_loss_usd = round(cost * 0.5, 2)
    else:
        take_profit_usd = np.nan
        stop_loss_usd = np.nan

    time_exit_dte = 21
    if dte is not None and not pd.isna(dte) and dte > 21:
        exit_date = datetime.now() + timedelta(days=dte - 21)
        time_exit_date = exit_date.strftime('%b %d, %Y')
        time_exit_msg = f"Close at 21 DTE → {time_exit_date}"
    elif dte is not None and not pd.isna(dte):
        time_exit_date = datetime.now().strftime('%b %d, %Y')
        time_exit_msg = "Exit now (past 21 DTE threshold)"
    else:
        time_exit_date = None
        time_exit_msg = "Close at 21 DTE"

    rsi_val = rsi if not pd.isna(rsi) else None
    rsi_str = f"RSI {rsi_val:.0f}" if rsi_val is not None else 'N/A'

    if is_call:
        tech_triggers = [
            {
                'label': 'RSI > 70 (overbought)',
                'triggered': rsi_val is not None and rsi_val > 70,
                'current_value': rsi_str,
            },
            {
                'label': 'Bearish BoS forms',
                'triggered': bool(row.get('smc_bos_bearish')),
                'current_value': 'active' if row.get('smc_bos_bearish') else 'not active',
            },
            {
                'label': 'Price enters Premium Zone',
                'triggered': bool(row.get('smc_premium_zone')),
                'current_value': 'active' if row.get('smc_premium_zone') else 'not active',
            },
            {
                'label': '穿 EMA20',
                'triggered': not bool(row.get('above_ema20', True)),
                'current_value': 'below' if not row.get('above_ema20', True) else 'above',
            },
            {
                'label': '穿 EMA50',
                'triggered': not bool(row.get('above_ema50', True)),
                'current_value': 'below' if not row.get('above_ema50', True) else 'above',
            },
        ]
    else:
        tech_triggers = [
            {
                'label': 'RSI < 30 (oversold)',
                'triggered': rsi_val is not None and rsi_val < 30,
                'current_value': rsi_str,
            },
            {
                'label': 'Bullish BoS forms',
                'triggered': bool(row.get('smc_bos_bullish')),
                'current_value': 'active' if row.get('smc_bos_bullish') else 'not active',
            },
            {
                'label': 'Price enters Discount Zone',
                'triggered': bool(row.get('smc_discount_zone')),
                'current_value': 'active' if row.get('smc_discount_zone') else 'not active',
            },
            {
                'label': '升穿 EMA20',
                'triggered': bool(row.get('above_ema20', False)),
                'current_value': 'above' if row.get('above_ema20', False) else 'below',
            },
            {
                'label': '升穿 EMA50',
                'triggered': bool(row.get('above_ema50', False)),
                'current_value': 'above' if row.get('above_ema50', False) else 'below',
            },
        ]

    return {
        'take_profit_usd': take_profit_usd,
        'stop_loss_usd': stop_loss_usd,
        'take_profit_pct': 1.0,
        'stop_loss_pct': 0.5,
        'time_exit_dte': time_exit_dte,
        'time_exit_date': time_exit_date,
        'time_exit_msg': time_exit_msg,
        'tech_triggers': tech_triggers,
    }


def compute_sell_verdict(exits: dict, rec: dict, entry_premium=None) -> dict:
    """
    Aggregate exit signals into a single SELL / TRIM / HOLD verdict.

    Args:
        exits: output of compute_exit_rules() — provides 'tech_triggers'.
        rec:   strike recommendation — provides 'cost' (current price estimate)
               and 'dte'.
        entry_premium: user's fill in dollars, same unit as rec['cost'].
               None / 0 / negative / NaN → condition-only verdict (no P/L).

    Returns:
        status:  'hold' | 'trim' | 'sell'  (most-severe across both layers)
        reasons: list[str] of plain-English drivers (empty for a clean hold)
        pnl_pct: float | None  (None when entry_premium unusable or cost NaN)
    """
    severity = {'hold': 0, 'trim': 1, 'sell': 2}
    reasons = []

    # ── Condition layer (always evaluated) ──
    active = [t for t in exits.get('tech_triggers', []) if t.get('triggered')]
    n = len(active)
    dte = rec.get('dte', None)
    dte_valid = dte is not None and not pd.isna(dte)

    if n >= 2 or (dte_valid and dte <= 21):
        cond_status = 'sell'
    elif n == 1 or (dte_valid and dte <= 25):
        cond_status = 'trim'
    else:
        cond_status = 'hold'

    if n >= 1:
        labels = ', '.join(t['label'] for t in active)
        reasons.append(f"{n} tech exit{'s' if n > 1 else ''} active: {labels}")
    if dte_valid and dte <= 21:
        reasons.append("Past 21 DTE — time stop")
    elif dte_valid and dte <= 25:
        reasons.append("Approaching 21 DTE")

    # ── P/L layer (only with a usable entry premium and cost) ──
    pnl_pct = None
    pnl_status = 'hold'
    cost = rec.get('cost', np.nan)
    if (entry_premium is not None and not pd.isna(entry_premium)
            and entry_premium > 0 and not pd.isna(cost)):
        pnl_pct = cost / entry_premium - 1.0
        if pnl_pct <= -0.50 or pnl_pct >= 1.00:
            pnl_status = 'sell'
        elif pnl_pct >= 0.50:
            pnl_status = 'trim'

        pct_str = f"{pnl_pct * 100:+.0f}%"
        if pnl_pct >= 1.00:
            reasons.append(f"{pct_str} — profit target hit")
        elif pnl_pct <= -0.50:
            reasons.append(f"{pct_str} — stop loss hit")
        elif pnl_pct >= 0.50:
            reasons.append(f"{pct_str} — lock partial")

    # ── Most-severe-wins ──
    status = cond_status if severity[cond_status] >= severity[pnl_status] else pnl_status

    return {'status': status, 'reasons': reasons, 'pnl_pct': pnl_pct}


def _active_smc_labels(row: dict, keys: list) -> str:
    label_map = {
        'smc_bos_bullish':     'BoS Bullish',
        'smc_near_bullish_ob': 'Near Bull OB',
        'smc_in_bullish_fvg':  'Bull FVG',
        'smc_bos_bearish':     'BoS Bearish',
        'smc_near_bearish_ob': 'Near Bear OB',
        'smc_in_bearish_fvg':  'Bear FVG',
    }
    return ', '.join(label_map[k] for k in keys if row.get(k))
