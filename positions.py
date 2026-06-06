# positions.py
"""
positions.py - Live analysis for tracked option positions.

Fetches the exact held contract's current price and reuses the screener's
signal functions plus signals.py's verdict to produce a live SELL/TRIM/HOLD
call and P/L for each stored position.
"""
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from screener import _compute_rsi, compute_smc_signals
from signals import compute_exit_rules, compute_sell_verdict


def _fetch_contract_price(ticker: str, strategy: str, strike: float, expiry: str) -> float:
    """Live per-share mid for the exact contract, or np.nan if unavailable.

    Uncached core logic; tested directly. Use get_contract_price in the app.
    """
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        opts = chain.calls if strategy == 'Long Call' else chain.puts
        match = opts[opts['strike'] == float(strike)]
        if match.empty:
            return float('nan')
        opt = match.iloc[0]
        bid = float(opt.get('bid') or 0)
        ask = float(opt.get('ask') or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        last = float(opt.get('lastPrice') or 0)
        return last if last > 0 else float('nan')
    except Exception:
        return float('nan')


@st.cache_data(ttl=120, show_spinner=False)
def get_contract_price(ticker: str, strategy: str, strike: float, expiry: str) -> float:
    """Cached wrapper around _fetch_contract_price (2 min TTL for live quotes)."""
    return _fetch_contract_price(ticker, strategy, strike, expiry)


@st.cache_data(ttl=900, show_spinner=False)
def _get_history(ticker: str):
    """Download ~3 months of OHLCV (15 min TTL). Separated for test mocking."""
    return yf.Ticker(ticker).history(period='3mo')


def _normalize_expiry(expiry: str):
    """Coerce any stored expiry (Google Sheets may reformat dates) to ISO
    'YYYY-MM-DD'. Returns None if unparseable."""
    ts = pd.to_datetime(expiry, errors='coerce')
    if pd.isna(ts):
        return None
    return ts.strftime('%Y-%m-%d')


def _days_to_expiry(expiry_iso: str) -> int:
    exp = datetime.strptime(expiry_iso, '%Y-%m-%d').date()
    return (exp - date.today()).days


_STRATEGIES = {'long call': 'Long Call', 'long put': 'Long Put'}


def _normalize_strategy(value):
    """Map a stored strategy to canonical 'Long Call'/'Long Put' (tolerating
    whitespace and case). Returns None if unrecognized."""
    if value is None:
        return None
    return _STRATEGIES.get(str(value).strip().lower())


def _error_result(dte, message: str) -> dict:
    """A safe analyze_position result for a row we cannot evaluate."""
    return {
        'current_price': float('nan'),
        'pnl_pct': None,
        'pnl_usd': None,
        'dte': dte,
        'verdict': {'status': 'hold', 'reasons': [], 'pnl_pct': None},
        'error': message,
    }


def analyze_position(pos: dict) -> dict:
    """Compute live price, P/L, DTE, and SELL/TRIM/HOLD verdict for a position."""
    ticker = pos['ticker']
    strategy = _normalize_strategy(pos.get('strategy'))
    strike = pos['strike']
    expiry = pos['expiry']
    entry = pos.get('entry_premium', 0)
    contracts = int(pos.get('contracts', 1) or 1)

    expiry_iso = _normalize_expiry(expiry)
    dte = _days_to_expiry(expiry_iso) if expiry_iso else None

    if strategy is None:
        return _error_result(dte, f"Unknown strategy: {pos.get('strategy')!r}")

    current = (get_contract_price(ticker, strategy, strike, expiry_iso)
               if expiry_iso else float('nan'))

    row = {}
    error = None
    try:
        hist = _get_history(ticker)
        if hist is not None and not hist.empty and len(hist) >= 20:
            row['rsi'] = _compute_rsi(hist['Close'])
            for key, val in compute_smc_signals(hist).items():
                row[f'smc_{key}'] = val
        else:
            error = 'signal data unavailable'
    except Exception:
        error = 'signal data unavailable'

    rec = {'cost': current, 'dte': dte}
    exits = compute_exit_rules(row, strategy, rec)
    ep = entry if (entry and entry > 0) else None
    verdict = compute_sell_verdict(exits, rec, ep)

    if np.isnan(current):
        error = error or 'price unavailable'
        pnl_usd = None
    else:
        pnl_usd = (current - entry) * 100 * contracts

    return {
        'current_price': current,
        'pnl_pct': verdict['pnl_pct'],
        'pnl_usd': pnl_usd,
        'dte': dte,
        'verdict': verdict,
        'error': error,
    }
