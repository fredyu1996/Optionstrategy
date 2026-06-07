# indicators.py
"""
indicators.py - EMA signal computation (daily, pure) and 4H status (networked).
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def compute_ema_signals(close) -> dict:
    """EMA 20/50/100/200 signals from a close-price series. NaN-safe."""
    s = pd.Series(close, dtype='float64').dropna()

    def ema(span):
        if len(s) < span:
            return np.nan
        return float(s.ewm(span=span, adjust=False).mean().iloc[-1])

    e20, e50, e100, e200 = ema(20), ema(50), ema(100), ema(200)
    last = float(s.iloc[-1]) if len(s) else np.nan

    def gt(a, b):
        return (not np.isnan(a)) and (not np.isnan(b)) and a > b

    bull = gt(e20, e50) and gt(e50, e100) and gt(e100, e200)
    bear = gt(e50, e20) and gt(e100, e50) and gt(e200, e100)

    return {
        'ema20': e20, 'ema50': e50, 'ema100': e100, 'ema200': e200,
        'ema_bull_stack': bool(bull),
        'ema_bear_stack': bool(bear),
        'above_ema20': bool(gt(last, e20)),
        'above_ema50': bool(gt(last, e50)),
    }
