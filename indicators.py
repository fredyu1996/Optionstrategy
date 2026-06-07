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


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_4h_close(ticker: str):
    """~60d of 1h bars resampled to 4H closes. Cached; mockable in tests."""
    df = yf.Ticker(ticker).history(period='60d', interval='1h')
    if df is None or df.empty:
        return None
    return df['Close'].resample('4h').last().dropna()


def fetch_4h_ema_status(ticker: str) -> dict:
    """Classify a ticker's 4H EMA posture for entry timing / exit warning.

    status: 'good' | 'wait' | 'avoid' | 'unknown'
    """
    try:
        close4h = _fetch_4h_close(ticker)
        if close4h is None or len(close4h) < 20:
            return {'status': 'unknown', 'label': '4H: 數據不足', 'signals': {}}
        sig = compute_ema_signals(close4h)
        if sig['ema_bull_stack'] and sig['above_ema20']:
            status, msg = 'good', '多頭排列, 企 EMA20 上 → 入場時機好'
        elif sig['above_ema50'] and not sig['above_ema20']:
            status, msg = 'wait', '穿 EMA20 但守 EMA50 → 等重奪'
        elif (not sig['above_ema50']) or sig['ema_bear_stack']:
            status, msg = 'avoid', '穿 EMA50 / 空頭排列 → 唔好追'
        else:
            status, msg = 'wait', '訊號中性 → 觀望'
        return {'status': status, 'label': f'4H: {msg}', 'signals': sig}
    except Exception:
        return {'status': 'unknown', 'label': '4H: 資料抓取失敗', 'signals': {}}
