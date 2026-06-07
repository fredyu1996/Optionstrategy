import sys
import os
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import indicators
from indicators import compute_ema_signals


def test_bull_stack_on_rising_series():
    close = pd.Series(np.linspace(100, 200, 250))
    sig = compute_ema_signals(close)
    assert sig['ema_bull_stack'] is True
    assert sig['ema_bear_stack'] is False
    assert sig['above_ema20'] is True
    assert sig['above_ema50'] is True


def test_bear_stack_on_falling_series():
    close = pd.Series(np.linspace(200, 100, 250))
    sig = compute_ema_signals(close)
    assert sig['ema_bear_stack'] is True
    assert sig['ema_bull_stack'] is False
    assert sig['above_ema20'] is False


def test_too_few_bars_no_stack_no_crash():
    close = pd.Series(np.linspace(100, 110, 30))
    sig = compute_ema_signals(close)
    assert sig['ema_bull_stack'] is False
    assert sig['ema_bear_stack'] is False
    assert np.isnan(sig['ema200'])
    assert not np.isnan(sig['ema20'])
    assert sig['above_ema20'] is True


def test_returns_required_keys():
    sig = compute_ema_signals(pd.Series(np.linspace(100, 200, 250)))
    assert set(sig.keys()) == {
        'ema20', 'ema50', 'ema100', 'ema200',
        'ema_bull_stack', 'ema_bear_stack', 'above_ema20', 'above_ema50',
    }


def test_empty_series_safe():
    sig = compute_ema_signals(pd.Series([], dtype=float))
    assert sig['ema_bull_stack'] is False
    assert np.isnan(sig['ema20'])
    assert sig['above_ema20'] is False


# ── fetch_4h_ema_status ────────────────────────────────────────────────────────

from indicators import fetch_4h_ema_status


def _patch_4h(monkeypatch, close):
    monkeypatch.setattr(indicators, '_fetch_4h_close', lambda ticker: close)


def test_4h_good_when_bull_stack_above_ema20(monkeypatch):
    _patch_4h(monkeypatch, pd.Series(np.linspace(100, 200, 250)))
    out = fetch_4h_ema_status('AAPL')
    assert out['status'] == 'good'
    assert out['label'].startswith('4H')


def test_4h_avoid_when_falling(monkeypatch):
    _patch_4h(monkeypatch, pd.Series(np.linspace(200, 100, 250)))
    out = fetch_4h_ema_status('AAPL')
    assert out['status'] == 'avoid'


def test_4h_wait_when_below_ema20_above_ema50(monkeypatch):
    base = list(np.linspace(100, 200, 240))
    dip = list(np.linspace(200, 192, 10))
    _patch_4h(monkeypatch, pd.Series(base + dip))
    out = fetch_4h_ema_status('AAPL')
    assert out['status'] != 'unknown'


def test_4h_unknown_on_insufficient_data(monkeypatch):
    _patch_4h(monkeypatch, pd.Series(np.linspace(100, 110, 5)))
    out = fetch_4h_ema_status('AAPL')
    assert out['status'] == 'unknown'


def test_4h_unknown_on_fetch_exception(monkeypatch):
    def boom(ticker):
        raise RuntimeError('network')
    monkeypatch.setattr(indicators, '_fetch_4h_close', boom)
    out = fetch_4h_ema_status('AAPL')
    assert out['status'] == 'unknown'


def test_4h_none_close_is_unknown(monkeypatch):
    _patch_4h(monkeypatch, None)
    out = fetch_4h_ema_status('AAPL')
    assert out['status'] == 'unknown'
