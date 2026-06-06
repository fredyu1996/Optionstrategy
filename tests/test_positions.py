import sys
import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import positions
from positions import analyze_position


def _pos(**over):
    base = {
        'id': 'p1', 'ticker': 'AAPL', 'strategy': 'Long Call', 'strike': 150.0,
        'expiry': (date.today() + timedelta(days=40)).isoformat(),
        'entry_premium': 4.0, 'contracts': 2, 'entry_date': '2026-06-05',
    }
    base.update(over)
    return base


def _patch(monkeypatch, price, hist=None):
    monkeypatch.setattr(positions, 'get_contract_price', lambda *a, **k: price)
    monkeypatch.setattr(positions, '_get_history',
                        lambda ticker: hist if hist is not None else pd.DataFrame())


def test_pnl_usd_uses_100x_multiplier(monkeypatch):
    _patch(monkeypatch, price=6.0)  # +50% on 4.0 entry
    data = analyze_position(_pos())
    assert data['pnl_usd'] == pytest.approx(400.0)
    assert data['pnl_pct'] == pytest.approx(0.5)


def test_verdict_status_flows_through(monkeypatch):
    _patch(monkeypatch, price=10.0)  # +150% -> profit target -> sell
    data = analyze_position(_pos())
    assert data['verdict']['status'] == 'sell'


def test_nan_price_sets_error_and_none_pnl(monkeypatch):
    _patch(monkeypatch, price=float('nan'))
    data = analyze_position(_pos())
    assert data['error'] is not None
    assert data['pnl_usd'] is None
    assert np.isnan(data['current_price'])


def test_dte_computed_from_expiry(monkeypatch):
    _patch(monkeypatch, price=5.0)
    data = analyze_position(_pos(expiry=(date.today() + timedelta(days=30)).isoformat()))
    assert data['dte'] == 30


def test_result_has_required_keys(monkeypatch):
    _patch(monkeypatch, price=5.0)
    data = analyze_position(_pos())
    assert set(data.keys()) == {
        'current_price', 'pnl_pct', 'pnl_usd', 'dte', 'verdict', 'error',
    }
