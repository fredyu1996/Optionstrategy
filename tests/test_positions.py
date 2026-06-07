import sys
import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import positions
from positions import analyze_position, _fetch_contract_price, _normalize_expiry


class _FakeChain:
    def __init__(self, calls, puts):
        self.calls = calls
        self.puts = puts


class _FakeTicker:
    def __init__(self, calls, puts):
        self._chain = _FakeChain(calls, puts)

    def option_chain(self, expiry):
        return self._chain


def _patch_chain(monkeypatch, calls, puts=None):
    puts = puts if puts is not None else pd.DataFrame()
    monkeypatch.setattr(positions.yf, 'Ticker', lambda t: _FakeTicker(calls, puts))


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


@pytest.fixture(autouse=True)
def _stub_4h(monkeypatch):
    monkeypatch.setattr(positions, 'fetch_4h_ema_status',
                        lambda t: {'status': 'unknown', 'label': '', 'signals': {}})


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


def test_unknown_strategy_returns_error_not_raise(monkeypatch):
    _patch(monkeypatch, price=5.0)
    data = analyze_position(_pos(strategy='Iron Condor'))
    assert data['error'] is not None
    assert data['verdict']['status'] == 'hold'
    assert data['pnl_usd'] is None


def test_strategy_normalized_whitespace_and_case(monkeypatch):
    # ' long call ' must normalize to 'Long Call' and follow the normal path
    # (pnl computed), not the unknown-strategy error path (which yields None pnl).
    _patch(monkeypatch, price=6.0)
    data = analyze_position(_pos(strategy=' long call '))
    assert data['pnl_usd'] == pytest.approx(400.0)
    assert not str(data['error']).startswith('Unknown strategy')


def test_normalize_expiry_variants():
    assert _normalize_expiry('2026-07-17') == '2026-07-17'
    assert _normalize_expiry('7/17/2026') == '2026-07-17'  # Sheets M/D/YYYY reformat
    assert _normalize_expiry('garbage') is None


def test_analyze_handles_sheets_reformatted_date(monkeypatch):
    # Google Sheets reformats ISO dates to M/D/YYYY on round-trip — must not crash
    _patch(monkeypatch, price=5.0)
    fut = date.today() + timedelta(days=30)
    sheets_fmt = f"{fut.month}/{fut.day}/{fut.year}"
    data = analyze_position(_pos(expiry=sheets_fmt))
    assert data['dte'] == 30


def test_expiry_normalized_to_iso_for_price_fetch(monkeypatch):
    captured = {}

    def fake_price(ticker, strategy, strike, expiry):
        captured['expiry'] = expiry
        return 5.0

    monkeypatch.setattr(positions, 'get_contract_price', fake_price)
    monkeypatch.setattr(positions, '_get_history', lambda t: pd.DataFrame())
    analyze_position(_pos(expiry='7/17/2026'))
    assert captured['expiry'] == '2026-07-17'  # yfinance requires ISO


def test_unparseable_expiry_sets_nan_price(monkeypatch):
    _patch(monkeypatch, price=5.0)  # price fn won't be called for bad expiry
    data = analyze_position(_pos(expiry='not-a-date'))
    assert data['dte'] is None
    assert np.isnan(data['current_price'])


def test_result_has_required_keys(monkeypatch):
    _patch(monkeypatch, price=5.0)
    data = analyze_position(_pos())
    assert set(data.keys()) == {
        'current_price', 'pnl_pct', 'pnl_usd', 'dte', 'verdict', 'ema4h', 'error',
    }


# ── _fetch_contract_price ──────────────────────────────────────────────────────

def test_fetch_contract_price_uses_bid_ask_mid(monkeypatch):
    calls = pd.DataFrame([{'strike': 150.0, 'bid': 4.0, 'ask': 5.0, 'lastPrice': 4.2}])
    _patch_chain(monkeypatch, calls)
    assert _fetch_contract_price('AAPL', 'Long Call', 150.0, '2026-07-17') == pytest.approx(4.5)


def test_fetch_contract_price_falls_back_to_last(monkeypatch):
    # no bid/ask -> use lastPrice
    calls = pd.DataFrame([{'strike': 150.0, 'bid': 0.0, 'ask': 0.0, 'lastPrice': 4.2}])
    _patch_chain(monkeypatch, calls)
    assert _fetch_contract_price('AAPL', 'Long Call', 150.0, '2026-07-17') == pytest.approx(4.2)


def test_fetch_contract_price_strike_not_found_returns_nan(monkeypatch):
    calls = pd.DataFrame([{'strike': 155.0, 'bid': 4.0, 'ask': 5.0, 'lastPrice': 4.2}])
    _patch_chain(monkeypatch, calls)
    assert np.isnan(_fetch_contract_price('AAPL', 'Long Call', 150.0, '2026-07-17'))


def test_fetch_contract_price_uses_puts_for_long_put(monkeypatch):
    calls = pd.DataFrame([{'strike': 150.0, 'bid': 9.0, 'ask': 9.0, 'lastPrice': 9.0}])
    puts = pd.DataFrame([{'strike': 150.0, 'bid': 3.0, 'ask': 5.0, 'lastPrice': 4.0}])
    _patch_chain(monkeypatch, calls, puts)
    assert _fetch_contract_price('AAPL', 'Long Put', 150.0, '2026-07-17') == pytest.approx(4.0)


def test_analyze_includes_ema4h(monkeypatch):
    monkeypatch.setattr(positions, 'get_contract_price', lambda *a, **k: 5.0)
    monkeypatch.setattr(positions, '_get_history', lambda t: pd.DataFrame())
    monkeypatch.setattr(positions, 'fetch_4h_ema_status',
                        lambda t: {'status': 'good', 'label': '4H: ok', 'signals': {}})
    data = analyze_position(_pos())
    assert data['ema4h'] == {'status': 'good', 'label': '4H: ok', 'signals': {}}


def test_analyze_adds_daily_ema_flags_to_exit(monkeypatch):
    n = 70
    falling = np.linspace(200, 120, n)
    hist = pd.DataFrame({'Open': falling, 'High': falling + 1,
                         'Low': falling - 1, 'Close': falling})
    monkeypatch.setattr(positions, 'get_contract_price', lambda *a, **k: 5.0)
    monkeypatch.setattr(positions, '_get_history', lambda t: hist)
    monkeypatch.setattr(positions, 'fetch_4h_ema_status',
                        lambda t: {'status': 'unknown', 'label': '', 'signals': {}})
    data = analyze_position(_pos())  # Long Call on a downtrend
    assert data['verdict']['status'] == 'sell'


def test_analyze_runs_signal_path_with_real_ohlcv(monkeypatch):
    # >=20 rows of OHLCV exercises the rsi + SMC remap branch (no error set there)
    n = 30
    rng = np.linspace(100, 130, n)
    hist = pd.DataFrame({
        'Open': rng, 'High': rng + 1, 'Low': rng - 1, 'Close': rng,
    })
    monkeypatch.setattr(positions, 'get_contract_price', lambda *a, **k: 5.0)
    monkeypatch.setattr(positions, '_get_history', lambda ticker: hist)
    data = analyze_position(_pos())
    # signal path ran (no 'signal data unavailable'); verdict produced
    assert data['error'] is None
    assert data['verdict']['status'] in {'hold', 'trim', 'sell'}
