import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import positions_store
from positions_store import (
    load_positions, add_position, delete_position, PositionsConfigError,
)


class FakeWorksheet:
    def __init__(self, records=None):
        self._records = records or []
        self.appended = []
        self.deleted = []

    def get_all_records(self):
        return [dict(r) for r in self._records]

    def append_row(self, row):
        self.appended.append(row)

    def delete_rows(self, idx):
        self.deleted.append(idx)


def _patch_ws(monkeypatch, ws):
    monkeypatch.setattr(positions_store, 'get_worksheet', lambda: ws)


def test_load_positions_types_fields(monkeypatch):
    ws = FakeWorksheet([{
        'id': 'a1', 'ticker': 'AAPL', 'strategy': 'Long Call',
        'strike': '150', 'expiry': '2026-07-17', 'entry_premium': '4.2',
        'contracts': '2', 'entry_date': '2026-06-05',
    }])
    _patch_ws(monkeypatch, ws)
    out = load_positions()
    assert len(out) == 1
    p = out[0]
    assert p['ticker'] == 'AAPL'
    assert isinstance(p['strike'], float) and p['strike'] == 150.0
    assert isinstance(p['entry_premium'], float) and p['entry_premium'] == 4.2
    assert isinstance(p['contracts'], int) and p['contracts'] == 2


def test_load_positions_handles_numeric_inputs(monkeypatch):
    # gspread returns native int/float for numeric cells, not strings
    ws = FakeWorksheet([{
        'id': 'a1', 'ticker': 'AAPL', 'strategy': 'Long Call',
        'strike': 150, 'expiry': '2026-07-17', 'entry_premium': 4.2,
        'contracts': 2, 'entry_date': '2026-06-05',
    }])
    _patch_ws(monkeypatch, ws)
    p = load_positions()[0]
    assert isinstance(p['strike'], float) and p['strike'] == 150.0
    assert isinstance(p['entry_premium'], float) and p['entry_premium'] == 4.2
    assert isinstance(p['contracts'], int) and p['contracts'] == 2


def test_load_positions_empty(monkeypatch):
    _patch_ws(monkeypatch, FakeWorksheet([]))
    assert load_positions() == []


def test_add_position_appends_and_autofills(monkeypatch):
    ws = FakeWorksheet([])
    _patch_ws(monkeypatch, ws)
    add_position({
        'ticker': 'MSFT', 'strategy': 'Long Put', 'strike': 400.0,
        'expiry': '2026-08-21', 'entry_premium': 6.5, 'contracts': 1,
    })
    assert len(ws.appended) == 1
    row = ws.appended[0]
    assert row[0]  # id
    assert row[1] == 'MSFT'
    assert row[7]  # entry_date
    assert len(row) == 8


def test_add_position_preserves_given_id(monkeypatch):
    ws = FakeWorksheet([])
    _patch_ws(monkeypatch, ws)
    add_position({
        'id': 'fixed-id', 'ticker': 'NVDA', 'strategy': 'Long Call',
        'strike': 120.0, 'expiry': '2026-09-18', 'entry_premium': 5.0,
        'contracts': 3, 'entry_date': '2026-06-01',
    })
    assert ws.appended[0][0] == 'fixed-id'
    assert ws.appended[0][7] == '2026-06-01'


def test_delete_position_removes_matching_row(monkeypatch):
    ws = FakeWorksheet([
        {'id': 'x1', 'ticker': 'AAPL', 'strategy': 'Long Call', 'strike': '150',
         'expiry': '2026-07-17', 'entry_premium': '4.2', 'contracts': '1', 'entry_date': '2026-06-05'},
        {'id': 'x2', 'ticker': 'TSLA', 'strategy': 'Long Put', 'strike': '200',
         'expiry': '2026-07-17', 'entry_premium': '7.0', 'contracts': '2', 'entry_date': '2026-06-05'},
    ])
    _patch_ws(monkeypatch, ws)
    delete_position('x2')
    assert ws.deleted == [3]


def test_delete_position_noop_when_absent(monkeypatch):
    ws = FakeWorksheet([
        {'id': 'x1', 'ticker': 'AAPL', 'strategy': 'Long Call', 'strike': '150',
         'expiry': '2026-07-17', 'entry_premium': '4.2', 'contracts': '1', 'entry_date': '2026-06-05'},
    ])
    _patch_ws(monkeypatch, ws)
    delete_position('does-not-exist')
    assert ws.deleted == []


def test_missing_secrets_raises_config_error(monkeypatch):
    monkeypatch.setattr(positions_store.st, 'secrets', {})
    with pytest.raises(PositionsConfigError):
        positions_store.get_worksheet()
