import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sheets
from sheets import open_worksheet, replace_rows


class _FakeSheet:
    def __init__(self, existing=None):
        self._existing = existing or {}
        self.added = []

    def worksheet(self, title):
        if title in self._existing:
            return self._existing[title]
        raise sheets.gspread.WorksheetNotFound()

    def add_worksheet(self, title, rows, cols):
        ws = f'WS:{title}'
        self._existing[title] = ws
        self.added.append(title)
        return ws


class _FakeClient:
    def __init__(self, sheet):
        self._sheet = sheet

    def open_by_key(self, key):
        return self._sheet


def _patch(monkeypatch, sheet):
    monkeypatch.setattr(sheets.Credentials, 'from_service_account_info',
                        lambda info, scopes=None: 'CREDS')
    monkeypatch.setattr(sheets.gspread, 'authorize', lambda creds: _FakeClient(sheet))


def test_open_worksheet_returns_existing(monkeypatch):
    sheet = _FakeSheet(existing={'positions': 'WS:positions'})
    _patch(monkeypatch, sheet)
    ws = open_worksheet({'x': 1}, 'KEY', 'positions')
    assert ws == 'WS:positions'
    assert sheet.added == []


def test_open_worksheet_creates_when_missing(monkeypatch):
    sheet = _FakeSheet()
    _patch(monkeypatch, sheet)
    ws = open_worksheet({'x': 1}, 'KEY', 'alert_state')
    assert ws == 'WS:alert_state'
    assert sheet.added == ['alert_state']


def test_replace_rows_clears_then_writes_header_first():
    class _WS:
        def __init__(self):
            self.cleared = False
            self.updated = None

        def clear(self):
            self.cleared = True

        def update(self, values):
            self.updated = values

    ws = _WS()
    replace_rows(ws, ['key', 'state'], [['a', 'enter'], ['b', 'sell']])
    assert ws.cleared is True
    assert ws.updated[0] == ['key', 'state']
    assert ws.updated[1] == ['a', 'enter']
