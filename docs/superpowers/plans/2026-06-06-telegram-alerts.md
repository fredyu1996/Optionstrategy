# Telegram Entry/Exit Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GitHub Actions cron job that scans the S&P 500 for fresh 🟢 Enter signals and held positions for TRIM/SELL, then pushes de-duplicated alerts to Telegram.

**Architecture:** Pure decision logic in `alerts.py` (entry/exit detection, dedup, formatting) is fully unit-tested. `telegram_bot.py` and `sheets.py` are thin IO wrappers. `notify.py` orchestrates: read env secrets, run scans (reusing screener/signals/positions), diff against the `alert_state` sheet, send, persist. A workflow runs it hourly.

**Tech Stack:** Python 3.11, gspread + google-auth (env creds), requests (Telegram), GitHub Actions cron, pytest. No new dependencies.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `telegram_bot.py` | Create | `send_message(token, chat_id, text)` |
| `tests/test_telegram_bot.py` | Create | send_message tests (mock requests) |
| `alerts.py` | Create | pure: entry/exit detection, dedup, formatting |
| `tests/test_alerts.py` | Create | pure-logic tests |
| `sheets.py` | Create | env-based gspread open/read/replace |
| `tests/test_sheets.py` | Create | worksheet open/replace tests (mock gspread) |
| `notify.py` | Create | orchestration (env → scans → diff → send → persist) |
| `.github/workflows/notify.yml` | Create | hourly cron workflow |
| `README.md` | Modify | Telegram + Actions secrets setup |

**Verified reuse facts:**
- `screener.get_sp500_tickers()`, `screener.batch_screen_fundamentals(tickers)->df` (adds trend/rsi/smc_*/ema_*), `screener.enrich_with_iv(df)->df` (adds iv_hv_ratio/days_to_earnings).
- `signals.compute_entry_readiness(row, strategy)->{status,met,total,checks}`.
- `positions.analyze_position(pos)->{...,'verdict':{status,reasons,pnl_pct},'ema4h',...}` (callable outside Streamlit; `@st.cache_data` deps just warn).
- `requirements.txt` already has gspread, google-auth, requests.

---

## Task 1: `telegram_bot.py`

**Files:**
- Create: `tests/test_telegram_bot.py`
- Create: `telegram_bot.py`

- [ ] **Step 1: Write failing tests** — create `tests/test_telegram_bot.py`:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import telegram_bot
from telegram_bot import send_message


class _Resp:
    def __init__(self, code):
        self.status_code = code


def test_send_message_success(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured['url'] = url
        captured['json'] = json
        return _Resp(200)

    monkeypatch.setattr(telegram_bot.requests, 'post', fake_post)
    ok = send_message('TOKEN123', '99', 'hello')
    assert ok is True
    assert 'TOKEN123' in captured['url']
    assert captured['json']['chat_id'] == '99'
    assert captured['json']['text'] == 'hello'


def test_send_message_non_200_returns_false(monkeypatch):
    monkeypatch.setattr(telegram_bot.requests, 'post', lambda *a, **k: _Resp(403))
    assert send_message('T', '1', 'x') is False


def test_send_message_exception_returns_false(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('network')
    monkeypatch.setattr(telegram_bot.requests, 'post', boom)
    assert send_message('T', '1', 'x') is False
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_telegram_bot.py -v`
Expected: `ModuleNotFoundError: No module named 'telegram_bot'`.

- [ ] **Step 3: Create `telegram_bot.py`**:
```python
# telegram_bot.py
"""telegram_bot.py - minimal Telegram Bot API sender."""
import requests


def send_message(token: str, chat_id: str, text: str) -> bool:
    """POST a message to a Telegram chat. Returns True on HTTP 200, else False.
    Never raises."""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={'chat_id': chat_id, 'text': text}, timeout=15)
        return resp.status_code == 200
    except Exception:
        return False
```

- [ ] **Step 4: Run, verify pass** — `python -m pytest tests/test_telegram_bot.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**
```bash
git add telegram_bot.py tests/test_telegram_bot.py
git commit -m "feat: add telegram_bot.send_message

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `alerts.py` (pure decision logic)

**Files:**
- Create: `tests/test_alerts.py`
- Create: `alerts.py`

- [ ] **Step 1: Write failing tests** — create `tests/test_alerts.py`:
```python
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from alerts import (
    entry_alerts, exit_alerts, diff_alerts, current_state_map,
    format_entry_msg, format_exit_msg,
)


def _enter_row():
    # a row that compute_entry_readiness scores as 'enter' for Long Call (>=6/7)
    return {
        'ticker': 'AAPL', 'trend': 'Up', 'rsi': 42.0, 'iv_hv_ratio': 0.7,
        'days_to_earnings': None,
        'smc_bos_bullish': True, 'smc_bos_bearish': False,
        'smc_choch_bullish': False, 'smc_choch_bearish': False,
        'smc_discount_zone': True, 'smc_premium_zone': False,
        'smc_near_bullish_ob': False, 'smc_near_bearish_ob': False,
        'smc_in_bullish_fvg': False, 'smc_in_bearish_fvg': False,
        'ema_bull_stack': True, 'ema_bear_stack': False,
        'above_ema20': True, 'above_ema50': True,
    }


def _dead_row():
    r = _enter_row()
    r['ticker'] = 'XYZ'
    r['trend'] = 'Down'
    r['rsi'] = 80.0
    r['iv_hv_ratio'] = 1.8
    r['smc_bos_bullish'] = False
    r['smc_discount_zone'] = False
    r['ema_bull_stack'] = False
    r['above_ema20'] = False
    return r


def test_entry_alerts_flags_enter_row():
    out = entry_alerts([_enter_row()])
    calls = [a for a in out if a['strategy'] == 'Long Call']
    assert len(calls) == 1
    a = calls[0]
    assert a['key'] == 'entry:AAPL:Long Call'
    assert a['kind'] == 'entry'
    assert a['state'] == 'enter'


def test_entry_alerts_ignores_dead_row():
    out = entry_alerts([_dead_row()])
    assert out == []


def test_exit_alerts_flags_sell_and_trim():
    analyzed = [
        {'pos': {'id': 'p1', 'ticker': 'NVDA', 'strike': 150.0, 'strategy': 'Long Call'},
         'data': {'verdict': {'status': 'sell', 'reasons': ['RSI > 70']}, 'pnl_pct': 0.3}},
        {'pos': {'id': 'p2', 'ticker': 'TSLA', 'strike': 200.0, 'strategy': 'Long Put'},
         'data': {'verdict': {'status': 'hold', 'reasons': []}, 'pnl_pct': 0.0}},
    ]
    out = exit_alerts(analyzed)
    assert len(out) == 1
    assert out[0]['key'] == 'exit:p1'
    assert out[0]['state'] == 'sell'


def test_diff_alerts_new_and_changed_only():
    current = [
        {'key': 'entry:AAPL:Long Call', 'state': 'enter'},
        {'key': 'exit:p1', 'state': 'sell'},
        {'key': 'exit:p2', 'state': 'trim'},
    ]
    stored = {'exit:p1': 'sell', 'exit:p2': 'hold'}
    out = diff_alerts(current, stored)
    keys = {a['key'] for a in out}
    assert keys == {'entry:AAPL:Long Call', 'exit:p2'}  # new + changed; p1 unchanged


def test_current_state_map():
    current = [{'key': 'a', 'state': 'enter'}, {'key': 'b', 'state': 'sell'}]
    assert current_state_map(current) == {'a': 'enter', 'b': 'sell'}


def test_format_entry_msg_contains_core_fields():
    a = {'ticker': 'AAPL', 'strategy': 'Long Call', 'met': 6, 'total': 7,
         'checks': [{'label': 'Trend', 'passed': True, 'value': 'Up'},
                    {'label': 'RSI < 50', 'passed': False, 'value': '80'}]}
    msg = format_entry_msg(a)
    assert 'ENTRY' in msg and 'AAPL' in msg and 'Long Call' in msg
    assert '6/7' in msg
    assert 'Trend' in msg          # passed check listed
    assert 'RSI < 50' not in msg   # failed check not listed


def test_format_exit_msg_sell_with_pnl():
    a = {'ticker': 'NVDA', 'strike': 150.0, 'strategy': 'Long Call',
         'status': 'sell', 'reasons': ['RSI > 70', '穿 EMA50'], 'pnl_pct': 0.3}
    msg = format_exit_msg(a)
    assert 'SELL' in msg and 'NVDA' in msg and '$150' in msg
    assert '+30%' in msg
    assert 'RSI > 70' in msg
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_alerts.py -v`
Expected: `ModuleNotFoundError: No module named 'alerts'`.

- [ ] **Step 3: Create `alerts.py`**:
```python
# alerts.py
"""
alerts.py - Pure decision logic for signal alerts: detect entry/exit signals,
de-duplicate against prior state, and format Telegram messages. No IO.
"""
from signals import compute_entry_readiness


def entry_alerts(rows: list) -> list:
    """Return entry alert dicts for rows that score 'enter' (Long Call/Put)."""
    out = []
    for row in rows:
        ticker = row.get('ticker')
        if not ticker:
            continue
        for strategy in ('Long Call', 'Long Put'):
            r = compute_entry_readiness(row, strategy)
            if r['status'] == 'enter':
                out.append({
                    'key': f'entry:{ticker}:{strategy}',
                    'kind': 'entry',
                    'state': 'enter',
                    'ticker': ticker,
                    'strategy': strategy,
                    'met': r['met'],
                    'total': r['total'],
                    'checks': r['checks'],
                })
    return out


def exit_alerts(analyzed: list) -> list:
    """Return exit alert dicts for analyzed positions with trim/sell verdicts.
    analyzed = [{'pos': pos_dict, 'data': analyze_position(pos)}]."""
    out = []
    for item in analyzed:
        pos = item['pos']
        data = item['data']
        status = data['verdict']['status']
        if status in ('trim', 'sell'):
            out.append({
                'key': f'exit:{pos["id"]}',
                'kind': 'exit',
                'state': status,
                'ticker': pos.get('ticker'),
                'strike': pos.get('strike'),
                'strategy': pos.get('strategy'),
                'status': status,
                'reasons': data['verdict'].get('reasons', []),
                'pnl_pct': data.get('pnl_pct'),
            })
    return out


def diff_alerts(current: list, stored: dict) -> list:
    """Return alerts whose state differs from the stored state (new/changed)."""
    return [a for a in current if stored.get(a['key']) != a['state']]


def current_state_map(current: list) -> dict:
    """Map each current alert key to its state string (to persist)."""
    return {a['key']: a['state'] for a in current}


def format_entry_msg(a: dict) -> str:
    passed = [c['label'] for c in a.get('checks', []) if c['passed']]
    head = f"🟢 ENTRY  {a['ticker']} {a['strategy']} ({a['met']}/{a['total']})"
    return head + "\n" + " · ".join(passed) if passed else head


def format_exit_msg(a: dict) -> str:
    is_sell = a['status'] == 'sell'
    emoji = '🔴' if is_sell else '🟡'
    label = 'SELL' if is_sell else 'TRIM'
    strike = a.get('strike')
    strike_str = f"${strike:.0f} " if isinstance(strike, (int, float)) else ''
    pnl = a.get('pnl_pct')
    pnl_str = f" · P/L {pnl * 100:+.0f}%" if pnl is not None else ''
    reasons = '; '.join(a.get('reasons') or []) or 'exit conditions'
    return f"{emoji} {label}  {a['ticker']} {strike_str}{a['strategy']}\n{reasons}{pnl_str}"
```

- [ ] **Step 4: Run, verify pass** — `python -m pytest tests/test_alerts.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**
```bash
git add alerts.py tests/test_alerts.py
git commit -m "feat: add alerts.py pure entry/exit detection + dedup + formatting

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `sheets.py` (env-based Google Sheets)

**Files:**
- Create: `tests/test_sheets.py`
- Create: `sheets.py`

- [ ] **Step 1: Write failing tests** — create `tests/test_sheets.py`:
```python
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
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_sheets.py -v`
Expected: `ModuleNotFoundError: No module named 'sheets'`.

- [ ] **Step 3: Create `sheets.py`**:
```python
# sheets.py
"""
sheets.py - Google Sheets access from an explicit service-account dict (for use
outside Streamlit, e.g. the notify.py GitHub Actions job).
"""
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


def open_worksheet(service_account_info: dict, sheet_key: str, title: str):
    """Authorize from a service-account dict and return the named worksheet,
    creating an empty one if it doesn't exist."""
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_key)
    try:
        return sheet.worksheet(title)
    except gspread.WorksheetNotFound:
        return sheet.add_worksheet(title=title, rows=100, cols=10)


def get_records(ws) -> list:
    """Return all data rows as dicts (first row is the header)."""
    return ws.get_all_records()


def replace_rows(ws, header: list, rows: list) -> None:
    """Clear the worksheet and write the header followed by all rows."""
    ws.clear()
    ws.update([header] + rows)
```

- [ ] **Step 4: Run, verify pass** — `python -m pytest tests/test_sheets.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**
```bash
git add sheets.py tests/test_sheets.py
git commit -m "feat: add sheets.py env-based gspread access

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `notify.py` (orchestration)

**Files:**
- Create: `notify.py`

- [ ] **Step 1: Create `notify.py`**:
```python
# notify.py
"""
notify.py - Background alert job (run by GitHub Actions, NOT Streamlit).

Scans S&P 500 for fresh entry signals and held positions for TRIM/SELL,
de-duplicates against the Google Sheets `alert_state` tab, and pushes new
signals to Telegram. Secrets come from environment variables.
"""
import json
import os
import sys

import alerts
import sheets
import telegram_bot
from positions import analyze_position


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: missing env var {name}", file=sys.stderr)
        sys.exit(1)
    return val


def _positions_to_dicts(records: list) -> list:
    """Coerce alert positions sheet records into the shape analyze_position needs."""
    out = []
    for r in records:
        out.append({
            'id': str(r.get('id', '')),
            'ticker': str(r.get('ticker', '')),
            'strategy': str(r.get('strategy', '')),
            'strike': float(r.get('strike') or 0),
            'expiry': str(r.get('expiry', '')),
            'entry_premium': float(r.get('entry_premium') or 0),
            'contracts': int(float(r.get('contracts') or 0)),
            'entry_date': str(r.get('entry_date', '')),
        })
    return out


def run_exit_scan(sa_info: dict, sheet_key: str) -> list:
    """Analyze held positions; return exit alert dicts."""
    try:
        ws = sheets.open_worksheet(sa_info, sheet_key, 'positions')
        positions = _positions_to_dicts(sheets.get_records(ws))
        analyzed = [{'pos': p, 'data': analyze_position(p)} for p in positions]
        return alerts.exit_alerts(analyzed)
    except Exception as exc:
        print(f"exit scan failed: {exc}", file=sys.stderr)
        return []


def run_entry_scan() -> list:
    """Screen the S&P 500; return entry alert dicts."""
    try:
        from screener import (
            get_sp500_tickers, batch_screen_fundamentals, enrich_with_iv,
        )
        tickers = get_sp500_tickers()
        df = batch_screen_fundamentals(tickers)
        df = enrich_with_iv(df)
        rows = df.to_dict('records')
        return alerts.entry_alerts(rows)
    except Exception as exc:
        print(f"entry scan failed: {exc}", file=sys.stderr)
        return []


def main() -> None:
    token = _require_env('TELEGRAM_BOT_TOKEN')
    chat_id = _require_env('TELEGRAM_CHAT_ID')
    sheet_key = _require_env('POSITIONS_SHEET_KEY')
    sa_info = json.loads(_require_env('GCP_SERVICE_ACCOUNT'))

    current = run_exit_scan(sa_info, sheet_key) + run_entry_scan()

    state_ws = sheets.open_worksheet(sa_info, sheet_key, 'alert_state')
    stored = {r['key']: r['state'] for r in sheets.get_records(state_ws) if r.get('key')}

    to_send = alerts.diff_alerts(current, stored)
    for a in to_send:
        text = (alerts.format_entry_msg(a) if a['kind'] == 'entry'
                else alerts.format_exit_msg(a))
        telegram_bot.send_message(token, chat_id, text)

    new_state = alerts.current_state_map(current)
    sheets.replace_rows(state_ws, ['key', 'state'],
                        [[k, v] for k, v in new_state.items()])
    print(f"sent {len(to_send)} alerts; tracked {len(new_state)} active signals")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Syntax + import check** (no network call without env):
Run: `python -c "import ast; ast.parse(open('notify.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Verify the module imports and missing-env exits cleanly**:
Run: `python -c "import notify; print('import OK')"`
Expected: `import OK` (importing does not call `main`).

- [ ] **Step 4: Verify missing-env guard** (should exit non-zero with message):
Run (bash): `python notify.py; echo "exit=$?"`
Expected: prints `ERROR: missing env var TELEGRAM_BOT_TOKEN` and `exit=1`.

- [ ] **Step 5: Run the full test suite** (notify has no unit tests; confirm nothing else broke):
Run: `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**
```bash
git add notify.py
git commit -m "feat: add notify.py alert orchestration job

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: GitHub Actions workflow + README

**Files:**
- Create: `.github/workflows/notify.yml`
- Modify: `README.md`

- [ ] **Step 1: Create `.github/workflows/notify.yml`**:
```yaml
name: Signal Alerts

on:
  schedule:
    - cron: '0 13-21 * * 1-5'   # hourly, ~US market hours (UTC), Mon-Fri
  workflow_dispatch: {}

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python notify.py
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          GCP_SERVICE_ACCOUNT: ${{ secrets.GCP_SERVICE_ACCOUNT }}
          POSITIONS_SHEET_KEY: ${{ secrets.POSITIONS_SHEET_KEY }}
```

- [ ] **Step 2: Validate the YAML parses**:
Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/notify.yml', encoding='utf-8')); print('YAML OK')"`
Expected: `YAML OK`. (If `yaml` is unavailable, instead run `python -c "import ast; print('skip')"` — the workflow is plain YAML and will be validated by GitHub on push.)

- [ ] **Step 3: Append the setup section to `README.md`**:
```markdown

## Signal Alerts (Telegram)

A GitHub Actions job (`notify.py`, hourly during US market hours) scans the
S&P 500 for fresh 🟢 Enter signals and your held positions for TRIM/SELL, then
sends new signals to Telegram. De-duplicated via an `alert_state` tab in the
positions sheet (auto-created).

One-time setup:

1. In Telegram, message **@BotFather** → `/newbot` → copy the **bot token**.
2. Get your **chat id**: message **@userinfobot**, or message your new bot and
   open `https://api.telegram.org/bot<token>/getUpdates` to read `chat.id`.
3. In the GitHub repo → **Settings → Secrets and variables → Actions → New
   repository secret**, add:
   - `TELEGRAM_BOT_TOKEN` — the bot token
   - `TELEGRAM_CHAT_ID` — your chat id
   - `POSITIONS_SHEET_KEY` — the same Google Sheet id used by the app
   - `GCP_SERVICE_ACCOUNT` — the full service-account JSON (paste as the value)
4. The service account already shares the sheet; the `alert_state` tab is created
   automatically on the first run.
5. Trigger a manual test: repo → **Actions → Signal Alerts → Run workflow**.
```

- [ ] **Step 4: Commit**
```bash
git add .github/workflows/notify.yml README.md
git commit -m "feat: add hourly Telegram alerts workflow + setup docs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - Telegram send → Task 1 `telegram_bot.py`
  - entry/exit detection, dedup, formatting (pure) → Task 2 `alerts.py`
  - env-based Sheets access + auto-create tab → Task 3 `sheets.py`
  - orchestration (env secrets, scans, diff, send, persist, partial-failure isolation) → Task 4 `notify.py`
  - hourly cron + workflow_dispatch + env secrets → Task 5 workflow
  - 🟢 Enter-only entry threshold → `entry_alerts` only emits `status=='enter'`
  - exit on trim/sell → `exit_alerts` filters those
  - dedup state in Google Sheets tab → `alert_state` read in `main`, persisted via `replace_rows`
  - README setup → Task 5 Step 3
  - tests → Tasks 1,2,3

- [x] **No placeholders:** every step has runnable code/commands.

- [x] **Type consistency:**
  - alert dicts carry `key`/`kind`/`state` consumed by `diff_alerts`/`current_state_map` and `main`'s `a['kind']` branch.
  - `entry_alerts` emits `met`/`total`/`checks` consumed by `format_entry_msg`; `exit_alerts` emits `status`/`reasons`/`pnl_pct`/`strike` consumed by `format_exit_msg`.
  - `open_worksheet(sa_info, sheet_key, title)` / `get_records(ws)` / `replace_rows(ws, header, rows)` signatures match `notify.py` calls.
  - `analyze_position(pos)` result `['verdict']['status']` consumed by `exit_alerts`.
