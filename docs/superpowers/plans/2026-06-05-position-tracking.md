# Manual + Persisted Position Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record open long call/put positions, persist them in Google Sheets, and show each with a live SELL/TRIM/HOLD verdict and P/L in a new "My Positions" view.

**Architecture:** `positions_store.py` does Google Sheets CRUD; `positions.py` fetches the exact contract's live price and reuses `screener.py` signal functions + `signals.py` verdict functions to analyze each position; `app.py` gets a sidebar radio nav and a positions page. Persistence is external (Google Sheets) because Streamlit Cloud wipes local files on redeploy.

**Tech Stack:** Python 3.11, Streamlit, yfinance, gspread, google-auth, pandas, numpy, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add `gspread`, `google-auth` |
| `positions_store.py` | Create | Google Sheets CRUD + `PositionsConfigError` |
| `positions.py` | Create | `get_contract_price`, `analyze_position` |
| `tests/test_positions_store.py` | Create | CRUD tests with fake worksheet |
| `tests/test_positions.py` | Create | analysis + P/L math tests with mocked IO |
| `app.py` | Modify | imports, `render_positions_page`, sidebar nav, view guard |
| `README.md` | Modify | Google Sheets setup steps |

**Key reuse facts (verified in current code):**
- `screener._compute_rsi(prices: pd.Series, period=14) -> float`
- `screener.compute_smc_signals(ohlcv_df) -> dict` — returns keys WITHOUT `smc_` prefix (`bos_bullish`, `premium_zone`, …); `compute_exit_rules`/`compute_sell_verdict` expect them WITH `smc_` prefix, so they must be remapped.
- `signals.compute_exit_rules(row, strategy, rec)` and `signals.compute_sell_verdict(exits, rec, entry_premium)` (already in repo).
- yfinance: `yf.Ticker(t).option_chain(expiry).calls/.puts` (cols `strike,bid,ask,lastPrice`); `yf.Ticker(t).history(period='3mo')` (cols `Open,High,Low,Close`).
- `app.py` already defines `_PLAYBOOK_VERDICT` (status→emoji/text/color) and runs screening at `if st.session_state.run_screen:` (~line 814), AFTER the `with st.sidebar:` block (ends ~line 734).

---

## Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append the two deps**

Add these lines to the end of `requirements.txt`:
```
gspread>=6.0.0
google-auth>=2.30.0
```

- [ ] **Step 2: Install them locally**

Run: `python -m pip install "gspread>=6.0.0" "google-auth>=2.30.0"`
Expected: installs successfully.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: add gspread + google-auth for position persistence

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `positions_store.py` — Google Sheets CRUD

**Files:**
- Create: `tests/test_positions_store.py`
- Create: `positions_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_positions_store.py`:
```python
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
    # id auto-generated (non-empty), entry_date auto-filled (non-empty)
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
    # row 1 = header, so record index 0 -> sheet row 2, index 1 -> sheet row 3
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
    # Empty secrets -> get_worksheet raises PositionsConfigError
    monkeypatch.setattr(positions_store.st, 'secrets', {})
    with pytest.raises(PositionsConfigError):
        positions_store.get_worksheet()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/test_positions_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'positions_store'`.

- [ ] **Step 3: Create `positions_store.py`**

```python
# positions_store.py
"""
positions_store.py - Google Sheets persistence for tracked option positions.

Storage backend is Google Sheets (via gspread) because Streamlit Cloud's local
filesystem is wiped on every redeploy. Secrets required in st.secrets:
    positions_sheet_key = "<sheet id>"
    [gcp_service_account] = { ...service account JSON... }
"""
import uuid
from datetime import date

import streamlit as st

HEADER = ['id', 'ticker', 'strategy', 'strike', 'expiry',
          'entry_premium', 'contracts', 'entry_date']


class PositionsConfigError(Exception):
    """Raised when Google Sheets credentials/secrets are not configured."""


def get_worksheet():
    """Authorize the service account and return the positions worksheet.

    Raises PositionsConfigError if secrets are missing or libs unavailable.
    """
    if 'gcp_service_account' not in st.secrets or 'positions_sheet_key' not in st.secrets:
        raise PositionsConfigError("Missing Google Sheets secrets")
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:
        raise PositionsConfigError("gspread/google-auth not installed") from exc

    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    creds = Credentials.from_service_account_info(
        dict(st.secrets['gcp_service_account']), scopes=scopes
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(st.secrets['positions_sheet_key'])
    return sheet.sheet1


def load_positions() -> list:
    """Return all positions as a list of typed dicts."""
    ws = get_worksheet()
    out = []
    for r in ws.get_all_records():
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


def add_position(pos: dict) -> None:
    """Append a position row, auto-filling id and entry_date when absent."""
    ws = get_worksheet()
    pid = pos.get('id') or str(uuid.uuid4())
    entry_date = pos.get('entry_date') or date.today().isoformat()
    row = [
        pid, pos['ticker'], pos['strategy'], pos['strike'], pos['expiry'],
        pos['entry_premium'], pos['contracts'], entry_date,
    ]
    ws.append_row(row)


def delete_position(position_id: str) -> None:
    """Delete the row whose id matches; no-op if not found."""
    ws = get_worksheet()
    records = ws.get_all_records()
    for idx, rec in enumerate(records, start=2):  # row 1 is the header
        if str(rec.get('id')) == str(position_id):
            ws.delete_rows(idx)
            return
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_positions_store.py -v`
Expected: all PASS. (The missing-secrets test relies on `st.secrets` being patched to `{}`; `'x' not in {}` is True, so it raises before importing gspread.)

- [ ] **Step 5: Commit**

```bash
git add positions_store.py tests/test_positions_store.py
git commit -m "feat: add positions_store for Google Sheets persistence

CRUD over a service-account-authorized sheet; typed load, uuid+date
autofill on add, id-based delete. PositionsConfigError when secrets absent.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `positions.py` — live price + analysis

**Files:**
- Create: `tests/test_positions.py`
- Create: `positions.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_positions.py`:
```python
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
    # (6 - 4) * 100 * 2 contracts = 400
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/test_positions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'positions'`.

- [ ] **Step 3: Create `positions.py`**

```python
# positions.py
"""
positions.py - Live analysis for tracked option positions.

Fetches the exact held contract's current price and reuses the screener's
signal functions plus signals.py's verdict to produce a live SELL/TRIM/HOLD
call and P/L for each stored position.
"""
from datetime import datetime, date

import numpy as np
import yfinance as yf

from screener import _compute_rsi, compute_smc_signals
from signals import compute_exit_rules, compute_sell_verdict


def get_contract_price(ticker: str, strategy: str, strike: float, expiry: str) -> float:
    """Live per-share mid for the exact contract, or np.nan if unavailable."""
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


def _get_history(ticker: str):
    """Download ~3 months of OHLCV. Separated for test mocking."""
    return yf.Ticker(ticker).history(period='3mo')


def _days_to_expiry(expiry: str) -> int:
    exp = datetime.strptime(expiry, '%Y-%m-%d').date()
    return (exp - date.today()).days


def analyze_position(pos: dict) -> dict:
    """Compute live price, P/L, DTE, and SELL/TRIM/HOLD verdict for a position."""
    ticker = pos['ticker']
    strategy = pos['strategy']
    strike = pos['strike']
    expiry = pos['expiry']
    entry = pos['entry_premium']
    contracts = int(pos.get('contracts', 1) or 1)

    dte = _days_to_expiry(expiry)
    current = get_contract_price(ticker, strategy, strike, expiry)

    # Build the signal `row` the verdict needs (rsi + smc_* flags).
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_positions.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (75 prior + new store/positions tests).

- [ ] **Step 6: Commit**

```bash
git add positions.py tests/test_positions.py
git commit -m "feat: add analyze_position for live position verdict + P/L

Fetches the exact contract mid from yfinance, rebuilds rsi+SMC signals
via screener, and runs compute_sell_verdict. P/L = (mid-entry)*100*qty;
graceful when price or signal data is unavailable.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `app.py` — My Positions page + nav, and README setup

**Files:**
- Modify: `app.py`
- Modify: `README.md`

### Step 1: Add imports

- [ ] **Add positions imports after the existing `from signals import ...` line**

Find:
```python
from signals import compute_entry_readiness, compute_exit_rules, compute_sell_verdict
```
Add immediately after:
```python
from positions_store import load_positions, add_position, delete_position, PositionsConfigError
from positions import analyze_position
```

### Step 2: Add `render_positions_page`

- [ ] **Add this function immediately AFTER the `_render_playbook_col` function** (it ends with the `for trigger in exits['tech_triggers']:` loop's `st.markdown(...)` block, before the next top-level statement)

```python
def render_positions_page():
    """Render the My Positions view: add form + live position cards."""
    st.markdown('<div class="section-header">📋 My Positions</div>', unsafe_allow_html=True)

    try:
        positions = load_positions()
    except PositionsConfigError:
        st.warning(
            "Google Sheets is not configured. Add `positions_sheet_key` and "
            "a `[gcp_service_account]` block to your Streamlit secrets. "
            "See the README → **Position Tracking Setup**."
        )
        return

    with st.expander("➕ Add a position", expanded=not positions):
        with st.form("add_position_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                ticker = st.text_input("Ticker").strip().upper()
                strategy = st.selectbox("Type", ["Long Call", "Long Put"])
            with c2:
                strike = st.number_input("Strike", min_value=0.0, step=1.0)
                expiry = st.date_input("Expiry")
            with c3:
                entry_premium = st.number_input("Entry premium ($/share)", min_value=0.0, step=0.05)
                contracts = st.number_input("Contracts", min_value=1, step=1, value=1)
            submitted = st.form_submit_button("Add", type="primary")
            if submitted:
                if not ticker or strike <= 0 or entry_premium <= 0:
                    st.error("Ticker, strike, and entry premium are required.")
                else:
                    add_position({
                        'ticker': ticker,
                        'strategy': strategy,
                        'strike': float(strike),
                        'expiry': expiry.isoformat(),
                        'entry_premium': float(entry_premium),
                        'contracts': int(contracts),
                    })
                    st.success(f"Added {ticker} ${strike:.0f} {strategy}.")
                    st.rerun()

    if not positions:
        st.info("No open positions. Add one above.")
        return

    for pos in positions:
        with st.spinner(f"Analyzing {pos['ticker']}…"):
            data = analyze_position(pos)

        v_emoji, v_text, v_color = _PLAYBOOK_VERDICT[data['verdict']['status']]
        price_str = (f"${data['current_price']:.2f}"
                     if not np.isnan(data['current_price']) else "n/a")

        if data['pnl_usd'] is not None and data['pnl_pct'] is not None:
            pnl_color = '#10b981' if data['pnl_usd'] >= 0 else '#ef4444'
            pnl_str = (f'<span style="color:{pnl_color};font-weight:700;">'
                       f'{data["pnl_pct"] * 100:+.0f}% (${data["pnl_usd"]:+.0f})</span>')
        else:
            pnl_str = '<span style="color:#64748b;">P/L n/a</span>'

        reasons_html = ''.join(
            f'<div style="font-size:0.74rem;color:#94a3b8;">• {r}</div>'
            for r in data['verdict']['reasons']
        ) or '<div style="font-size:0.74rem;color:#64748b;">No exit signals active</div>'

        err_html = (f'<div style="font-size:0.72rem;color:#f59e0b;">⚠ {data["error"]}</div>'
                    if data['error'] else '')

        st.markdown(
            f'<div style="background:#1e293b;border:1px solid {v_color}55;'
            f'border-left:4px solid {v_color};border-radius:0.5rem;'
            f'padding:0.75rem 1rem;margin:0.5rem 0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="font-size:1.0rem;font-weight:700;color:#e2e8f0;">'
            f'{pos["ticker"]} ${pos["strike"]:.0f} {pos["strategy"]} · exp {pos["expiry"]}</div>'
            f'<div style="font-size:1.1rem;font-weight:800;color:{v_color};">{v_emoji} {v_text}</div>'
            f'</div>'
            f'<div style="font-size:0.82rem;color:#cbd5e1;margin-top:0.3rem;">'
            f'{pos["contracts"]}x · entry ${pos["entry_premium"]:.2f} · now {price_str} · '
            f'{data["dte"]} DTE · {pnl_str}</div>'
            f'{reasons_html}{err_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Close position", key=f"close_{pos['id']}"):
            delete_position(pos['id'])
            st.rerun()
```

### Step 3: Add the nav radio inside the sidebar

- [ ] **Add a view selector as the FIRST element inside `with st.sidebar:`**

Find:
```python
with st.sidebar:
    st.markdown("## ⚙️ Screener Settings")
```
Replace with:
```python
with st.sidebar:
    view = st.radio("View", ["🔍 Screener", "📋 My Positions"], key="view_nav")
    st.markdown("---")
    st.markdown("## ⚙️ Screener Settings")
```

### Step 4: Add the view guard after the sidebar block

- [ ] **Insert the routing guard right after the `with st.sidebar:` block ends and before `# SECTION 1: Macro Overview`**

Find:
```python


# ──────────────────────────────────────────────
# SECTION 1: Macro Overview (always visible)
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">🌍 Macro Environment</div>', unsafe_allow_html=True)
```
Replace with:
```python


# ──────────────────────────────────────────────
# Top-level view routing
# ──────────────────────────────────────────────
if view == "📋 My Positions":
    render_positions_page()
    st.stop()


# ──────────────────────────────────────────────
# SECTION 1: Macro Overview (always visible)
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">🌍 Macro Environment</div>', unsafe_allow_html=True)
```

### Step 5: Syntax check

- [ ] **Parse app.py**

Run: `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

### Step 6: Confirm tests still pass

- [ ] **Full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass.

### Step 7: Add README setup section

- [ ] **Append a "Position Tracking Setup" section to `README.md`**

Add at the end of `README.md`:
```markdown

## Position Tracking Setup (Google Sheets)

The **📋 My Positions** view persists positions in a Google Sheet so they survive
Streamlit Cloud redeploys. One-time setup:

1. In [Google Cloud Console](https://console.cloud.google.com/), create (or pick) a project.
2. Enable **Google Sheets API** and **Google Drive API**.
3. Create a **service account**, then create a **JSON key** for it and download it.
4. Create a new Google Sheet. In row 1, add these headers (columns A–H):
   `id`, `ticker`, `strategy`, `strike`, `expiry`, `entry_premium`, `contracts`, `entry_date`
5. Click **Share** on the sheet and give **Editor** access to the service
   account's `client_email` (from the JSON).
6. Copy the sheet ID from its URL (`.../d/<SHEET_ID>/edit`).
7. Add to Streamlit secrets (`.streamlit/secrets.toml` locally, or the app's
   **Secrets** on Streamlit Cloud):

   ```toml
   positions_sheet_key = "<SHEET_ID>"

   [gcp_service_account]
   type = "service_account"
   project_id = "..."
   private_key_id = "..."
   private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
   client_email = "...@....iam.gserviceaccount.com"
   client_id = "..."
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "..."
   ```

Until secrets are set, the My Positions view shows a setup prompt instead of crashing.
```

- [ ] **Step 8: Commit**

```bash
git add app.py README.md
git commit -m "feat: add My Positions view with live verdict + Google Sheets

Sidebar radio routes to a positions page: add form persists to the
sheet, each open position renders a live SELL/TRIM/HOLD card with P/L
and a close button. README documents the Google Sheets setup.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - Google Sheets persistence (CRUD, schema, PositionsConfigError) → Task 2
  - Exact-contract live price → Task 3 `get_contract_price`
  - Reuse rsi + SMC + verdict, `smc_` remap → Task 3 `analyze_position`
  - P/L ×100×contracts → Task 3 + test `test_pnl_usd_uses_100x_multiplier`
  - Sidebar radio nav, screener body skipped via `st.stop()` → Task 4 Steps 3–4
  - Add form + live cards + close → Task 4 Step 2 `render_positions_page`
  - Config-missing friendly message → Task 4 Step 2 (catches PositionsConfigError)
  - Price/data unavailable fallback → Task 3 (`error`, condition-only verdict) + Task 4 card `err_html`
  - Deps gspread/google-auth → Task 1
  - README setup → Task 4 Step 7
  - Tests → Task 2 & 3

- [x] **No placeholders:** every step has runnable code/commands.

- [x] **Type consistency:**
  - `load_positions`/`add_position`/`delete_position` signatures match between `positions_store.py`, its tests, and `app.py` call sites.
  - `analyze_position` returns `{current_price, pnl_pct, pnl_usd, dte, verdict, error}` — consumed with exactly those keys in `render_positions_page`.
  - `verdict` is the `compute_sell_verdict` dict (`status`/`reasons`/`pnl_pct`); `_PLAYBOOK_VERDICT[data['verdict']['status']]` keys (hold/trim/sell) match.
  - `compute_smc_signals` keys remapped to `smc_*` before `compute_exit_rules` consumes them.
  - Both `rec['cost']` (current mid) and `entry_premium` are per-share, so `compute_sell_verdict`'s ratio is correct; ×100 applies only to `pnl_usd`.
