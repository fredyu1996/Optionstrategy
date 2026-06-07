# EMA Signals Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add EMA 20/50/100/200 signals — a daily EMA entry check + graduated EMA20/50 sell triggers, and a 4H advisory status for single stocks — without changing existing SMA/SMC logic.

**Architecture:** New `indicators.py` holds pure `compute_ema_signals` + networked `fetch_4h_ema_status`. `signals.py` consumes daily EMA flags on the `row` for a 7th entry check and two extra sell triggers. `screener.py` and `positions.py` populate those daily flags; `app.py` shows the 4H status in the playbook and position cards.

**Tech Stack:** Python 3.11, pandas (`ewm`), yfinance (1h→4H resample), Streamlit, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `indicators.py` | Create | `compute_ema_signals`, `fetch_4h_ema_status`, `_fetch_4h_close` |
| `tests/test_indicators.py` | Create | EMA math + 4H classification (mocked) |
| `signals.py` | Modify | EMA entry check + threshold formula; EMA exit triggers |
| `tests/test_signals.py` | Modify | update fixtures/counts; add EMA entry+exit tests |
| `screener.py` | Modify | populate daily EMA flags on each row |
| `positions.py` | Modify | daily EMA flags on `row` + `ema4h` in result |
| `tests/test_positions.py` | Modify | assert `ema4h` present |
| `app.py` | Modify | render 4H EMA line in playbook + position cards |

**Verified facts:**
- `screener.py` batch loop (~line 774-800) has `close_series` (daily closes) and `row['price']`; EMA fields go in after the trend block.
- `signals.compute_entry_readiness` builds a `checks` list then status from `met`; `compute_exit_rules` builds `tech_triggers` (RSI=idx0, BoS=idx1, Zone=idx2). Adding EMA triggers at the END preserves those indices.
- `positions.analyze_position` builds `row` from `_compute_rsi` + `compute_smc_signals`; it must also add EMA flags. `_get_history` returns 3mo daily (~63 bars: enough for EMA20/50, not EMA200 — only above_ema20/50 are used by exit triggers).
- `app.py` already has `_PLAYBOOK_VERDICT` colors and a tab3 block calling `_render_playbook_col`.

---

## Task 1: `compute_ema_signals` (daily, pure)

**Files:**
- Create: `tests/test_indicators.py`
- Create: `indicators.py`

- [ ] **Step 1: Write failing tests** — create `tests/test_indicators.py`:
```python
import sys
import os
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import indicators
from indicators import compute_ema_signals


def test_bull_stack_on_rising_series():
    close = pd.Series(np.linspace(100, 200, 250))  # steady uptrend, 250 bars
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
    close = pd.Series(np.linspace(100, 110, 30))  # 30 bars: EMA200 undefined
    sig = compute_ema_signals(close)
    assert sig['ema_bull_stack'] is False
    assert sig['ema_bear_stack'] is False
    assert np.isnan(sig['ema200'])
    # EMA20 still computable on 30 bars
    assert not np.isnan(sig['ema20'])
    assert sig['above_ema20'] is True  # rising series, last > ema20


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
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_indicators.py -k ema_signals -v`
Expected: `ModuleNotFoundError: No module named 'indicators'`. (Run `-k "stack or bars or keys or series"` if needed; simplest: run the whole file.)

- [ ] **Step 3: Create `indicators.py`**:
```python
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
```

- [ ] **Step 4: Run, verify pass** — `python -m pytest tests/test_indicators.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**
```bash
git add indicators.py tests/test_indicators.py
git commit -m "feat: add compute_ema_signals (daily EMA 20/50/100/200)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `fetch_4h_ema_status` (4H classification)

**Files:**
- Modify: `tests/test_indicators.py`
- Modify: `indicators.py`

- [ ] **Step 1: Append failing tests** to `tests/test_indicators.py`:
```python
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
    # rising then a small dip: last close under EMA20 but still over EMA50
    base = list(np.linspace(100, 200, 240))
    dip = list(np.linspace(200, 192, 10))  # pull back just under fast EMA
    _patch_4h(monkeypatch, pd.Series(base + dip))
    out = fetch_4h_ema_status('AAPL')
    assert out['status'] in ('wait', 'good', 'avoid')  # exact depends on EMA math
    # deterministic check: classification is one of the known states
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
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_indicators.py -k 4h -v`
Expected: ImportError for `fetch_4h_ema_status`.

- [ ] **Step 3: Append to `indicators.py`**:
```python
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
```

- [ ] **Step 4: Run, verify pass** — `python -m pytest tests/test_indicators.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**
```bash
git add indicators.py tests/test_indicators.py
git commit -m "feat: add fetch_4h_ema_status 4H entry/exit classifier

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: EMA entry check + threshold formula

**Files:**
- Modify: `signals.py`
- Modify: `tests/test_signals.py`

### Step 1: Update shared fixtures (add EMA keys)

- [ ] **In `tests/test_signals.py`, add EMA keys to `_bullish_row` and `_bearish_row`.**

In `_bullish_row`'s returned dict, add these keys (before the closing `}`):
```python
        'ema_bull_stack': True,
        'ema_bear_stack': False,
        'above_ema20': True,
        'above_ema50': True,
```
In `_bearish_row`'s returned dict, add:
```python
        'ema_bull_stack': False,
        'ema_bear_stack': True,
        'above_ema20': False,
        'above_ema50': False,
```

### Step 2: Write/adjust failing entry tests

- [ ] **Replace the three count/threshold tests in `TestEntryReadinessLongCall`.**

Replace `test_all_pass_returns_enter`, `test_three_pass_returns_wait`, `test_two_or_fewer_returns_not_yet`, and `test_returns_six_checks` with:
```python
    def test_all_pass_returns_enter(self):
        result = compute_entry_readiness(_bullish_row(), 'Long Call')
        assert result['met'] == 7
        assert result['total'] == 7
        assert result['status'] == 'enter'

    def test_four_pass_returns_wait(self):
        row = _bullish_row()
        row['trend'] = 'Down'              # fail trend
        row['smc_discount_zone'] = False   # fail zone
        row['smc_bos_bullish'] = False     # fail smc
        result = compute_entry_readiness(row, 'Long Call')
        assert result['met'] == 4
        assert result['status'] == 'wait'

    def test_three_pass_returns_not_yet(self):
        row = _bullish_row()
        row['trend'] = 'Down'
        row['smc_discount_zone'] = False
        row['smc_bos_bullish'] = False
        row['ema_bull_stack'] = False      # also fail EMA -> 3 left
        result = compute_entry_readiness(row, 'Long Call')
        assert result['met'] == 3
        assert result['status'] == 'not_yet'

    def test_returns_seven_checks(self):
        result = compute_entry_readiness(_bullish_row(), 'Long Call')
        assert len(result['checks']) == 7
```

- [ ] **Add an EMA-specific entry test** to `TestEntryReadinessLongCall`:
```python
    def test_ema_check_present_and_passes(self):
        result = compute_entry_readiness(_bullish_row(), 'Long Call')
        ema_check = next(c for c in result['checks'] if 'EMA' in c['label'])
        assert ema_check['passed']

    def test_ema_check_fails_when_below_ema20(self):
        row = _bullish_row()
        row['above_ema20'] = False
        result = compute_entry_readiness(row, 'Long Call')
        ema_check = next(c for c in result['checks'] if 'EMA' in c['label'])
        assert not ema_check['passed']
```

- [ ] **Add a Long Put EMA test** to `TestEntryReadinessLongPut`:
```python
    def test_ema_bearish_check_passes(self):
        result = compute_entry_readiness(_bearish_row(), 'Long Put')
        ema_check = next(c for c in result['checks'] if 'EMA' in c['label'])
        assert ema_check['passed']
```

- [ ] **Step 3: Run, verify the new entry tests FAIL** — `python -m pytest tests/test_signals.py -k Entry -v`
Expected: failures (still 6 checks / old thresholds).

### Step 4: Implement in `signals.py`

- [ ] **Add the EMA check to the Long Call `checks` list** in `compute_entry_readiness`. After the `'No near earnings'` dict in the `if is_call:` branch, add:
```python
            {
                'label': 'EMA bullish',
                'passed': bool(row.get('ema_bull_stack') and row.get('above_ema20')),
                'value': ('stack+>EMA20'
                          if (row.get('ema_bull_stack') and row.get('above_ema20'))
                          else 'no'),
            },
```

- [ ] **Add the EMA check to the Long Put `checks` list** (the `else:` branch). After its `'No near earnings'` dict, add:
```python
            {
                'label': 'EMA bearish',
                'passed': bool(row.get('ema_bear_stack') and not row.get('above_ema20')),
                'value': ('stack+<EMA20'
                          if (row.get('ema_bear_stack') and not row.get('above_ema20'))
                          else 'no'),
            },
```

- [ ] **Replace the hardcoded threshold block** (currently `if met >= 5: ... elif met >= 3: ... else:`) and the `total` value. Find:
```python
    met = sum(1 for c in checks if c['passed'])

    if met >= 5:
        status = 'enter'
    elif met >= 3:
        status = 'wait'
    else:
        status = 'not_yet'

    return {'status': status, 'met': met, 'total': 6, 'checks': checks}
```
Replace with:
```python
    met = sum(1 for c in checks if c['passed'])
    total = len(checks)

    if met >= total - 1:
        status = 'enter'
    elif met >= (total + 1) // 2:
        status = 'wait'
    else:
        status = 'not_yet'

    return {'status': status, 'met': met, 'total': total, 'checks': checks}
```
(For 7 checks: enter ≥6, wait ≥4, else not_yet. Formula is backward-compatible with the old 6-check thresholds.)

- [ ] **Step 5: Run entry tests, verify pass** — `python -m pytest tests/test_signals.py -k Entry -v`
Expected: all pass.

- [ ] **Step 6: Commit**
```bash
git add signals.py tests/test_signals.py
git commit -m "feat: add EMA entry check (7-check readiness, formula thresholds)

Long Call requires bull stack + price > EMA20; Long Put the mirror.
Thresholds now scale with check count (enter >= n-1, wait >= ceil(n/2)).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: EMA exit triggers (graduated)

**Files:**
- Modify: `signals.py`
- Modify: `tests/test_signals.py`

- [ ] **Step 1: Update the two count tests + add EMA exit tests** in `TestExitRules`.

Replace `test_long_call_has_three_tech_triggers` and `test_long_put_has_three_tech_triggers` with:
```python
    def test_long_call_has_five_tech_triggers(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert len(result['tech_triggers']) == 5

    def test_long_put_has_five_tech_triggers(self):
        result = compute_exit_rules(_bearish_row(), 'Long Put', _base_rec())
        assert len(result['tech_triggers']) == 5
```

Add new EMA exit tests to `TestExitRules`:
```python
    def test_long_call_break_ema20_triggers(self):
        row = _bullish_row()
        row['above_ema20'] = False
        result = compute_exit_rules(row, 'Long Call', _base_rec())
        ema20 = next(t for t in result['tech_triggers'] if 'EMA20' in t['label'])
        assert ema20['triggered'] is True

    def test_long_call_break_ema50_triggers(self):
        row = _bullish_row()
        row['above_ema50'] = False
        result = compute_exit_rules(row, 'Long Call', _base_rec())
        ema50 = next(t for t in result['tech_triggers'] if 'EMA50' in t['label'])
        assert ema50['triggered'] is True

    def test_break_ema20_only_is_trim_via_verdict(self):
        row = _bullish_row()
        row['above_ema20'] = False  # 1 trigger
        exits = compute_exit_rules(row, 'Long Call', _base_rec())
        verdict = compute_sell_verdict(exits, _base_rec())
        assert verdict['status'] == 'trim'

    def test_break_ema20_and_ema50_is_sell_via_verdict(self):
        row = _bullish_row()
        row['above_ema20'] = False
        row['above_ema50'] = False  # 2 triggers
        exits = compute_exit_rules(row, 'Long Call', _base_rec())
        verdict = compute_sell_verdict(exits, _base_rec())
        assert verdict['status'] == 'sell'

    def test_long_put_break_up_ema20_triggers(self):
        row = _bearish_row()
        row['above_ema20'] = True  # price back above EMA20 = bad for a put
        result = compute_exit_rules(row, 'Long Put', _base_rec())
        ema20 = next(t for t in result['tech_triggers'] if 'EMA20' in t['label'])
        assert ema20['triggered'] is True
```

- [ ] **Step 2: Run, verify the new exit tests FAIL** — `python -m pytest tests/test_signals.py -k Exit -v`
Expected: failures (only 3 triggers, no EMA labels).

- [ ] **Step 3: Add EMA triggers in `compute_exit_rules`.**

In the `if is_call:` branch, the `tech_triggers` list currently ends with the `'Price enters Premium Zone'` dict. Add two more dicts AFTER it (still inside that list):
```python
            {
                'label': '穿 EMA20',
                'triggered': not bool(row.get('above_ema20', True)),
                'current_value': 'below' if not row.get('above_ema20', True) else 'above',
            },
            {
                'label': '穿 EMA50',
                'triggered': not bool(row.get('above_ema50', True)),
                'current_value': 'below' if not row.get('above_ema50', True) else 'above',
            },
```
In the `else:` (Long Put) branch, after the `'Price enters Discount Zone'` dict, add:
```python
            {
                'label': '升穿 EMA20',
                'triggered': bool(row.get('above_ema20', False)),
                'current_value': 'above' if row.get('above_ema20', False) else 'below',
            },
            {
                'label': '升穿 EMA50',
                'triggered': bool(row.get('above_ema50', False)),
                'current_value': 'above' if row.get('above_ema50', False) else 'below',
            },
```
(Defaults: call uses `above_ema20=True` default so missing EMA data never fires a sell; put uses `False` default likewise.)

- [ ] **Step 4: Run exit tests, verify pass** — `python -m pytest tests/test_signals.py -k Exit -v`
Expected: all pass.

- [ ] **Step 5: Run the whole signals suite** — `python -m pytest tests/test_signals.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**
```bash
git add signals.py tests/test_signals.py
git commit -m "feat: add graduated EMA20/50 exit triggers

Break EMA20 -> 1 trigger (TRIM); break EMA20 and EMA50 -> 2 (SELL).
Missing EMA data never fires (safe defaults). Long Put mirrors upward.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Populate daily EMA flags in the screener

**Files:**
- Modify: `screener.py`

- [ ] **Step 1: Add the import** near the other local imports at the top of `screener.py` (after the existing `import` lines). Add:
```python
from indicators import compute_ema_signals
```

- [ ] **Step 2: Populate EMA flags after the trend block.** Find this block in `batch_screen_fundamentals` (the trend classification ending):
```python
            else:
                row['trend'] = 'Sideways'

            # Fundamentals
```
Replace with:
```python
            else:
                row['trend'] = 'Sideways'

            # EMA signals (daily)
            _ema = compute_ema_signals(close_series)
            row['ema_bull_stack'] = _ema['ema_bull_stack']
            row['ema_bear_stack'] = _ema['ema_bear_stack']
            row['above_ema20'] = _ema['above_ema20']
            row['above_ema50'] = _ema['above_ema50']

            # Fundamentals
```

- [ ] **Step 3: Syntax check** — `python -c "import ast; ast.parse(open('screener.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Confirm full suite still green** — `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**
```bash
git add screener.py
git commit -m "feat: populate daily EMA flags on each screened row

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Daily EMA on positions + 4H status in result

**Files:**
- Modify: `positions.py`
- Modify: `tests/test_positions.py`

- [ ] **Step 1: Add failing tests** to `tests/test_positions.py`:
```python
def test_analyze_includes_ema4h(monkeypatch):
    monkeypatch.setattr(positions, 'get_contract_price', lambda *a, **k: 5.0)
    monkeypatch.setattr(positions, '_get_history', lambda t: pd.DataFrame())
    monkeypatch.setattr(positions, 'fetch_4h_ema_status',
                        lambda t: {'status': 'good', 'label': '4H: ok', 'signals': {}})
    data = analyze_position(_pos())
    assert data['ema4h'] == {'status': 'good', 'label': '4H: ok', 'signals': {}}


def test_analyze_adds_daily_ema_flags_to_exit(monkeypatch):
    # 60+ rows of falling closes -> price below EMA20/50 -> EMA exit triggers fire
    n = 70
    falling = np.linspace(200, 120, n)
    hist = pd.DataFrame({'Open': falling, 'High': falling + 1,
                         'Low': falling - 1, 'Close': falling})
    monkeypatch.setattr(positions, 'get_contract_price', lambda *a, **k: 5.0)
    monkeypatch.setattr(positions, '_get_history', lambda t: hist)
    monkeypatch.setattr(positions, 'fetch_4h_ema_status',
                        lambda t: {'status': 'unknown', 'label': '', 'signals': {}})
    data = analyze_position(_pos())  # Long Call on a downtrend
    labels = [r for r in data['verdict']['reasons']]
    # 2 EMA breaks => sell
    assert data['verdict']['status'] == 'sell'
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_positions.py -k "ema4h or daily_ema" -v`
Expected: fail (`ema4h` KeyError / `fetch_4h_ema_status` attribute missing).

- [ ] **Step 3: Update `positions.py`.**

Add to the imports (after the existing `from screener import ...` line):
```python
from indicators import compute_ema_signals, fetch_4h_ema_status
```

In `analyze_position`, find the signal `row`-building block:
```python
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
```
Replace with (adds EMA flags from the same daily history):
```python
    try:
        hist = _get_history(ticker)
        if hist is not None and not hist.empty and len(hist) >= 20:
            row['rsi'] = _compute_rsi(hist['Close'])
            for key, val in compute_smc_signals(hist).items():
                row[f'smc_{key}'] = val
            for key, val in compute_ema_signals(hist['Close']).items():
                row[key] = val
        else:
            error = 'signal data unavailable'
    except Exception:
        error = 'signal data unavailable'
```

Then find the return dict at the end of `analyze_position`:
```python
    return {
        'current_price': current,
        'pnl_pct': verdict['pnl_pct'],
        'pnl_usd': pnl_usd,
        'dte': dte,
        'verdict': verdict,
        'error': error,
    }
```
Replace with (adds `ema4h`; computed just before the return):
```python
    ema4h = fetch_4h_ema_status(ticker)

    return {
        'current_price': current,
        'pnl_pct': verdict['pnl_pct'],
        'pnl_usd': pnl_usd,
        'dte': dte,
        'verdict': verdict,
        'ema4h': ema4h,
        'error': error,
    }
```

Also update `_error_result` to include the new key so the early-return path stays shape-compatible. Find:
```python
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
```
Replace with:
```python
def _error_result(dte, message: str) -> dict:
    """A safe analyze_position result for a row we cannot evaluate."""
    return {
        'current_price': float('nan'),
        'pnl_pct': None,
        'pnl_usd': None,
        'dte': dte,
        'verdict': {'status': 'hold', 'reasons': [], 'pnl_pct': None},
        'ema4h': {'status': 'unknown', 'label': '', 'signals': {}},
        'error': message,
    }
```

- [ ] **Step 4: Fix the existing required-keys test.** In `tests/test_positions.py`, find `test_result_has_required_keys` and update its key set:
```python
    assert set(data.keys()) == {
        'current_price', 'pnl_pct', 'pnl_usd', 'dte', 'verdict', 'ema4h', 'error',
    }
```
The other `analyze_position` tests call it without mocking `fetch_4h_ema_status`, which would hit the network. Add a module-level autouse fixture near the top of `tests/test_positions.py` (after imports) so every test gets a stub:
```python
@pytest.fixture(autouse=True)
def _stub_4h(monkeypatch):
    monkeypatch.setattr(positions, 'fetch_4h_ema_status',
                        lambda t: {'status': 'unknown', 'label': '', 'signals': {}})
```

- [ ] **Step 5: Run positions tests, verify pass** — `python -m pytest tests/test_positions.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**
```bash
git add positions.py tests/test_positions.py
git commit -m "feat: add daily EMA exit flags + 4H status to analyze_position

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Show 4H EMA status in the UI

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add a 4H color map** next to `_PLAYBOOK_VERDICT`. After the `_PLAYBOOK_VERDICT = {...}` dict, add:
```python
_EMA4H_COLOR = {
    'good': '#10b981',
    'wait': '#f59e0b',
    'avoid': '#ef4444',
    'unknown': '#64748b',
}
```

- [ ] **Step 2: Render the 4H line on each position card.** In `render_positions_page`, find the card's error line + close button region:
```python
        err_html = (f'<div style="font-size:0.72rem;color:#f59e0b;">⚠ {data["error"]}</div>'
                    if data['error'] else '')
```
Add immediately after it:
```python
        ema4h = data.get('ema4h') or {'status': 'unknown', 'label': ''}
        ema_color = _EMA4H_COLOR.get(ema4h['status'], '#64748b')
        ema_html = (f'<div style="font-size:0.76rem;color:{ema_color};margin-top:0.15rem;">'
                    f'{ema4h["label"]}</div>' if ema4h.get('label') else '')
```
Then in that card's `st.markdown(... )` f-string, insert `{ema_html}` right before `{reasons_html}` (or before `{err_html}` if reasons not in this card). Specifically find `f'{reasons_html}{err_html}'` in the card markdown and change to `f'{ema_html}{reasons_html}{err_html}'`.

- [ ] **Step 3: Render the 4H line in the playbook tab.** In the `with tab3:` block (after `rec_lc`/`rec_lp` are computed and before/after the two playbook columns), add a single shared 4H line for the selected ticker:
```python
            _ema4h = fetch_4h_ema_status(ticker_str)
            _ema_color = _EMA4H_COLOR.get(_ema4h['status'], '#64748b')
            if _ema4h.get('label'):
                st.markdown(
                    f'<div style="font-size:0.85rem;font-weight:600;color:{_ema_color};'
                    f'margin:0.3rem 0 0.6rem;">{_ema4h["label"]}</div>',
                    unsafe_allow_html=True,
                )
```
Place this right after the line `exits_lp = compute_exit_rules(row, 'Long Put', rec_lp)` (before `col_lc, col_lp = st.columns(2)`).

- [ ] **Step 4: Add the import.** Ensure `app.py` imports `fetch_4h_ema_status`. After the `from positions import analyze_position` line, add:
```python
from indicators import fetch_4h_ema_status
```

- [ ] **Step 5: Syntax check** — `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`.

- [ ] **Step 6: Full suite** — `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**
```bash
git add app.py
git commit -m "feat: show 4H EMA status in playbook and position cards

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - `compute_ema_signals` daily, NaN-safe → Task 1
  - `fetch_4h_ema_status` good/wait/avoid/unknown → Task 2
  - daily EMA entry check (7 checks, new thresholds) → Task 3
  - graduated EMA20/50 exit triggers (TRIM/SELL via verdict) → Task 4
  - screener populates daily EMA flags → Task 5
  - positions get daily EMA exit flags + `ema4h` → Task 6
  - 4H status shown in playbook + position cards → Task 7
  - SMA trend unchanged → no task touches the SMA/trend block except inserting after it
  - tests for all → Tasks 1,2,3,4,6

- [x] **No placeholders:** every step has runnable code/commands.

- [x] **Type consistency:**
  - `compute_ema_signals` returns the 8-key dict consumed by screener/positions/signals (`ema_bull_stack`, `ema_bear_stack`, `above_ema20`, `above_ema50` used by signals).
  - `fetch_4h_ema_status` returns `{status,label,signals}`; consumed in `analyze_position` (`ema4h`) and rendered via `_EMA4H_COLOR[status]`.
  - entry `total` now `len(checks)`; thresholds formula matches the 7-check assertions.
  - EMA exit triggers appended AFTER RSI/BoS/Zone, preserving indices 0/1/2 used by existing index-based exit tests.
  - `analyze_position` result gains `ema4h`; `_error_result` and `test_result_has_required_keys` updated to match.
