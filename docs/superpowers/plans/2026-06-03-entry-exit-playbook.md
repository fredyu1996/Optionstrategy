# Entry/Exit Playbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "🎯 Entry/Exit Playbook" tab to the Strategy Detail view showing entry readiness badge + exit rules for Long Call and Long Put.

**Architecture:** New `signals.py` module computes entry readiness (6-check score → 🟢/🟡/🔴 badge) and exit rules (price targets, tech triggers, time rule) from existing screener row data. `app.py` adds a third tab that renders two columns (Long Call | Long Put) using these results. No new API calls — reuses cached `get_strike_recommendation()` data.

**Tech Stack:** Python 3.11, Streamlit, numpy, yfinance (no new deps)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `signals.py` | **Create** | `compute_entry_readiness`, `compute_exit_rules`, `_active_smc_labels` |
| `tests/test_signals.py` | **Create** | Unit tests for both public functions |
| `app.py` | **Modify** | Add import, `_render_playbook_col` helper, tab3 block |

---

## Task 1: Write failing tests for `signals.py`

**Files:**
- Create: `tests/test_signals.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_signals.py
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from signals import compute_entry_readiness, compute_exit_rules


def _bullish_row():
    return {
        'trend': 'Up',
        'rsi': 42.0,
        'iv_hv_ratio': 0.7,
        'days_to_earnings': None,
        'smc_bos_bullish': True,
        'smc_bos_bearish': False,
        'smc_choch_bullish': False,
        'smc_choch_bearish': False,
        'smc_discount_zone': True,
        'smc_premium_zone': False,
        'smc_near_bullish_ob': False,
        'smc_near_bearish_ob': False,
        'smc_in_bullish_fvg': False,
        'smc_in_bearish_fvg': False,
    }


def _bearish_row():
    return {
        'trend': 'Strong Down',
        'rsi': 58.0,
        'iv_hv_ratio': 0.7,
        'days_to_earnings': None,
        'smc_bos_bullish': False,
        'smc_bos_bearish': True,
        'smc_choch_bullish': False,
        'smc_choch_bearish': False,
        'smc_discount_zone': False,
        'smc_premium_zone': True,
        'smc_near_bullish_ob': False,
        'smc_near_bearish_ob': False,
        'smc_in_bullish_fvg': False,
        'smc_in_bearish_fvg': False,
    }


def _base_rec():
    return {'cost': 420.0, 'dte': 45, 'strike': 150.0}


# ── compute_entry_readiness ──────────────────────────────────────────────────

class TestEntryReadinessLongCall:
    def test_all_pass_returns_enter(self):
        result = compute_entry_readiness(_bullish_row(), 'Long Call')
        assert result['met'] == 6
        assert result['total'] == 6
        assert result['status'] == 'enter'

    def test_three_pass_returns_wait(self):
        row = _bullish_row()
        row['trend'] = 'Down'
        row['smc_discount_zone'] = False
        row['smc_bos_bullish'] = False
        result = compute_entry_readiness(row, 'Long Call')
        assert result['met'] == 3
        assert result['status'] == 'wait'

    def test_two_or_fewer_returns_not_yet(self):
        row = _bullish_row()
        row['trend'] = 'Strong Down'
        row['rsi'] = 72.0
        row['smc_discount_zone'] = False
        row['smc_bos_bullish'] = False
        row['iv_hv_ratio'] = 1.5
        result = compute_entry_readiness(row, 'Long Call')
        assert result['met'] <= 2
        assert result['status'] == 'not_yet'

    def test_returns_six_checks(self):
        result = compute_entry_readiness(_bullish_row(), 'Long Call')
        assert len(result['checks']) == 6

    def test_nan_rsi_fails_rsi_check(self):
        row = _bullish_row()
        row['rsi'] = float('nan')
        result = compute_entry_readiness(row, 'Long Call')
        rsi_check = next(c for c in result['checks'] if 'RSI' in c['label'])
        assert not rsi_check['passed']

    def test_near_earnings_fails_earnings_check(self):
        row = _bullish_row()
        row['days_to_earnings'] = 10
        result = compute_entry_readiness(row, 'Long Call')
        earn_check = next(c for c in result['checks'] if 'earnings' in c['label'].lower())
        assert not earn_check['passed']

    def test_smc_fvg_satisfies_smc_check(self):
        row = _bullish_row()
        row['smc_bos_bullish'] = False
        row['smc_near_bullish_ob'] = False
        row['smc_in_bullish_fvg'] = True
        result = compute_entry_readiness(row, 'Long Call')
        smc_check = next(c for c in result['checks'] if 'SMC' in c['label'])
        assert smc_check['passed']

    def test_checks_have_required_keys(self):
        result = compute_entry_readiness(_bullish_row(), 'Long Call')
        for check in result['checks']:
            assert 'label' in check
            assert 'passed' in check
            assert 'value' in check


class TestEntryReadinessLongPut:
    def test_all_pass_returns_enter(self):
        result = compute_entry_readiness(_bearish_row(), 'Long Put')
        assert result['met'] == 6
        assert result['status'] == 'enter'

    def test_uptrend_fails_trend_check(self):
        row = _bearish_row()
        row['trend'] = 'Strong Up'
        result = compute_entry_readiness(row, 'Long Put')
        trend_check = next(c for c in result['checks'] if c['label'] == 'Trend')
        assert not trend_check['passed']

    def test_rsi_below_50_fails_put_rsi_check(self):
        row = _bearish_row()
        row['rsi'] = 45.0
        result = compute_entry_readiness(row, 'Long Put')
        rsi_check = next(c for c in result['checks'] if 'RSI' in c['label'])
        assert not rsi_check['passed']


# ── compute_exit_rules ───────────────────────────────────────────────────────

class TestExitRules:
    def test_price_targets_computed_from_cost(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert result['take_profit_usd'] == pytest.approx(840.0)
        assert result['stop_loss_usd'] == pytest.approx(210.0)
        assert result['take_profit_pct'] == 1.0
        assert result['stop_loss_pct'] == 0.5

    def test_time_exit_dte_is_21(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert result['time_exit_dte'] == 21

    def test_time_exit_date_set_when_dte_above_21(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert result['time_exit_date'] is not None

    def test_time_exit_msg_when_dte_below_threshold(self):
        rec = {'cost': 420.0, 'dte': 10, 'strike': 150.0}
        result = compute_exit_rules(_bullish_row(), 'Long Call', rec)
        assert 'Exit now' in result['time_exit_msg']

    def test_nan_cost_returns_nan_targets(self):
        rec = {'cost': float('nan'), 'dte': 45, 'strike': 150.0}
        result = compute_exit_rules(_bullish_row(), 'Long Call', rec)
        assert np.isnan(result['take_profit_usd'])
        assert np.isnan(result['stop_loss_usd'])

    def test_long_call_has_three_tech_triggers(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert len(result['tech_triggers']) == 3

    def test_long_put_has_three_tech_triggers(self):
        result = compute_exit_rules(_bearish_row(), 'Long Put', _base_rec())
        assert len(result['tech_triggers']) == 3

    def test_long_call_rsi_trigger_fires_above_70(self):
        row = _bullish_row()
        row['rsi'] = 75.0
        result = compute_exit_rules(row, 'Long Call', _base_rec())
        rsi_trigger = result['tech_triggers'][0]
        assert rsi_trigger['triggered'] is True

    def test_long_call_bearish_bos_trigger(self):
        row = _bullish_row()
        row['smc_bos_bearish'] = True
        result = compute_exit_rules(row, 'Long Call', _base_rec())
        bos_trigger = result['tech_triggers'][1]
        assert bos_trigger['triggered'] is True

    def test_long_call_premium_zone_trigger(self):
        row = _bullish_row()
        row['smc_premium_zone'] = True
        result = compute_exit_rules(row, 'Long Call', _base_rec())
        zone_trigger = result['tech_triggers'][2]
        assert zone_trigger['triggered'] is True

    def test_long_put_rsi_trigger_fires_below_30(self):
        row = _bearish_row()
        row['rsi'] = 25.0
        result = compute_exit_rules(row, 'Long Put', _base_rec())
        rsi_trigger = result['tech_triggers'][0]
        assert rsi_trigger['triggered'] is True

    def test_long_put_bullish_bos_trigger(self):
        row = _bearish_row()
        row['smc_bos_bullish'] = True
        result = compute_exit_rules(row, 'Long Put', _base_rec())
        bos_trigger = result['tech_triggers'][1]
        assert bos_trigger['triggered'] is True

    def test_long_put_discount_zone_trigger(self):
        row = _bearish_row()
        row['smc_discount_zone'] = True
        result = compute_exit_rules(row, 'Long Put', _base_rec())
        zone_trigger = result['tech_triggers'][2]
        assert zone_trigger['triggered'] is True

    def test_tech_triggers_have_required_keys(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        for t in result['tech_triggers']:
            assert 'label' in t
            assert 'triggered' in t
            assert 'current_value' in t

    def test_no_triggers_active_when_conditions_safe(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert all(not t['triggered'] for t in result['tech_triggers'])
```

- [ ] **Step 2: Run tests, verify they all fail with ImportError**

```
cd C:\Users\fredy\Optionstrategy
pytest tests/test_signals.py -v
```

Expected: All tests fail with `ModuleNotFoundError: No module named 'signals'`

---

## Task 2: Implement `signals.py`

**Files:**
- Create: `signals.py`

- [ ] **Step 1: Create `signals.py`**

```python
# signals.py
"""
signals.py - Entry readiness and exit rules for Long Call / Long Put positions.
"""
from datetime import datetime, timedelta
import numpy as np


def compute_entry_readiness(row: dict, strategy: str) -> dict:
    """
    Evaluate 6 entry conditions for Long Call or Long Put.

    Returns:
        status: 'enter' | 'wait' | 'not_yet'
        met: int
        total: int  (always 6)
        checks: list of {label, passed, value}
    """
    is_call = (strategy == 'Long Call')
    rsi = row.get('rsi', np.nan)
    iv_hv = row.get('iv_hv_ratio', np.nan)
    trend = row.get('trend', 'Unknown')
    days_earn = row.get('days_to_earnings', None)

    if is_call:
        checks = [
            {
                'label': 'Trend',
                'passed': trend in ('Up', 'Strong Up'),
                'value': trend,
            },
            {
                'label': 'RSI < 50',
                'passed': not (isinstance(rsi, float) and np.isnan(rsi)) and rsi < 50,
                'value': f"{rsi:.0f}" if not (isinstance(rsi, float) and np.isnan(rsi)) else 'N/A',
            },
            {
                'label': 'Bullish SMC signal',
                'passed': bool(
                    row.get('smc_bos_bullish')
                    or row.get('smc_near_bullish_ob')
                    or row.get('smc_in_bullish_fvg')
                ),
                'value': _active_smc_labels(
                    row, ['smc_bos_bullish', 'smc_near_bullish_ob', 'smc_in_bullish_fvg']
                ) or 'none',
            },
            {
                'label': 'Discount Zone',
                'passed': bool(row.get('smc_discount_zone')),
                'value': 'active' if row.get('smc_discount_zone') else 'not active',
            },
            {
                'label': 'IV/HV < 1.0',
                'passed': not (isinstance(iv_hv, float) and np.isnan(iv_hv)) and iv_hv < 1.0,
                'value': f"{iv_hv:.2f}" if not (isinstance(iv_hv, float) and np.isnan(iv_hv)) else 'N/A',
            },
            {
                'label': 'No near earnings',
                'passed': days_earn is None or days_earn > 14,
                'value': f"{days_earn}d" if days_earn is not None else 'none',
            },
        ]
    else:
        checks = [
            {
                'label': 'Trend',
                'passed': trend in ('Down', 'Strong Down'),
                'value': trend,
            },
            {
                'label': 'RSI > 50',
                'passed': not (isinstance(rsi, float) and np.isnan(rsi)) and rsi > 50,
                'value': f"{rsi:.0f}" if not (isinstance(rsi, float) and np.isnan(rsi)) else 'N/A',
            },
            {
                'label': 'Bearish SMC signal',
                'passed': bool(
                    row.get('smc_bos_bearish')
                    or row.get('smc_near_bearish_ob')
                    or row.get('smc_in_bearish_fvg')
                ),
                'value': _active_smc_labels(
                    row, ['smc_bos_bearish', 'smc_near_bearish_ob', 'smc_in_bearish_fvg']
                ) or 'none',
            },
            {
                'label': 'Premium Zone',
                'passed': bool(row.get('smc_premium_zone')),
                'value': 'active' if row.get('smc_premium_zone') else 'not active',
            },
            {
                'label': 'IV/HV < 1.0',
                'passed': not (isinstance(iv_hv, float) and np.isnan(iv_hv)) and iv_hv < 1.0,
                'value': f"{iv_hv:.2f}" if not (isinstance(iv_hv, float) and np.isnan(iv_hv)) else 'N/A',
            },
            {
                'label': 'No near earnings',
                'passed': days_earn is None or days_earn > 14,
                'value': f"{days_earn}d" if days_earn is not None else 'none',
            },
        ]

    met = sum(1 for c in checks if c['passed'])

    if met >= 5:
        status = 'enter'
    elif met >= 3:
        status = 'wait'
    else:
        status = 'not_yet'

    return {'status': status, 'met': met, 'total': 6, 'checks': checks}


def compute_exit_rules(row: dict, strategy: str, rec: dict) -> dict:
    """
    Compute exit rules from screener row and strike recommendation.

    rec: result of get_strike_recommendation() — needs 'cost' and 'dte'.

    Returns:
        take_profit_usd, stop_loss_usd, take_profit_pct, stop_loss_pct,
        time_exit_dte, time_exit_date, time_exit_msg,
        tech_triggers: list of {label, triggered, current_value}
    """
    is_call = (strategy == 'Long Call')
    rsi = row.get('rsi', np.nan)
    cost = rec.get('cost', np.nan)
    dte = rec.get('dte', None)

    if cost is not None and not (isinstance(cost, float) and np.isnan(cost)):
        take_profit_usd = round(cost * 2.0, 2)
        stop_loss_usd = round(cost * 0.5, 2)
    else:
        take_profit_usd = np.nan
        stop_loss_usd = np.nan

    time_exit_dte = 21
    if dte is not None and dte > 21:
        exit_date = datetime.now() + timedelta(days=dte - 21)
        time_exit_date = exit_date.strftime('%b %d, %Y')
        time_exit_msg = f"Close at 21 DTE → {time_exit_date}"
    elif dte is not None:
        time_exit_date = datetime.now().strftime('%b %d, %Y')
        time_exit_msg = "Exit now (past 21 DTE threshold)"
    else:
        time_exit_date = None
        time_exit_msg = "Close at 21 DTE"

    rsi_val = rsi if not (isinstance(rsi, float) and np.isnan(rsi)) else None
    rsi_str = f"RSI {rsi_val:.0f}" if rsi_val is not None else 'N/A'

    if is_call:
        tech_triggers = [
            {
                'label': 'RSI > 70 (overbought)',
                'triggered': rsi_val is not None and rsi_val > 70,
                'current_value': rsi_str,
            },
            {
                'label': 'Bearish BoS forms',
                'triggered': bool(row.get('smc_bos_bearish')),
                'current_value': 'active' if row.get('smc_bos_bearish') else 'not active',
            },
            {
                'label': 'Price enters Premium Zone',
                'triggered': bool(row.get('smc_premium_zone')),
                'current_value': 'active' if row.get('smc_premium_zone') else 'not active',
            },
        ]
    else:
        tech_triggers = [
            {
                'label': 'RSI < 30 (oversold)',
                'triggered': rsi_val is not None and rsi_val < 30,
                'current_value': rsi_str,
            },
            {
                'label': 'Bullish BoS forms',
                'triggered': bool(row.get('smc_bos_bullish')),
                'current_value': 'active' if row.get('smc_bos_bullish') else 'not active',
            },
            {
                'label': 'Price enters Discount Zone',
                'triggered': bool(row.get('smc_discount_zone')),
                'current_value': 'active' if row.get('smc_discount_zone') else 'not active',
            },
        ]

    return {
        'take_profit_usd': take_profit_usd,
        'stop_loss_usd': stop_loss_usd,
        'take_profit_pct': 1.0,
        'stop_loss_pct': 0.5,
        'time_exit_dte': time_exit_dte,
        'time_exit_date': time_exit_date,
        'time_exit_msg': time_exit_msg,
        'tech_triggers': tech_triggers,
    }


def _active_smc_labels(row: dict, keys: list) -> str:
    label_map = {
        'smc_bos_bullish':     'BoS Bullish',
        'smc_near_bullish_ob': 'Near Bull OB',
        'smc_in_bullish_fvg':  'Bull FVG',
        'smc_bos_bearish':     'BoS Bearish',
        'smc_near_bearish_ob': 'Near Bear OB',
        'smc_in_bearish_fvg':  'Bear FVG',
    }
    return ', '.join(label_map[k] for k in keys if row.get(k))
```

- [ ] **Step 2: Run tests — verify all pass**

```
cd C:\Users\fredy\Optionstrategy
pytest tests/test_signals.py -v
```

Expected: All tests PASS. If any fail, fix `signals.py` before continuing.

- [ ] **Step 3: Commit**

```bash
git add signals.py tests/test_signals.py
git commit -m "feat: add signals.py with entry readiness and exit rules

compute_entry_readiness: 6-check scoring -> enter/wait/not_yet status
compute_exit_rules: price targets, tech triggers, 21 DTE time exit

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Add Entry/Exit Playbook tab to `app.py`

**Files:**
- Modify: `app.py`

### Step 1: Add import

- [ ] **Add `signals` import after the existing local imports (around line 25)**

Find this block in `app.py`:
```python
from strategies import suggest_strategies, get_specific_contracts, get_option_chain_display
from chatbot import render_chat_button
```

Change to:
```python
from strategies import suggest_strategies, get_specific_contracts, get_option_chain_display
from chatbot import render_chat_button
from signals import compute_entry_readiness, compute_exit_rules
```

### Step 2: Add `_render_playbook_col` helper

- [ ] **Add helper function after `_build_rec_html` (around line 358)**

Find the end of `_build_rec_html` — it ends with:
```python
        f'{crush_html}'
        f'<div style="font-size:0.72rem;color:#475569;margin-top:0.2rem;">{rec["reason"]}</div>'
        '</div>'
    )
```

Insert after that closing parenthesis + newline:

```python

_PLAYBOOK_STATUS = {
    'enter':   ('🟢', 'Enter Now',             '#10b981'),
    'wait':    ('🟡', 'Wait for Confirmation', '#f59e0b'),
    'not_yet': ('🔴', 'Not Yet',               '#ef4444'),
}


def _render_playbook_col(readiness: dict, exits: dict, strategy_label: str) -> None:
    """Render entry readiness + exit rules for one strategy column."""
    emoji, status_text, status_color = _PLAYBOOK_STATUS[readiness['status']]
    met = readiness['met']
    total = readiness['total']

    st.markdown(f"#### {strategy_label}")

    st.markdown(
        f'<div style="background:#1e293b;border:1px solid {status_color}55;'
        f'border-left:4px solid {status_color};border-radius:0.5rem;'
        f'padding:0.75rem 1rem;margin-bottom:0.75rem;">'
        f'<div style="font-size:0.78rem;color:#94a3b8;font-weight:700;'
        f'margin-bottom:0.3rem;">ENTRY READINESS</div>'
        f'<div style="font-size:1.2rem;font-weight:700;color:{status_color};">'
        f'{emoji} {status_text}</div>'
        f'<div style="font-size:0.75rem;color:#64748b;margin-top:0.15rem;">'
        f'{met}/{total} conditions met</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for check in readiness['checks']:
        icon = '✓' if check['passed'] else '✗'
        color = '#10b981' if check['passed'] else '#ef4444'
        st.markdown(
            f'<div style="font-size:0.82rem;margin:0.15rem 0;">'
            f'<span style="color:{color};font-weight:700;">{icon}</span> '
            f'<span style="color:#e2e8f0;">{check["label"]}</span> '
            f'<span style="color:#64748b;">({check["value"]})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="margin:0.75rem 0;border-top:1px solid #334155;"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.78rem;color:#94a3b8;font-weight:700;'
        'margin-bottom:0.4rem;">EXIT RULES</div>',
        unsafe_allow_html=True,
    )

    tp_usd = exits['take_profit_usd']
    sl_usd = exits['stop_loss_usd']
    tp_str = f"${tp_usd:.0f}" if not (isinstance(tp_usd, float) and np.isnan(tp_usd)) else 'N/A'
    sl_str = f"${sl_usd:.0f}" if not (isinstance(sl_usd, float) and np.isnan(sl_usd)) else 'N/A'

    st.markdown(
        f'<div style="font-size:0.82rem;margin:0.2rem 0;color:#e2e8f0;">'
        f'💰 <strong>Take Profit</strong> +100% → '
        f'<span style="color:#10b981;">{tp_str}</span></div>'
        f'<div style="font-size:0.82rem;margin:0.2rem 0;color:#e2e8f0;">'
        f'🛑 <strong>Stop Loss</strong> -50% → '
        f'<span style="color:#ef4444;">{sl_str}</span></div>'
        f'<div style="font-size:0.82rem;margin:0.2rem 0;color:#e2e8f0;">'
        f'⏰ <strong>Time</strong> {exits["time_exit_msg"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.78rem;color:#94a3b8;margin:0.5rem 0 0.25rem;">'
        'Tech exits (watch for):</div>',
        unsafe_allow_html=True,
    )

    for trigger in exits['tech_triggers']:
        t_icon = '⚠️' if trigger['triggered'] else '✓'
        t_color = '#f59e0b' if trigger['triggered'] else '#64748b'
        t_status = 'TRIGGERED' if trigger['triggered'] else 'safe'
        st.markdown(
            f'<div style="font-size:0.79rem;margin:0.12rem 0;">'
            f'<span style="color:{t_color};">{t_icon}</span> '
            f'{trigger["label"]} — '
            f'<span style="color:#94a3b8;">{trigger["current_value"]}</span> '
            f'<span style="color:{t_color};font-weight:600;">({t_status})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
```

### Step 3: Add third tab

- [ ] **Change tab definition (around line 1037)**

Find:
```python
        tab1, tab2 = st.tabs(["📈 Strategy Recommendations", "⛓️ Option Chain"])
```

Replace with:
```python
        tab1, tab2, tab3 = st.tabs([
            "📈 Strategy Recommendations",
            "⛓️ Option Chain",
            "🎯 Entry/Exit Playbook",
        ])
```

### Step 4: Add `with tab3` block

- [ ] **Add tab3 block immediately after the `with tab2:` block**

The `with tab2:` block ends at approximately line 1183 (the `else: st.info("No put data available.")` line). Add the following immediately after it:

```python
        # ── Tab 3: Entry/Exit Playbook ──
        with tab3:
            smc_tuple = (
                bool(row.get('smc_bos_bullish')),    bool(row.get('smc_bos_bearish')),
                bool(row.get('smc_choch_bullish')),  bool(row.get('smc_choch_bearish')),
                bool(row.get('smc_discount_zone')),  bool(row.get('smc_premium_zone')),
                bool(row.get('smc_near_bullish_ob')), bool(row.get('smc_near_bearish_ob')),
                bool(row.get('smc_in_bullish_fvg')), bool(row.get('smc_in_bearish_fvg')),
            )
            iv_hv_val = row.get('iv_hv_ratio', np.nan)
            iv_hv_for_rec = float(iv_hv_val) if not (isinstance(iv_hv_val, float) and np.isnan(iv_hv_val)) else 0.9

            rec_lc = get_strike_recommendation(
                ticker_str, 'Long Call', iv_hv_for_rec, smc_tuple, budget_config['max_risk_usd'],
            )
            rec_lp = get_strike_recommendation(
                ticker_str, 'Long Put', iv_hv_for_rec, smc_tuple, budget_config['max_risk_usd'],
            )

            readiness_lc = compute_entry_readiness(row, 'Long Call')
            readiness_lp = compute_entry_readiness(row, 'Long Put')
            exits_lc = compute_exit_rules(row, 'Long Call', rec_lc)
            exits_lp = compute_exit_rules(row, 'Long Put', rec_lp)

            col_lc, col_lp = st.columns(2)
            with col_lc:
                _render_playbook_col(readiness_lc, exits_lc, '📈 Long Call')
            with col_lp:
                _render_playbook_col(readiness_lp, exits_lp, '📉 Long Put')
```

### Step 5: Verify app starts without errors

- [ ] **Run a syntax check**

```
cd C:\Users\fredy\Optionstrategy
python -c "import app" 2>&1 | head -20
```

Expected: No output (no import errors). If errors appear, fix them before continuing.

- [ ] **Run all tests to confirm nothing broken**

```
cd C:\Users\fredy\Optionstrategy
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Commit**

```bash
git add app.py
git commit -m "feat: add Entry/Exit Playbook tab to Strategy Detail view

New tab shows entry readiness badge (green/yellow/red) with 6-check
checklist, plus exit rules: price targets, tech triggers, 21 DTE
time exit for both Long Call and Long Put.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - Entry readiness badge (🟢/🟡/🔴) → Task 2 `compute_entry_readiness` + Task 3 tab3
  - 6-check checklist per strategy → Task 2 `checks` list + Task 3 `_render_playbook_col`
  - Price targets (take profit +100%, stop loss -50%) → Task 2 `compute_exit_rules`
  - Technical exit triggers (RSI, BoS, Zone) → Task 2 `tech_triggers`
  - Time exit at 21 DTE → Task 2 `time_exit_msg`
  - New tab in Strategy Detail → Task 3 Steps 3 & 4
  - Unit tests → Task 1 & 2
  - No new API calls → ✓ `get_strike_recommendation` is cached (`@st.cache_data`)

- [x] **No placeholders:** All code blocks are complete and runnable.

- [x] **Type consistency:**
  - `compute_entry_readiness` returns `{'status', 'met', 'total', 'checks'}` — used correctly in `_render_playbook_col`
  - `compute_exit_rules` returns `{'take_profit_usd', 'stop_loss_usd', 'time_exit_msg', 'tech_triggers', ...}` — all keys accessed correctly in `_render_playbook_col`
  - `_PLAYBOOK_STATUS` dict keys match possible values of `readiness['status']`
