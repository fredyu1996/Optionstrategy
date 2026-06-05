# Live Sell Verdict Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single 🟢 HOLD / 🟡 TRIM / 🔴 SELL verdict badge to each Long Call / Long Put column in the Entry/Exit Playbook tab, aggregating the existing exit signals plus an optional entry-premium P/L check.

**Architecture:** One new pure function `compute_sell_verdict` in `signals.py` consumes the `exits` dict (from `compute_exit_rules`) and the strike `rec` (for `dte` and live `cost`), plus an optional `entry_premium`. It returns one of three statuses via most-severe-wins across a condition layer and a P/L layer. `app.py`'s `_render_playbook_col` renders the badge (in a placeholder above a single `number_input`) and wires it in tab3. No new modules, no new API calls.

**Tech Stack:** Python 3.11, Streamlit, numpy, pandas (`pd.isna` guards, matching existing `signals.py` style)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tests/test_signals.py` | Modify | Add `TestSellVerdict` class |
| `signals.py` | Modify | Add `compute_sell_verdict` |
| `app.py` | Modify | Import fn, add `_PLAYBOOK_VERDICT` map, badge + input in `_render_playbook_col`, pass `rec`/`key_prefix` from tab3 |

---

## Task 1: Write failing tests for `compute_sell_verdict`

**Files:**
- Modify: `tests/test_signals.py`

- [ ] **Step 1: Add import for the new function**

In `tests/test_signals.py`, find:
```python
from signals import compute_entry_readiness, compute_exit_rules
```
Change to:
```python
from signals import compute_entry_readiness, compute_exit_rules, compute_sell_verdict
```

- [ ] **Step 2: Append the `TestSellVerdict` class to the end of `tests/test_signals.py`**

These helpers build `exits`/`rec` shapes directly so the verdict tests don't depend on `compute_exit_rules` internals.

```python
# ── compute_sell_verdict ──────────────────────────────────────────────────────

def _exits_with_triggers(n_active: int) -> dict:
    """Build an exits dict with n_active triggered tech triggers (max 3)."""
    labels = ['RSI > 70 (overbought)', 'Bearish BoS forms', 'Price enters Premium Zone']
    triggers = [
        {'label': labels[i], 'triggered': i < n_active, 'current_value': 'x'}
        for i in range(3)
    ]
    return {'tech_triggers': triggers}


class TestSellVerdict:
    # ── condition layer: tech triggers ──
    def test_two_triggers_returns_sell(self):
        result = compute_sell_verdict(_exits_with_triggers(2), {'cost': 420.0, 'dte': 45})
        assert result['status'] == 'sell'

    def test_one_trigger_returns_trim(self):
        result = compute_sell_verdict(_exits_with_triggers(1), {'cost': 420.0, 'dte': 45})
        assert result['status'] == 'trim'

    def test_zero_triggers_far_dte_returns_hold(self):
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 420.0, 'dte': 45})
        assert result['status'] == 'hold'
        assert result['reasons'] == []

    # ── condition layer: time ──
    def test_past_21_dte_returns_sell(self):
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 420.0, 'dte': 21})
        assert result['status'] == 'sell'

    def test_dte_24_returns_trim(self):
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 420.0, 'dte': 24})
        assert result['status'] == 'trim'

    def test_dte_none_no_time_contribution(self):
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 420.0, 'dte': None})
        assert result['status'] == 'hold'

    # ── P/L layer ──
    def test_pnl_target_hit_returns_sell(self):
        # cost 924 / entry 420 - 1 = +1.20
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 924.0, 'dte': 45}, entry_premium=420.0)
        assert result['status'] == 'sell'
        assert result['pnl_pct'] == pytest.approx(1.2)

    def test_pnl_stop_hit_returns_sell(self):
        # cost 180 / entry 420 - 1 = -0.571
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 180.0, 'dte': 45}, entry_premium=420.0)
        assert result['status'] == 'sell'

    def test_pnl_partial_returns_trim(self):
        # cost 672 / entry 420 - 1 = +0.60
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 672.0, 'dte': 45}, entry_premium=420.0)
        assert result['status'] == 'trim'

    def test_pnl_layer_beats_condition_layer(self):
        # zero triggers, far dte (cond=hold) but +110% P/L → sell
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 882.0, 'dte': 45}, entry_premium=420.0)
        assert result['status'] == 'sell'

    def test_condition_layer_beats_pnl_layer(self):
        # two triggers (cond=sell) but P/L only +10% (pnl=hold) → sell
        result = compute_sell_verdict(_exits_with_triggers(2), {'cost': 462.0, 'dte': 45}, entry_premium=420.0)
        assert result['status'] == 'sell'

    # ── entry premium guards ──
    def test_entry_premium_none_pnl_is_none(self):
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 420.0, 'dte': 45})
        assert result['pnl_pct'] is None

    def test_entry_premium_zero_treated_as_none(self):
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 420.0, 'dte': 45}, entry_premium=0.0)
        assert result['pnl_pct'] is None

    def test_nan_cost_pnl_is_none(self):
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': float('nan'), 'dte': 45}, entry_premium=420.0)
        assert result['pnl_pct'] is None

    # ── reasons ──
    def test_reasons_lists_triggered_labels(self):
        result = compute_sell_verdict(_exits_with_triggers(2), {'cost': 420.0, 'dte': 45})
        joined = ' '.join(result['reasons'])
        assert 'RSI > 70 (overbought)' in joined
        assert 'Bearish BoS forms' in joined

    def test_reasons_includes_time_stop(self):
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 420.0, 'dte': 21})
        assert any('21 DTE' in r for r in result['reasons'])

    def test_result_has_required_keys(self):
        result = compute_sell_verdict(_exits_with_triggers(0), {'cost': 420.0, 'dte': 45})
        assert set(result.keys()) == {'status', 'reasons', 'pnl_pct'}
```

- [ ] **Step 3: Run the new tests and verify they fail with ImportError**

Run: `python -m pytest tests/test_signals.py -k SellVerdict -v`
Expected: collection/import error — `ImportError: cannot import name 'compute_sell_verdict'`

---

## Task 2: Implement `compute_sell_verdict`

**Files:**
- Modify: `signals.py`

- [ ] **Step 1: Add the function to `signals.py`**

Insert immediately after `compute_exit_rules` (before the `_active_smc_labels` helper near line 210):

```python
def compute_sell_verdict(exits: dict, rec: dict, entry_premium=None) -> dict:
    """
    Aggregate exit signals into a single SELL / TRIM / HOLD verdict.

    Args:
        exits: output of compute_exit_rules() — provides 'tech_triggers'.
        rec:   strike recommendation — provides 'cost' (current price estimate)
               and 'dte'.
        entry_premium: user's fill in dollars, same unit as rec['cost'].
               None / 0 / negative / NaN → condition-only verdict (no P/L).

    Returns:
        status:  'hold' | 'trim' | 'sell'  (most-severe across both layers)
        reasons: list[str] of plain-English drivers (empty for a clean hold)
        pnl_pct: float | None  (None when entry_premium unusable or cost NaN)
    """
    severity = {'hold': 0, 'trim': 1, 'sell': 2}
    reasons = []

    # ── Condition layer (always evaluated) ──
    active = [t for t in exits.get('tech_triggers', []) if t.get('triggered')]
    n = len(active)
    dte = rec.get('dte', None)
    dte_valid = dte is not None and not pd.isna(dte)

    if n >= 2 or (dte_valid and dte <= 21):
        cond_status = 'sell'
    elif n == 1 or (dte_valid and dte <= 25):
        cond_status = 'trim'
    else:
        cond_status = 'hold'

    if n >= 1:
        labels = ', '.join(t['label'] for t in active)
        reasons.append(f"{n} tech exit{'s' if n > 1 else ''} active: {labels}")
    if dte_valid and dte <= 21:
        reasons.append("Past 21 DTE — time stop")
    elif dte_valid and dte <= 25:
        reasons.append("Approaching 21 DTE")

    # ── P/L layer (only with a usable entry premium and cost) ──
    pnl_pct = None
    pnl_status = 'hold'
    cost = rec.get('cost', np.nan)
    if (entry_premium is not None and not pd.isna(entry_premium)
            and entry_premium > 0 and not pd.isna(cost)):
        pnl_pct = cost / entry_premium - 1.0
        if pnl_pct <= -0.50 or pnl_pct >= 1.00:
            pnl_status = 'sell'
        elif pnl_pct >= 0.50:
            pnl_status = 'trim'

        pct_str = f"{pnl_pct * 100:+.0f}%"
        if pnl_pct >= 1.00:
            reasons.append(f"{pct_str} — profit target hit")
        elif pnl_pct <= -0.50:
            reasons.append(f"{pct_str} — stop loss hit")
        elif pnl_pct >= 0.50:
            reasons.append(f"{pct_str} — lock partial")

    # ── Most-severe-wins ──
    status = cond_status if severity[cond_status] >= severity[pnl_status] else pnl_status

    return {'status': status, 'reasons': reasons, 'pnl_pct': pnl_pct}
```

- [ ] **Step 2: Run the verdict tests and verify they pass**

Run: `python -m pytest tests/test_signals.py -k SellVerdict -v`
Expected: all `TestSellVerdict` tests PASS. Fix `signals.py` before continuing if any fail.

- [ ] **Step 3: Run the full signals suite to confirm nothing broke**

Run: `python -m pytest tests/test_signals.py -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add signals.py tests/test_signals.py
git commit -m "feat: add compute_sell_verdict for live SELL/TRIM/HOLD call

Aggregates tech triggers + 21 DTE time stop (condition layer) with
optional entry-premium P/L (target/stop/partial). Most-severe-wins.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Wire verdict badge into the Playbook tab

**Files:**
- Modify: `app.py`

### Step 1: Add the import

- [ ] **Update the signals import**

Find in `app.py`:
```python
from signals import compute_entry_readiness, compute_exit_rules
```
Change to:
```python
from signals import compute_entry_readiness, compute_exit_rules, compute_sell_verdict
```

### Step 2: Add the verdict status map

- [ ] **Add `_PLAYBOOK_VERDICT` right after the existing `_PLAYBOOK_STATUS` dict (after line 365)**

Find:
```python
_PLAYBOOK_STATUS = {
    'enter':   ('🟢', 'Enter Now',             '#10b981'),
    'wait':    ('🟡', 'Wait for Confirmation', '#f59e0b'),
    'not_yet': ('🔴', 'Not Yet',               '#ef4444'),
}
```
Add immediately below it:
```python
_PLAYBOOK_VERDICT = {
    'hold': ('🟢', 'HOLD', '#10b981'),
    'trim': ('🟡', 'TRIM', '#f59e0b'),
    'sell': ('🔴', 'SELL', '#ef4444'),
}
```

### Step 3: Change `_render_playbook_col` signature

- [ ] **Add `rec` and `key_prefix` parameters**

Find:
```python
def _render_playbook_col(readiness: dict, exits: dict, strategy_label: str) -> None:
    """Render entry readiness + exit rules for one strategy column in the Playbook tab."""
```
Change to:
```python
def _render_playbook_col(readiness: dict, exits: dict, rec: dict, strategy_label: str, key_prefix: str) -> None:
    """Render entry readiness, sell verdict, and exit rules for one strategy column."""
```

### Step 4: Insert the verdict badge + entry-premium input

- [ ] **Add the verdict block between the `EXIT RULES` header and the price-target computation**

Find this block (the `EXIT RULES` header followed by `tp_usd = ...`):
```python
    st.markdown(
        '<div style="font-size:0.78rem;color:#94a3b8;font-weight:700;'
        'margin-bottom:0.4rem;">EXIT RULES</div>',
        unsafe_allow_html=True,
    )

    tp_usd = exits['take_profit_usd']
```
Replace it with:
```python
    st.markdown(
        '<div style="font-size:0.78rem;color:#94a3b8;font-weight:700;'
        'margin-bottom:0.4rem;">EXIT RULES</div>',
        unsafe_allow_html=True,
    )

    # ── Sell verdict badge (rendered above the input via a placeholder) ──
    verdict_slot = st.empty()
    entry_premium = st.number_input(
        "Your entry premium ($, per contract)",
        min_value=0.0, value=0.0, step=1.0,
        key=f"{key_prefix}_entry_prem",
        help="Total $ you paid per contract. Leave 0 to judge by conditions only.",
    )
    ep = entry_premium if entry_premium > 0 else None
    verdict = compute_sell_verdict(exits, rec, ep)
    v_emoji, v_text, v_color = _PLAYBOOK_VERDICT[verdict['status']]

    pnl_html = ''
    if verdict['pnl_pct'] is not None:
        pnl = verdict['pnl_pct']
        pnl_color = '#10b981' if pnl >= 0 else '#ef4444'
        pnl_html = (
            f'<div style="font-size:0.8rem;margin-top:0.2rem;color:{pnl_color};'
            f'font-weight:600;">Current P/L: {pnl * 100:+.0f}%</div>'
        )

    if verdict['reasons']:
        reasons_html = ''.join(
            f'<div style="font-size:0.74rem;color:#94a3b8;margin-top:0.12rem;">• {r}</div>'
            for r in verdict['reasons']
        )
    else:
        reasons_html = (
            '<div style="font-size:0.74rem;color:#64748b;margin-top:0.12rem;">'
            'No exit signals active</div>'
        )

    verdict_slot.markdown(
        f'<div style="background:#0f172a;border:1px solid {v_color}55;'
        f'border-left:4px solid {v_color};border-radius:0.5rem;'
        f'padding:0.6rem 0.9rem;margin-bottom:0.6rem;">'
        f'<div style="font-size:1.25rem;font-weight:800;color:{v_color};">'
        f'{v_emoji} {v_text}</div>'
        f'{pnl_html}{reasons_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    tp_usd = exits['take_profit_usd']
```

### Step 5: Update the tab3 call sites

- [ ] **Pass `rec` and a unique `key_prefix` to both columns**

Find:
```python
            col_lc, col_lp = st.columns(2)
            with col_lc:
                _render_playbook_col(readiness_lc, exits_lc, '📈 Long Call')
            with col_lp:
                _render_playbook_col(readiness_lp, exits_lp, '📉 Long Put')
```
Replace with:
```python
            col_lc, col_lp = st.columns(2)
            with col_lc:
                _render_playbook_col(readiness_lc, exits_lc, rec_lc, '📈 Long Call', f"pb_{ticker_str}_lc")
            with col_lp:
                _render_playbook_col(readiness_lp, exits_lp, rec_lp, '📉 Long Put', f"pb_{ticker_str}_lp")
```

### Step 6: Syntax check

- [ ] **Import the module to catch syntax/indentation errors**

Run: `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

### Step 7: Run the full test suite

- [ ] **Confirm nothing broke**

Run: `python -m pytest tests/ -q`
Expected: all tests pass (58 prior + new `TestSellVerdict`).

- [ ] **Commit**

```bash
git add app.py
git commit -m "feat: add live SELL/TRIM/HOLD verdict to Playbook tab

Each Long Call/Put column now shows a verdict badge above the exit
rules, driven by compute_sell_verdict. Optional entry-premium input
layers in live P/L; reasons list explains the call.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - 3-tier SELL/TRIM/HOLD verdict → Task 2 `compute_sell_verdict` status logic
  - Condition layer (2+ triggers / 21 DTE → sell; 1 trigger / ≤25 DTE → trim) → Task 2
  - P/L layer (≤−50% or ≥+100% → sell; ≥+50% → trim) → Task 2
  - Most-severe-wins → Task 2 `severity` comparison + Task 1 `test_*_beats_*` tests
  - Single entry-premium input, current price auto from `rec['cost']` → Task 3 Step 4
  - Verdict badge atop Exit Rules column, above input via `st.empty()` → Task 3 Step 4
  - Plain-English `reasons` + Current P/L line → Task 3 Step 4 + Task 2
  - Edge cases (entry 0/None/NaN, NaN cost, None dte) → Task 1 guard tests + Task 2 guards
  - Tests → Task 1 `TestSellVerdict`

- [x] **No placeholders:** every step has runnable code/commands.

- [x] **Type consistency:**
  - `compute_sell_verdict(exits, rec, entry_premium=None)` → `{'status','reasons','pnl_pct'}`; consumed in `_render_playbook_col` with those exact keys.
  - `_PLAYBOOK_VERDICT` keys (`hold`/`trim`/`sell`) match `verdict['status']` values.
  - `_render_playbook_col(readiness, exits, rec, strategy_label, key_prefix)` signature matches both tab3 call sites.
