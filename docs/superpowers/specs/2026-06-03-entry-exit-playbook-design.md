# Entry/Exit Playbook — Design Spec
**Date:** 2026-06-03  
**Feature:** Entry readiness badge + exit rules tab for Long Call / Long Put

---

## Overview

Add a new "🎯 Entry/Exit Playbook" tab to the Strategy Detail view that tells the user:
1. **When to enter** — live readiness badge (🟢/🟡/🔴) + per-condition checklist
2. **When to exit** — price targets, technical triggers, time-based rule

---

## Architecture

### New file: `signals.py`

Two public functions:

#### `compute_entry_readiness(row: dict, strategy: str) -> dict`

Evaluates 6 conditions from the screener row. Returns:
```python
{
  'status': 'enter' | 'wait' | 'not_yet',
  'met': int,        # conditions passing
  'total': 6,
  'checks': [
    {'label': str, 'passed': bool, 'value': str}
  ]
}
```

**Long Call checks:**

| # | Label | Pass condition |
|---|---|---|
| 1 | Trend | `trend` in (`Up`, `Strong Up`) |
| 2 | RSI | `rsi` < 50 |
| 3 | SMC signal | `smc_bos_bullish` OR `smc_near_bullish_ob` OR `smc_in_bullish_fvg` |
| 4 | Zone | `smc_discount_zone` = True |
| 5 | IV/HV | `iv_hv_ratio` < 1.0 |
| 6 | Earnings | `days_to_earnings` is None OR > 14 |

**Long Put checks** (mirror):

| # | Label | Pass condition |
|---|---|---|
| 1 | Trend | `trend` in (`Down`, `Strong Down`) |
| 2 | RSI | `rsi` > 50 |
| 3 | SMC signal | `smc_bos_bearish` OR `smc_near_bearish_ob` OR `smc_in_bearish_fvg` |
| 4 | Zone | `smc_premium_zone` = True |
| 5 | IV/HV | `iv_hv_ratio` < 1.0 |
| 6 | Earnings | `days_to_earnings` is None OR > 14 |

**Status thresholds:**
- 5–6 met → `'enter'` → 🟢 Enter Now
- 3–4 met → `'wait'` → 🟡 Wait for Confirmation
- 0–2 met → `'not_yet'` → 🔴 Not Yet

---

#### `compute_exit_rules(row: dict, strategy: str, rec: dict) -> dict`

`rec` = result of `get_strike_recommendation()` (already fetched in detail view).

Returns:
```python
{
  'take_profit_usd': float,   # rec['cost'] * 2.0
  'stop_loss_usd': float,     # rec['cost'] * 0.5
  'take_profit_pct': 1.0,
  'stop_loss_pct': 0.5,
  'time_exit_dte': 21,
  'time_exit_date': str,      # today + max(0, rec['dte'] - 21) days; if rec['dte'] <= 21 → "Exit now (past time threshold)"
  'tech_triggers': [
    {'label': str, 'current_value': str, 'triggered': bool}
  ]
}
```

**Long Call tech triggers:**

| Trigger | Triggered when | Current value shown |
|---|---|---|
| RSI > 70 (overbought) | `rsi` > 70 | `f"RSI {rsi:.0f}"` |
| Bearish BoS | `smc_bos_bearish` = True | active / not active |
| Enters Premium Zone | `smc_premium_zone` = True | active / not active |

**Long Put tech triggers:**

| Trigger | Triggered when | Current value shown |
|---|---|---|
| RSI < 30 (oversold) | `rsi` < 30 | `f"RSI {rsi:.0f}"` |
| Bullish BoS | `smc_bos_bullish` = True | active / not active |
| Enters Discount Zone | `smc_discount_zone` = True | active / not active |

If `rec['strike']` is None (no recommendation available), exit price targets are shown as N/A.

---

### Modified file: `app.py`

#### Tab change (Strategy Detail section, ~line 1037):
```python
# Before:
tab1, tab2 = st.tabs(["📈 Strategy Recommendations", "⛓️ Option Chain"])

# After:
tab1, tab2, tab3 = st.tabs([
    "📈 Strategy Recommendations",
    "⛓️ Option Chain",
    "🎯 Entry/Exit Playbook",
])
```

#### New tab rendering (`tab3`):
- Import `compute_entry_readiness`, `compute_exit_rules` from `signals`
- Fetch `rec_lc` and `rec_lp` via `get_strike_recommendation()` (same call already made in Top 10 — repeated here for the detail view)
- Two columns: left = Long Call, right = Long Put
- Each column:
  1. **Entry Readiness** header with colored badge
  2. Checklist: ✓/✗ per condition with current value
  3. **Exit Rules** header
  4. Price targets (take profit / stop loss in USD and %)
  5. Time exit (DTE + estimated date)
  6. Tech triggers table (label | current value | status)

---

## Data flow

```
screener row (already in session_state.screening_results)
    │
    ├─► compute_entry_readiness(row, 'Long Call')  → readiness_lc
    ├─► compute_entry_readiness(row, 'Long Put')   → readiness_lp
    │
    ├─► get_strike_recommendation(ticker, 'Long Call', ...)  → rec_lc
    ├─► get_strike_recommendation(ticker, 'Long Put', ...)   → rec_lp
    │
    ├─► compute_exit_rules(row, 'Long Call', rec_lc)  → exits_lc
    └─► compute_exit_rules(row, 'Long Put',  rec_lp)  → exits_lp
```

`get_strike_recommendation` is `@st.cache_data(ttl=900)` so repeated calls in the same session are free.

---

## Files changed

| File | Change |
|---|---|
| `signals.py` | **New** — `compute_entry_readiness`, `compute_exit_rules` |
| `app.py` | Add `tab3`, import from `signals`, render playbook UI |
| `tests/test_signals.py` | **New** — unit tests for both functions |

---

## Out of scope

- Entry/exit badges on Top 10 cards (user chose detail-view only)
- Price chart annotations
- Alerts / notifications
- Historical backtesting of signals
