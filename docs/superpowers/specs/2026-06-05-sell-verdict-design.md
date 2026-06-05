# Live Sell Verdict — Design Spec

**Date:** 2026-06-05
**Status:** Approved (design)

## Problem

The Entry/Exit Playbook tab lists exit *rules* (take profit +100%, stop loss
−50%, 21 DTE time exit, three technical triggers) but gives no single answer to
the question a holder actually asks: **"Should I sell my long call/put right
now, or keep holding?"** The user reads a list and still has to decide.

## Goal

Add a prominent **SELL / TRIM / HOLD** verdict badge at the top of each Exit
Rules column (Long Call and Long Put) that aggregates the existing exit signals
into one live call, with an optional entry-premium input that layers in
profit/loss targets.

## Non-Goals (YAGNI)

- No position persistence / portfolio tracking across sessions.
- No live per-position option price fetch beyond the screener's existing cost
  estimate.
- No alerting / notifications.

## Architecture

One new pure function in `signals.py`, consumed by the existing tab3 renderer in
`app.py`. No new modules, no new API calls — reuses `compute_exit_rules` output
and the cached `get_strike_recommendation` cost.

### New function: `compute_sell_verdict`

```python
def compute_sell_verdict(
    exits: dict,
    rec: dict,
    entry_premium: float | None = None,
) -> dict:
    """
    Aggregate exit signals into a single SELL / TRIM / HOLD verdict.

    Args:
        exits: output of compute_exit_rules() — provides tech_triggers and
               time_exit_dte / dte state.
        rec:   strike recommendation — provides 'cost' (current live price
               estimate) and 'dte'.
        entry_premium: user's actual fill in dollars (per contract, same unit
               as rec['cost']). None → condition-only verdict.

    Returns:
        {
            'status': 'hold' | 'trim' | 'sell',
            'reasons': list[str],     # plain-English drivers, may be empty for hold
            'pnl_pct': float | None,  # None when entry_premium not given or cost NaN
        }
    """
```

### Verdict logic — most-severe-wins across two layers

Severity order: `sell` > `trim` > `hold`. Compute each layer's status, return
the worst.

**Condition layer (always evaluated):**

Let `n` = count of `exits['tech_triggers']` where `triggered` is True.
Let `dte` = `rec.get('dte')`.

- `sell`  if `n >= 2` OR (`dte is not None` and `dte <= 21`)
- `trim`  if `n == 1` OR (`dte is not None` and `dte <= 25`)
- `hold`  otherwise

**P/L layer (only when `entry_premium` is a positive number and `rec['cost']` is
valid):**

`pnl_pct = rec['cost'] / entry_premium - 1.0`

- `sell`  if `pnl_pct <= -0.50` OR `pnl_pct >= 1.00`
- `trim`  if `pnl_pct >= 0.50`
- else contributes `hold`

**reasons[]** — append one short string per active driver, e.g.:
- `"2 tech exits active: RSI > 70, Bearish BoS"` (joins triggered labels)
- `"1 tech exit active: Premium Zone"`
- `"Past 21 DTE — time stop"` / `"Approaching 21 DTE"`
- `"+112% — profit target hit"` / `"−54% — stop loss hit"` / `"+63% — lock partial"`

`pnl_pct` is returned (float) when computed, else `None`.

### Edge cases

- `entry_premium` None, 0, negative, or NaN → P/L layer skipped, `pnl_pct = None`.
- `rec['cost']` NaN/None → P/L layer skipped even if entry_premium given.
- `dte` None → time conditions skipped (neither sell nor trim from time).
- No triggers, no P/L, dte > 25 → `hold` with empty `reasons`.

## UI changes (`app.py`, tab3 `_render_playbook_col`)

Add to the **top of the Exit Rules section** (after the existing `EXIT RULES`
header, before the price targets):

1. **Verdict badge** — same visual language as the entry readiness badge:
   - 🟢 HOLD (`#10b981`), 🟡 TRIM (`#f59e0b`), 🔴 SELL (`#ef4444`)
   - Below the headline: the `reasons[]` rendered as small lines; if empty,
     show `"No exit signals active"`.
   - If `pnl_pct` present: show `Current P/L: +NN%` colored green/red.

2. **Entry premium input** — a single `st.number_input` labeled
   `"Your entry premium ($, per contract)"`, `min_value=0.0`, `value=0.0`,
   `step=1.0`, unique `key` per strategy column. `0.0` is treated as "not set"
   (passes `None` to the verdict). Streamlit reruns on change → badge refreshes.

`_render_playbook_col` signature gains the verdict dict; the tab3 block calls
`compute_sell_verdict` for each leg using the `rec_lc` / `rec_lp` already fetched
and the `exits_lc` / `exits_lp` already computed, reading the entry premium from
the new inputs.

## Testing (`tests/test_signals.py`)

New `TestSellVerdict` class:

- Two triggers active → `sell`.
- One trigger active → `trim`.
- Zero triggers, dte 45 → `hold`, empty reasons.
- dte 21 → `sell` (time stop).
- dte 24 → `trim`.
- P/L +120% (cost 924, entry 420) → `sell`.
- P/L −55% → `sell`.
- P/L +60% with zero triggers and dte 45 → `trim`.
- P/L layer beats condition layer: zero triggers + entry premium giving +110%
  → `sell`.
- `entry_premium=None` → `pnl_pct is None`, verdict from conditions only.
- `entry_premium=0.0` treated as None.
- NaN cost with entry premium given → `pnl_pct is None`.
- reasons contains triggered labels when tech exits fire.

## File Map

| File | Action | Responsibility |
|---|---|---|
| `signals.py` | Modify | Add `compute_sell_verdict` |
| `tests/test_signals.py` | Modify | Add `TestSellVerdict` |
| `app.py` | Modify | Verdict badge + entry-premium input in `_render_playbook_col`; wire in tab3 |
