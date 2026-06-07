# EMA Signals Integration — Design Spec

**Date:** 2026-06-06
**Status:** Approved (design)

## Problem

The screener derives trend and signals from SMA (daily) and SMC structure, but
the user trades off EMA 20/50/100/200 (as on their TradingView 4H chart). The
app currently can't see short-term momentum shifts like "price broke below
EMA20" that drive their entry/exit timing. EMA should inform **both** entry and
exit.

## Goal

Add EMA 20/50/100/200 signals to the screener and the entry/exit logic, on two
timeframes:
- **Daily** EMA for all screened stocks (entry checklist + sell triggers).
- **4-hour** EMA for a single stock (Strategy Detail playbook + position cards)
  as an advisory entry-timing / exit-warning indicator.

SMA-based trend is kept unchanged; EMA is added as new, independent signals.

## Non-Goals (YAGNI)

- No 4H EMA across the full 500-stock scan (too slow / intraday rate limits).
- No EMA-based auto-trading or alerts (alerts are a separate later project).
- No change to existing SMA trend, SMC, RSI, or IV/HV logic.

## Constraints

- yfinance intraday history is limited (~60 days for hourly); 4H is resampled
  from 1h bars and only fetched for a single ticker on demand.
- EMA needs enough bars: EMA200 daily needs ≥200 daily closes (the screener
  already fetches ~1y); 4H EMA200 needs ~200 4H bars (~33 trading days of 1h
  data, within the 60-day intraday window).

## Architecture

New module `indicators.py` owns all EMA math and the 4H fetch. `signals.py`
consumes the daily EMA flags already present on the screener `row`. `screener.py`
populates daily EMA fields during enrichment. `app.py` renders the 4H status.

```
indicators.py   ── compute_ema_signals(close_series) [daily, pure]
                   fetch_4h_ema_status(ticker)       [4H, network]
screener.py     ── add daily EMA fields to each row
signals.py      ── EMA entry check + EMA sell triggers (consume row flags)
app.py          ── show 4H EMA status in playbook + position cards
positions.py    ── include 4H status in analyze_position result
```

### Module: `indicators.py`

```python
def compute_ema_signals(close: pd.Series) -> dict:
    """Pure EMA signal computation from a close-price series.

    Returns (NaN-safe; values default False / nan when too few bars):
        ema20, ema50, ema100, ema200: float
        ema_bull_stack: bool   # ema20 > ema50 > ema100 > ema200
        ema_bear_stack: bool   # ema20 < ema50 < ema100 < ema200
        above_ema20: bool      # last close > ema20
        above_ema50: bool
    """

def fetch_4h_ema_status(ticker: str) -> dict:
    """Fetch ~60d of 1h bars, resample to 4H, compute EMA signals, and classify.

    Returns:
        status: 'good' | 'wait' | 'avoid' | 'unknown'
        label:  human string, e.g. '多頭排列, 企 EMA20 上 → 入場時機好'
        signals: dict from compute_ema_signals (or empty when unavailable)
    """
```

**4H classification (`fetch_4h_ema_status`):**
- 🟢 `good` — `ema_bull_stack` and `above_ema20`
- 🟡 `wait` — `above_ema50` but not `above_ema20` (broke EMA20, holding EMA50)
- 🔴 `avoid` — not `above_ema50` or `ema_bear_stack`
- `unknown` — fetch failed / too few bars

This single 4H indicator serves both sides: an entry shopper reads 🟢 as a good
timing, a holder reads 🔴 as an exit warning.

### Daily EMA on the screener row

`screener.py` enrichment computes `compute_ema_signals` from the daily closes it
already downloads and writes these onto each `row`:
`ema_bull_stack`, `ema_bear_stack`, `above_ema20`, `above_ema50`
(plus the four EMA values for display). NaN-safe: missing → False.

### `signals.py` changes

**Entry — add one EMA check (now 7 checks):**
- Long Call: **EMA bullish** = `ema_bull_stack and above_ema20`
- Long Put: **EMA bearish** = `ema_bear_stack and not above_ema20`

New thresholds (scale with 7 checks):
- 🟢 enter: met ≥ 6
- 🟡 wait: met ≥ 4
- 🔴 not_yet: met ≤ 3

**Exit — add two daily EMA tech triggers (graduated):**
- Long Call: `穿 EMA20` = `not above_ema20`; `穿 EMA50` = `not above_ema50`
- Long Put: `升穿 EMA20` = `above_ema20`; `升穿 EMA50` = `above_ema50`

These join the existing 3 triggers (RSI, BoS, Zone). Because
`compute_sell_verdict` already maps trigger-count to severity, the EMA triggers
grade cleanly:
- break **EMA20 only** → 1 trigger → 🟡 TRIM (short-term caution)
- break **EMA20 + EMA50** → 2 triggers → 🔴 SELL (trend weakening)

No change to `compute_sell_verdict` thresholds.

### `app.py` / `positions.py` — 4H display

- `positions.analyze_position` adds `ema4h` = `fetch_4h_ema_status(ticker)` to
  its result (cached fetch); position cards render a 4H status line.
- Strategy Detail playbook (`_render_playbook_col` or the tab3 block) shows the
  same 4H status line for the selected ticker (fetched once per ticker, cached).
- Rendering: colored line using existing verdict colors (good=green,
  wait=amber, avoid=red, unknown=grey).

## Testing

**`tests/test_indicators.py`** (pure / mocked):
- `compute_ema_signals` bull stack: synthetic rising series → `ema_bull_stack`
  True, `above_ema20` True.
- bear stack: falling series → `ema_bear_stack` True.
- too few bars (<200) → EMA values nan / stacks False, no crash.
- `fetch_4h_ema_status` classification: monkeypatch the 1h fetch + resample to
  feed known closes → assert `good`/`wait`/`avoid`; fetch exception → `unknown`.

**`tests/test_signals.py`** (extend):
- Long Call entry: bull stack + above_ema20 → EMA check passes; 7 checks total.
- entry thresholds: 6/7 → enter, 4/7 → wait, 3/7 → not_yet.
- exit: `not above_ema20` only → 1 EMA trigger active → verdict trim;
  `not above_ema20` and `not above_ema50` → 2 triggers → sell.
- Long Put mirror.

**`tests/test_positions.py`** (extend):
- `analyze_position` includes an `ema4h` key (mock `fetch_4h_ema_status`).

## Error Handling

- `compute_ema_signals` with <200 bars: return nan EMAs and False stacks; entry
  EMA check simply fails, no crash.
- `fetch_4h_ema_status` network/parse failure → `status='unknown'`, neutral grey
  line; never raises into the page.
- Existing `analyze_position` try/except already isolates per-position failures.

## File Map

| File | Action | Responsibility |
|---|---|---|
| `indicators.py` | Create | `compute_ema_signals`, `fetch_4h_ema_status` |
| `tests/test_indicators.py` | Create | EMA math + 4H classification tests |
| `screener.py` | Modify | populate daily EMA flags on each row |
| `signals.py` | Modify | EMA entry check + EMA exit triggers + new thresholds |
| `tests/test_signals.py` | Modify | EMA entry/exit tests |
| `positions.py` | Modify | add `ema4h` to `analyze_position` |
| `tests/test_positions.py` | Modify | assert `ema4h` present |
| `app.py` | Modify | render 4H EMA line in playbook + position cards |
