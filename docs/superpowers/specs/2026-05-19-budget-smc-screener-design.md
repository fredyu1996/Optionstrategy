# Design: Budget-Aware Screener + SMC Scoring

**Date:** 2026-05-19  
**Scope:** S&P 500 Options Strategy Screener — `app.py`, `screener.py`  
**Strategies:** Long Call, Long Put only

---

## 1. Goals

1. Make the screener budget-aware — user has 500 CAD (~$370 USD), max risk 33% per trade (~$120 USD).
2. Show cost per contract, affordability, position sizing, and risk/reward on every pick.
3. Fix flawed scoring: remove VIX double-counting, remove useless delta bonus, rebalance trend vs IV weights.
4. Add SMC (Smart Money Concepts) signals: BoS, CHoCH, Discount/Premium zone, Order Blocks, FVGs.
5. Remove Quick Screen mode — Full Screen (with IV) is the only mode.

---

## 2. Sidebar — "My Account" Panel

New section added above "Screen Now" button.

```
💰 My Account
─────────────────────────────
Account Size (CAD):  [500    ]
CAD → USD Rate:      [0.73   ]   ← auto-fetched from yfinance CADUSD=X, user-editable
Risk per Trade:      [33%  ▼ ]   options: 10% / 20% / 33%
─────────────────────────────
Budget (USD):   $365
Max Risk/Trade: $120
```

- `budget_usd = account_cad × fx_rate`
- `max_risk_usd = budget_usd × risk_pct`
- FX rate fetched once on load via `yf.download('CADUSD=X', period='1d')`, cached. Falls back to 0.73 if fetch fails.
- `budget_config` dict passed to Top 10 rendering and table filter.

**Removed:** "Screening Mode" radio (Quick Screen / Full Screen). Full Screen is now the only mode.

**Added:** "Affordable only" checkbox — hides rows where `cost_per_contract > max_risk_usd`.

---

## 3. Screener Table — New Columns

Two new columns added to the existing results table:

| Column | Source | Format | Notes |
|--------|--------|--------|-------|
| `Cost/Contract` | `atm_mid_price × 100` (USD) | `$142.00` | ATM call mid = (bid + ask) / 2 |
| `Affordable` | `cost <= max_risk_usd` | `✓` / `✗` | Based on My Account settings |

`atm_mid_price` stored in `enrich_with_iv` alongside existing ATM data.  
Bid-ask spread % also stored: `atm_spread_pct = (ask - bid) / mid`.

---

## 4. Top 10 Picks Cards — Budget Info

Each card gains a budget row:

```
AMD                                          Score: 71
$112.40  |  📈 Up  |  1M: +4.1%
[Bullish BoS]  [Discount Zone]  [Near OB]
IV/HV: 0.68  |  Δ 0.47  |  Θ $-5.10/day  |  OI: 8,200
─────────────────────────────────────────────────────
💰 Cost: $98/contract  |  ✓ Affordable
   Max Loss: $98  |  2:1 Target: +$196
```

Over-budget picks:
```
💰 Cost: $187/contract  |  ✗ Over Budget ($120 max)
```

- Affordable picks show: cost, max loss, 2:1 profit target.
- Over-budget picks show cost + red label. Still visible unless "Affordable only" is checked.
- SMC active signals shown as tags between price row and Greeks row.

---

## 5. Revised Scoring System

### Removed
- VIX bonus (+20/+10) — redundant, IV/HV already captures market vol level.
- ATM delta range bonus (+10) — ATM options always delta ~0.45–0.55, discriminates nothing.
- Raw 1M momentum bonus — replaced by trend-quality combined signal (see below).

### Shared Base (both Long Call + Long Put)

| Signal | Points |
|--------|--------|
| IV/HV < 0.6 (very cheap) | +25 |
| IV/HV < 0.8 (cheap) | +15 |
| IV/HV < 1.0 (fair) | +8 |
| No earnings within 30d | +10 |
| Earnings < 14d away | -15 |
| Theta < $10/day | +5 |
| Theta $25–50/day | -8 |
| Theta > $50/day | -15 |
| Bid-ask spread > 20% of mid | -10 |
| Bid-ask spread 10–20% of mid | -5 |

### Trend-Quality Signal (replaces raw momentum + RSI separate bonuses)

Rationale: strong 1M momentum (>10%) often means overbought — bad entry for a new long position.
Best Long Call entry: trend is UP but RSI has pulled back to 45–60 (not extended).

**Long Call trend-quality:**

| Condition | Points |
|-----------|--------|
| Trend Up/Strong Up AND RSI 45–60 | +20 (clean entry in uptrend) |
| Trend Up/Strong Up AND RSI 60–70 | +10 (acceptable) |
| Trend Up/Strong Up AND RSI > 70 | -5 (chasing, overbought) |
| Trend Up/Strong Up AND RSI < 45 | +15 (dip within uptrend) |
| 1M return 2–8% (steady, not extended) | +8 |
| 1M return > 10% (extended move) | -5 |
| Trend Sideways | +2 |
| Trend Down | -5 |
| Trend Strong Down | -10 |

**Long Put trend-quality:** mirrors above for bearish direction.  
Best entry: trend DOWN, RSI 30–55 (not yet oversold), 1M return -2% to -8%.

### Directional Bonuses

**Long Call:**

| Signal | Points |
|--------|--------|
| Market bias bullish | +5 |
| Call OI > 5000 | +10 |
| Call OI > 1000 | +5 |

**Long Put:**

| Signal | Points |
|--------|--------|
| Market bias bearish | +5 |
| Put OI > 5000 | +10 |
| Put OI > 1000 | +5 |

---

## 6. SMC Signals

### Data Required
`batch_screen_fundamentals` currently stores only `Close`. Update to also extract and store `Open`, `High`, `Low` from the existing `yf.download` result (already fetched, not stored).

### New Function: `compute_smc_signals(ohlcv_df) -> dict`

**Break of Structure (BoS)**
- Identify swing highs/lows using 5-candle lookback each side.
- Bullish BoS: most recent close > previous swing high.
- Bearish BoS: most recent close < previous swing low.

**Change of Character (CHoCH)**
- Track sequence of swing highs/lows (HH/HL for uptrend, LL/LH for downtrend).
- Bullish CHoCH: downtrend makes first Higher High.
- Bearish CHoCH: uptrend makes first Lower Low.

**Discount / Premium Zone**
- Uses the same 5-candle swing points identified by BoS logic. `range_high` = most recent swing high, `range_low` = most recent swing low.
- Equilibrium = 50% of range.
- Discount: current price < equilibrium → good Long Call entry.
- Premium: current price > equilibrium → good Long Put entry.

**Order Blocks (OB)**
- Bullish OB: last bearish (red) candle before a significant up-move (≥3 consecutive up candles or single candle body > 1.5× 14-period ATR).
- Price within bullish OB zone (OB high/low) = near support.
- Bearish OB: last bullish candle before a significant down-move.
- Tolerance: price within 1% of OB zone.

**Fair Value Gaps (FVG)**
- Scan last 20 candles for 3-candle FVG patterns.
- Bullish FVG: `candle[i+2].low > candle[i].high` (gap up).
- Bearish FVG: `candle[i+2].high < candle[i].low` (gap down).
- "Unfilled": current price has not retraced into the gap zone.
- `in_bullish_fvg`: current price within an unfilled bullish FVG.
- `in_bearish_fvg`: current price within an unfilled bearish FVG.

### SMC Scoring

| SMC Signal | Long Call | Long Put |
|------------|-----------|----------|
| Bullish BoS confirmed | +15 | -10 |
| Bearish BoS confirmed | -10 | +15 |
| Bullish CHoCH | +5 | -8 |
| Bearish CHoCH | -8 | +5 |
| Price in Discount zone | +10 | -5 |
| Price in Premium zone | -5 | +10 |
| Price at/near Bullish OB | +15 | -10 |
| Price at/near Bearish OB | -10 | +15 |
| Price in Bullish FVG | +10 | -5 |
| Price in Bearish FVG | -5 | +10 |

### Display
Active SMC signals shown as tags on Top 10 cards and in screener table tooltip:
- `[Bullish BoS]` `[Discount Zone]` `[Near Bullish OB]` `[In FVG]`

---

## 7. Files Changed

| File | Changes |
|------|---------|
| `screener.py` | Add OHLCV storage in `batch_screen_fundamentals`; add `compute_smc_signals()`; add `atm_mid_price`, `atm_spread_pct` to `enrich_with_iv`; update `score_strategies` with new weights + SMC signals |
| `app.py` | Add "My Account" sidebar panel; remove Quick Screen mode; add `Cost/Contract` + `Affordable` columns to table; update Top 10 cards with budget row + SMC tags; add "Affordable only" filter |

No new files. No new dependencies.

---

## 8. Out of Scope

- Vertical spreads (user chose Long Call / Long Put only)
- Liquidity sweep detection (too noisy to automate reliably)
- Paper trading / trade log
- Push notifications / alerts
