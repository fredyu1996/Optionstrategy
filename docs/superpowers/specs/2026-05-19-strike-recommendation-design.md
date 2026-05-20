# Design: Strike Recommendation + Expandable Options Chain

**Date:** 2026-05-19  
**Scope:** S&P 500 Options Strategy Screener — `app.py`, `screener.py`  
**Strategies:** Long Call, Long Put only

---

## 1. Goals

1. For each Top 10 pick, recommend a specific strike price with evidence.
2. Show why that strike was chosen: IV condition, SMC strength, delta target, cost vs budget.
3. Let the user expand to see the surrounding 5-strike chain for comparison.

---

## 2. Strike Selection Logic

### Step 1 — Delta target range based on IV/HV

| IV/HV | Condition | Target Delta Range |
|-------|-----------|--------------------|
| < 0.8 | Cheap vol | 0.45–0.55 (ATM) |
| 0.8–1.0 | Fair vol | 0.35–0.45 |
| > 1.0 | Expensive vol | 0.25–0.35 |

For Long Put, target delta is the negative mirror: –0.45 to –0.55, –0.35 to –0.45, –0.25 to –0.35.

### Step 2 — SMC confidence adjustment

Count active bullish SMC signals for Long Call (or bearish for Long Put):
- `bullish_bos`, `discount_zone`, `near_bullish_ob`, `in_bullish_fvg`, `bullish_choch`

If ≥ 3 signals active → shift delta target –0.05 (more OTM, higher conviction)  
If 0–1 signals → shift +0.05 toward ATM (less conviction, reduce risk)

### Step 3 — Budget filter + selection

From strikes at the target expiry:
1. Filter to strikes within the adjusted delta target range.
2. From those, keep only affordable: `cost_per_contract = mid_price × 100 ≤ max_risk_usd`.
3. Pick the strike with delta closest to the center of the range.
4. If no affordable strike in range: expand to all affordable strikes, pick closest delta. Flag as "outside ideal range".
5. If no affordable strikes at all: pick cheapest OTM strike, flag as "over budget".

### Evidence string

Build a human-readable reason shown on the card:
> "IV/HV 0.68 (cheap vol) → ATM target. 3 SMC signals → shifted slightly OTM. Best affordable: Δ 0.41."

---

## 3. New Function: `get_strike_recommendation`

```python
def get_strike_recommendation(
    ticker_obj,          # yf.Ticker
    strategy: str,       # "Long Call" | "Long Put"
    iv_hv: float,        # IV/HV ratio from enrich_with_iv
    smc: dict,           # smc_* signals dict from batch_screen_fundamentals
    budget_config: dict, # {max_risk_usd: float}
    expiry: str,         # expiry date string (best 30-DTE expiry)
) -> dict:
```

**Returns:**
```python
{
    "strike": float,          # e.g. 185.0
    "expiry": str,            # e.g. "2025-06-20"
    "dte": int,               # days to expiry
    "delta": float,           # e.g. 0.41
    "gamma": float,
    "theta": float,           # negative number, e.g. -6.20
    "cost": float,            # cost per contract in USD
    "affordable": bool,
    "breakeven": float,       # strike + cost/100 for call; strike - cost/100 for put
    "iv_crush_warning": bool, # True if iv_hv > 1.0
    "delta_target_center": float,  # center of the chosen target range
    "smc_active_count": int,
    "reason": str,            # human-readable evidence string
    "chain_df": pd.DataFrame, # 5 nearest strikes for expandable table
}
```

**`chain_df` columns:** `strike`, `delta`, `cost`, `theta`, `affordable`  
Nearest 5 strikes: 2 above + 2 below the recommended strike + the recommended strike itself.  
Sorted by strike ascending.

---

## 4. UI — Recommendation Block on Top 10 Cards

Added below the existing SMC tags row, above the budget row.

```
Recommended Entry
─────────────────────────────────────────────────────
Buy $185 Call  |  Jun 20  (31 DTE)
Δ 0.41  |  Γ 0.041  |  Θ $-6.20/day  |  Breakeven: $188.40
Cost: $94/contract  ✓ within $120 max
⚠ IV elevated (IV/HV 1.12) — risk of IV crush
Reason: IV fair vol → Δ 0.35–0.45 target. 3 SMC signals → shifted OTM. Best affordable: Δ 0.41.
```

- Strike label: `Buy $XXX Call` or `Sell $XXX Put` (direction matches strategy)
- IV crush warning shown only when `iv_crush_warning=True`
- "Outside ideal range" or "Over budget" badge shown when applicable

---

## 5. UI — Expandable Options Chain

`st.expander("View options chain")` below the recommendation block.

```
Strike  | Delta | Cost    | Θ/day   | Affordable
$175    | 0.61  | $162    | $-9.40  | ✗
$180    | 0.52  | $128    | $-7.80  | ✗
$185 ★  | 0.41  | $94     | $-6.20  | ✓   ← recommended
$190    | 0.28  | $63     | $-4.50  | ✓
$195    | 0.17  | $39     | $-2.90  | ✓
```

- Recommended row highlighted (★ marker + bold or colored row via `st.dataframe` styling)
- `Affordable` column shows ✓ / ✗
- Cost column shows `$XX` USD

---

## 6. Files Changed

| File | Changes |
|------|---------|
| `screener.py` | Add `get_strike_recommendation()` |
| `app.py` | Update Top 10 Long Call + Long Put card rendering to call `get_strike_recommendation()` and display recommendation block + expander |

No new files. No new dependencies (yfinance options chain already used).

---

## 7. Out of Scope

- Recommending specific number of contracts (position sizing beyond max risk)
- Multi-leg strategies
- Historical back-test of recommended strikes
- Real-time Greeks update (uses data from the last screen run)
