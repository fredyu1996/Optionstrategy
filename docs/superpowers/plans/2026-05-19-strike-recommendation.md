# Strike Recommendation + Expandable Options Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For each Top 10 pick, recommend a specific strike price based on IV/HV, SMC strength, and budget, and let the user expand to see the surrounding options chain.

**Architecture:** Two pure helper functions (`_compute_delta_range`, `_select_best_strike`) hold all selection logic and are fully unit-testable. A cached orchestrator (`get_strike_recommendation`) fetches live options data and calls the helpers. `app.py` calls the orchestrator per card and renders the recommendation block + `st.expander` chain table.

**Tech Stack:** Python 3.14, yfinance, pandas, numpy, Streamlit 1.57, pytest

---

## File Map

| File | Changes |
|------|---------|
| `screener.py` | Add `_empty_recommendation`, `_compute_delta_range`, `_select_best_strike`, `get_strike_recommendation` |
| `app.py` | Update Top 10 card loop (Long Call + Long Put) to call recommendation, render block + expander, replace old budget line |
| `tests/test_screener.py` | Add 6 tests for `_compute_delta_range` and `_select_best_strike` |

---

## Task 1: Pure helpers + tests in `screener.py`

**Files:**
- Modify: `screener.py` (append after `_empty_smc`)
- Test: `tests/test_screener.py` (append)

### Step 1 — Write failing tests

Append to `tests/test_screener.py`. Add `import pytest` to the imports at the top of the file first:

```python
import pytest
from screener import _compute_delta_range, _select_best_strike, _empty_recommendation


# ── _compute_delta_range ────────────────────────────────────────────────────

def test_delta_range_cheap_iv_targets_atm():
    """IV/HV < 0.8 → delta range 0.45–0.55 for calls (before SMC adj)."""
    lo, hi, center = _compute_delta_range(iv_hv=0.6, smc_count=2, is_call=True)
    assert lo == pytest.approx(0.45), f"lo={lo}"
    assert hi == pytest.approx(0.55), f"hi={hi}"


def test_delta_range_expensive_iv_targets_otm():
    """IV/HV > 1.0 → delta range 0.25–0.35 for calls (before SMC adj)."""
    lo, hi, center = _compute_delta_range(iv_hv=1.2, smc_count=2, is_call=True)
    assert lo == pytest.approx(0.25), f"lo={lo}"
    assert hi == pytest.approx(0.35), f"hi={hi}"


def test_delta_range_high_smc_shifts_otm():
    """3+ SMC signals should shift range -0.05 (more OTM)."""
    lo_low_smc, hi_low_smc, _ = _compute_delta_range(iv_hv=0.9, smc_count=2, is_call=True)
    lo_high_smc, hi_high_smc, _ = _compute_delta_range(iv_hv=0.9, smc_count=3, is_call=True)
    assert lo_high_smc < lo_low_smc
    assert hi_high_smc < hi_low_smc


def test_delta_range_put_negated():
    """For puts, delta range should be negative (mirror of call range)."""
    lo, hi, center = _compute_delta_range(iv_hv=0.9, smc_count=2, is_call=False)
    assert lo < 0
    assert hi < 0
    assert center < 0


# ── _select_best_strike ─────────────────────────────────────────────────────

def _make_strikes(deltas, costs, stock_price=100.0):
    """Helper to build strike_data list for _select_best_strike tests."""
    return [
        {
            'strike': stock_price * (1 + (d - 0.5) * 0.2),  # rough strike from delta
            'delta': d,
            'gamma': 0.04,
            'theta': -5.0,
            'cost': c,
            'affordable': c <= 120.0,
        }
        for d, c in zip(deltas, costs)
    ]


def test_select_best_strike_picks_closest_to_center():
    """Should pick the affordable strike with delta closest to range center."""
    strikes = _make_strikes(
        deltas=[0.25, 0.38, 0.45, 0.55, 0.65],
        costs=[50, 80, 100, 130, 160],
    )
    # target range 0.35–0.45, center 0.40, max_risk 120
    chosen, flag = _select_best_strike(strikes, 0.35, 0.45, 0.40, 120.0, True, 100.0)
    assert chosen['delta'] == pytest.approx(0.38)
    assert flag is None


def test_select_best_strike_falls_back_outside_range():
    """No affordable strike in range → pick closest affordable, flag 'outside_ideal_range'."""
    strikes = _make_strikes(
        deltas=[0.25, 0.38, 0.45, 0.55, 0.65],
        costs=[50, 200, 200, 200, 200],  # only delta=0.25 is affordable
    )
    chosen, flag = _select_best_strike(strikes, 0.35, 0.45, 0.40, 120.0, True, 100.0)
    assert flag == 'outside_ideal_range'
    assert chosen['affordable'] is True


def test_select_best_strike_over_budget_flag():
    """Nothing affordable → cheapest OTM strike, flag 'over_budget'."""
    strikes = _make_strikes(
        deltas=[0.25, 0.38, 0.55],
        costs=[200, 300, 400],
    )
    chosen, flag = _select_best_strike(strikes, 0.35, 0.45, 0.40, 120.0, True, 100.0)
    assert flag == 'over_budget'
    assert chosen['cost'] == min(200, 300, 400)
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd C:\Users\fredy\Optionstrategy
python -m pytest tests/test_screener.py::test_delta_range_cheap_iv_targets_atm tests/test_screener.py::test_select_best_strike_picks_closest_to_center -v
```

Expected: `ImportError: cannot import name '_compute_delta_range'`

- [ ] **Step 3: Implement helpers in `screener.py`**

Add after the `_empty_smc` function (around line 216):

```python
def _empty_recommendation(max_risk_usd: float) -> dict:
    return {
        'strike': None, 'expiry': None, 'dte': None,
        'delta': np.nan, 'gamma': np.nan, 'theta': np.nan,
        'cost': np.nan, 'affordable': False, 'breakeven': np.nan,
        'iv_crush_warning': False, 'delta_target_center': 0.40,
        'smc_active_count': 0, 'reason': 'No options data available.',
        'chain_df': pd.DataFrame(), 'flag': 'no_data',
    }


def _compute_delta_range(iv_hv: float, smc_count: int, is_call: bool) -> tuple:
    """Return (lo, hi, center) for the desired delta range."""
    if np.isnan(iv_hv) or iv_hv <= 0:
        lo, hi = 0.35, 0.45
    elif iv_hv < 0.8:
        lo, hi = 0.45, 0.55
    elif iv_hv <= 1.0:
        lo, hi = 0.35, 0.45
    else:
        lo, hi = 0.25, 0.35

    if smc_count >= 3:
        lo -= 0.05
        hi -= 0.05
    elif smc_count <= 1:
        lo += 0.05
        hi += 0.05

    lo = max(0.05, lo)
    hi = min(0.95, hi)
    center = (lo + hi) / 2

    if not is_call:
        lo, hi, center = -hi, -lo, -center
    return lo, hi, center


def _select_best_strike(
    strike_data: list,
    delta_lo: float,
    delta_hi: float,
    delta_center: float,
    max_risk_usd: float,
    is_call: bool,
    stock_price: float,
) -> tuple:
    """
    From list of strike dicts (keys: strike, delta, cost, affordable, gamma, theta),
    return (chosen_dict, flag). flag: None | 'outside_ideal_range' | 'over_budget'.
    """
    in_range = [s for s in strike_data if delta_lo <= s['delta'] <= delta_hi and s['affordable']]
    if in_range:
        return min(in_range, key=lambda x: abs(x['delta'] - delta_center)), None

    affordable = [s for s in strike_data if s['affordable']]
    if affordable:
        return min(affordable, key=lambda x: abs(x['delta'] - delta_center)), 'outside_ideal_range'

    candidates = [s for s in strike_data if s['strike'] >= stock_price] if is_call \
        else [s for s in strike_data if s['strike'] <= stock_price]
    if not candidates:
        candidates = strike_data
    return min(candidates, key=lambda x: x['cost']), 'over_budget'
```

- [ ] **Step 4: Run tests — expect PASS**

```
python -m pytest tests/test_screener.py::test_delta_range_cheap_iv_targets_atm tests/test_screener.py::test_delta_range_expensive_iv_targets_otm tests/test_screener.py::test_delta_range_high_smc_shifts_otm tests/test_screener.py::test_delta_range_put_negated tests/test_screener.py::test_select_best_strike_picks_closest_to_center tests/test_screener.py::test_select_best_strike_falls_back_outside_range tests/test_screener.py::test_select_best_strike_over_budget_flag -v
```

Expected: 7 PASSED

- [ ] **Step 5: Run full test suite — all 13 prior tests still pass**

```
python -m pytest tests/test_screener.py -v
```

Expected: 20 PASSED (13 prior + 7 new)

- [ ] **Step 6: Commit**

```
git add screener.py tests/test_screener.py
git commit -m "feat: add _compute_delta_range and _select_best_strike helpers with tests"
```

---

## Task 2: `get_strike_recommendation` orchestrator in `screener.py`

**Files:**
- Modify: `screener.py` (append after `_select_best_strike`)

- [ ] **Step 1: Implement `get_strike_recommendation`**

Add after `_select_best_strike`:

```python
@st.cache_data(ttl=900)
def get_strike_recommendation(
    ticker_str: str,
    strategy: str,
    iv_hv: float,
    smc_tuple: tuple,
    max_risk_usd: float,
) -> dict:
    """
    Recommend a specific strike for ticker based on IV/HV, SMC signals, and budget.

    smc_tuple order: (bos_bull, bos_bear, choch_bull, choch_bear, disc, prem,
                      ob_bull, ob_bear, fvg_bull, fvg_bear)
    """
    smc_keys = [
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
        'discount_zone', 'premium_zone', 'near_bullish_ob', 'near_bearish_ob',
        'in_bullish_fvg', 'in_bearish_fvg',
    ]
    smc = dict(zip(smc_keys, smc_tuple))
    is_call = (strategy == 'Long Call')

    relevant_keys = (
        ['bos_bullish', 'discount_zone', 'near_bullish_ob', 'in_bullish_fvg', 'choch_bullish']
        if is_call else
        ['bos_bearish', 'premium_zone', 'near_bearish_ob', 'in_bearish_fvg', 'choch_bearish']
    )
    smc_count = sum(1 for k in relevant_keys if smc.get(k, False))

    delta_lo, delta_hi, delta_center = _compute_delta_range(iv_hv, smc_count, is_call)

    try:
        t = yf.Ticker(ticker_str)
        expiry = _find_best_expiry(t, target_dte=30)
        if expiry is None:
            return _empty_recommendation(max_risk_usd)

        today = datetime.now().date()
        exp_date = datetime.strptime(expiry, '%Y-%m-%d').date()
        dte = max((exp_date - today).days, 1)
        T = dte / 365

        hist = t.history(period='1d')
        S = float(hist['Close'].iloc[-1]) if not hist.empty else 0.0
        if S <= 0:
            return _empty_recommendation(max_risk_usd)

        chain = t.option_chain(expiry)
        options = chain.calls if is_call else chain.puts
        if options.empty:
            return _empty_recommendation(max_risk_usd)
    except Exception:
        return _empty_recommendation(max_risk_usd)

    strike_data = []
    for _, opt_row in options.iterrows():
        try:
            K = float(opt_row['strike'])
            iv_val = float(opt_row.get('impliedVolatility') or 0)
            bid = float(opt_row.get('bid') or 0)
            ask = float(opt_row.get('ask') or 0)
            if iv_val <= 0 or bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            cost = mid * 100
            g = compute_greeks(S, K, T, 0.045, iv_val, 'call' if is_call else 'put')
            if np.isnan(g['delta']):
                continue
            strike_data.append({
                'strike': K,
                'delta': g['delta'],
                'gamma': g['gamma'],
                'theta': g['theta'],
                'cost': cost,
                'affordable': cost <= max_risk_usd,
            })
        except Exception:
            continue

    if not strike_data:
        return _empty_recommendation(max_risk_usd)

    strike_data.sort(key=lambda x: x['strike'])
    chosen, flag = _select_best_strike(
        strike_data, delta_lo, delta_hi, delta_center, max_risk_usd, is_call, S
    )

    # Build human-readable reason
    if np.isnan(iv_hv) or iv_hv <= 0:
        iv_desc = "fair vol → Δ 0.35–0.45 target"
    elif iv_hv < 0.8:
        iv_desc = "cheap vol → ATM target"
    elif iv_hv <= 1.0:
        iv_desc = "fair vol → Δ 0.35–0.45 target"
    else:
        iv_desc = "expensive vol → Δ 0.25–0.35 target"

    if smc_count >= 3:
        smc_adj = f"{smc_count} SMC signals → shifted OTM"
    elif smc_count <= 1:
        smc_adj = f"{smc_count} SMC signals → shifted toward ATM"
    else:
        smc_adj = f"{smc_count} SMC signals → no adjustment"

    reason = f"IV/HV {iv_hv:.2f} ({iv_desc}). {smc_adj}. Best affordable: Δ {chosen['delta']:.2f}."

    cost_per_share = chosen['cost'] / 100
    breakeven = (chosen['strike'] + cost_per_share if is_call
                 else chosen['strike'] - cost_per_share)

    # 5-strike chain: 2 below + chosen + 2 above
    strikes_list = [s['strike'] for s in strike_data]
    try:
        idx = strikes_list.index(chosen['strike'])
    except ValueError:
        idx = len(strike_data) // 2
    subset = strike_data[max(0, idx - 2): min(len(strike_data), idx + 3)]

    chain_df = pd.DataFrame([{
        'Strike': f"${s['strike']:.0f}" + (" ★" if s['strike'] == chosen['strike'] else ""),
        'Delta': round(s['delta'], 3),
        'Cost': f"${s['cost']:.0f}",
        'Θ/day': f"${s['theta']:.2f}",
        'Affordable': "✓" if s['affordable'] else "✗",
    } for s in subset])

    return {
        'strike': chosen['strike'],
        'expiry': expiry,
        'dte': dte,
        'delta': chosen['delta'],
        'gamma': chosen['gamma'],
        'theta': chosen['theta'],
        'cost': chosen['cost'],
        'affordable': chosen['affordable'],
        'breakeven': breakeven,
        'iv_crush_warning': (not (np.isnan(iv_hv) or iv_hv <= 0) and iv_hv > 1.0),
        'delta_target_center': delta_center,
        'smc_active_count': smc_count,
        'reason': reason,
        'chain_df': chain_df,
        'flag': flag,
    }
```

- [ ] **Step 2: Run full test suite — all tests still pass**

```
python -m pytest tests/test_screener.py -v
```

Expected: 20 PASSED

- [ ] **Step 3: Commit**

```
git add screener.py
git commit -m "feat: add get_strike_recommendation orchestrator"
```

---

## Task 3: UI update in `app.py`

**Files:**
- Modify: `app.py`

### Step 1 — Add import

In `app.py`, update the `from screener import (...)` block (around line 16) to add `get_strike_recommendation`:

```python
from screener import (
    get_sp500_tickers,
    get_macro_data,
    batch_screen_fundamentals,
    enrich_with_iv,
    score_strategies,
    get_strike_recommendation,
)
```

- [ ] **Step 2: Add `_build_rec_html` helper to `app.py`**

Add this function near `build_smc_tags` (around line 267):

```python
def _build_rec_html(rec: dict, strategy: str, iv_hv_val: float, max_risk_usd: float) -> str:
    """Build HTML for the recommended entry block inside a Top 10 card."""
    if rec['strike'] is None:
        return (
            '<div style="border-top:1px solid #334155;margin-top:0.4rem;padding-top:0.4rem;'
            'font-size:0.75rem;color:#475569;">No recommendation available.</div>'
        )

    is_call = (strategy == 'Long Call')
    try:
        expiry_label = datetime.strptime(rec['expiry'], '%Y-%m-%d').strftime('%b %d')
    except Exception:
        expiry_label = rec['expiry'] or ''

    dte_str = f"{rec['dte']} DTE" if rec['dte'] else ''
    direction = "Buy" if is_call else "Buy"
    option_type = "Call" if is_call else "Put"

    flag_html = ''
    if rec.get('flag') == 'outside_ideal_range':
        flag_html = ' <span style="color:#f59e0b;font-size:0.72rem;">⚠ outside ideal range</span>'
    elif rec.get('flag') == 'over_budget':
        flag_html = ' <span style="color:#ef4444;font-size:0.72rem;">⚠ over budget</span>'

    aff_color = '#10b981' if rec['affordable'] else '#ef4444'
    aff_label = '✓ Affordable' if rec['affordable'] else f'✗ Over Budget (${max_risk_usd:.0f} max)'

    gamma_str = f"{rec['gamma']:.4f}" if not np.isnan(rec['gamma']) else 'N/A'
    theta_str = f"${rec['theta']:.2f}/day" if not np.isnan(rec['theta']) else 'N/A'
    breakeven_str = f"${rec['breakeven']:.2f}" if not np.isnan(rec['breakeven']) else 'N/A'
    cost_str = f"${rec['cost']:.0f}" if not np.isnan(rec['cost']) else 'N/A'
    target_str = f"+${rec['cost'] * 2:.0f}" if not np.isnan(rec['cost']) else 'N/A'

    crush_html = ''
    if rec.get('iv_crush_warning'):
        crush_html = (
            f'<div style="font-size:0.75rem;color:#f59e0b;margin-top:0.15rem;">'
            f'⚠ IV elevated (IV/HV {iv_hv_val:.2f}) — risk of IV crush</div>'
        )

    return (
        '<div style="border-top:1px solid #334155;margin-top:0.4rem;padding-top:0.4rem;">'
        '<div style="font-size:0.75rem;color:#94a3b8;font-weight:700;margin-bottom:0.15rem;">Recommended Entry</div>'
        f'<div style="font-size:0.85rem;color:#e2e8f0;">'
        f'{direction} ${rec["strike"]:.0f} {option_type} &nbsp;|&nbsp; {expiry_label} ({dte_str}){flag_html}'
        f'</div>'
        f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:0.1rem;">'
        f'Δ {rec["delta"]:.3f} &nbsp;|&nbsp; Γ {gamma_str} &nbsp;|&nbsp; Θ {theta_str} &nbsp;|&nbsp; Breakeven: {breakeven_str}'
        f'</div>'
        f'<div style="font-size:0.78rem;margin-top:0.1rem;">'
        f'Cost: <strong>{cost_str}/contract</strong> &nbsp;|&nbsp; '
        f'<span style="color:{aff_color};">{aff_label}</span> &nbsp;|&nbsp; '
        f'2:1 Target: <strong style="color:#10b981;">{target_str}</strong>'
        f'</div>'
        f'{crush_html}'
        f'<div style="font-size:0.72rem;color:#475569;margin-top:0.2rem;">{rec["reason"]}</div>'
        '</div>'
    )
```

- [ ] **Step 3: Update Long Call card loop**

Find the Long Call card loop starting around line 786. The loop starts `for _, r in top_lc.iterrows():`.

Replace the entire loop body (from the variables block through `st.markdown(...)`) with the updated version below. Key changes:
1. Add `smc_tuple` construction and `get_strike_recommendation` call
2. Replace old `budget_line` HTML with `_build_rec_html` call
3. Add `st.expander` chain table after the card markdown

```python
    with col_lc:
        st.markdown("#### 📈 Top 5 Long Call")
        for _, r in top_lc.iterrows():
            trend_val = r.get('trend', '—')
            iv_hv_val = r.get('iv_hv_ratio', np.nan)
            iv_str = f"{iv_hv_val:.2f}" if not np.isnan(iv_hv_val) else "N/A"
            delta_val = r.get('atm_delta', np.nan)
            delta_str = f"{delta_val:.3f}" if not np.isnan(delta_val) else "N/A"
            theta_val = r.get('atm_theta', np.nan)
            theta_str = f"${theta_val:.2f}/day" if not np.isnan(theta_val) else "N/A"
            call_oi_val = r.get('atm_call_oi', np.nan)
            oi_str = f"{int(call_oi_val):,}" if not np.isnan(call_oi_val) else "N/A"
            ret_val = r.get('ret_1m', np.nan)
            ret_str = f"{ret_val:+.1f}%" if not np.isnan(ret_val) else "N/A"
            score = r.get('lc_score', 0)
            price_val = r.get('price', 0)

            smc_tuple = (
                bool(r.get('smc_bos_bullish')), bool(r.get('smc_bos_bearish')),
                bool(r.get('smc_choch_bullish')), bool(r.get('smc_choch_bearish')),
                bool(r.get('smc_discount_zone')), bool(r.get('smc_premium_zone')),
                bool(r.get('smc_near_bullish_ob')), bool(r.get('smc_near_bearish_ob')),
                bool(r.get('smc_in_bullish_fvg')), bool(r.get('smc_in_bearish_fvg')),
            )
            rec = get_strike_recommendation(
                r['ticker'], 'Long Call',
                float(iv_hv_val) if not np.isnan(iv_hv_val) else 0.9,
                smc_tuple,
                budget_config['max_risk_usd'],
            )
            rec_html = _build_rec_html(rec, 'Long Call', iv_hv_val, budget_config['max_risk_usd'])

            smc_tags = build_smc_tags(r.to_dict())
            smc_row = f'<div style="margin:0.25rem 0;">{smc_tags}</div>' if smc_tags else ''

            with st.container():
                st.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-left:4px solid #10b981;border-radius:0.5rem;padding:0.75rem 1rem;margin-bottom:0.5rem;">
<div style="display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">{r['ticker']}</span>
  <span style="font-size:0.85rem;font-weight:700;color:#10b981;">Score: {score:.0f}</span>
</div>
<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.25rem;">
  ${price_val:.2f} &nbsp;|&nbsp; {_trend_emoji(trend_val)} {trend_val} &nbsp;|&nbsp; 1M: {ret_str}
</div>
{smc_row}
<div style="font-size:0.78rem;color:#64748b;margin-top:0.2rem;">
  IV/HV: {iv_str} &nbsp;|&nbsp; Δ {delta_str} &nbsp;|&nbsp; Θ {theta_str} &nbsp;|&nbsp; OI: {oi_str}
</div>
{rec_html}
</div>""", unsafe_allow_html=True)
                if not rec['chain_df'].empty:
                    with st.expander(f"Options chain — {r['ticker']}"):
                        st.dataframe(rec['chain_df'], use_container_width=True, hide_index=True)
```

- [ ] **Step 4: Update Long Put card loop**

Find the Long Put card loop starting around line 844. Apply the identical pattern — only these differ:
- `strategy = 'Long Put'`
- `score = r.get('lp_score', 0)`
- `call_oi_val` → `put_oi_val = r.get('atm_put_oi', np.nan)` and `oi_str` uses that
- Card border color: `#ef4444`
- Score color: `#ef4444`
- `get_strike_recommendation(r['ticker'], 'Long Put', ...)`
- `_build_rec_html(rec, 'Long Put', ...)`

Full Long Put loop body (replacing lines ~844–900):

```python
    with col_lp:
        st.markdown("#### 📉 Top 5 Long Put")
        for _, r in top_lp.iterrows():
            trend_val = r.get('trend', '—')
            iv_hv_val = r.get('iv_hv_ratio', np.nan)
            iv_str = f"{iv_hv_val:.2f}" if not np.isnan(iv_hv_val) else "N/A"
            delta_val = r.get('atm_delta', np.nan)
            delta_str = f"{delta_val:.3f}" if not np.isnan(delta_val) else "N/A"
            theta_val = r.get('atm_theta', np.nan)
            theta_str = f"${theta_val:.2f}/day" if not np.isnan(theta_val) else "N/A"
            put_oi_val = r.get('atm_put_oi', np.nan)
            oi_str = f"{int(put_oi_val):,}" if not np.isnan(put_oi_val) else "N/A"
            ret_val = r.get('ret_1m', np.nan)
            ret_str = f"{ret_val:+.1f}%" if not np.isnan(ret_val) else "N/A"
            score = r.get('lp_score', 0)
            price_val = r.get('price', 0)

            smc_tuple = (
                bool(r.get('smc_bos_bullish')), bool(r.get('smc_bos_bearish')),
                bool(r.get('smc_choch_bullish')), bool(r.get('smc_choch_bearish')),
                bool(r.get('smc_discount_zone')), bool(r.get('smc_premium_zone')),
                bool(r.get('smc_near_bullish_ob')), bool(r.get('smc_near_bearish_ob')),
                bool(r.get('smc_in_bullish_fvg')), bool(r.get('smc_in_bearish_fvg')),
            )
            rec = get_strike_recommendation(
                r['ticker'], 'Long Put',
                float(iv_hv_val) if not np.isnan(iv_hv_val) else 0.9,
                smc_tuple,
                budget_config['max_risk_usd'],
            )
            rec_html = _build_rec_html(rec, 'Long Put', iv_hv_val, budget_config['max_risk_usd'])

            smc_tags = build_smc_tags(r.to_dict())
            smc_row = f'<div style="margin:0.25rem 0;">{smc_tags}</div>' if smc_tags else ''

            with st.container():
                st.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-left:4px solid #ef4444;border-radius:0.5rem;padding:0.75rem 1rem;margin-bottom:0.5rem;">
<div style="display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">{r['ticker']}</span>
  <span style="font-size:0.85rem;font-weight:700;color:#ef4444;">Score: {score:.0f}</span>
</div>
<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.25rem;">
  ${price_val:.2f} &nbsp;|&nbsp; {_trend_emoji(trend_val)} {trend_val} &nbsp;|&nbsp; 1M: {ret_str}
</div>
{smc_row}
<div style="font-size:0.78rem;color:#64748b;margin-top:0.2rem;">
  IV/HV: {iv_str} &nbsp;|&nbsp; Δ {delta_str} &nbsp;|&nbsp; Θ {theta_str} &nbsp;|&nbsp; OI: {oi_str}
</div>
{rec_html}
</div>""", unsafe_allow_html=True)
                if not rec['chain_df'].empty:
                    with st.expander(f"Options chain — {r['ticker']}"):
                        st.dataframe(rec['chain_df'], use_container_width=True, hide_index=True)
```

- [ ] **Step 5: Run full test suite**

```
python -m pytest tests/test_screener.py -v
```

Expected: 20 PASSED

- [ ] **Step 6: Commit**

```
git add app.py
git commit -m "feat: add strike recommendation block and expandable options chain to Top 10 cards"
```
