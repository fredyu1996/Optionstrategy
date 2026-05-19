# Budget-Aware SMC Screener Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add budget awareness (500 CAD / max $120 USD risk per trade), SMC signals (BoS, CHoCH, Discount/Premium, Order Blocks, FVGs), and a fixed scoring system to the S&P 500 options screener.

**Architecture:** All computation lives in `screener.py` (data fetching, SMC signals, scoring). All display/UI lives in `app.py`. No new files. No new pip dependencies — all logic uses existing numpy/pandas.

**Tech Stack:** Python 3.14, Streamlit, yfinance, pandas, numpy

---

## File Map

| File | What changes |
|------|-------------|
| `screener.py` | Store OHLCV (not just Close); add `_compute_atr`, `_empty_smc`, `compute_smc_signals`; add `atm_mid_price` + `atm_spread_pct` to `enrich_with_iv`; rewrite `score_strategies` |
| `app.py` | Remove Quick Screen mode; add My Account sidebar panel with CADUSD FX fetch; add `Cost/Contract` + `Affordable` columns to table; update Top 10 cards with budget row + SMC tags |
| `tests/test_screener.py` | Unit tests for `compute_smc_signals`, `score_strategies`, `_compute_atr` |
| `requirements.txt` | Add `pytest>=7.0.0` |

---

## Task 1: Install pytest + scaffold test file

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_screener.py`

- [ ] **Step 1: Add pytest to requirements.txt**

Open `requirements.txt` and add at the bottom:
```
pytest>=7.0.0
```

- [ ] **Step 2: Install pytest**

```bash
pip install pytest
```

Expected output: `Successfully installed pytest-...`

- [ ] **Step 3: Create tests/__init__.py**

Create empty file `tests/__init__.py` (makes tests a package).

- [ ] **Step 4: Create scaffold test file**

Create `tests/test_screener.py`:
```python
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from screener import _compute_atr, _empty_smc, compute_smc_signals, score_strategies
```

- [ ] **Step 5: Verify import works**

```bash
python -m pytest tests/test_screener.py --collect-only
```

Expected: `0 tests collected` (no failures — import succeeded)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/
git commit -m "chore: add pytest and test scaffold"
```

---

## Task 2: Add OHLCV storage + ATM mid/spread to screener.py

**Files:**
- Modify: `screener.py` — `batch_screen_fundamentals` (lines ~357–443), `enrich_with_iv` (lines ~446–545)

- [ ] **Step 1: Write failing tests for OHLCV columns**

Add to `tests/test_screener.py`:
```python
def test_batch_screen_fundamentals_stores_ohlcv():
    """batch_screen_fundamentals must store high_arr, low_arr, open_arr columns."""
    from screener import batch_screen_fundamentals
    df = batch_screen_fundamentals(['AAPL'])
    assert 'high_arr' in df.columns, "high_arr column missing"
    assert 'low_arr' in df.columns, "low_arr column missing"
    assert 'open_arr' in df.columns, "open_arr column missing"
    assert isinstance(df.iloc[0]['high_arr'], np.ndarray)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_screener.py::test_batch_screen_fundamentals_stores_ohlcv -v
```

Expected: `FAILED` — `AssertionError: high_arr column missing`

- [ ] **Step 3: Add OHLCV storage in batch_screen_fundamentals**

In `screener.py`, inside the `for ticker in tickers:` loop in `batch_screen_fundamentals`, after the existing `close_series` extraction block, add extraction for High, Low, Open and store as numpy arrays:

```python
# Extract High, Low, Open arrays (needed for SMC signals)
high_series = None
low_series = None
open_series = None
if not raw.empty:
    if len(tickers) == 1:
        if 'High' in raw.columns:
            high_series = raw['High'].dropna()
        if 'Low' in raw.columns:
            low_series = raw['Low'].dropna()
        if 'Open' in raw.columns:
            open_series = raw['Open'].dropna()
    elif ticker in raw.columns.get_level_values(0):
        high_series = raw[ticker]['High'].dropna()
        low_series = raw[ticker]['Low'].dropna()
        open_series = raw[ticker]['Open'].dropna()
    elif (isinstance(raw.columns, pd.MultiIndex) and
          'High' in raw.columns.get_level_values(0)):
        high_series = raw['High'][ticker].dropna()
        low_series = raw['Low'][ticker].dropna()
        open_series = raw['Open'][ticker].dropna()

row['high_arr'] = high_series.values if high_series is not None else np.array([])
row['low_arr'] = low_series.values if low_series is not None else np.array([])
row['open_arr'] = open_series.values if open_series is not None else np.array([])
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_screener.py::test_batch_screen_fundamentals_stores_ohlcv -v
```

Expected: `PASSED`

- [ ] **Step 5: Write failing tests for atm_mid_price and atm_spread_pct**

Add to `tests/test_screener.py`:
```python
def test_enrich_with_iv_adds_mid_and_spread():
    """enrich_with_iv must add atm_mid_price and atm_spread_pct columns."""
    from screener import batch_screen_fundamentals, enrich_with_iv
    df = batch_screen_fundamentals(['AAPL'])
    df = enrich_with_iv(df)
    assert 'atm_mid_price' in df.columns, "atm_mid_price column missing"
    assert 'atm_spread_pct' in df.columns, "atm_spread_pct column missing"
```

- [ ] **Step 6: Run test to confirm it fails**

```bash
python -m pytest tests/test_screener.py::test_enrich_with_iv_adds_mid_and_spread -v
```

Expected: `FAILED` — `AssertionError: atm_mid_price column missing`

- [ ] **Step 7: Add atm_mid_price + atm_spread_pct to enrich_with_iv**

In `screener.py`, inside `enrich_with_iv`, add two new lists before the `for i, row in df.iterrows():` loop:

```python
atm_mid_prices = []
atm_spread_pcts = []
```

Inside the loop, after the existing IV fetch, add ATM mid and spread computation. Find the block where `get_atm_iv` is called and add after it:

```python
# Compute ATM mid price and spread %
atm_mid = np.nan
atm_spread = np.nan
try:
    expirations = t.options
    if expirations:
        today = datetime.now().date()
        best_exp = None
        best_diff = float('inf')
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                dte_days = (exp_date - today).days
                if dte_days < 7:
                    continue
                diff = abs(dte_days - 30)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str
            except Exception:
                continue
        if best_exp:
            chain = t.option_chain(best_exp)
            calls = chain.calls
            price_val = row.get('price', 0)
            if not calls.empty and price_val > 0:
                strikes = calls['strike'].values
                atm_idx = int(np.argmin(np.abs(strikes - price_val)))
                atm_row = calls.iloc[atm_idx]
                bid = float(atm_row.get('bid', np.nan))
                ask = float(atm_row.get('ask', np.nan))
                if not np.isnan(bid) and not np.isnan(ask) and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    atm_mid = mid
                    atm_spread = (ask - bid) / mid if mid > 0 else np.nan
except Exception:
    pass

atm_mid_prices.append(atm_mid)
atm_spread_pcts.append(atm_spread)
```

After the loop, add to the df assignment block:
```python
df['atm_mid_price'] = atm_mid_prices
df['atm_spread_pct'] = atm_spread_pcts
```

- [ ] **Step 8: Run test to confirm it passes**

```bash
python -m pytest tests/test_screener.py::test_enrich_with_iv_adds_mid_and_spread -v
```

Expected: `PASSED`

- [ ] **Step 9: Commit**

```bash
git add screener.py tests/test_screener.py requirements.txt
git commit -m "feat: store OHLCV arrays and ATM mid/spread in screener"
```

---

## Task 3: Add _compute_atr, _empty_smc, compute_smc_signals to screener.py

**Files:**
- Modify: `screener.py` — add three new functions after `compute_greeks`

- [ ] **Step 1: Write failing tests for SMC signals**

Add to `tests/test_screener.py`:
```python
def _make_ohlcv(n=60, trend='up'):
    """Generate synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    base = 100.0
    prices = []
    for i in range(n):
        if trend == 'up':
            base += np.random.uniform(0, 1.5)
        elif trend == 'down':
            base -= np.random.uniform(0, 1.5)
        else:
            base += np.random.uniform(-0.5, 0.5)
        prices.append(base)
    closes = np.array(prices)
    highs = closes + np.random.uniform(0.5, 1.5, n)
    lows = closes - np.random.uniform(0.5, 1.5, n)
    opens = closes - np.random.uniform(-0.5, 0.5, n)
    return pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes})


def test_compute_smc_returns_all_keys():
    signals = compute_smc_signals(_make_ohlcv())
    expected_keys = [
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
        'discount_zone', 'premium_zone', 'near_bullish_ob', 'near_bearish_ob',
        'in_bullish_fvg', 'in_bearish_fvg',
    ]
    for key in expected_keys:
        assert key in signals, f"Missing key: {key}"
    for key in expected_keys:
        assert isinstance(signals[key], bool), f"{key} must be bool"


def test_compute_smc_uptrend_bos_bullish():
    """Strong uptrend should produce bullish BoS."""
    signals = compute_smc_signals(_make_ohlcv(n=60, trend='up'))
    assert signals['bos_bullish'] is True


def test_compute_smc_downtrend_bos_bearish():
    """Strong downtrend should produce bearish BoS."""
    signals = compute_smc_signals(_make_ohlcv(n=60, trend='down'))
    assert signals['bos_bearish'] is True


def test_empty_smc_all_false():
    result = _empty_smc()
    assert all(v is False for v in result.values())


def test_compute_atr_returns_positive():
    df = _make_ohlcv()
    atr = _compute_atr(df['High'].values, df['Low'].values, df['Close'].values)
    assert atr > 0


def test_compute_smc_short_data_returns_empty():
    """Less than 20 candles → all False."""
    df = _make_ohlcv(n=10)
    signals = compute_smc_signals(df)
    assert all(v is False for v in signals.values())
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_screener.py -k "smc or atr or empty_smc" -v
```

Expected: Multiple `FAILED` — `ImportError: cannot import name 'compute_smc_signals'`

- [ ] **Step 3: Add _compute_atr helper to screener.py**

Add after `compute_greeks` function (around line 189):

```python
def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Compute Average True Range over last `period` candles."""
    n = len(closes)
    if n < 2:
        return float(np.mean(highs - lows)) if len(highs) > 0 else 1.0
    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) >= period:
        return float(np.mean(trs[-period:]))
    return float(np.mean(trs)) if trs else 1.0


def _empty_smc() -> dict:
    """Return all-False SMC signal dict (used when data is insufficient)."""
    return {k: False for k in [
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
        'discount_zone', 'premium_zone', 'near_bullish_ob', 'near_bearish_ob',
        'in_bullish_fvg', 'in_bearish_fvg',
    ]}
```

- [ ] **Step 4: Add compute_smc_signals to screener.py**

Add immediately after `_empty_smc`:

```python
def compute_smc_signals(ohlcv_df: pd.DataFrame) -> dict:
    """
    Compute SMC signals from OHLCV DataFrame.
    Requires columns: Open, High, Low, Close. Min 20 rows.
    Returns dict of bool signals.
    """
    if len(ohlcv_df) < 20:
        return _empty_smc()

    highs = ohlcv_df['High'].values.astype(float)
    lows = ohlcv_df['Low'].values.astype(float)
    closes = ohlcv_df['Close'].values.astype(float)
    opens = ohlcv_df['Open'].values.astype(float)
    n = len(closes)
    current_price = closes[-1]

    # ── Swing highs/lows (5-candle lookback each side) ──────────────────────
    swing_highs = []  # list of (index, price)
    swing_lows = []
    for i in range(5, n - 5):
        if highs[i] == max(highs[i - 5:i + 6]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - 5:i + 6]):
            swing_lows.append((i, lows[i]))

    # ── Break of Structure ───────────────────────────────────────────────────
    bos_bullish = bool(swing_highs and current_price > swing_highs[-1][1])
    bos_bearish = bool(swing_lows and current_price < swing_lows[-1][1])

    # ── Change of Character ──────────────────────────────────────────────────
    choch_bullish = False
    choch_bearish = False
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        recent_highs = [x[1] for x in swing_highs[-3:]]
        recent_lows = [x[1] for x in swing_lows[-3:]]
        was_uptrend = len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[-2]
        was_downtrend = len(recent_highs) >= 2 and recent_highs[-1] < recent_highs[-2]
        if was_uptrend and len(recent_lows) >= 2 and recent_lows[-1] < recent_lows[-2]:
            choch_bearish = True
        if was_downtrend and len(recent_lows) >= 2 and recent_lows[-1] > recent_lows[-2]:
            choch_bullish = True

    # ── Discount / Premium Zone ──────────────────────────────────────────────
    discount_zone = False
    premium_zone = False
    if swing_highs and swing_lows:
        range_high = swing_highs[-1][1]
        range_low = swing_lows[-1][1]
        if range_high > range_low:
            equilibrium = (range_high + range_low) / 2
            discount_zone = bool(current_price < equilibrium)
            premium_zone = bool(current_price > equilibrium)

    # ── Order Blocks ─────────────────────────────────────────────────────────
    atr = _compute_atr(highs, lows, closes, period=14)
    near_bullish_ob = False
    near_bearish_ob = False
    lookback_start = max(0, n - 50)

    for i in range(lookback_start, n - 3):
        is_bearish_candle = closes[i] < opens[i]
        is_bullish_candle = closes[i] > opens[i]
        ob_high = max(opens[i], closes[i])
        ob_low = min(opens[i], closes[i])

        if is_bearish_candle and not near_bullish_ob:
            up_count = sum(1 for j in range(i + 1, min(i + 4, n)) if closes[j] > closes[j - 1])
            large_up = (i + 1 < n and closes[i + 1] - opens[i + 1] > 1.5 * atr
                        and closes[i + 1] > opens[i + 1])
            if up_count >= 3 or large_up:
                if ob_low * 0.99 <= current_price <= ob_high * 1.01:
                    near_bullish_ob = True

        if is_bullish_candle and not near_bearish_ob:
            down_count = sum(1 for j in range(i + 1, min(i + 4, n)) if closes[j] < closes[j - 1])
            large_down = (i + 1 < n and opens[i + 1] - closes[i + 1] > 1.5 * atr
                          and closes[i + 1] < opens[i + 1])
            if down_count >= 3 or large_down:
                if ob_low * 0.99 <= current_price <= ob_high * 1.01:
                    near_bearish_ob = True

    # ── Fair Value Gaps ──────────────────────────────────────────────────────
    in_bullish_fvg = False
    in_bearish_fvg = False
    fvg_start = max(0, n - 20)

    for i in range(fvg_start, n - 2):
        # Bullish FVG: candle[i+2].low > candle[i].high
        if lows[i + 2] > highs[i]:
            fvg_low, fvg_high = highs[i], lows[i + 2]
            if fvg_low <= current_price <= fvg_high:
                in_bullish_fvg = True
        # Bearish FVG: candle[i+2].high < candle[i].low
        if highs[i + 2] < lows[i]:
            fvg_low, fvg_high = highs[i + 2], lows[i]
            if fvg_low <= current_price <= fvg_high:
                in_bearish_fvg = True

    return {
        'bos_bullish': bos_bullish,
        'bos_bearish': bos_bearish,
        'choch_bullish': choch_bullish,
        'choch_bearish': choch_bearish,
        'discount_zone': discount_zone,
        'premium_zone': premium_zone,
        'near_bullish_ob': near_bullish_ob,
        'near_bearish_ob': near_bearish_ob,
        'in_bullish_fvg': in_bullish_fvg,
        'in_bearish_fvg': in_bearish_fvg,
    }
```

- [ ] **Step 5: Run SMC tests**

```bash
python -m pytest tests/test_screener.py -k "smc or atr or empty_smc" -v
```

Expected: All `PASSED`

- [ ] **Step 6: Call compute_smc_signals inside batch_screen_fundamentals**

Inside `batch_screen_fundamentals`, after the OHLCV arrays are stored on `row` (end of the try block, before `results.append(row)`), add:

```python
# Compute SMC signals
if (len(row.get('high_arr', [])) >= 20 and
        len(row.get('low_arr', [])) >= 20 and
        len(row.get('open_arr', [])) >= 20):
    ohlcv_for_smc = pd.DataFrame({
        'Open': row['open_arr'],
        'High': row['high_arr'],
        'Low': row['low_arr'],
        'Close': close_series.values,
    })
    smc = compute_smc_signals(ohlcv_for_smc)
else:
    smc = _empty_smc()

row.update({f'smc_{k}': v for k, v in smc.items()})
```

- [ ] **Step 7: Commit**

```bash
git add screener.py tests/test_screener.py
git commit -m "feat: add SMC signals (BoS, CHoCH, OB, FVG, discount/premium)"
```

---

## Task 4: Rewrite score_strategies with new weights + SMC

**Files:**
- Modify: `screener.py` — `score_strategies` function (lines ~548–714)

- [ ] **Step 1: Write failing tests for new scoring**

Add to `tests/test_screener.py`:
```python
def _make_row(**kwargs):
    """Build a minimal screener row dict for score_strategies testing."""
    defaults = {
        'ticker': 'TEST',
        'iv_hv_ratio': 0.7,
        'days_to_earnings': None,
        'rsi': 55.0,
        'ret_1m': 4.0,
        'trend': 'Up',
        'atm_delta': 0.48,
        'atm_theta': -8.0,
        'atm_call_oi': 6000.0,
        'atm_put_oi': 6000.0,
        'atm_spread_pct': 0.05,
        # SMC defaults (all False)
        'smc_bos_bullish': False, 'smc_bos_bearish': False,
        'smc_choch_bullish': False, 'smc_choch_bearish': False,
        'smc_discount_zone': False, 'smc_premium_zone': False,
        'smc_near_bullish_ob': False, 'smc_near_bearish_ob': False,
        'smc_in_bullish_fvg': False, 'smc_in_bearish_fvg': False,
    }
    defaults.update(kwargs)
    return defaults


def _score_one(row, macro=None):
    if macro is None:
        macro = {'vix_current': 18.0, 'market_bias': 'neutral'}
    import pandas as pd
    df = pd.DataFrame([row])
    result = score_strategies(df, macro)
    return result.iloc[0]['lc_score'], result.iloc[0]['lp_score']


def test_vix_bonus_removed():
    """Low VIX should NOT inflate scores beyond IV/HV contribution."""
    row_low_vix = _make_row(iv_hv_ratio=np.nan)
    row_high_vix = _make_row(iv_hv_ratio=np.nan)
    lc_low, _ = _score_one(row_low_vix, {'vix_current': 12.0, 'market_bias': 'neutral'})
    lc_high, _ = _score_one(row_high_vix, {'vix_current': 25.0, 'market_bias': 'neutral'})
    assert lc_low == lc_high, "VIX level should not affect score when IV/HV is NaN"


def test_overbought_penalizes_long_call():
    """RSI > 70 in uptrend should score lower than RSI 50 in uptrend."""
    lc_good, _ = _score_one(_make_row(trend='Up', rsi=52.0, ret_1m=4.0))
    lc_bad, _ = _score_one(_make_row(trend='Up', rsi=75.0, ret_1m=12.0))
    assert lc_good > lc_bad, "Overbought + extended momentum should score lower than clean entry"


def test_smc_bullish_bos_boosts_long_call():
    """Bullish BoS should increase LC score."""
    lc_no_bos, _ = _score_one(_make_row(smc_bos_bullish=False))
    lc_bos, _ = _score_one(_make_row(smc_bos_bullish=True))
    assert lc_bos > lc_no_bos


def test_wide_spread_penalizes_both():
    """Wide bid-ask spread (>20%) should penalize both LC and LP."""
    lc_tight, lp_tight = _score_one(_make_row(atm_spread_pct=0.05))
    lc_wide, lp_wide = _score_one(_make_row(atm_spread_pct=0.25))
    assert lc_tight > lc_wide
    assert lp_tight > lp_wide


def test_best_strategy_field_exists():
    lc, lp = _score_one(_make_row(trend='Up', smc_bos_bullish=True))
    import pandas as pd
    df = pd.DataFrame([_make_row(trend='Up', smc_bos_bullish=True)])
    macro = {'vix_current': 18.0, 'market_bias': 'neutral'}
    result = score_strategies(df, macro)
    assert 'best_strategy' in result.columns
    assert result.iloc[0]['best_strategy'] in ('Long Call', 'Long Put')
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_screener.py -k "vix_bonus or overbought or smc_bullish or wide_spread or best_strategy" -v
```

Expected: Most `FAILED` (scoring hasn't changed yet)

- [ ] **Step 3: Rewrite score_strategies**

Replace the entire `score_strategies` function body in `screener.py` with:

```python
def score_strategies(df: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """
    Score each stock for Long Call / Long Put suitability.
    Returns df with columns: lc_score, lp_score, best_strategy.
    """
    df = df.copy()
    lc_scores, lp_scores, best_strategies = [], [], []

    market_bias = macro.get('market_bias', 'neutral')

    for _, row in df.iterrows():
        iv_hv      = row.get('iv_hv_ratio', np.nan)
        days_earn  = row.get('days_to_earnings', None)
        rsi        = row.get('rsi', np.nan)
        ret_1m     = row.get('ret_1m', np.nan)
        trend      = row.get('trend', 'Unknown')
        atm_theta  = row.get('atm_theta', np.nan)
        call_oi    = row.get('atm_call_oi', np.nan)
        put_oi     = row.get('atm_put_oi', np.nan)
        spread_pct = row.get('atm_spread_pct', np.nan)

        # ── Shared base ───────────────────────────────────────────────────
        # IV cheapness
        iv_pts = 0.0
        if not np.isnan(iv_hv):
            if iv_hv < 0.6:
                iv_pts = 25
            elif iv_hv < 0.8:
                iv_pts = 15
            elif iv_hv < 1.0:
                iv_pts = 8

        # Earnings safety
        earn_pts = 0.0
        if days_earn is None or days_earn > 30:
            earn_pts = 10
        elif days_earn < 14:
            earn_pts = -15

        # Theta (daily decay per contract)
        theta_pts = 0.0
        if not np.isnan(atm_theta):
            daily_decay = abs(atm_theta)
            if daily_decay < 10:
                theta_pts = 5
            elif daily_decay > 50:
                theta_pts = -15
            elif daily_decay > 25:
                theta_pts = -8

        # Bid-ask spread penalty
        spread_pts = 0.0
        if not np.isnan(spread_pct):
            if spread_pct > 0.20:
                spread_pts = -10
            elif spread_pct > 0.10:
                spread_pts = -5

        shared = iv_pts + earn_pts + theta_pts + spread_pts

        # ── Long Call ─────────────────────────────────────────────────────
        lc = shared

        # Trend-quality combined signal
        if trend in ('Up', 'Strong Up'):
            if not np.isnan(rsi):
                if rsi < 45:
                    lc += 15   # dip within uptrend — best entry
                elif rsi <= 60:
                    lc += 20   # clean entry
                elif rsi <= 70:
                    lc += 10   # acceptable
                else:
                    lc -= 5    # overbought — chasing
            else:
                lc += 10       # trend confirmed, no RSI data
        elif trend == 'Sideways':
            lc += 2
        elif trend == 'Down':
            lc -= 5
        elif trend == 'Strong Down':
            lc -= 10

        # 1M momentum quality
        if not np.isnan(ret_1m):
            if 2 <= ret_1m <= 8:
                lc += 8    # steady, not extended
            elif ret_1m > 10:
                lc -= 5    # extended move — risky entry

        # Directional bonuses
        if market_bias == 'bullish':
            lc += 5
        if not np.isnan(call_oi):
            if call_oi > 5000:
                lc += 10
            elif call_oi > 1000:
                lc += 5

        # SMC signals
        if row.get('smc_bos_bullish'):    lc += 15
        if row.get('smc_bos_bearish'):    lc -= 10
        if row.get('smc_choch_bullish'):  lc += 5
        if row.get('smc_choch_bearish'):  lc -= 8
        if row.get('smc_discount_zone'):  lc += 10
        if row.get('smc_premium_zone'):   lc -= 5
        if row.get('smc_near_bullish_ob'): lc += 15
        if row.get('smc_near_bearish_ob'): lc -= 10
        if row.get('smc_in_bullish_fvg'): lc += 10
        if row.get('smc_in_bearish_fvg'): lc -= 5

        # ── Long Put ──────────────────────────────────────────────────────
        lp = shared

        # Trend-quality (bearish mirror)
        if trend in ('Down', 'Strong Down'):
            if not np.isnan(rsi):
                if rsi > 55:
                    lp += 15   # still elevated — best short entry
                elif rsi >= 45:
                    lp += 20   # clean entry
                elif rsi >= 30:
                    lp += 10   # acceptable
                else:
                    lp -= 5    # oversold — chasing
            else:
                lp += 10
        elif trend == 'Sideways':
            lp += 2
        elif trend == 'Up':
            lp -= 5
        elif trend == 'Strong Up':
            lp -= 10

        # 1M momentum (bearish)
        if not np.isnan(ret_1m):
            if -8 <= ret_1m <= -2:
                lp += 8
            elif ret_1m < -10:
                lp -= 5    # extended drop — risky put entry

        # Directional bonuses
        if market_bias == 'bearish':
            lp += 5
        if not np.isnan(put_oi):
            if put_oi > 5000:
                lp += 10
            elif put_oi > 1000:
                lp += 5

        # SMC signals
        if row.get('smc_bos_bearish'):    lp += 15
        if row.get('smc_bos_bullish'):    lp -= 10
        if row.get('smc_choch_bearish'):  lp += 5
        if row.get('smc_choch_bullish'):  lp -= 8
        if row.get('smc_premium_zone'):   lp += 10
        if row.get('smc_discount_zone'):  lp -= 5
        if row.get('smc_near_bearish_ob'): lp += 15
        if row.get('smc_near_bullish_ob'): lp -= 10
        if row.get('smc_in_bearish_fvg'): lp += 10
        if row.get('smc_in_bullish_fvg'): lp -= 5

        lc = max(0.0, min(100.0, lc))
        lp = max(0.0, min(100.0, lp))

        lc_scores.append(round(lc, 1))
        lp_scores.append(round(lp, 1))
        best_strategies.append('Long Call' if lc >= lp else 'Long Put')

    df['lc_score'] = lc_scores
    df['lp_score'] = lp_scores
    df['best_strategy'] = best_strategies
    return df
```

- [ ] **Step 4: Run scoring tests**

```bash
python -m pytest tests/test_screener.py -k "vix_bonus or overbought or smc_bullish or wide_spread or best_strategy" -v
```

Expected: All `PASSED`

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All `PASSED`

- [ ] **Step 6: Commit**

```bash
git add screener.py tests/test_screener.py
git commit -m "feat: rewrite scoring with trend-quality, SMC signals, spread penalty"
```

---

## Task 5: Remove Quick Screen mode + add My Account panel in app.py

**Files:**
- Modify: `app.py` — sidebar section (lines ~396–448)

- [ ] **Step 1: Add CADUSD FX fetch helper at top of app.py**

After the existing imports block (after `from strategies import ...`), add:

```python
@st.cache_data(ttl=3600)
def get_fx_rate() -> float:
    """Fetch CAD→USD rate from yfinance. Falls back to 0.73."""
    try:
        data = yf.download('CADUSD=X', period='1d', progress=False, auto_adjust=True)
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception:
        pass
    return 0.73
```

- [ ] **Step 2: Remove Quick Screen mode radio from sidebar**

In `app.py`, delete these lines from the sidebar section (around lines 422–427):

```python
screen_mode = st.radio(
    "Screening Mode",
    ["Quick Screen (HV only, fast)", "Full Screen (with IV from options chain)"],
    help="Quick mode uses only historical volatility. Full mode fetches live IV from options chains (slower)."
)
```

- [ ] **Step 3: Add My Account panel to sidebar**

In `app.py`, inside `with st.sidebar:`, add the My Account section directly above the `if st.button("🔍 Screen Now"...)` line:

```python
st.markdown("---")
st.markdown("## 💰 My Account")

fx_rate_default = get_fx_rate()
account_cad = st.number_input(
    "Account Size (CAD)",
    min_value=100, max_value=100000,
    value=500, step=50,
)
fx_rate = st.number_input(
    "CAD → USD Rate",
    min_value=0.50, max_value=1.00,
    value=float(round(fx_rate_default, 4)),
    step=0.001, format="%.4f",
)
risk_pct_label = st.selectbox(
    "Risk per Trade",
    options=["10%", "20%", "33%"],
    index=2,
)
risk_pct = int(risk_pct_label.replace('%', '')) / 100

budget_usd = account_cad * fx_rate
max_risk_usd = budget_usd * risk_pct

st.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:0.5rem;padding:0.75rem;margin-top:0.5rem;">
  <div style="font-size:0.8rem;color:#94a3b8;">Budget (USD): <strong style="color:#e2e8f0;">${budget_usd:.0f}</strong></div>
  <div style="font-size:0.8rem;color:#94a3b8;">Max Risk/Trade: <strong style="color:#f59e0b;">${max_risk_usd:.0f}</strong></div>
</div>
""", unsafe_allow_html=True)

budget_config = {
    'account_cad': account_cad,
    'fx_rate': fx_rate,
    'budget_usd': budget_usd,
    'max_risk_usd': max_risk_usd,
}

affordable_only = st.checkbox("Affordable only", value=False,
    help=f"Hide contracts costing more than ${max_risk_usd:.0f}")
```

- [ ] **Step 4: Remove Quick Screen fallback block in screening logic**

In `app.py`, find and delete the entire `else:` block that sets IV columns to NaN (Quick Screen fallback, around lines 576–590):

```python
else:
    # Quick mode: set IV columns to NaN
    fundamentals_df['atm_iv'] = np.nan
    ...
    status_text.markdown("**Step 2/3:** Quick mode — skipping IV fetch.")
```

Also remove the `is_full_screen` variable and the `if is_full_screen:` conditional wrapper — just call `enrich_with_iv` directly:

```python
status_text.markdown("**Step 2/3:** Fetching implied volatility from options chains…")
iv_progress = st.progress(0)

def iv_progress_callback(done, total):
    iv_progress.progress(int(done / total * 100))
    status_text.markdown(f"**Step 2/3:** Fetching IV… ({done}/{total})")

fundamentals_df = enrich_with_iv(fundamentals_df, progress_callback=iv_progress_callback)
iv_progress.empty()
```

- [ ] **Step 5: Apply affordable_only filter after existing filters**

In `app.py`, after the existing `iv_filter` application block (around line 606–611), add:

```python
if affordable_only and 'atm_mid_price' in filtered_df.columns:
    cost_series = filtered_df['atm_mid_price'].fillna(0) * 100
    filtered_df = filtered_df[cost_series <= max_risk_usd]
```

- [ ] **Step 6: Start app and verify sidebar renders correctly**

```bash
python -m streamlit run app.py
```

Open http://localhost:8501. Confirm:
- "My Account" section visible in sidebar
- Budget/Max Risk computed values update when inputs change
- "Screening Mode" radio is gone
- "Affordable only" checkbox present

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: add My Account sidebar panel, remove Quick Screen mode"
```

---

## Task 6: Add Cost/Contract + Affordable columns to screener table

**Files:**
- Modify: `app.py` — display_df construction block (lines ~639–701)

- [ ] **Step 1: Add cost_per_contract and affordable columns to display_df**

In `app.py`, inside the display_df construction block (after `display_df['Put OI'] = ...`), add:

```python
cost_usd = safe_col(results_df, 'atm_mid_price') * 100
display_df['Cost/Contract'] = cost_usd.round(2)
display_df['Affordable'] = cost_usd.apply(
    lambda c: '✓' if (not np.isnan(c) and c <= max_risk_usd) else ('✗' if not np.isnan(c) else 'N/A')
)
```

- [ ] **Step 2: Add column_config entries for new columns**

In `app.py`, inside the `st.dataframe(...)` call's `column_config` dict, add:

```python
'Cost/Contract': st.column_config.NumberColumn('Cost/Contract ($)', format="$%.2f",
    help="Estimated cost per 1 contract (ATM call mid × 100)"),
'Affordable': st.column_config.TextColumn('Affordable',
    help=f"✓ = within ${max_risk_usd:.0f} max risk budget"),
```

- [ ] **Step 3: Verify in browser**

Run app, screen a few stocks (Full Screen mode). Confirm:
- "Cost/Contract" column shows dollar amounts
- "Affordable" column shows ✓/✗ based on My Account settings
- Changing "Risk per Trade" in sidebar and re-screening updates the ✓/✗

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add Cost/Contract and Affordable columns to screener table"
```

---

## Task 7: Update Top 10 Picks cards with budget row + SMC tags

**Files:**
- Modify: `app.py` — Top 10 Picks rendering (lines ~708–776)

- [ ] **Step 1: Add SMC tag builder helper in app.py**

After the existing helper functions (after `risk_badge`), add:

```python
def build_smc_tags(row: dict) -> str:
    """Return HTML string of active SMC signal tags for a screener row."""
    tag_map = {
        'smc_bos_bullish':     ('Bullish BoS', '#10b981'),
        'smc_bos_bearish':     ('Bearish BoS', '#ef4444'),
        'smc_choch_bullish':   ('Bullish CHoCH', '#3b82f6'),
        'smc_choch_bearish':   ('Bearish CHoCH', '#f97316'),
        'smc_discount_zone':   ('Discount Zone', '#10b981'),
        'smc_premium_zone':    ('Premium Zone', '#ef4444'),
        'smc_near_bullish_ob': ('Near Bull OB', '#10b981'),
        'smc_near_bearish_ob': ('Near Bear OB', '#ef4444'),
        'smc_in_bullish_fvg':  ('Bull FVG', '#60a5fa'),
        'smc_in_bearish_fvg':  ('Bear FVG', '#f87171'),
    }
    tags = []
    for key, (label, color) in tag_map.items():
        if row.get(key):
            tags.append(
                f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
                f'padding:1px 7px;border-radius:10px;font-size:0.7rem;font-weight:600;">'
                f'{label}</span>'
            )
    return ' '.join(tags)
```

- [ ] **Step 2: Update Long Call Top 5 cards**

In `app.py`, inside the `with col_lc:` block, replace the existing card HTML template with:

```python
cost_val = r.get('atm_mid_price', np.nan)
cost_usd = cost_val * 100 if not np.isnan(cost_val) else np.nan
is_affordable = not np.isnan(cost_usd) and cost_usd <= budget_config['max_risk_usd']
if not np.isnan(cost_usd):
    budget_line = (
        f'<div style="border-top:1px solid #334155;margin-top:0.4rem;padding-top:0.4rem;font-size:0.78rem;">'
        f'💰 Cost: <strong>${cost_usd:.0f}/contract</strong> &nbsp;|&nbsp; '
        + (f'<span style="color:#10b981;">✓ Affordable</span> &nbsp;|&nbsp; '
           f'Max Loss: <strong>${cost_usd:.0f}</strong> &nbsp;|&nbsp; '
           f'2:1 Target: <strong style="color:#10b981;">+${cost_usd*2:.0f}</strong>'
           if is_affordable else
           f'<span style="color:#ef4444;">✗ Over Budget (${budget_config["max_risk_usd"]:.0f} max)</span>')
        + '</div>'
    )
else:
    budget_line = ''

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
{budget_line}
</div>""", unsafe_allow_html=True)
```

- [ ] **Step 3: Update Long Put Top 5 cards**

Apply the same pattern to the `with col_lp:` block, replacing `#10b981` border with `#ef4444` and using `put_oi_val` / `lp_score`:

```python
cost_val = r.get('atm_mid_price', np.nan)
cost_usd = cost_val * 100 if not np.isnan(cost_val) else np.nan
is_affordable = not np.isnan(cost_usd) and cost_usd <= budget_config['max_risk_usd']
if not np.isnan(cost_usd):
    budget_line = (
        f'<div style="border-top:1px solid #334155;margin-top:0.4rem;padding-top:0.4rem;font-size:0.78rem;">'
        f'💰 Cost: <strong>${cost_usd:.0f}/contract</strong> &nbsp;|&nbsp; '
        + (f'<span style="color:#10b981;">✓ Affordable</span> &nbsp;|&nbsp; '
           f'Max Loss: <strong>${cost_usd:.0f}</strong> &nbsp;|&nbsp; '
           f'2:1 Target: <strong style="color:#10b981;">+${cost_usd*2:.0f}</strong>'
           if is_affordable else
           f'<span style="color:#ef4444;">✗ Over Budget (${budget_config["max_risk_usd"]:.0f} max)</span>')
        + '</div>'
    )
else:
    budget_line = ''

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
{budget_line}
</div>""", unsafe_allow_html=True)
```

- [ ] **Step 4: Verify in browser**

Run app, screen stocks, check Top 10:
- SMC tags appear on cards (e.g., `[Bullish BoS]` `[Discount Zone]`)
- Budget row shows cost + ✓/✗
- Affordable picks show Max Loss and 2:1 target
- Over-budget picks show red label with max allowed

- [ ] **Step 5: Run all tests one final time**

```bash
python -m pytest tests/ -v
```

Expected: All `PASSED`

- [ ] **Step 6: Final commit**

```bash
git add app.py screener.py tests/
git commit -m "feat: Top 10 cards show SMC tags + budget row (cost, max loss, 2:1 target)"
```

---

## Spec Coverage Checklist

| Spec requirement | Task |
|-----------------|------|
| My Account sidebar panel (CAD, FX, risk%) | Task 5 |
| CADUSD=X auto-fetch | Task 5 |
| Remove Quick Screen mode | Task 5 |
| Cost/Contract column | Task 6 |
| Affordable column + filter | Task 5, 6 |
| Top 10 budget row (cost, max loss, 2:1 target) | Task 7 |
| SMC tags on Top 10 cards | Task 7 |
| OHLCV stored in batch_screen_fundamentals | Task 2 |
| atm_mid_price + atm_spread_pct | Task 2 |
| _compute_atr, _empty_smc | Task 3 |
| compute_smc_signals (BoS, CHoCH, Discount/Premium, OB, FVG) | Task 3 |
| SMC signals in score_strategies | Task 4 |
| Revised scoring (remove VIX bonus, delta bonus; rebalance) | Task 4 |
| Trend-quality combined signal | Task 4 |
| Bid-ask spread penalty | Task 4 |
