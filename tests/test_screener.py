import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from screener import score_strategies, compute_smc_signals, _empty_smc, _compute_atr, \
    _compute_delta_range, _select_best_strike, _empty_recommendation


def test_batch_screen_fundamentals_stores_ohlcv():
    """batch_screen_fundamentals must store high_arr, low_arr, open_arr columns."""
    from screener import batch_screen_fundamentals
    df = batch_screen_fundamentals(['AAPL'])
    assert 'high_arr' in df.columns, "high_arr column missing"
    assert 'low_arr' in df.columns, "low_arr column missing"
    assert 'open_arr' in df.columns, "open_arr column missing"
    assert isinstance(df.iloc[0]['high_arr'], np.ndarray)


def test_enrich_with_iv_adds_mid_and_spread():
    """enrich_with_iv must add atm_mid_price and atm_spread_pct columns."""
    from screener import batch_screen_fundamentals, enrich_with_iv
    df = batch_screen_fundamentals(['AAPL'])
    df = enrich_with_iv(df)
    assert 'atm_mid_price' in df.columns, "atm_mid_price column missing"
    assert 'atm_spread_pct' in df.columns, "atm_spread_pct column missing"


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
    df = pd.DataFrame([row])
    result = score_strategies(df, macro)
    return result.iloc[0]['lc_score'], result.iloc[0]['lp_score']


def test_vix_bonus_removed():
    """VIX level should NOT affect scores when IV/HV is NaN."""
    row = _make_row(iv_hv_ratio=np.nan)
    lc_low, _ = _score_one(row, {'vix_current': 12.0, 'market_bias': 'neutral'})
    lc_high, _ = _score_one(row, {'vix_current': 25.0, 'market_bias': 'neutral'})
    assert lc_low == lc_high, "VIX level should not affect score when IV/HV is NaN"


def test_overbought_penalizes_long_call():
    """RSI > 70 in uptrend + extended momentum should score lower than RSI 50."""
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
    df = pd.DataFrame([_make_row(trend='Up', smc_bos_bullish=True)])
    macro = {'vix_current': 18.0, 'market_bias': 'neutral'}
    result = score_strategies(df, macro)
    assert 'best_strategy' in result.columns
    assert result.iloc[0]['best_strategy'] in ('Long Call', 'Long Put')


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
            'strike': stock_price * (1 + (d - 0.5) * 0.2),
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
    chosen, flag = _select_best_strike(strikes, 0.35, 0.45, 0.40, 120.0, True, 100.0)
    assert chosen['delta'] == pytest.approx(0.38)
    assert flag is None


def test_select_best_strike_falls_back_outside_range():
    """No affordable strike in range → pick closest affordable, flag 'outside_ideal_range'."""
    strikes = _make_strikes(
        deltas=[0.25, 0.38, 0.45, 0.55, 0.65],
        costs=[50, 200, 200, 200, 200],
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
