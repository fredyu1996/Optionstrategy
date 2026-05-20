import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from screener import score_strategies, compute_smc_signals, _empty_smc, _compute_atr


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
