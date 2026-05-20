import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from screener import score_strategies


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
