import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from chatbot import _build_system_prompt


def _make_context(top_lc=None, top_lp=None, focused_ticker=None, focused_row=None):
    return {
        "macro": {
            "vix_current": 18.5,
            "vix_regime": "normal",
            "market_regime": "Normal",
            "spy_price": 520.0,
            "spy_trend": "Bullish",
            "spy_ret_1m": 2.3,
            "tnx_yield": 4.5,
        },
        "top_lc": top_lc or [],
        "top_lp": top_lp or [],
        "focused_ticker": focused_ticker,
        "focused_row": focused_row,
    }


def test_system_prompt_contains_macro():
    prompt = _build_system_prompt(_make_context())
    assert "18.5" in prompt
    assert "Bullish" in prompt
    assert "4.5" in prompt


def test_system_prompt_no_results_note():
    prompt = _build_system_prompt(_make_context())
    assert "No screener results yet" in prompt


def test_system_prompt_with_picks():
    lc = [{"ticker": "AAPL", "lc_score": 75, "iv_hv_ratio": 0.82, "trend": "Up", "atm_delta": 0.51}]
    lp = [{"ticker": "TSLA", "lp_score": 68, "iv_hv_ratio": 0.75, "trend": "Down", "atm_delta": -0.49}]
    prompt = _build_system_prompt(_make_context(top_lc=lc, top_lp=lp))
    assert "AAPL" in prompt
    assert "TSLA" in prompt
    assert "No screener results yet" not in prompt


def test_system_prompt_focused_block_absent_when_no_ticker():
    prompt = _build_system_prompt(_make_context())
    assert "FOCUSED STOCK" not in prompt


def test_system_prompt_focused_block_present_when_ticker_set():
    row = {
        "lc_score": 80, "lp_score": 55,
        "iv_hv_ratio": 0.78, "trend": "Strong Up", "ret_1m": 5.1,
        "atm_delta": 0.52, "atm_theta": -0.15, "atm_vega": 0.22,
        "smc_bos_bullish": True, "smc_bos_bearish": False,
        "smc_choch_bullish": False, "smc_choch_bearish": False,
        "smc_discount_zone": True, "smc_premium_zone": False,
        "smc_near_bullish_ob": False, "smc_near_bearish_ob": False,
        "smc_in_bullish_fvg": True, "smc_in_bearish_fvg": False,
        "best_strategy": "Long Call",
    }
    prompt = _build_system_prompt(_make_context(focused_ticker="NVDA", focused_row=row))
    assert "FOCUSED STOCK: NVDA" in prompt
    assert "Bullish BoS" in prompt
    assert "Discount Zone" in prompt
    assert "Bull FVG" in prompt


def test_system_prompt_verdict_instruction():
    prompt = _build_system_prompt(_make_context())
    assert "PROCEED" in prompt
    assert "CAUTION" in prompt
    assert "SKIP" in prompt
