import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from alerts import (
    entry_alerts, exit_alerts, diff_alerts, current_state_map,
    format_entry_msg, format_exit_msg,
)


def _enter_row():
    return {
        'ticker': 'AAPL', 'trend': 'Up', 'rsi': 42.0, 'iv_hv_ratio': 0.7,
        'days_to_earnings': None,
        'smc_bos_bullish': True, 'smc_bos_bearish': False,
        'smc_choch_bullish': False, 'smc_choch_bearish': False,
        'smc_discount_zone': True, 'smc_premium_zone': False,
        'smc_near_bullish_ob': False, 'smc_near_bearish_ob': False,
        'smc_in_bullish_fvg': False, 'smc_in_bearish_fvg': False,
        'ema_bull_stack': True, 'ema_bear_stack': False,
        'above_ema20': True, 'above_ema50': True,
    }


def _dead_row():
    r = _enter_row()
    r['ticker'] = 'XYZ'
    r['trend'] = 'Down'
    r['rsi'] = 80.0
    r['iv_hv_ratio'] = 1.8
    r['smc_bos_bullish'] = False
    r['smc_discount_zone'] = False
    r['ema_bull_stack'] = False
    r['above_ema20'] = False
    return r


def test_entry_alerts_flags_enter_row():
    out = entry_alerts([_enter_row()])
    calls = [a for a in out if a['strategy'] == 'Long Call']
    assert len(calls) == 1
    a = calls[0]
    assert a['key'] == 'entry:AAPL:Long Call'
    assert a['kind'] == 'entry'
    assert a['state'] == 'enter'


def test_entry_alerts_ignores_dead_row():
    out = entry_alerts([_dead_row()])
    assert out == []


def test_exit_alerts_flags_sell_and_trim():
    analyzed = [
        {'pos': {'id': 'p1', 'ticker': 'NVDA', 'strike': 150.0, 'strategy': 'Long Call'},
         'data': {'verdict': {'status': 'sell', 'reasons': ['RSI > 70']}, 'pnl_pct': 0.3}},
        {'pos': {'id': 'p2', 'ticker': 'TSLA', 'strike': 200.0, 'strategy': 'Long Put'},
         'data': {'verdict': {'status': 'hold', 'reasons': []}, 'pnl_pct': 0.0}},
    ]
    out = exit_alerts(analyzed)
    assert len(out) == 1
    assert out[0]['key'] == 'exit:p1'
    assert out[0]['state'] == 'sell'


def test_diff_alerts_new_and_changed_only():
    current = [
        {'key': 'entry:AAPL:Long Call', 'state': 'enter'},
        {'key': 'exit:p1', 'state': 'sell'},
        {'key': 'exit:p2', 'state': 'trim'},
    ]
    stored = {'exit:p1': 'sell', 'exit:p2': 'hold'}
    out = diff_alerts(current, stored)
    keys = {a['key'] for a in out}
    assert keys == {'entry:AAPL:Long Call', 'exit:p2'}


def test_current_state_map():
    current = [{'key': 'a', 'state': 'enter'}, {'key': 'b', 'state': 'sell'}]
    assert current_state_map(current) == {'a': 'enter', 'b': 'sell'}


def test_format_entry_msg_contains_core_fields():
    a = {'ticker': 'AAPL', 'strategy': 'Long Call', 'met': 6, 'total': 7,
         'checks': [{'label': 'Trend', 'passed': True, 'value': 'Up'},
                    {'label': 'RSI < 50', 'passed': False, 'value': '80'}]}
    msg = format_entry_msg(a)
    assert 'ENTRY' in msg and 'AAPL' in msg and 'Long Call' in msg
    assert '6/7' in msg
    assert 'Trend' in msg
    assert 'RSI < 50' not in msg


def test_format_exit_msg_sell_with_pnl():
    a = {'ticker': 'NVDA', 'strike': 150.0, 'strategy': 'Long Call',
         'status': 'sell', 'reasons': ['RSI > 70', '穿 EMA50'], 'pnl_pct': 0.3}
    msg = format_exit_msg(a)
    assert 'SELL' in msg and 'NVDA' in msg and '$150' in msg
    assert '+30%' in msg
    assert 'RSI > 70' in msg
