# tests/test_signals.py
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from signals import compute_entry_readiness, compute_exit_rules


def _bullish_row():
    return {
        'trend': 'Up',
        'rsi': 42.0,
        'iv_hv_ratio': 0.7,
        'days_to_earnings': None,
        'smc_bos_bullish': True,
        'smc_bos_bearish': False,
        'smc_choch_bullish': False,
        'smc_choch_bearish': False,
        'smc_discount_zone': True,
        'smc_premium_zone': False,
        'smc_near_bullish_ob': False,
        'smc_near_bearish_ob': False,
        'smc_in_bullish_fvg': False,
        'smc_in_bearish_fvg': False,
    }


def _bearish_row():
    return {
        'trend': 'Strong Down',
        'rsi': 58.0,
        'iv_hv_ratio': 0.7,
        'days_to_earnings': None,
        'smc_bos_bullish': False,
        'smc_bos_bearish': True,
        'smc_choch_bullish': False,
        'smc_choch_bearish': False,
        'smc_discount_zone': False,
        'smc_premium_zone': True,
        'smc_near_bullish_ob': False,
        'smc_near_bearish_ob': False,
        'smc_in_bullish_fvg': False,
        'smc_in_bearish_fvg': False,
    }


def _base_rec():
    return {'cost': 420.0, 'dte': 45, 'strike': 150.0}


# ── compute_entry_readiness ──────────────────────────────────────────────────

class TestEntryReadinessLongCall:
    def test_all_pass_returns_enter(self):
        result = compute_entry_readiness(_bullish_row(), 'Long Call')
        assert result['met'] == 6
        assert result['total'] == 6
        assert result['status'] == 'enter'

    def test_three_pass_returns_wait(self):
        row = _bullish_row()
        row['trend'] = 'Down'
        row['smc_discount_zone'] = False
        row['smc_bos_bullish'] = False
        result = compute_entry_readiness(row, 'Long Call')
        assert result['met'] == 3
        assert result['status'] == 'wait'

    def test_two_or_fewer_returns_not_yet(self):
        row = _bullish_row()
        row['trend'] = 'Strong Down'
        row['rsi'] = 72.0
        row['smc_discount_zone'] = False
        row['smc_bos_bullish'] = False
        row['iv_hv_ratio'] = 1.5
        result = compute_entry_readiness(row, 'Long Call')
        assert result['met'] <= 2
        assert result['status'] == 'not_yet'

    def test_returns_six_checks(self):
        result = compute_entry_readiness(_bullish_row(), 'Long Call')
        assert len(result['checks']) == 6

    def test_nan_rsi_fails_rsi_check(self):
        row = _bullish_row()
        row['rsi'] = float('nan')
        result = compute_entry_readiness(row, 'Long Call')
        rsi_check = next(c for c in result['checks'] if 'RSI' in c['label'])
        assert not rsi_check['passed']

    def test_near_earnings_fails_earnings_check(self):
        row = _bullish_row()
        row['days_to_earnings'] = 10
        result = compute_entry_readiness(row, 'Long Call')
        earn_check = next(c for c in result['checks'] if 'earnings' in c['label'].lower())
        assert not earn_check['passed']

    def test_smc_fvg_satisfies_smc_check(self):
        row = _bullish_row()
        row['smc_bos_bullish'] = False
        row['smc_near_bullish_ob'] = False
        row['smc_in_bullish_fvg'] = True
        result = compute_entry_readiness(row, 'Long Call')
        smc_check = next(c for c in result['checks'] if 'SMC' in c['label'])
        assert smc_check['passed']

    def test_checks_have_required_keys(self):
        result = compute_entry_readiness(_bullish_row(), 'Long Call')
        for check in result['checks']:
            assert 'label' in check
            assert 'passed' in check
            assert 'value' in check


class TestEntryReadinessLongPut:
    def test_all_pass_returns_enter(self):
        result = compute_entry_readiness(_bearish_row(), 'Long Put')
        assert result['met'] == 6
        assert result['status'] == 'enter'

    def test_uptrend_fails_trend_check(self):
        row = _bearish_row()
        row['trend'] = 'Strong Up'
        result = compute_entry_readiness(row, 'Long Put')
        trend_check = next(c for c in result['checks'] if c['label'] == 'Trend')
        assert not trend_check['passed']

    def test_rsi_below_50_fails_put_rsi_check(self):
        row = _bearish_row()
        row['rsi'] = 45.0
        result = compute_entry_readiness(row, 'Long Put')
        rsi_check = next(c for c in result['checks'] if 'RSI' in c['label'])
        assert not rsi_check['passed']


# ── compute_exit_rules ───────────────────────────────────────────────────────

class TestExitRules:
    def test_price_targets_computed_from_cost(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert result['take_profit_usd'] == pytest.approx(840.0)
        assert result['stop_loss_usd'] == pytest.approx(210.0)
        assert result['take_profit_pct'] == 1.0
        assert result['stop_loss_pct'] == 0.5

    def test_time_exit_dte_is_21(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert result['time_exit_dte'] == 21

    def test_time_exit_date_set_when_dte_above_21(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert result['time_exit_date'] is not None

    def test_time_exit_msg_when_dte_below_threshold(self):
        rec = {'cost': 420.0, 'dte': 10, 'strike': 150.0}
        result = compute_exit_rules(_bullish_row(), 'Long Call', rec)
        assert 'Exit now' in result['time_exit_msg']

    def test_nan_cost_returns_nan_targets(self):
        rec = {'cost': float('nan'), 'dte': 45, 'strike': 150.0}
        result = compute_exit_rules(_bullish_row(), 'Long Call', rec)
        assert np.isnan(result['take_profit_usd'])
        assert np.isnan(result['stop_loss_usd'])

    def test_long_call_has_three_tech_triggers(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert len(result['tech_triggers']) == 3

    def test_long_put_has_three_tech_triggers(self):
        result = compute_exit_rules(_bearish_row(), 'Long Put', _base_rec())
        assert len(result['tech_triggers']) == 3

    def test_long_call_rsi_trigger_fires_above_70(self):
        row = _bullish_row()
        row['rsi'] = 75.0
        result = compute_exit_rules(row, 'Long Call', _base_rec())
        rsi_trigger = next(t for t in result['tech_triggers'] if 'RSI' in t['label'])
        assert rsi_trigger['triggered'] is True

    def test_long_call_bearish_bos_trigger(self):
        row = _bullish_row()
        row['smc_bos_bearish'] = True
        result = compute_exit_rules(row, 'Long Call', _base_rec())
        bos_trigger = next(t for t in result['tech_triggers'] if 'BoS' in t['label'])
        assert bos_trigger['triggered'] is True

    def test_long_call_premium_zone_trigger(self):
        row = _bullish_row()
        row['smc_premium_zone'] = True
        result = compute_exit_rules(row, 'Long Call', _base_rec())
        zone_trigger = next(t for t in result['tech_triggers'] if 'Premium' in t['label'])
        assert zone_trigger['triggered'] is True

    def test_long_put_rsi_trigger_fires_below_30(self):
        row = _bearish_row()
        row['rsi'] = 25.0
        result = compute_exit_rules(row, 'Long Put', _base_rec())
        rsi_trigger = next(t for t in result['tech_triggers'] if 'RSI' in t['label'])
        assert rsi_trigger['triggered'] is True

    def test_long_put_bullish_bos_trigger(self):
        row = _bearish_row()
        row['smc_bos_bullish'] = True
        result = compute_exit_rules(row, 'Long Put', _base_rec())
        bos_trigger = next(t for t in result['tech_triggers'] if 'BoS' in t['label'])
        assert bos_trigger['triggered'] is True

    def test_long_put_discount_zone_trigger(self):
        row = _bearish_row()
        row['smc_discount_zone'] = True
        result = compute_exit_rules(row, 'Long Put', _base_rec())
        zone_trigger = next(t for t in result['tech_triggers'] if 'Discount' in t['label'])
        assert zone_trigger['triggered'] is True

    def test_tech_triggers_have_required_keys(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        for t in result['tech_triggers']:
            assert 'label' in t
            assert 'triggered' in t
            assert 'current_value' in t

    def test_no_triggers_active_when_conditions_safe(self):
        result = compute_exit_rules(_bullish_row(), 'Long Call', _base_rec())
        assert all(not t['triggered'] for t in result['tech_triggers'])


class TestValidation:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_entry_readiness(_bullish_row(), 'Long Strangle')

    def test_invalid_strategy_exit_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_exit_rules(_bullish_row(), 'Long Strangle', _base_rec())
