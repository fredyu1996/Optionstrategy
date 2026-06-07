"""
app.py - S&P 500 Options Strategy Screener - Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

from screener import (
    get_sp500_tickers,
    get_macro_data,
    batch_screen_fundamentals,
    enrich_with_iv,
    score_strategies,
    get_strike_recommendation,
)
from strategies import suggest_strategies, get_specific_contracts, get_option_chain_display
from chatbot import render_chat_button
from signals import compute_entry_readiness, compute_exit_rules, compute_sell_verdict
from positions_store import load_positions, add_position, delete_position, PositionsConfigError
from positions import analyze_position
from indicators import fetch_4h_ema_status


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

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="S&P 500 Options Strategy Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
.main { padding-top: 0.5rem; }
[data-testid="stSidebar"] { background-color: #0e1117; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 0.75rem;
    margin-bottom: 1.5rem;
    border: 1px solid #2d3748;
}
.app-title {
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* Metric cards */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
}
.metric-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.1rem;
}

/* Market banner */
.banner-premium-sell {
    background: linear-gradient(90deg, #065f46, #064e3b);
    border-left: 4px solid #10b981;
    padding: 0.75rem 1.25rem;
    border-radius: 0.5rem;
    color: #d1fae5;
    font-size: 1rem;
    font-weight: 600;
    margin: 1rem 0;
}
.banner-premium-buy {
    background: linear-gradient(90deg, #1e3a5f, #1e3799);
    border-left: 4px solid #3b82f6;
    padding: 0.75rem 1.25rem;
    border-radius: 0.5rem;
    color: #dbeafe;
    font-size: 1rem;
    font-weight: 600;
    margin: 1rem 0;
}
.banner-mixed {
    background: linear-gradient(90deg, #44337a, #553c9a);
    border-left: 4px solid #a78bfa;
    padding: 0.75rem 1.25rem;
    border-radius: 0.5rem;
    color: #ede9fe;
    font-size: 1rem;
    font-weight: 600;
    margin: 1rem 0;
}

/* Strategy cards */
.strategy-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 0.75rem;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.strategy-card-high {
    border-left: 4px solid #ef4444;
}
.strategy-card-low {
    border-left: 4px solid #10b981;
}
.strategy-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
}
.strategy-score {
    font-size: 1.4rem;
    font-weight: 700;
}
.score-high { color: #10b981; }
.score-med  { color: #f59e0b; }
.score-low  { color: #64748b; }

/* Risk badges */
.badge-low  { background: #065f46; color: #d1fae5; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
.badge-high { background: #7f1d1d; color: #fee2e2; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
.badge-med  { background: #78350f; color: #fef3c7; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }

/* IV/HV labels */
.ivhv-very-exp { color: #ef4444; font-weight: 700; }
.ivhv-exp      { color: #f97316; font-weight: 600; }
.ivhv-fair     { color: #94a3b8; }
.ivhv-cheap    { color: #3b82f6; font-weight: 600; }

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    padding: 0.5rem 0;
    border-bottom: 1px solid #334155;
    margin-bottom: 1rem;
}

/* Disclaimer */
.disclaimer {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper Utilities
# ──────────────────────────────────────────────
def iv_hv_label(ratio):
    if ratio is None or (isinstance(ratio, float) and np.isnan(ratio)):
        return "N/A"
    if ratio > 1.5:
        return "Very Expensive"
    elif ratio > 1.2:
        return "Expensive"
    elif ratio > 0.8:
        return "Fair"
    else:
        return "Cheap"


def iv_hv_css_class(ratio):
    if ratio is None or (isinstance(ratio, float) and np.isnan(ratio)):
        return "ivhv-fair"
    if ratio > 1.5:
        return "ivhv-very-exp"
    elif ratio > 1.2:
        return "ivhv-exp"
    elif ratio > 0.8:
        return "ivhv-fair"
    else:
        return "ivhv-cheap"


def fmt_pct(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val*100:.{decimals}f}%"


def fmt_num(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def fmt_market_cap(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    elif val >= 1e9:
        return f"${val/1e9:.1f}B"
    elif val >= 1e6:
        return f"${val/1e6:.1f}M"
    return f"${val:.0f}"


def score_color(score):
    if score >= 65:
        return "score-high"
    elif score >= 40:
        return "score-med"
    else:
        return "score-low"


def risk_badge(risk_level):
    cls = {
        'Low': 'badge-low',
        'High': 'badge-high',
        'Medium': 'badge-med',
    }.get(risk_level, 'badge-med')
    return f'<span class="{cls}">{risk_level}</span>'


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
    option_type = "Call" if is_call else "Put"

    flag_html = ''
    if rec.get('flag') == 'outside_ideal_range':
        flag_html = ' <span style="color:#f59e0b;font-size:0.72rem;">⚠ outside ideal range</span>'
    elif rec.get('flag') == 'over_budget':
        flag_html = ' <span style="color:#ef4444;font-size:0.72rem;">⚠ over budget</span>'

    aff_color = '#10b981' if rec['affordable'] else '#ef4444'
    aff_label = '✓ Affordable' if rec['affordable'] else f'✗ Over Budget (${max_risk_usd:.0f} max)'

    delta_val = rec['delta']
    gamma_val = rec['gamma']
    theta_val = rec['theta']
    breakeven_val = rec['breakeven']
    cost_val = rec['cost']

    delta_str = f"{delta_val:.3f}" if not (isinstance(delta_val, float) and np.isnan(delta_val)) else 'N/A'
    gamma_str = f"{gamma_val:.4f}" if not (isinstance(gamma_val, float) and np.isnan(gamma_val)) else 'N/A'
    theta_str = f"${theta_val:.2f}/day" if not (isinstance(theta_val, float) and np.isnan(theta_val)) else 'N/A'
    breakeven_str = f"${breakeven_val:.2f}" if not (isinstance(breakeven_val, float) and np.isnan(breakeven_val)) else 'N/A'
    cost_str = f"${cost_val:.0f}" if not (isinstance(cost_val, float) and np.isnan(cost_val)) else 'N/A'
    target_str = f"+${cost_val * 2:.0f}" if not (isinstance(cost_val, float) and np.isnan(cost_val)) else 'N/A'

    crush_html = ''
    if rec.get('iv_crush_warning') and not (isinstance(iv_hv_val, float) and np.isnan(iv_hv_val)):
        crush_html = (
            f'<div style="font-size:0.75rem;color:#f59e0b;margin-top:0.15rem;">'
            f'⚠ IV elevated (IV/HV {iv_hv_val:.2f}) — risk of IV crush</div>'
        )

    return (
        '<div style="border-top:1px solid #334155;margin-top:0.4rem;padding-top:0.4rem;">'
        '<div style="font-size:0.75rem;color:#94a3b8;font-weight:700;margin-bottom:0.15rem;">Recommended Entry</div>'
        f'<div style="font-size:0.85rem;color:#e2e8f0;">'
        f'Buy ${rec["strike"]:.0f} {option_type} &nbsp;|&nbsp; {expiry_label} ({dte_str}){flag_html}'
        f'</div>'
        f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:0.1rem;">'
        f'Δ {delta_str} &nbsp;|&nbsp; Γ {gamma_str} &nbsp;|&nbsp; Θ {theta_str} &nbsp;|&nbsp; Breakeven: {breakeven_str}'
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


_PLAYBOOK_STATUS = {
    'enter':   ('🟢', 'Enter Now',             '#10b981'),
    'wait':    ('🟡', 'Wait for Confirmation', '#f59e0b'),
    'not_yet': ('🔴', 'Not Yet',               '#ef4444'),
}

_PLAYBOOK_VERDICT = {
    'hold': ('🟢', 'HOLD', '#10b981'),
    'trim': ('🟡', 'TRIM', '#f59e0b'),
    'sell': ('🔴', 'SELL', '#ef4444'),
}

_EMA4H_COLOR = {
    'good': '#10b981',
    'wait': '#f59e0b',
    'avoid': '#ef4444',
    'unknown': '#64748b',
}


def _render_playbook_col(readiness: dict, exits: dict, rec: dict, strategy_label: str, key_prefix: str) -> None:
    """Render entry readiness, sell verdict, and exit rules for one strategy column."""
    emoji, status_text, status_color = _PLAYBOOK_STATUS[readiness['status']]
    met = readiness['met']
    total = readiness['total']

    st.markdown(f"#### {strategy_label}")

    st.markdown(
        f'<div style="background:#1e293b;border:1px solid {status_color}55;'
        f'border-left:4px solid {status_color};border-radius:0.5rem;'
        f'padding:0.75rem 1rem;margin-bottom:0.75rem;">'
        f'<div style="font-size:0.78rem;color:#94a3b8;font-weight:700;'
        f'margin-bottom:0.3rem;">ENTRY READINESS</div>'
        f'<div style="font-size:1.2rem;font-weight:700;color:{status_color};">'
        f'{emoji} {status_text}</div>'
        f'<div style="font-size:0.75rem;color:#64748b;margin-top:0.15rem;">'
        f'{met}/{total} conditions met</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for check in readiness['checks']:
        icon = '✓' if check['passed'] else '✗'
        color = '#10b981' if check['passed'] else '#ef4444'
        st.markdown(
            f'<div style="font-size:0.82rem;margin:0.15rem 0;">'
            f'<span style="color:{color};font-weight:700;">{icon}</span> '
            f'<span style="color:#e2e8f0;">{check["label"]}</span> '
            f'<span style="color:#64748b;">({check["value"]})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="margin:0.75rem 0;border-top:1px solid #334155;"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.78rem;color:#94a3b8;font-weight:700;'
        'margin-bottom:0.4rem;">EXIT RULES</div>',
        unsafe_allow_html=True,
    )

    # ── Sell verdict badge (rendered above the input via a placeholder) ──
    verdict_slot = st.empty()
    entry_premium = st.number_input(
        "Your entry premium ($, per contract)",
        min_value=0.0, value=0.0, step=1.0,
        key=f"{key_prefix}_entry_prem",
        help="Total $ you paid per contract. Leave 0 to judge by conditions only.",
    )
    ep = entry_premium if entry_premium > 0 else None
    verdict = compute_sell_verdict(exits, rec, ep)
    v_emoji, v_text, v_color = _PLAYBOOK_VERDICT[verdict['status']]

    pnl_html = ''
    if verdict['pnl_pct'] is not None:
        pnl = verdict['pnl_pct']
        pnl_color = '#10b981' if pnl >= 0 else '#ef4444'
        pnl_html = (
            f'<div style="font-size:0.8rem;margin-top:0.2rem;color:{pnl_color};'
            f'font-weight:600;">Current P/L: {pnl * 100:+.0f}%</div>'
        )

    if verdict['reasons']:
        reasons_html = ''.join(
            f'<div style="font-size:0.74rem;color:#94a3b8;margin-top:0.12rem;">• {r}</div>'
            for r in verdict['reasons']
        )
    else:
        reasons_html = (
            '<div style="font-size:0.74rem;color:#64748b;margin-top:0.12rem;">'
            'No exit signals active</div>'
        )

    verdict_slot.markdown(
        f'<div style="background:#0f172a;border:1px solid {v_color}55;'
        f'border-left:4px solid {v_color};border-radius:0.5rem;'
        f'padding:0.6rem 0.9rem;margin-bottom:0.6rem;">'
        f'<div style="font-size:1.25rem;font-weight:800;color:{v_color};">'
        f'{v_emoji} {v_text}</div>'
        f'{pnl_html}{reasons_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    tp_usd = exits['take_profit_usd']
    sl_usd = exits['stop_loss_usd']
    tp_str = f"${tp_usd:.0f}" if not (isinstance(tp_usd, float) and np.isnan(tp_usd)) else 'N/A'
    sl_str = f"${sl_usd:.0f}" if not (isinstance(sl_usd, float) and np.isnan(sl_usd)) else 'N/A'

    st.markdown(
        f'<div style="font-size:0.82rem;margin:0.2rem 0;color:#e2e8f0;">'
        f'💰 <strong>Take Profit</strong> +100% → '
        f'<span style="color:#10b981;">{tp_str}</span></div>'
        f'<div style="font-size:0.82rem;margin:0.2rem 0;color:#e2e8f0;">'
        f'🛑 <strong>Stop Loss</strong> -50% → '
        f'<span style="color:#ef4444;">{sl_str}</span></div>'
        f'<div style="font-size:0.82rem;margin:0.2rem 0;color:#e2e8f0;">'
        f'⏰ <strong>Time</strong> {exits["time_exit_msg"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.78rem;color:#94a3b8;margin:0.5rem 0 0.25rem;">'
        'Tech exits (watch for):</div>',
        unsafe_allow_html=True,
    )

    for trigger in exits['tech_triggers']:
        t_icon = '⚠️' if trigger['triggered'] else '✓'
        t_color = '#f59e0b' if trigger['triggered'] else '#64748b'
        t_status = 'TRIGGERED' if trigger['triggered'] else 'safe'
        st.markdown(
            f'<div style="font-size:0.79rem;margin:0.12rem 0;">'
            f'<span style="color:{t_color};">{t_icon}</span> '
            f'{trigger["label"]} — '
            f'<span style="color:#94a3b8;">{trigger["current_value"]}</span> '
            f'<span style="color:{t_color};font-weight:600;">({t_status})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_positions_page():
    """Render the My Positions view: add form + live position cards."""
    st.markdown('<div class="section-header">📋 My Positions</div>', unsafe_allow_html=True)

    try:
        positions = load_positions()
    except PositionsConfigError:
        st.warning(
            "Google Sheets is not configured. Add `positions_sheet_key` and "
            "a `[gcp_service_account]` block to your Streamlit secrets. "
            "See the README → **Position Tracking Setup**."
        )
        return

    with st.expander("➕ Add a position", expanded=not positions):
        with st.form("add_position_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                ticker = st.text_input("Ticker").strip().upper()
                strategy = st.selectbox("Type", ["Long Call", "Long Put"])
            with c2:
                strike = st.number_input("Strike", min_value=0.0, step=1.0)
                expiry = st.date_input("Expiry")
            with c3:
                entry_premium = st.number_input("Entry premium ($/share)", min_value=0.0, step=0.05)
                contracts = st.number_input("Contracts", min_value=1, step=1, value=1)
            submitted = st.form_submit_button("Add", type="primary")
            if submitted:
                if not ticker or strike <= 0 or entry_premium <= 0:
                    st.error("Ticker, strike, and entry premium are required.")
                else:
                    add_position({
                        'ticker': ticker,
                        'strategy': strategy,
                        'strike': float(strike),
                        'expiry': expiry.isoformat(),
                        'entry_premium': float(entry_premium),
                        'contracts': int(contracts),
                    })
                    st.success(f"Added {ticker} ${strike:.0f} {strategy}.")
                    st.rerun()

    if not positions:
        st.info("No open positions. Add one above.")
        return

    for pos in positions:
        with st.spinner(f"Analyzing {pos['ticker']}…"):
            try:
                data = analyze_position(pos)
            except Exception as exc:  # one bad row must not kill the page
                st.error(f"Could not analyze {pos.get('ticker', '?')}: {exc}")
                if st.button("Close position", key=f"close_{pos['id']}"):
                    delete_position(pos['id'])
                    st.rerun()
                continue

        v_emoji, v_text, v_color = _PLAYBOOK_VERDICT[data['verdict']['status']]
        price_str = (f"${data['current_price']:.2f}"
                     if not np.isnan(data['current_price']) else "n/a")

        if data['pnl_usd'] is not None and data['pnl_pct'] is not None:
            pnl_color = '#10b981' if data['pnl_usd'] >= 0 else '#ef4444'
            pnl_str = (f'<span style="color:{pnl_color};font-weight:700;">'
                       f'{data["pnl_pct"] * 100:+.0f}% (${data["pnl_usd"]:+.0f})</span>')
        else:
            pnl_str = '<span style="color:#64748b;">P/L n/a</span>'

        reasons_html = ''.join(
            f'<div style="font-size:0.74rem;color:#94a3b8;">• {r}</div>'
            for r in data['verdict']['reasons']
        ) or '<div style="font-size:0.74rem;color:#64748b;">No exit signals active</div>'

        err_html = (f'<div style="font-size:0.72rem;color:#f59e0b;">⚠ {data["error"]}</div>'
                    if data['error'] else '')
        ema4h = data.get('ema4h') or {'status': 'unknown', 'label': ''}
        ema_color = _EMA4H_COLOR.get(ema4h['status'], '#64748b')
        ema_html = (f'<div style="font-size:0.76rem;color:{ema_color};margin-top:0.15rem;">'
                    f'{ema4h["label"]}</div>' if ema4h.get('label') else '')

        st.markdown(
            f'<div style="background:#1e293b;border:1px solid {v_color}55;'
            f'border-left:4px solid {v_color};border-radius:0.5rem;'
            f'padding:0.75rem 1rem;margin:0.5rem 0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="font-size:1.0rem;font-weight:700;color:#e2e8f0;">'
            f'{pos["ticker"]} ${pos["strike"]:.0f} {pos["strategy"]} · exp {pos["expiry"]}</div>'
            f'<div style="font-size:1.1rem;font-weight:800;color:{v_color};">{v_emoji} {v_text}</div>'
            f'</div>'
            f'<div style="font-size:0.82rem;color:#cbd5e1;margin-top:0.3rem;">'
            f'{pos["contracts"]}x · entry ${pos["entry_premium"]:.2f} · now {price_str} · '
            f'{data["dte"]} DTE · {pnl_str}</div>'
            f'{ema_html}{reasons_html}{err_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Close position", key=f"close_{pos['id']}"):
            delete_position(pos['id'])
            st.rerun()


# ──────────────────────────────────────────────
# VIX Chart
# ──────────────────────────────────────────────
def build_vix_chart(macro: dict) -> go.Figure:
    vix_hist = macro.get('vix_history')
    fig = go.Figure()

    if vix_hist is not None and not vix_hist.empty:
        dates = vix_hist.index
        values = vix_hist.values

        # Regime zones
        fig.add_hrect(y0=0, y1=15, fillcolor="#10b981", opacity=0.08, line_width=0, annotation_text="Low", annotation_position="left")
        fig.add_hrect(y0=15, y1=20, fillcolor="#94a3b8", opacity=0.06, line_width=0, annotation_text="Normal", annotation_position="left")
        fig.add_hrect(y0=20, y1=30, fillcolor="#f59e0b", opacity=0.08, line_width=0, annotation_text="Elevated", annotation_position="left")
        fig.add_hrect(y0=30, y1=80, fillcolor="#ef4444", opacity=0.08, line_width=0, annotation_text="High Fear", annotation_position="left")

        # Reference lines
        for level, color, dash in [(15, '#10b981', 'dot'), (20, '#f59e0b', 'dash'), (30, '#ef4444', 'dash')]:
            fig.add_hline(y=level, line_color=color, line_dash=dash, line_width=1, opacity=0.5)

        # VIX line
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='lines',
            line=dict(color='#60a5fa', width=2),
            fill='tozeroy',
            fillcolor='rgba(96, 165, 250, 0.1)',
            name='VIX',
            hovertemplate='%{x|%b %d}<br>VIX: %{y:.1f}<extra></extra>',
        ))

        # Current VIX marker
        fig.add_trace(go.Scatter(
            x=[dates[-1]], y=[values[-1]],
            mode='markers',
            marker=dict(color='#f59e0b', size=8, symbol='circle'),
            name=f'Current: {values[-1]:.1f}',
            hovertemplate=f'Current VIX: {values[-1]:.1f}<extra></extra>',
        ))
    else:
        fig.add_annotation(text="VIX history unavailable", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#64748b"))

    fig.update_layout(
        title=dict(text="VIX - 90 Day History with Regime Zones", font=dict(color="#e2e8f0", size=14)),
        paper_bgcolor='#1e293b',
        plot_bgcolor='#0f172a',
        font=dict(color='#94a3b8'),
        height=280,
        margin=dict(l=50, r=20, t=40, b=30),
        xaxis=dict(showgrid=False, color='#475569'),
        yaxis=dict(showgrid=True, gridcolor='#1e293b', color='#475569', range=[0, max(40, float(vix_hist.max()) * 1.1) if vix_hist is not None and not vix_hist.empty else 40]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='#94a3b8')),
    )
    return fig


# ──────────────────────────────────────────────
# IV Smile Chart
# ──────────────────────────────────────────────
def build_iv_smile_chart(chain_data: dict) -> go.Figure:
    fig = go.Figure()
    calls = chain_data.get('calls', pd.DataFrame())
    puts = chain_data.get('puts', pd.DataFrame())
    current_price = chain_data.get('current_price')

    if not calls.empty and 'impliedVolatility' in calls.columns:
        fig.add_trace(go.Scatter(
            x=calls['strike'], y=calls['impliedVolatility'] * 100,
            mode='lines+markers', name='Calls IV',
            line=dict(color='#10b981', width=2),
            marker=dict(size=5),
            hovertemplate='Strike: $%{x}<br>IV: %{y:.1f}%<extra>Call</extra>',
        ))

    if not puts.empty and 'impliedVolatility' in puts.columns:
        fig.add_trace(go.Scatter(
            x=puts['strike'], y=puts['impliedVolatility'] * 100,
            mode='lines+markers', name='Puts IV',
            line=dict(color='#ef4444', width=2),
            marker=dict(size=5),
            hovertemplate='Strike: $%{x}<br>IV: %{y:.1f}%<extra>Put</extra>',
        ))

    if current_price:
        fig.add_vline(x=current_price, line_color='#f59e0b', line_dash='dash', line_width=1.5,
                      annotation_text=f"  ${current_price:.2f}", annotation_font_color='#f59e0b')

    fig.update_layout(
        title=dict(text="Implied Volatility Smile", font=dict(color="#e2e8f0", size=13)),
        paper_bgcolor='#1e293b', plot_bgcolor='#0f172a',
        font=dict(color='#94a3b8'), height=300,
        margin=dict(l=50, r=20, t=40, b=30),
        xaxis=dict(title="Strike", showgrid=True, gridcolor='#1e293b', color='#475569'),
        yaxis=dict(title="IV (%)", showgrid=True, gridcolor='#1e293b', color='#475569'),
        legend=dict(font=dict(color='#94a3b8')),
    )
    return fig


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = None
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None
if 'macro_data' not in st.session_state:
    st.session_state.macro_data = None
if 'sp500_df' not in st.session_state:
    st.session_state.sp500_df = None
if 'top_lc_picks' not in st.session_state:
    st.session_state.top_lc_picks = []
if 'top_lp_picks' not in st.session_state:
    st.session_state.top_lp_picks = []


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">📊 S&P 500 Options Strategy Screener</div>
  <div class="app-subtitle">
    Screen S&P 500 stocks for optimal option strategies using IV/HV analysis, momentum, fundamentals & macro data
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load macro & S&P 500 list (auto, cached)
# ──────────────────────────────────────────────
with st.spinner("Loading macro data & S&P 500 tickers…"):
    if st.session_state.macro_data is None:
        st.session_state.macro_data = get_macro_data()
    if st.session_state.sp500_df is None:
        st.session_state.sp500_df = get_sp500_tickers()

macro = st.session_state.macro_data
sp500_df = st.session_state.sp500_df


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    view = st.radio("View", ["🔍 Screener", "📋 My Positions"], key="view_nav")
    st.markdown("---")
    st.markdown("## ⚙️ Screener Settings")
    st.markdown("---")

    # Sector filter
    all_sectors = sorted(sp500_df['sector'].dropna().unique().tolist())
    sector_options = ["All Sectors"] + all_sectors
    selected_sectors = st.multiselect(
        "Sectors",
        options=sector_options,
        default=["All Sectors"],
        help="Filter by GICS sector"
    )

    max_stocks = st.slider(
        "Max Stocks to Screen",
        min_value=10, max_value=100,
        value=50, step=10,
        help="More stocks = longer screening time"
    )

    iv_filter = st.selectbox(
        "IV Filter",
        ["All", "Low IV only (IV/HV < 0.8)"],
        help="Low IV = cheap options, better for Long Call / Long Put"
    )

    exclude_earnings = st.checkbox(
        "Exclude Near-Earnings Stocks",
        value=True,
        help="Exclude stocks with earnings within 14 days"
    )

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
    risk_pct_int = st.slider(
        "Risk per Trade (%)",
        min_value=10, max_value=100, value=33, step=1,
    )
    risk_pct = risk_pct_int / 100

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

    st.markdown("---")

    if st.button("🔍 Screen Now", type="primary", use_container_width=True):
        st.session_state.run_screen = True
    else:
        if 'run_screen' not in st.session_state:
            st.session_state.run_screen = False

    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("- Price & Options: [yfinance](https://pypi.org/project/yfinance/)")
    st.markdown("- S&P 500 list: Wikipedia")
    st.markdown("- Macro: Yahoo Finance")
    st.markdown(f"\n*Last refreshed: {datetime.now().strftime('%H:%M:%S')}*")


# ──────────────────────────────────────────────
# Top-level view routing
# ──────────────────────────────────────────────
if view == "📋 My Positions":
    render_positions_page()
    st.stop()


# ──────────────────────────────────────────────
# SECTION 1: Macro Overview (always visible)
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">🌍 Macro Environment</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

vix_val = macro.get('vix_current', 20.0)
vix_regime = macro.get('vix_regime', 'normal')
regime_labels = {
    'low': ('< 15', '#10b981'),
    'normal': ('15–20', '#94a3b8'),
    'high': ('20–30', '#f59e0b'),
    'extreme': ('> 30', '#ef4444'),
}
vix_range_str, vix_color = regime_labels.get(vix_regime, ('15–20', '#94a3b8'))
spy_price = macro.get('spy_price', 450.0)
spy_trend = "Bullish" if macro.get('spy_above_200ma') else "Bearish"
spy_trend_color = "#10b981" if spy_trend == "Bullish" else "#ef4444"
spy_1m = macro.get('spy_ret_1m', 0.0)
tnx = macro.get('tnx_yield', 4.5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">VIX Level</div>
      <div class="metric-value" style="color:{vix_color}">{vix_val:.1f}</div>
      <div class="metric-sub">Regime: {macro.get('market_regime','Normal')}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">VIX 30d Avg</div>
      <div class="metric-value">{macro.get('vix_30d_avg', 20.0):.1f}</div>
      <div class="metric-sub">Range: {vix_range_str}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">SPY Trend</div>
      <div class="metric-value" style="color:{spy_trend_color}">{spy_trend}</div>
      <div class="metric-sub">${spy_price:.2f} | 1M: {spy_1m:+.1f}%</div>
    </div>""", unsafe_allow_html=True)

with col4:
    rate_env = "High" if tnx > 5 else ("Moderate" if tnx > 3 else "Low")
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">10Y Rate (TNX)</div>
      <div class="metric-value">{tnx:.2f}%</div>
      <div class="metric-sub">Rate Env: {rate_env}</div>
    </div>""", unsafe_allow_html=True)

# Market bias banner
market_bias = macro.get('market_bias', 'neutral')
if vix_val < 15:
    banner_class = "banner-premium-buy"
    banner_text = "💡 Low VIX — Options Are Cheap. Ideal environment for Long Calls and Long Puts."
elif vix_val > 20:
    banner_class = "banner-mixed"
    banner_text = "⚠️ Elevated VIX — Options Are Expensive. Premium paid will be higher; size positions carefully."
else:
    banner_class = "banner-mixed"
    banner_text = "⚖️ Normal Volatility — Screen individual stocks for low IV/HV ratio to find cheap options."

st.markdown(f'<div class="{banner_class}">{banner_text}</div>', unsafe_allow_html=True)

# VIX Chart
st.plotly_chart(build_vix_chart(macro), use_container_width=True)


# ──────────────────────────────────────────────
# SECTION 2: Screener Results
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Screener Results</div>', unsafe_allow_html=True)

if st.session_state.run_screen:
    st.session_state.run_screen = False  # Reset flag

    # Determine tickers to screen
    filtered_sp500 = sp500_df.copy()

    if selected_sectors and "All Sectors" not in selected_sectors:
        filtered_sp500 = filtered_sp500[filtered_sp500['sector'].isin(selected_sectors)]

    tickers_to_screen = filtered_sp500['ticker'].tolist()[:max_stocks]

    if not tickers_to_screen:
        st.warning("No tickers match the selected filters.")
    else:
        st.info(f"Screening {len(tickers_to_screen)} stocks…")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Batch fundamentals
        status_text.markdown("**Step 1/3:** Fetching price history & fundamentals…")
        with st.spinner(""):
            fundamentals_df = batch_screen_fundamentals(tickers_to_screen)
        progress_bar.progress(33)

        if fundamentals_df.empty:
            st.error("Could not fetch fundamental data. Please try again.")
        else:
            # Merge with sector info
            fundamentals_df = fundamentals_df.merge(
                filtered_sp500[['ticker', 'name', 'sector', 'sub_industry']],
                on='ticker', how='left'
            )

            # Step 2: Enrich with IV
            status_text.markdown("**Step 2/3:** Fetching implied volatility from options chains…")
            iv_progress = st.progress(0)

            def iv_progress_callback(done, total):
                iv_progress.progress(int(done / total * 100))
                status_text.markdown(f"**Step 2/3:** Fetching IV… ({done}/{total})")

            fundamentals_df = enrich_with_iv(fundamentals_df, progress_callback=iv_progress_callback)
            iv_progress.empty()

            progress_bar.progress(66)

            # Step 3: Score strategies
            status_text.markdown("**Step 3/3:** Scoring strategies…")
            results_df = score_strategies(fundamentals_df, macro)
            progress_bar.progress(100)

            # Apply filters
            filtered_df = results_df.copy()

            if exclude_earnings:
                mask = (filtered_df['days_to_earnings'].isna()) | (filtered_df['days_to_earnings'] > 14)
                filtered_df = filtered_df[mask]

            if iv_filter == "Low IV only (IV/HV < 0.8)":
                filtered_df = filtered_df[filtered_df['iv_hv_ratio'] < 0.8]

            if affordable_only and 'atm_mid_price' in filtered_df.columns:
                cost_series = filtered_df['atm_mid_price'].fillna(0) * 100
                filtered_df = filtered_df[cost_series <= max_risk_usd]

            filtered_df['_top_score'] = filtered_df[['lc_score', 'lp_score']].max(axis=1)
            filtered_df = filtered_df.sort_values('_top_score', ascending=False).drop(columns=['_top_score'])

            st.session_state.screening_results = filtered_df
            progress_bar.empty()
            status_text.empty()

if st.session_state.screening_results is not None:
    results_df = st.session_state.screening_results

    n_total = len(st.session_state.sp500_df) if st.session_state.sp500_df is not None else 0
    n_screened = len(results_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Stocks Screened", n_screened)
    with c2:
        lc_top = (results_df['lc_score'] >= 60).sum() if 'lc_score' in results_df.columns else 0
        st.metric("Long Call Picks", int(lc_top))
    with c3:
        lp_top = (results_df['lp_score'] >= 60).sum() if 'lp_score' in results_df.columns else 0
        st.metric("Long Put Picks", int(lp_top))
    with c4:
        near_earn = results_df['days_to_earnings'].notna().sum() if 'days_to_earnings' in results_df.columns else 0
        st.metric("Near Earnings", int(near_earn))

    st.markdown("---")

    # Build display table
    display_cols = []
    col_map = {}

    def safe_col(df, col_name, default=np.nan):
        if col_name in df.columns:
            return df[col_name]
        return pd.Series([default] * len(df))

    display_df = pd.DataFrame()
    display_df['Ticker'] = results_df['ticker']
    display_df['Name'] = results_df['name'].fillna(results_df['ticker']) if 'name' in results_df.columns else results_df['ticker']
    display_df['Sector'] = results_df['sector'].fillna('Unknown') if 'sector' in results_df.columns else 'Unknown'
    display_df['Price'] = safe_col(results_df, 'price').round(2)
    display_df['HV30'] = (safe_col(results_df, 'hv30') * 100).round(1)
    display_df['ATM IV'] = (safe_col(results_df, 'atm_iv') * 100).round(1)
    display_df['IV/HV'] = safe_col(results_df, 'iv_hv_ratio').round(2)
    display_df['RSI'] = safe_col(results_df, 'rsi').round(1)
    display_df['Mom 1M %'] = safe_col(results_df, 'ret_1m').round(1)
    display_df['Trend'] = safe_col(results_df, 'trend', default='—')
    display_df['Days→Earn'] = safe_col(results_df, 'days_to_earnings', default=None)
    display_df['Δ Delta'] = safe_col(results_df, 'atm_delta').round(3)
    display_df['Γ Gamma'] = safe_col(results_df, 'atm_gamma').round(4)
    display_df['Θ Theta'] = safe_col(results_df, 'atm_theta').round(2)
    display_df['V Vega'] = safe_col(results_df, 'atm_vega').round(2)
    display_df['Call OI'] = safe_col(results_df, 'atm_call_oi').round(0)
    display_df['Put OI'] = safe_col(results_df, 'atm_put_oi').round(0)
    display_df['Best'] = safe_col(results_df, 'best_strategy', default='—')
    display_df['LC Score'] = safe_col(results_df, 'lc_score').round(0)
    display_df['LP Score'] = safe_col(results_df, 'lp_score').round(0)
    cost_usd = safe_col(results_df, 'atm_mid_price') * 100
    display_df['Cost/Contract'] = cost_usd.round(2)
    display_df['Affordable'] = cost_usd.apply(
        lambda c: '✓' if (not (isinstance(c, float) and np.isnan(c)) and c <= max_risk_usd) else ('✗' if not (isinstance(c, float) and np.isnan(c)) else 'N/A')
    )

    # Rename HV/IV columns with % suffix for display
    display_df = display_df.rename(columns={'HV30': 'HV30 %', 'ATM IV': 'ATM IV %'})

    # Replace NaN with N/A for display
    display_df = display_df.replace({np.nan: None})

    st.dataframe(
        display_df,
        use_container_width=True,
        height=450,
        column_config={
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            'Name': st.column_config.TextColumn('Company Name'),
            'Sector': st.column_config.TextColumn('Sector'),
            'Price': st.column_config.NumberColumn('Price ($)', format="$%.2f"),
            'HV30 %': st.column_config.NumberColumn('HV30 (%)', format="%.1f%%"),
            'ATM IV %': st.column_config.NumberColumn('ATM IV (%)', format="%.1f%%"),
            'IV/HV': st.column_config.NumberColumn('IV/HV', format="%.2f"),
            'RSI': st.column_config.NumberColumn('RSI', format="%.0f"),
            'Mom 1M %': st.column_config.NumberColumn('Mom 1M (%)', format="%+.1f%%"),
            'Trend': st.column_config.TextColumn('Trend'),
            'Days→Earn': st.column_config.NumberColumn('Days→Earn', format="%d"),
            'Δ Delta': st.column_config.NumberColumn('Δ Delta (ATM Call)', format="%.3f", help="ATM call delta at 30 DTE. ~0.5 for ATM options."),
            'Γ Gamma': st.column_config.NumberColumn('Γ Gamma', format="%.4f", help="Rate of delta change per $1 move."),
            'Θ Theta': st.column_config.NumberColumn('Θ Theta ($/day)', format="$%.2f", help="Daily time decay per contract (100 shares)."),
            'V Vega': st.column_config.NumberColumn('V Vega ($/1%IV)', format="$%.2f", help="$ change per 1% IV move per contract."),
            'Call OI': st.column_config.NumberColumn('Call OI (ATM)', format="%d"),
            'Put OI': st.column_config.NumberColumn('Put OI (ATM)', format="%d"),
            'Best': st.column_config.TextColumn('Best Strategy'),
            'LC Score': st.column_config.ProgressColumn('Long Call Score', min_value=0, max_value=100, format="%.0f"),
            'LP Score': st.column_config.ProgressColumn('Long Put Score', min_value=0, max_value=100, format="%.0f"),
            'Cost/Contract': st.column_config.NumberColumn('Cost/Contract ($)', format="$%.2f",
                help="Estimated cost per 1 contract (ATM call mid × 100)"),
            'Affordable': st.column_config.TextColumn('Affordable',
                help=f"✓ = within ${max_risk_usd:.0f} max risk budget"),
        },
        hide_index=True,
    )

    # Top 10 Picks
    st.markdown("---")
    st.markdown('<div class="section-header">🏆 Top 10 Picks</div>', unsafe_allow_html=True)
    st.caption("Top 5 Long Call and Top 5 Long Put candidates ranked by composite score (IV, trend, momentum, OI, Greeks).")

    top_lc = results_df.nlargest(5, 'lc_score')
    top_lp = results_df.nlargest(5, 'lp_score')
    st.session_state.top_lc_picks = top_lc.to_dict('records')
    st.session_state.top_lp_picks = top_lp.to_dict('records')

    col_lc, col_lp = st.columns(2)

    def _trend_emoji(t):
        return {'Strong Up': '🚀', 'Up': '📈', 'Sideways': '➡️', 'Down': '📉', 'Strong Down': '🔻'}.get(t, '❓')

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

    # Ticker selector for detail view
    st.markdown("---")
    ticker_options = results_df['ticker'].tolist()
    selected = st.selectbox(
        "Select a stock for detailed strategy analysis:",
        options=["— Select a ticker —"] + ticker_options,
        key="ticker_selector",
    )
    if selected != "— Select a ticker —":
        st.session_state.selected_ticker = selected

else:
    st.info("👈 Configure your filters in the sidebar and click **Screen Now** to begin.")


# ──────────────────────────────────────────────
# SECTION 3: Strategy Detail View
# ──────────────────────────────────────────────
if st.session_state.selected_ticker and st.session_state.screening_results is not None:
    ticker_str = st.session_state.selected_ticker
    results_df = st.session_state.screening_results

    row_df = results_df[results_df['ticker'] == ticker_str]
    if row_df.empty:
        st.warning(f"No data found for {ticker_str}")
    else:
        row = row_df.iloc[0].to_dict()

        st.markdown(f'<div class="section-header">📋 Strategy Detail: {ticker_str}</div>', unsafe_allow_html=True)

        # Stock summary header
        company_name = row.get('name', ticker_str) or ticker_str
        sector = row.get('sector', 'Unknown') or 'Unknown'
        price = row.get('price', 0) or 0
        hv30 = row.get('hv30', np.nan)
        atm_iv = row.get('atm_iv', np.nan)
        iv_hv = row.get('iv_hv_ratio', np.nan)
        rsi = row.get('rsi', np.nan)
        ret_1m = row.get('ret_1m', np.nan)
        market_cap = row.get('market_cap', np.nan)
        pe_ratio = row.get('pe_ratio', np.nan)
        next_earn = row.get('next_earnings')
        days_earn = row.get('days_to_earnings')

        iv_hv_str = iv_hv_label(iv_hv)
        iv_hv_css = iv_hv_css_class(iv_hv)

        with st.container():
            hcol1, hcol2, hcol3, hcol4, hcol5, hcol6 = st.columns(6)
            with hcol1:
                st.metric("Price", f"${price:.2f}")
            with hcol2:
                st.metric("HV30", f"{hv30*100:.1f}%" if not np.isnan(hv30) else "N/A")
            with hcol3:
                st.metric("ATM IV", f"{atm_iv*100:.1f}%" if not np.isnan(atm_iv) else "N/A")
            with hcol4:
                iv_hv_display = f"{iv_hv:.2f} ({iv_hv_str})" if not np.isnan(iv_hv) else "N/A"
                st.metric("IV/HV", iv_hv_display)
            with hcol5:
                st.metric("RSI", f"{rsi:.0f}" if not np.isnan(rsi) else "N/A")
            with hcol6:
                earn_str = f"{days_earn}d ({next_earn})" if days_earn is not None else "Unknown"
                st.metric("Next Earnings", earn_str)

            st.markdown(f"**{company_name}** | Sector: {sector} | Market Cap: {fmt_market_cap(market_cap)} | P/E: {fmt_num(pe_ratio)} | 1M Return: {fmt_num(ret_1m)}%")

        st.markdown("---")

        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "📈 Strategy Recommendations",
            "⛓️ Option Chain",
            "🎯 Entry/Exit Playbook",
        ])

        # ── Tab 1: Strategy Recommendations ──
        with tab1:
            strategies = suggest_strategies(row, macro)

            if not strategies:
                st.info("No strategy recommendations available.")
            else:
                strategies = sorted(strategies, key=lambda x: -x['score'])

                for strat in strategies:
                    score = strat['score']
                    risk_lvl = strat['risk_level']
                    card_class = "strategy-card-high" if risk_lvl == 'High' else "strategy-card-low"
                    score_cls = score_color(score)

                    with st.expander(
                        f"{'⭐ ' if score >= 65 else ''}{strat['strategy']} — Score: {score}  |  {risk_lvl} Risk",
                        expanded=(score >= 60)
                    ):
                        c_left, c_right = st.columns([2, 1])

                        with c_left:
                            st.markdown(f"**Description:** {strat['description']}")
                            st.markdown(f"**Rationale:** {strat['rationale']}")
                            st.markdown(f"**Setup:** {strat['setup']}")
                            st.markdown(f"**Ideal Conditions:** {strat['ideal_conditions']}")

                        with c_right:
                            st.markdown(f"**Max Profit:** {strat['max_profit']}")
                            st.markdown(f"**Max Loss:** {strat['max_loss']}")
                            st.markdown(f"**Timeframe:** {strat['timeframe']}")
                            tags_html = " ".join([f'<span style="background:#1e3a5f;color:#93c5fd;padding:2px 8px;border-radius:12px;font-size:0.75rem;">{t}</span>' for t in strat['tags']])
                            st.markdown(tags_html, unsafe_allow_html=True)

                        # Specific contracts
                        st.markdown("---")
                        if st.button(f"🔗 Fetch Specific Contracts for {strat['strategy']}", key=f"contracts_{strat['strategy']}_{ticker_str}"):
                            with st.spinner("Fetching option chain for specific contracts…"):
                                contracts = get_specific_contracts(ticker_str, strat['strategy'], row)

                            if contracts.get('error'):
                                st.warning(f"Could not fetch contracts: {contracts['error']}")
                            else:
                                exp = contracts.get('expiry', 'N/A')
                                dte_val = contracts.get('dte', 'N/A')
                                est_premium = contracts.get('estimated_premium')
                                max_profit = contracts.get('max_profit')
                                max_loss_val = contracts.get('max_loss')
                                breakeven = contracts.get('breakeven', [])

                                st.markdown(f"**Expiry:** {exp} ({dte_val} DTE)")
                                if est_premium is not None:
                                    prem_label = "Net Credit" if est_premium > 0 else "Net Debit"
                                    st.markdown(f"**{prem_label}:** ${abs(est_premium):.2f} per contract")
                                if max_profit is not None:
                                    st.markdown(f"**Max Profit:** {max_profit if isinstance(max_profit, str) else f'${max_profit:.2f}'}")
                                if max_loss_val is not None:
                                    st.markdown(f"**Max Loss:** {max_loss_val if isinstance(max_loss_val, str) else f'${max_loss_val:.2f}'}")
                                if breakeven:
                                    st.markdown(f"**Breakeven(s):** {', '.join([f'${b:.2f}' for b in breakeven])}")

                                legs = contracts.get('legs', [])
                                if legs:
                                    st.markdown("**Contract Legs:**")
                                    legs_df = pd.DataFrame(legs)
                                    if not legs_df.empty:
                                        disp_cols = [c for c in ['action', 'type', 'strike', 'bid', 'ask', 'mid', 'iv', 'delta', 'gamma', 'theta', 'vega', 'volume', 'oi'] if c in legs_df.columns]
                                        legs_df_disp = legs_df[disp_cols].copy()
                                        if 'iv' in legs_df_disp.columns:
                                            legs_df_disp['iv'] = (legs_df_disp['iv'] * 100).round(1).astype(str) + '%'
                                        st.dataframe(legs_df_disp, use_container_width=True, hide_index=True)

        # ── Tab 2: Option Chain ──
        with tab2:
            col_exp, col_btn = st.columns([3, 1])
            with col_exp:
                chain_dte_target = st.slider("Target DTE for chain display", 14, 90, 30, key=f"dte_{ticker_str}")
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                load_chain = st.button("Load Option Chain", key=f"load_chain_{ticker_str}")

            if load_chain:
                with st.spinner(f"Fetching option chain for {ticker_str}…"):
                    chain_data = get_option_chain_display(ticker_str, target_dte=chain_dte_target)

                if chain_data.get('error'):
                    st.warning(f"Option chain error: {chain_data['error']}")
                else:
                    chain_exp = chain_data.get('expiry', 'N/A')
                    chain_dte = chain_data.get('dte', 'N/A')
                    cur_price = chain_data.get('current_price', 0)
                    calls_chain = chain_data.get('calls', pd.DataFrame())
                    puts_chain = chain_data.get('puts', pd.DataFrame())

                    st.markdown(f"**Expiry:** {chain_exp} ({chain_dte} DTE) | **Current Price:** ${cur_price:.2f}")

                    # IV Smile chart
                    st.plotly_chart(build_iv_smile_chart(chain_data), use_container_width=True)

                    st.markdown("#### Calls")
                    if not calls_chain.empty:
                        calls_display = calls_chain.copy()
                        if 'impliedVolatility' in calls_display.columns:
                            calls_display['impliedVolatility'] = (calls_display['impliedVolatility'] * 100).round(1)
                            calls_display = calls_display.rename(columns={'impliedVolatility': 'IV %'})
                        # Highlight ATM row
                        st.dataframe(
                            calls_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'strike': st.column_config.NumberColumn('Strike', format="$%.2f"),
                                'lastPrice': st.column_config.NumberColumn('Last', format="$%.2f"),
                                'bid': st.column_config.NumberColumn('Bid', format="$%.2f"),
                                'ask': st.column_config.NumberColumn('Ask', format="$%.2f"),
                                'volume': st.column_config.NumberColumn('Volume'),
                                'openInterest': st.column_config.NumberColumn('OI'),
                                'IV %': st.column_config.NumberColumn('IV (%)', format="%.1f%%"),
                            }
                        )
                    else:
                        st.info("No call data available.")

                    st.markdown("#### Puts")
                    if not puts_chain.empty:
                        puts_display = puts_chain.copy()
                        if 'impliedVolatility' in puts_display.columns:
                            puts_display['impliedVolatility'] = (puts_display['impliedVolatility'] * 100).round(1)
                            puts_display = puts_display.rename(columns={'impliedVolatility': 'IV %'})
                        st.dataframe(
                            puts_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'strike': st.column_config.NumberColumn('Strike', format="$%.2f"),
                                'lastPrice': st.column_config.NumberColumn('Last', format="$%.2f"),
                                'bid': st.column_config.NumberColumn('Bid', format="$%.2f"),
                                'ask': st.column_config.NumberColumn('Ask', format="$%.2f"),
                                'volume': st.column_config.NumberColumn('Volume'),
                                'openInterest': st.column_config.NumberColumn('OI'),
                                'IV %': st.column_config.NumberColumn('IV (%)', format="%.1f%%"),
                            }
                        )
                    else:
                        st.info("No put data available.")

        # ── Tab 3: Entry/Exit Playbook ──
        with tab3:
            smc_tuple = (
                bool(row.get('smc_bos_bullish')),     bool(row.get('smc_bos_bearish')),
                bool(row.get('smc_choch_bullish')),   bool(row.get('smc_choch_bearish')),
                bool(row.get('smc_discount_zone')),   bool(row.get('smc_premium_zone')),
                bool(row.get('smc_near_bullish_ob')), bool(row.get('smc_near_bearish_ob')),
                bool(row.get('smc_in_bullish_fvg')),  bool(row.get('smc_in_bearish_fvg')),
            )
            iv_hv_val = row.get('iv_hv_ratio', np.nan)
            iv_hv_for_rec = float(iv_hv_val) if not (isinstance(iv_hv_val, float) and np.isnan(iv_hv_val)) else 0.9

            rec_lc = get_strike_recommendation(
                ticker_str, 'Long Call', iv_hv_for_rec, smc_tuple, budget_config['max_risk_usd'],
            )
            rec_lp = get_strike_recommendation(
                ticker_str, 'Long Put', iv_hv_for_rec, smc_tuple, budget_config['max_risk_usd'],
            )

            readiness_lc = compute_entry_readiness(row, 'Long Call')
            readiness_lp = compute_entry_readiness(row, 'Long Put')
            exits_lc = compute_exit_rules(row, 'Long Call', rec_lc)
            exits_lp = compute_exit_rules(row, 'Long Put', rec_lp)

            _ema4h = fetch_4h_ema_status(ticker_str)
            _ema_color = _EMA4H_COLOR.get(_ema4h['status'], '#64748b')
            if _ema4h.get('label'):
                st.markdown(
                    f'<div style="font-size:0.85rem;font-weight:600;color:{_ema_color};'
                    f'margin:0.3rem 0 0.6rem;">{_ema4h["label"]}</div>',
                    unsafe_allow_html=True,
                )

            col_lc, col_lp = st.columns(2)
            with col_lc:
                _render_playbook_col(readiness_lc, exits_lc, rec_lc, '📈 Long Call', f"pb_{ticker_str}_lc")
            with col_lp:
                _render_playbook_col(readiness_lp, exits_lp, rec_lp, '📉 Long Put', f"pb_{ticker_str}_lp")


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Disclaimer:</strong> This tool is for <strong>educational and informational purposes only</strong>.
  Options trading involves significant risk and is not suitable for all investors.
  Nothing on this platform constitutes financial advice. Always conduct your own research and
  consult a qualified financial advisor before making investment decisions.
  Data is sourced from Yahoo Finance via yfinance and may be delayed or inaccurate.
  Past performance is not indicative of future results.
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Floating Analyst Chatbot
# ──────────────────────────────────────────────
_focused_ticker = st.session_state.get('selected_ticker')
_focused_row = None
if _focused_ticker and st.session_state.screening_results is not None:
    _row_df = st.session_state.screening_results
    _match = _row_df[_row_df['ticker'] == _focused_ticker]
    if not _match.empty:
        _focused_row = _match.iloc[0].to_dict()

render_chat_button({
    "macro": st.session_state.macro_data or {},
    "top_lc": st.session_state.top_lc_picks,
    "top_lp": st.session_state.top_lp_picks,
    "focused_ticker": _focused_ticker,
    "focused_row": _focused_row,
})
