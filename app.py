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

warnings.filterwarnings('ignore')

from screener import (
    get_sp500_tickers,
    get_macro_data,
    batch_screen_fundamentals,
    enrich_with_iv,
    score_strategies,
)
from strategies import suggest_strategies, get_specific_contracts, get_option_chain_display

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
    st.markdown("## ⚙️ Screener Settings")
    st.markdown("---")

    risk_profile = st.selectbox(
        "Risk Profile",
        ["All Strategies", "Low Risk (Premium Earning)", "High Risk (High Reward)"],
        help="Filter suggested strategies by risk appetite"
    )

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
        ["All", "High IV only (IV/HV > 1.2)", "Low IV only (IV/HV < 0.8)"],
        help="Filter by IV/HV ratio signal"
    )

    screen_mode = st.radio(
        "Screening Mode",
        ["Quick Screen (HV only, fast)", "Full Screen (with IV from options chain)"],
        help="Quick mode uses only historical volatility. Full mode fetches live IV from options chains (slower)."
    )

    exclude_earnings = st.checkbox(
        "Exclude Near-Earnings Stocks",
        value=True,
        help="Exclude stocks with earnings within 14 days"
    )

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
if vix_val > 20 or (not np.isnan(vix_val) and macro.get('vix_regime') in ['high', 'extreme']):
    banner_class = "banner-premium-sell"
    banner_text = "📈 Market Favors: Premium Selling — High VIX = Expensive Options. Sell premium with iron condors, bull put spreads, CSPs."
elif vix_val < 15:
    banner_class = "banner-premium-buy"
    banner_text = "💡 Market Favors: Premium Buying — Low VIX = Cheap Options. Good environment for long calls/puts or straddles."
else:
    banner_class = "banner-mixed"
    banner_text = "⚖️ Market Favors: Mixed / Selective — Normal volatility. Screen individual stocks for IV/HV signals."

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

            # Step 2: Enrich with IV (Full Screen mode only)
            is_full_screen = "Full Screen" in screen_mode

            if is_full_screen:
                status_text.markdown("**Step 2/3:** Fetching implied volatility from options chains…")
                iv_progress = st.progress(0)

                def iv_progress_callback(done, total):
                    iv_progress.progress(int(done / total * 100))
                    status_text.markdown(f"**Step 2/3:** Fetching IV… ({done}/{total})")

                fundamentals_df = enrich_with_iv(fundamentals_df, progress_callback=iv_progress_callback)
                iv_progress.empty()
            else:
                # Quick mode: set IV columns to NaN
                fundamentals_df['atm_iv'] = np.nan
                fundamentals_df['iv_hv_ratio'] = np.nan
                fundamentals_df['next_earnings'] = None
                fundamentals_df['days_to_earnings'] = None
                status_text.markdown("**Step 2/3:** Quick mode — skipping IV fetch.")

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

            if iv_filter == "High IV only (IV/HV > 1.2)":
                filtered_df = filtered_df[filtered_df['iv_hv_ratio'] > 1.2]
            elif iv_filter == "Low IV only (IV/HV < 0.8)":
                filtered_df = filtered_df[filtered_df['iv_hv_ratio'] < 0.8]

            if risk_profile == "Low Risk (Premium Earning)":
                filtered_df = filtered_df.sort_values('low_risk_score', ascending=False)
            elif risk_profile == "High Risk (High Reward)":
                filtered_df = filtered_df.sort_values('high_risk_score', ascending=False)
            else:
                filtered_df['combined_score'] = filtered_df[['low_risk_score', 'high_risk_score']].max(axis=1)
                filtered_df = filtered_df.sort_values('combined_score', ascending=False)

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
        low_risk_top = (results_df['low_risk_score'] >= 60).sum() if 'low_risk_score' in results_df.columns else 0
        st.metric("Low Risk Picks", int(low_risk_top))
    with c3:
        high_risk_top = (results_df['high_risk_score'] >= 60).sum() if 'high_risk_score' in results_df.columns else 0
        st.metric("High Risk Picks", int(high_risk_top))
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
    display_df['Days→Earn'] = safe_col(results_df, 'days_to_earnings', default=None)
    display_df['Top Strategy'] = safe_col(results_df, 'best_strategy', default='—')
    display_df['Low Risk Score'] = safe_col(results_df, 'low_risk_score').round(0)
    display_df['High Risk Score'] = safe_col(results_df, 'high_risk_score').round(0)

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
            'Days→Earn': st.column_config.NumberColumn('Days→Earn', format="%d"),
            'Top Strategy': st.column_config.TextColumn('Top Strategy'),
            'Low Risk Score': st.column_config.ProgressColumn('Low Risk Score', min_value=0, max_value=100, format="%.0f"),
            'High Risk Score': st.column_config.ProgressColumn('High Risk Score', min_value=0, max_value=100, format="%.0f"),
        },
        hide_index=True,
    )

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
        tab1, tab2 = st.tabs(["📈 Strategy Recommendations", "⛓️ Option Chain"])

        # ── Tab 1: Strategy Recommendations ──
        with tab1:
            strategies = suggest_strategies(row, macro)

            if not strategies:
                st.info("No strategy recommendations available.")
            else:
                # Apply risk profile filter for display
                if risk_profile == "Low Risk (Premium Earning)":
                    strategies = [s for s in strategies if s['risk_level'] == 'Low']
                elif risk_profile == "High Risk (High Reward)":
                    strategies = [s for s in strategies if s['risk_level'] == 'High']

                strategies = sorted(strategies, key=lambda x: -x['score'])

                for strat in strategies[:6]:  # Show top 6
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
                                        disp_cols = [c for c in ['action', 'type', 'strike', 'bid', 'ask', 'mid', 'iv', 'volume', 'oi'] if c in legs_df.columns]
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
