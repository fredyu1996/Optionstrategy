import math

import streamlit as st

_SMC_LABELS = {
    'smc_bos_bullish':     'Bullish BoS',
    'smc_bos_bearish':     'Bearish BoS',
    'smc_choch_bullish':   'Bullish CHoCH',
    'smc_choch_bearish':   'Bearish CHoCH',
    'smc_discount_zone':   'Discount Zone',
    'smc_premium_zone':    'Premium Zone',
    'smc_near_bullish_ob': 'Near Bull OB',
    'smc_near_bearish_ob': 'Near Bear OB',
    'smc_in_bullish_fvg':  'Bull FVG',
    'smc_in_bearish_fvg':  'Bear FVG',
}

_FLOAT_CSS = """
<style>
div[data-testid="stPopover"]:last-of-type {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    z-index: 9999;
}
</style>
"""


def _build_system_prompt(context: dict) -> str:
    macro = context["macro"]
    top_lc = context.get("top_lc", [])
    top_lp = context.get("top_lp", [])
    focused_ticker = context.get("focused_ticker")
    focused_row = context.get("focused_row")

    def _fmt(val, fmt):
        try:
            return format(val, fmt) if not math.isnan(float(val)) else "N/A"
        except (TypeError, ValueError):
            return "N/A"

    def _pick_line(rank, row, score_key):
        ticker = row.get("ticker", "?")
        score = row.get(score_key, 0)
        iv_hv = row.get("iv_hv_ratio", float("nan"))
        trend = row.get("trend", "—")
        delta = row.get("atm_delta", float("nan"))
        return f"{rank}. {ticker} | Score={score:.0f} | IV/HV={_fmt(iv_hv, '.2f')} | Trend={trend} | Δ={_fmt(delta, '.3f')}"

    if top_lc:
        lc_summary = "\n".join(_pick_line(i + 1, r, "lc_score") for i, r in enumerate(top_lc))
    else:
        lc_summary = "No screener results yet."

    if top_lp:
        lp_summary = "\n".join(_pick_line(i + 1, r, "lp_score") for i, r in enumerate(top_lp))
    else:
        lp_summary = "No screener results yet."

    focused_block = ""
    if focused_ticker and focused_row:
        smc_active = [label for key, label in _SMC_LABELS.items() if focused_row.get(key)]
        smc_str = ", ".join(smc_active) if smc_active else "None"
        iv_hv = focused_row.get("iv_hv_ratio", float("nan"))
        ret_1m = focused_row.get("ret_1m", float("nan"))
        delta = focused_row.get("atm_delta", float("nan"))
        theta = focused_row.get("atm_theta", float("nan"))
        vega = focused_row.get("atm_vega", float("nan"))
        iv_str = _fmt(iv_hv, '.2f')
        ret_str = _fmt(ret_1m, '+.1f')
        delta_str = _fmt(delta, '.3f')
        theta_str = _fmt(theta, '.2f')
        vega_str = _fmt(vega, '.2f')
        focused_block = (
            f"\nFOCUSED STOCK: {focused_ticker}\n"
            f"- Score: LC={focused_row.get('lc_score', 0):.0f} | LP={focused_row.get('lp_score', 0):.0f}\n"
            f"- IV/HV: {iv_str} | Trend: {focused_row.get('trend', '—')} | 1M: {ret_str}%\n"
            f"- Greeks: Δ={delta_str} | Θ={theta_str}/day | V={vega_str}\n"
            f"- SMC Signals: {smc_str}\n"
            f"- Best Strategy: {focused_row.get('best_strategy', '—')}\n"
        )

    return (
        "You are a professional options analyst assistant for an S&P 500 options screener.\n\n"
        "MARKET CONTEXT:\n"
        f"- VIX: {macro['vix_current']} (Regime: {macro['vix_regime']} | {macro['market_regime']})\n"
        f"- SPY: {macro['spy_trend']} at ${macro['spy_price']:.2f} | 1M return: {macro['spy_ret_1m']:+.1f}%\n"
        f"- 10Y Rate: {macro['tnx_yield']:.2f}%\n\n"
        "TOP LONG CALL PICKS:\n"
        f"{lc_summary}\n\n"
        "TOP LONG PUT PICKS:\n"
        f"{lp_summary}\n"
        f"{focused_block}\n"
        "Your role: verify strategy quality, assess whether the long call or long put is worth\n"
        "entering, and flag key risks. Be concise and data-driven. Use options terminology.\n"
        "No filler. End every response with a clear verdict:\n"
        "PROCEED / CAUTION / SKIP — and one sentence of reasoning."
    )


def render_chat_button(context: dict) -> None:
    st.markdown(_FLOAT_CSS, unsafe_allow_html=True)

    with st.popover("🤖 Analyst"):
        st.markdown("#### Options Analyst")
        question = st.text_area(
            "Your question",
            placeholder="Ask about a strategy, pick, or market condition…",
            height=100,
            label_visibility="collapsed",
        )
        if st.button("Analyze", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Enter a question first.")
                return

            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                st.error("ANTHROPIC_API_KEY not set in Streamlit secrets.")
                return

            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                system_prompt = _build_system_prompt(context)
                with st.spinner("Analyzing…"):
                    message = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=512,
                        system=system_prompt,
                        messages=[{"role": "user", "content": question}],
                    )
                if not message.content:
                    st.error("Claude API returned an empty response.")
                    return
                response = message.content[0].text
                st.markdown(response)
            except Exception as e:
                st.error(f"Claude API error: {e}")
