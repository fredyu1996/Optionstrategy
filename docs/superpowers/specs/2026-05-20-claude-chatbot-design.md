# Claude Analyst Chatbot — Design Spec
**Date:** 2026-05-20
**Status:** Approved

## Overview

Add a Claude-powered floating analyst chatbot to the S&P 500 Options Strategy Screener. The chatbot verifies strategy quality and advises whether a long call or long put is worth entering, using live screener data as context.

---

## Architecture

### New file: `chatbot.py`

Single public function:
```python
def render_chat_button(context: dict) -> None
```

Called once at the bottom of `app.py` after screener results are available.

### Context dict shape
```python
{
    "macro": {
        "vix_current": float,
        "vix_regime": str,        # low/normal/high/extreme
        "market_regime": str,
        "spy_price": float,
        "spy_trend": str,         # Bullish/Bearish
        "spy_ret_1m": float,
        "tnx_yield": float,
    },
    "top_lc": list[dict],         # top 5 long call picks, each with ticker/scores/Greeks/SMC
    "top_lp": list[dict],         # top 5 long put picks
    "focused_ticker": str | None, # ticker of last-expanded stock card, or None
    "focused_row": dict | None,   # full screener row for focused_ticker, or None
}
```

### Integration in `app.py`
- `st.session_state.selected_ticker` already exists (set via selectbox at line 964) — used as `focused_ticker`, no new session state needed
- `focused_row` derived by filtering `st.session_state.screening_results` on `selected_ticker`
- `render_chat_button(context)` called at bottom of `app.py`, always rendered

---

## UI

### Floating button
`st.popover("🤖 Analyst")` positioned fixed bottom-right via injected CSS.
Must target only the chatbot popover (rendered last on the page):
```css
div[data-testid="stPopover"]:last-of-type {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    z-index: 9999;
}
```

### Inside the popover
- Title: "Options Analyst"
- `st.text_area` for user question (placeholder: "Ask about a strategy, pick, or market condition…")
- `st.button("Analyze")` — triggers Claude API call
- Response rendered via `st.markdown`
- No chat history — each submit is a fresh call

---

## Claude API Integration

**Model:** `claude-sonnet-4-6`

**API key:** Read from `st.secrets["ANTHROPIC_API_KEY"]` (set in Streamlit Cloud secrets).

**Call pattern:** Single-turn, no history. System prompt carries full context; user message is the question.

### System prompt template
```
You are a professional options analyst assistant for an S&P 500 options screener.

MARKET CONTEXT:
- VIX: {vix_current} (Regime: {vix_regime} | {market_regime})
- SPY: {spy_trend} at ${spy_price:.2f} | 1M return: {spy_ret_1m:+.1f}%
- 10Y Rate: {tnx_yield:.2f}%

TOP LONG CALL PICKS:
{top_lc_summary}

TOP LONG PUT PICKS:
{top_lp_summary}

{focused_block}

Your role: verify strategy quality, assess whether the long call or long put is worth
entering, and flag key risks. Be concise and data-driven. Use options terminology.
No filler. End every response with a clear verdict:
PROCEED / CAUTION / SKIP — and one sentence of reasoning.
```

`focused_block` is included only when `focused_ticker` is not None:
```
FOCUSED STOCK: {ticker}
- Score: LC={lc_score} | LP={lp_score}
- IV/HV: {iv_hv_ratio:.2f} | Trend: {trend} | 1M: {ret_1m:+.1f}%
- Greeks: Δ={delta:.3f} | Θ={theta:.2f}/day | V={vega:.2f}
- SMC Signals: {smc_summary}
- Strategies: {strategy_names}
```

`top_lc_summary` / `top_lp_summary` format (one line per pick):
```
{rank}. {ticker} | Score={score:.0f} | IV/HV={iv_hv:.2f} | Trend={trend} | Δ={delta:.3f}
```

### Pre-screener state
If `top_lc` and `top_lp` are empty, system prompt notes: "No screener results yet." Bot responds to general questions but tells user to run screener for stock-specific analysis.

---

## Error Handling

- Missing API key → display `st.error("ANTHROPIC_API_KEY not set in Streamlit secrets.")`
- API call fails → display `st.error(f"Claude API error: {e}")`
- Empty question → do not submit, show `st.warning("Enter a question first.")`

---

## Dependencies

Add to `requirements.txt`:
```
anthropic>=0.25.0
streamlit>=1.31.0
```

(`st.popover` requires Streamlit ≥ 1.31)

---

## Files Changed

| File | Change |
|------|--------|
| `chatbot.py` | New — `render_chat_button(context)` |
| `app.py` | Read existing `selected_ticker` session state; derive `focused_row` from `screening_results`; call `render_chat_button` at bottom |
| `requirements.txt` | Add `anthropic>=0.25.0`, bump `streamlit>=1.31.0` |

---

## Out of Scope

- Chat history / multi-turn conversation
- Streaming response (single blocking call)
- User authentication
- Cost tracking / token limits
