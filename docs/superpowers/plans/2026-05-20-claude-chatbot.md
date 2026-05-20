# Claude Analyst Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a floating Claude-powered analyst chatbot (st.popover) that verifies options strategies and advises on long call/put entries using live screener data.

**Architecture:** A new `chatbot.py` module exposes one function `render_chat_button(context)`. It renders a `st.popover("🤖 Analyst")` button fixed bottom-right via injected CSS. On submit it fires a single-turn Claude API call with macro + screener context as system prompt and returns a PROCEED/CAUTION/SKIP verdict.

**Tech Stack:** Python 3, Streamlit ≥ 1.31, `anthropic` SDK ≥ 0.25, yfinance (existing)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `requirements.txt` | Modify | Add `anthropic>=0.25.0`, bump `streamlit>=1.31.0` |
| `chatbot.py` | Create | `_build_system_prompt(context)` + `render_chat_button(context)` |
| `tests/test_chatbot.py` | Create | Unit tests for `_build_system_prompt` (pure function, no API calls) |
| `app.py` | Modify | Init `top_lc_picks`/`top_lp_picks` in session state; store them after computing; call `render_chat_button` after footer |

---

## Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Replace the file contents with:
```
streamlit>=1.31.0
yfinance>=0.2.31
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
requests>=2.31.0
lxml>=4.9.0
html5lib>=1.1
pytest>=7.0.0
anthropic>=0.25.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add anthropic SDK, bump streamlit to 1.31"
```

---

## Task 2: Write Failing Tests for chatbot.py

**Files:**
- Create: `tests/test_chatbot.py`

- [ ] **Step 1: Create the test file**

```python
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
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/test_chatbot.py -v
```

Expected: `ModuleNotFoundError: No module named 'chatbot'` or similar import error.

---

## Task 3: Implement chatbot.py

**Files:**
- Create: `chatbot.py`

- [ ] **Step 1: Create chatbot.py**

```python
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

    def _pick_line(rank, row, score_key):
        ticker = row.get("ticker", "?")
        score = row.get(score_key, 0)
        iv_hv = row.get("iv_hv_ratio", float("nan"))
        trend = row.get("trend", "—")
        delta = row.get("atm_delta", float("nan"))
        iv_str = f"{iv_hv:.2f}" if iv_hv == iv_hv else "N/A"
        delta_str = f"{delta:.3f}" if delta == delta else "N/A"
        return f"{rank}. {ticker} | Score={score:.0f} | IV/HV={iv_str} | Trend={trend} | Δ={delta_str}"

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
        vega = focused_row.get("atm_vega", focused_row.get("atm_vega", float("nan")))
        focused_block = (
            f"\nFOCUSED STOCK: {focused_ticker}\n"
            f"- Score: LC={focused_row.get('lc_score', 0):.0f} | LP={focused_row.get('lp_score', 0):.0f}\n"
            f"- IV/HV: {iv_hv:.2f if iv_hv == iv_hv else 'N/A'} | Trend: {focused_row.get('trend', '—')} | 1M: {ret_1m:+.1f if ret_1m == ret_1m else 'N/A'}%\n"
            f"- Greeks: Δ={delta:.3f if delta == delta else 'N/A'} | Θ={theta:.2f if theta == theta else 'N/A'}/day | V={vega:.2f if vega == vega else 'N/A'}\n"
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
                response = message.content[0].text
                st.markdown(response)
            except Exception as e:
                st.error(f"Claude API error: {e}")
```

- [ ] **Step 2: Run tests — confirm they pass**

```bash
pytest tests/test_chatbot.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add chatbot.py tests/test_chatbot.py
git commit -m "feat: add Claude analyst chatbot module with tests"
```

---

## Task 4: Wire chatbot into app.py

**Files:**
- Modify: `app.py`

### Step 1 — Add session state init for top picks

- [ ] **Step 1: Find the session state init block (lines ~462–469) and add two new entries**

After this existing block:
```python
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = None
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None
if 'macro_data' not in st.session_state:
    st.session_state.macro_data = None
if 'sp500_df' not in st.session_state:
    st.session_state.sp500_df = None
```

Add immediately after:
```python
if 'top_lc_picks' not in st.session_state:
    st.session_state.top_lc_picks = []
if 'top_lp_picks' not in st.session_state:
    st.session_state.top_lp_picks = []
```

### Step 2 — Store top_lc / top_lp after they're computed

- [ ] **Step 2: Find where top_lc and top_lp are computed (around line 843)**

The existing code is:
```python
top_lc = results_df.nlargest(5, 'lc_score')
top_lp = results_df.nlargest(5, 'lp_score')
```

Add two lines immediately after:
```python
top_lc = results_df.nlargest(5, 'lc_score')
top_lp = results_df.nlargest(5, 'lp_score')
st.session_state.top_lc_picks = top_lc.to_dict('records')
st.session_state.top_lp_picks = top_lp.to_dict('records')
```

### Step 3 — Add import and call render_chat_button

- [ ] **Step 3: Add import at the top of app.py**

After the existing local imports (around line 24):
```python
from strategies import suggest_strategies, get_specific_contracts, get_option_chain_display
```

Add:
```python
from chatbot import render_chat_button
```

- [ ] **Step 4: Call render_chat_button at the very end of app.py (after the footer block)**

The file currently ends at line 1192:
```python
""", unsafe_allow_html=True)
```

Add after it:
```python
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
```

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: wire Claude analyst chatbot into app with floating popover"
```

---

## Task 5: Deploy and Verify

- [ ] **Step 1: Push to remote**

```bash
git push
```

- [ ] **Step 2: Add API key in Streamlit Cloud**

Go to: share.streamlit.io → your app → **Settings** → **Secrets**

Add:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Click **Save**. App will reboot automatically.

- [ ] **Step 3: Verify floating button appears**

Open the app. Bottom-right corner should show a **🤖 Analyst** button.

- [ ] **Step 4: Verify pre-screener state**

Click the button without running screener. Type "What's the best strategy right now?" → submit. Response should mention "No screener results yet" and answer based on macro only.

- [ ] **Step 5: Verify post-screener state**

Run the screener. Select a stock. Click 🤖 Analyst. Ask "Is this worth a long call?" → response should reference the focused stock's IV/HV, trend, Greeks, and end with PROCEED / CAUTION / SKIP.

- [ ] **Step 6: Verify error handling**

Temporarily remove API key from secrets → reboot → open chatbot → submit → should show red error "ANTHROPIC_API_KEY not set in Streamlit secrets." Restore key afterward.
