# Manual + Persisted Position Tracking — Design Spec

**Date:** 2026-06-05
**Status:** Approved (design)

## Problem

The app screens trades and (via the sell verdict) judges a hypothetical
position from a manually typed premium, but it does not remember positions the
user actually holds. There is no list of "my open longs" and no live verdict per
held contract. The user wants to record each long call/put once and have the app
track it — live P/L and a SELL/TRIM/HOLD verdict — across sessions and redeploys.

## Goal

Let the user record open long-option positions (manual entry), persist them in
Google Sheets (survives Streamlit Cloud redeploys), and view each with a live
SELL/TRIM/HOLD verdict, current price, P/L %, and total $ P/L. Closing a position
removes it.

## Non-Goals (YAGNI)

- No closed-position / realized-P/L history (open positions only).
- No multi-user auth (personal single-user tool).
- No broker API sync.
- No editing a position in place (close + re-add instead).

## Constraints

- **Streamlit Cloud filesystem is ephemeral** — local files (SQLite, CSV) are
  wiped on every reboot/redeploy. Persistence MUST be external → Google Sheets.
- Both the entry premium and the fetched current contract price are **per-share**
  (e.g. `4.20`). US equity options = 100 shares/contract, so total dollars
  multiply by 100.

## Architecture

Two new modules with distinct responsibilities, plus a new top-level view in
`app.py`. Live analysis reuses existing signal functions — no duplication.

```
positions_store.py   ── Google Sheets persistence (CRUD)
positions.py         ── analyze_position(): live price + verdict, reuses signals.py/screener.py
app.py               ── sidebar radio nav + "My Positions" page
```

### Module: `positions_store.py`

Google Sheets via `gspread` + `google-auth` service account.

**Sheet schema** (first row = header), one worksheet:

| Column | Type | Notes |
|---|---|---|
| `id` | str | UUID4 generated on add |
| `ticker` | str | e.g. `AAPL` |
| `strategy` | str | `Long Call` or `Long Put` |
| `strike` | float | option strike |
| `expiry` | str | ISO `YYYY-MM-DD` |
| `entry_premium` | float | per-share price paid |
| `contracts` | int | number of contracts |
| `entry_date` | str | ISO date, auto-set on add |

**Functions:**

```python
def get_worksheet():
    """Authorize via st.secrets['gcp_service_account'], open sheet by
    st.secrets['positions_sheet_key'], return worksheet 1. Cached with
    st.cache_resource. Raises PositionsConfigError if secrets missing."""

def load_positions() -> list[dict]:
    """Return all rows as list of dicts with typed fields (strike/entry_premium
    float, contracts int). Empty list if only header."""

def add_position(pos: dict) -> None:
    """Append a row. Generates id (uuid4) and entry_date (today) if absent."""

def delete_position(position_id: str) -> None:
    """Find row by id and delete it. No-op if not found."""
```

`PositionsConfigError` is a module-level exception so the UI can catch it and
show setup instructions instead of crashing.

### Module: `positions.py`

```python
def get_contract_price(ticker: str, strategy: str, strike: float, expiry: str) -> float:
    """Live per-share mid for the exact contract via yfinance option chain.
    Returns the bid/ask mid, or lastPrice if no quotes, or np.nan if the
    contract/expiry isn't found."""

def analyze_position(pos: dict) -> dict:
    """
    Compute live state for one stored position. Reuses:
      - screener._compute_rsi + screener.compute_smc_signals → build `row`
        (trend/rsi/smc flags) from downloaded OHLCV
      - signals.compute_exit_rules(row, strategy, rec)
      - signals.compute_sell_verdict(exits, rec, entry_premium)
    where rec = {'cost': current_mid, 'dte': days_to_expiry}.

    Returns:
        current_price: float | nan   (per-share mid)
        pnl_pct: float | None        (from verdict)
        pnl_usd: float | None        ((current_mid - entry_premium) * 100 * contracts)
        dte: int
        verdict: dict                (status/reasons/pnl_pct from compute_sell_verdict)
        error: str | None            (set when price/data unavailable)
    """
```

**Unit consistency:** `rec['cost']` and `entry_premium` are both per-share, so
`compute_sell_verdict`'s `cost/entry_premium - 1` ratio is correct. The ×100
multiplier applies only to the displayed dollar P/L (`pnl_usd`).

**Trend:** derive the same way the screener does (reuse its trend logic if
exposed; otherwise compute from moving-average comparison in a small helper).
The condition-layer verdict only strictly needs `rsi` and the `smc_*` flags plus
`dte`; `trend` is not consumed by `compute_exit_rules`/`compute_sell_verdict`, so
a missing trend does not break the verdict.

### UI changes — `app.py`

1. **Sidebar top-level nav** (above existing Screener Settings):
   ```python
   view = st.sidebar.radio("View", ["🔍 Screener", "📋 My Positions"])
   ```
   The existing screener body is wrapped under `if view == "🔍 Screener":`.
   The new positions page renders under `elif view == "📋 My Positions":`.

2. **My Positions page:**
   - **Add form** (`st.form`): ticker (text, upper-cased), strategy
     (selectbox Long Call/Long Put), strike (number), expiry (date_input),
     entry premium per share (number, min 0.0), contracts (number, min 1,
     step 1). Submit → `add_position` → `st.rerun()`.
   - **Open positions:** for each `load_positions()` row, call
     `analyze_position` and render a card: verdict badge (reuse
     `_PLAYBOOK_VERDICT` colors), `TICKER strike Call/Put exp YYYY-MM-DD`,
     current price, P/L % (colored), total $ P/L (colored), DTE, reasons list,
     and a **Close** button (`delete_position(id)` → `st.rerun()`).
   - **No positions:** info message prompting to add one.
   - **Config missing:** catch `PositionsConfigError`, show setup steps + link
     to README section.

### Secrets / setup (one-time, user-performed)

Documented in README. Required `st.secrets`:
```toml
positions_sheet_key = "<google-sheet-id>"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "..."
client_email = "...@....iam.gserviceaccount.com"
# ...full service-account JSON fields...
```
Steps: create GCP project → enable Google Sheets API + Drive API → create
service account + JSON key → create a Google Sheet, add header row, share it
(Editor) with the service-account `client_email` → paste sheet id + JSON into
Streamlit secrets.

## Dependencies

Add to `requirements.txt`: `gspread`, `google-auth`.

## Testing

**`tests/test_positions_store.py`** (mock gspread worksheet):
- `load_positions` types fields (strike/entry_premium→float, contracts→int).
- `load_positions` returns `[]` for header-only sheet.
- `add_position` appends a row and auto-fills `id` + `entry_date`.
- `delete_position` removes the matching id; no-op when id absent.
- Missing secrets → `PositionsConfigError`.

**`tests/test_positions.py`** (mock `get_contract_price` + OHLCV/signal calls):
- `analyze_position` computes `pnl_usd = (mid - entry) * 100 * contracts`.
- `pnl_pct` matches verdict's pnl_pct.
- verdict status flows through from `compute_sell_verdict`.
- contract price `nan` → `error` set, condition-only verdict, `pnl_usd` None.
- DTE computed correctly from expiry.

## Error Handling

- Missing/invalid secrets → `PositionsConfigError` → friendly setup message.
- Contract or expiry not found in option chain → `current_price = nan`, card
  shows "price unavailable", verdict falls back to condition layer only.
- yfinance/network failure in `analyze_position` → caught, `error` populated,
  card renders what it can rather than crashing the page.

## File Map

| File | Action | Responsibility |
|---|---|---|
| `positions_store.py` | Create | Google Sheets CRUD + `PositionsConfigError` |
| `positions.py` | Create | `get_contract_price`, `analyze_position` |
| `tests/test_positions_store.py` | Create | CRUD tests (mocked gspread) |
| `tests/test_positions.py` | Create | analysis + P/L math tests (mocked IO) |
| `app.py` | Modify | sidebar radio nav; wrap screener body; My Positions page |
| `requirements.txt` | Modify | add `gspread`, `google-auth` |
| `README.md` | Modify | Google Sheets setup steps |
