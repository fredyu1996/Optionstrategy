# Telegram Entry/Exit Alerts — Design Spec

**Date:** 2026-06-06
**Status:** Approved (design)

## Problem

The Streamlit app only computes when someone has it open, so the user gets no
notification when a stock becomes a good entry or a held position turns into a
TRIM/SELL. They want push alerts without keeping the app open.

## Goal

A background job (GitHub Actions, hourly during US market hours) that scans the
S&P 500 for fresh 🟢 Enter signals and re-checks held positions for TRIM/SELL,
then sends new signals to Telegram — de-duplicated so the same signal isn't
repeated every hour.

## Non-Goals (YAGNI)

- No email channel (Telegram only).
- No in-app notification UI (this runs outside Streamlit).
- No per-user config UI; thresholds are fixed (entry = 🟢 Enter only).
- No intraday tick streaming; hourly cron is the resolution.

## Constraints

- **Runs outside Streamlit** → secrets come from `os.environ` (GitHub Actions
  secrets), NOT `st.secrets`. gspread is authorized from an env-provided service
  account JSON.
- Reused functions decorated with `@st.cache_data`/`@st.cache_resource`
  (screener pipeline, `analyze_position`) still *run* without a Streamlit
  runtime — they just skip caching and emit a benign warning. Acceptable for a
  batch script.
- Full-500 entry scan is minutes long; GitHub Actions handles it.

## Architecture

```
sheets.py        ── env-based gspread access (open_worksheet, read/replace rows)
telegram_bot.py  ── send_message(text) via Telegram Bot API (requests)
alerts.py        ── PURE decision logic (entry/exit detection, dedup, formatting)
notify.py        ── orchestration: load data, run scans, diff, send, persist state
.github/workflows/notify.yml ── hourly cron, runs `python notify.py`
```

Decision logic lives in `alerts.py` as pure functions (fully unit-tested);
`notify.py` only wires I/O (sheets, screener, telegram).

### Module: `sheets.py` (env-based Google Sheets)

```python
def open_worksheet(service_account_info: dict, sheet_key: str, title: str):
    """Authorize gspread from a service-account dict and return the worksheet
    named `title`, creating it (with no header) if absent."""

def get_records(ws) -> list[dict]:
    """ws.get_all_records()."""

def replace_rows(ws, header: list[str], rows: list[list]):
    """Clear the worksheet and write header + rows (used for alert_state)."""
```

`notify.py` builds `service_account_info` via
`json.loads(os.environ['GCP_SERVICE_ACCOUNT'])` and `sheet_key` from
`os.environ['POSITIONS_SHEET_KEY']`. (The existing `positions_store.py`
Streamlit path is left untouched.)

### Module: `telegram_bot.py`

```python
def send_message(token: str, chat_id: str, text: str) -> bool:
    """POST to https://api.telegram.org/bot<token>/sendMessage.
    Returns True on HTTP 200, False otherwise (never raises)."""
```

### Module: `alerts.py` (pure)

```python
def entry_alerts(rows: list[dict]) -> list[dict]:
    """For each screened row, run compute_entry_readiness for Long Call and
    Long Put; return [{key, kind:'entry', ticker, strategy, met, total, checks}]
    for those whose status == 'enter'. key = f'entry:{ticker}:{strategy}'."""

def exit_alerts(analyzed: list[dict]) -> list[dict]:
    """analyzed = [{'pos': pos, 'data': analyze_position(pos)}]. Return
    [{key, kind:'exit', ticker, strike, strategy, status, reasons, pnl_pct}]
    for positions whose verdict status in ('trim','sell').
    key = f'exit:{pos[\"id\"]}'."""

def diff_alerts(current: list[dict], stored: dict) -> list[dict]:
    """current = list of alert dicts (each has 'key' + a 'state' string:
    entry->'enter', exit->the verdict status). stored = {key: last_state}.
    Return the subset whose state differs from stored[key] (new or changed).
    This is the de-dup gate."""

def current_state_map(current: list[dict]) -> dict:
    """{key: state} for all current alerts, to persist as the new stored state."""

def format_entry_msg(alert: dict) -> str:
    """e.g. '🟢 ENTRY  AAPL Long Call (6/7)\n<passed check labels>'."""

def format_exit_msg(alert: dict) -> str:
    """e.g. '🔴 SELL  NVDA $150 Long Put\n<reasons> · P/L +30%'."""
```

**Alert state string:** for entry, `'enter'`; for exit, the verdict status
(`'trim'`/`'sell'`). `diff_alerts` notifies when `stored.get(key) != state`.
When a signal clears (no longer enter/trim/sell), it simply drops out of
`current`; its key is removed from the persisted state so a future re-trigger
notifies again.

### `notify.py` orchestration

1. Read env secrets (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
   `GCP_SERVICE_ACCOUNT`, `POSITIONS_SHEET_KEY`).
2. **Exit scan:** open `positions` worksheet → `get_records` → for each row,
   `positions.analyze_position(pos)` → `exit_alerts`.
3. **Entry scan:** `screener.get_sp500_tickers()` → `batch_screen_fundamentals`
   → `enrich_with_iv` → `score_strategies` → rows → `entry_alerts`.
4. Load `alert_state` worksheet → `{key: state}`.
5. `to_send = diff_alerts(entry+exit alerts, stored)`.
6. For each, `telegram_bot.send_message(format_*_msg(...))`.
7. `replace_rows(alert_state, ['key','state'], current_state_map(...))`.

If the entry scan raises (network/data), log and continue with exit alerts (and
vice versa) so one failing half doesn't kill the run.

### `.github/workflows/notify.yml`

```yaml
name: Signal Alerts
on:
  schedule:
    - cron: '0 13-21 * * 1-5'   # hourly, ~US market hours (UTC), Mon-Fri
  workflow_dispatch: {}          # manual run button
jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python notify.py
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          GCP_SERVICE_ACCOUNT: ${{ secrets.GCP_SERVICE_ACCOUNT }}
          POSITIONS_SHEET_KEY: ${{ secrets.POSITIONS_SHEET_KEY }}
```

(The cron window over-covers EST/EDT; acceptable. Hourly during 13–21 UTC.)

## User setup (one-time)

1. Telegram → **@BotFather** → `/newbot` → copy the **bot token**.
2. Get your **chat id**: message **@userinfobot**, or message your new bot then
   open `https://api.telegram.org/bot<token>/getUpdates` and read `chat.id`.
3. GitHub repo → **Settings → Secrets and variables → Actions** → add:
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `POSITIONS_SHEET_KEY`, and
   `GCP_SERVICE_ACCOUNT` (paste the full service-account JSON as the value).
4. The same service account already shares the positions sheet; the workflow
   auto-creates the `alert_state` tab on first run.

## Testing

**`tests/test_alerts.py`** (pure, no IO):
- `entry_alerts`: a row that yields `enter` for Long Call → one entry alert with
  the right key; a non-enter row → none.
- `exit_alerts`: analyzed list with a `sell` verdict → one exit alert; `hold` →
  none.
- `diff_alerts`: new key → included; unchanged state → excluded; changed state
  (trim→sell) → included.
- `current_state_map`: maps keys to their state strings.
- `format_entry_msg` / `format_exit_msg`: contain ticker, strategy, and the
  key numbers/reasons.

**`tests/test_telegram_bot.py`**:
- `send_message` mocks `requests.post`; asserts URL contains the token and
  payload has `chat_id` + `text`; returns True on 200, False on non-200, False
  on exception.

**`tests/test_sheets.py`**:
- `open_worksheet` creates the tab when missing (mock gspread client/spreadsheet
  with a `worksheet` that raises `WorksheetNotFound` then `add_worksheet`).
- `replace_rows` calls clear + update with header first.

`notify.py` itself (orchestration) is integration glue; covered by manual
`workflow_dispatch` run, not unit-tested.

## Error Handling

- `send_message` and the scan halves are individually try/wrapped so partial
  failures still deliver what they can.
- Missing env secret → `notify.py` exits non-zero with a clear message (so the
  Actions run shows red).
- `alert_state` tab missing → auto-created empty (treated as no prior state →
  first run may notify current signals; acceptable).

## File Map

| File | Action | Responsibility |
|---|---|---|
| `sheets.py` | Create | env-based gspread open/read/replace |
| `telegram_bot.py` | Create | Telegram send_message |
| `alerts.py` | Create | pure entry/exit detection, dedup, formatting |
| `notify.py` | Create | orchestration (env → scans → diff → send → persist) |
| `.github/workflows/notify.yml` | Create | hourly cron workflow |
| `tests/test_alerts.py` | Create | pure-logic tests |
| `tests/test_telegram_bot.py` | Create | send_message tests |
| `tests/test_sheets.py` | Create | worksheet open/replace tests |
| `README.md` | Modify | Telegram + Actions secrets setup |
