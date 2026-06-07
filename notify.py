# notify.py
"""
notify.py - Background alert job (run by GitHub Actions, NOT Streamlit).

Scans S&P 500 for fresh entry signals and held positions for TRIM/SELL,
de-duplicates against the Google Sheets `alert_state` tab, and pushes new
signals to Telegram. Secrets come from environment variables.
"""
import json
import os
import sys
import traceback

import alerts
import sheets
import telegram_bot
from positions import analyze_position


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: missing env var {name}", file=sys.stderr)
        sys.exit(1)
    return val


def _positions_to_dicts(records: list) -> list:
    """Coerce alert positions sheet records into the shape analyze_position needs."""
    out = []
    for r in records:
        out.append({
            'id': str(r.get('id', '')),
            'ticker': str(r.get('ticker', '')),
            'strategy': str(r.get('strategy', '')),
            'strike': float(r.get('strike') or 0),
            'expiry': str(r.get('expiry', '')),
            'entry_premium': float(r.get('entry_premium') or 0),
            'contracts': int(float(r.get('contracts') or 0)),
            'entry_date': str(r.get('entry_date', '')),
        })
    return out


def run_exit_scan(sa_info: dict, sheet_key: str) -> list:
    """Analyze held positions; return exit alert dicts."""
    try:
        ws = sheets.open_worksheet(sa_info, sheet_key, 'positions')
        positions = _positions_to_dicts(sheets.get_records(ws))
        analyzed = [{'pos': p, 'data': analyze_position(p)} for p in positions]
        return alerts.exit_alerts(analyzed)
    except Exception as exc:
        print(f"exit scan failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return []


def run_entry_scan() -> list:
    """Screen the S&P 500; return entry alert dicts."""
    try:
        from screener import (
            get_sp500_tickers, batch_screen_fundamentals, enrich_with_iv,
        )
        sp500 = get_sp500_tickers()
        tickers = sp500['ticker'].tolist()  # get_sp500_tickers returns a DataFrame
        df = batch_screen_fundamentals(tickers)
        df = enrich_with_iv(df)
        rows = df.to_dict('records')
        return alerts.entry_alerts(rows)
    except Exception as exc:
        print(f"entry scan failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return []


def main() -> None:
    token = _require_env('TELEGRAM_BOT_TOKEN')
    chat_id = _require_env('TELEGRAM_CHAT_ID')
    sheet_key = _require_env('POSITIONS_SHEET_KEY')
    sa_info = json.loads(_require_env('GCP_SERVICE_ACCOUNT'))

    current = run_exit_scan(sa_info, sheet_key) + run_entry_scan()

    state_ws = sheets.open_worksheet(sa_info, sheet_key, 'alert_state')
    stored = {r['key']: r.get('state', '') for r in sheets.get_records(state_ws) if r.get('key')}

    to_send = alerts.diff_alerts(current, stored)
    for a in to_send:
        text = (alerts.format_entry_msg(a) if a['kind'] == 'entry'
                else alerts.format_exit_msg(a))
        telegram_bot.send_message(token, chat_id, text)

    new_state = alerts.current_state_map(current)
    sheets.replace_rows(state_ws, ['key', 'state'],
                        [[k, v] for k, v in new_state.items()])
    print(f"sent {len(to_send)} alerts; tracked {len(new_state)} active signals")


if __name__ == '__main__':
    main()
