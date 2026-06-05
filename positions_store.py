# positions_store.py
"""
positions_store.py - Google Sheets persistence for tracked option positions.

Storage backend is Google Sheets (via gspread) because Streamlit Cloud's local
filesystem is wiped on every redeploy. Secrets required in st.secrets:
    positions_sheet_key = "<sheet id>"
    [gcp_service_account] = { ...service account JSON... }
"""
import uuid
from datetime import date

import streamlit as st

HEADER = ['id', 'ticker', 'strategy', 'strike', 'expiry',
          'entry_premium', 'contracts', 'entry_date']


class PositionsConfigError(Exception):
    """Raised when Google Sheets credentials/secrets are not configured."""


def get_worksheet():
    """Authorize the service account and return the positions worksheet.

    Raises PositionsConfigError if secrets are missing or libs unavailable.
    """
    if 'gcp_service_account' not in st.secrets or 'positions_sheet_key' not in st.secrets:
        raise PositionsConfigError("Missing Google Sheets secrets")
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:
        raise PositionsConfigError("gspread/google-auth not installed") from exc

    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    creds = Credentials.from_service_account_info(
        dict(st.secrets['gcp_service_account']), scopes=scopes
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(st.secrets['positions_sheet_key'])
    return sheet.sheet1


def load_positions() -> list:
    """Return all positions as a list of typed dicts."""
    ws = get_worksheet()
    out = []
    for r in ws.get_all_records():
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


def add_position(pos: dict) -> None:
    """Append a position row, auto-filling id and entry_date when absent."""
    ws = get_worksheet()
    pid = pos.get('id') or str(uuid.uuid4())
    entry_date = pos.get('entry_date') or date.today().isoformat()
    row = [
        pid, pos['ticker'], pos['strategy'], pos['strike'], pos['expiry'],
        pos['entry_premium'], pos['contracts'], entry_date,
    ]
    ws.append_row(row)


def delete_position(position_id: str) -> None:
    """Delete the row whose id matches; no-op if not found."""
    ws = get_worksheet()
    records = ws.get_all_records()
    for idx, rec in enumerate(records, start=2):  # row 1 is the header
        if str(rec.get('id')) == str(position_id):
            ws.delete_rows(idx)
            return
