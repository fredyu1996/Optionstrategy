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


@st.cache_resource(show_spinner=False)
def _get_client():
    """Authorize the service account (cached — OAuth runs once per session)."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    creds = Credentials.from_service_account_info(
        dict(st.secrets['gcp_service_account']), scopes=scopes
    )
    return gspread.authorize(creds)


def get_worksheet():
    """Return the positions worksheet, authorizing via a cached client.

    Raises PositionsConfigError if secrets are missing or libs unavailable.
    """
    if 'gcp_service_account' not in st.secrets or 'positions_sheet_key' not in st.secrets:
        raise PositionsConfigError("Missing Google Sheets secrets")
    try:
        client = _get_client()
    except ImportError as exc:
        raise PositionsConfigError("gspread/google-auth not installed") from exc
    return client.open_by_key(st.secrets['positions_sheet_key']).sheet1


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
    record = {
        'id': pos.get('id') or str(uuid.uuid4()),
        'ticker': pos['ticker'],
        'strategy': pos['strategy'],
        'strike': pos['strike'],
        'expiry': pos['expiry'],
        'entry_premium': pos['entry_premium'],
        'contracts': pos['contracts'],
        'entry_date': pos.get('entry_date') or date.today().isoformat(),
    }
    ws.append_row([record[col] for col in HEADER])


def delete_position(position_id: str) -> None:
    """Delete the row whose id matches; no-op if not found."""
    ws = get_worksheet()
    records = ws.get_all_records()
    for idx, rec in enumerate(records, start=2):  # row 1 is the header
        if str(rec.get('id')) == str(position_id):
            ws.delete_rows(idx)
            return
