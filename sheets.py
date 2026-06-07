# sheets.py
"""
sheets.py - Google Sheets access from an explicit service-account dict (for use
outside Streamlit, e.g. the notify.py GitHub Actions job).
"""
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


def open_worksheet(service_account_info: dict, sheet_key: str, title: str):
    """Authorize from a service-account dict and return the named worksheet,
    creating an empty one if it doesn't exist."""
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_key)
    try:
        return sheet.worksheet(title)
    except gspread.WorksheetNotFound:
        return sheet.add_worksheet(title=title, rows=100, cols=10)


def get_records(ws) -> list:
    """Return all data rows as dicts (first row is the header)."""
    return ws.get_all_records()


def replace_rows(ws, header: list, rows: list) -> None:
    """Clear the worksheet and write the header followed by all rows."""
    ws.clear()
    ws.update([header] + rows)
