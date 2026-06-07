# telegram_bot.py
"""telegram_bot.py - minimal Telegram Bot API sender."""
import requests


def send_message(token: str, chat_id: str, text: str) -> bool:
    """POST a message to a Telegram chat. Returns True on HTTP 200, else False.
    Never raises."""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={'chat_id': chat_id, 'text': text}, timeout=15)
        return resp.status_code == 200
    except Exception:
        return False
