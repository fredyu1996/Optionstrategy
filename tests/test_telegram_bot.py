import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import telegram_bot
from telegram_bot import send_message


class _Resp:
    def __init__(self, code):
        self.status_code = code


def test_send_message_success(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured['url'] = url
        captured['json'] = json
        return _Resp(200)

    monkeypatch.setattr(telegram_bot.requests, 'post', fake_post)
    ok = send_message('TOKEN123', '99', 'hello')
    assert ok is True
    assert 'TOKEN123' in captured['url']
    assert captured['json']['chat_id'] == '99'
    assert captured['json']['text'] == 'hello'


def test_send_message_non_200_returns_false(monkeypatch):
    monkeypatch.setattr(telegram_bot.requests, 'post', lambda *a, **k: _Resp(403))
    assert send_message('T', '1', 'x') is False


def test_send_message_exception_returns_false(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('network')
    monkeypatch.setattr(telegram_bot.requests, 'post', boom)
    assert send_message('T', '1', 'x') is False
