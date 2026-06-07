# alerts.py
"""
alerts.py - Pure decision logic for signal alerts: detect entry/exit signals,
de-duplicate against prior state, and format Telegram messages. No IO.
"""
from signals import compute_entry_readiness


def entry_alerts(rows: list) -> list:
    """Return entry alert dicts for rows that score 'enter' (Long Call/Put)."""
    out = []
    for row in rows:
        ticker = row.get('ticker')
        if not ticker:
            continue
        for strategy in ('Long Call', 'Long Put'):
            r = compute_entry_readiness(row, strategy)
            if r['status'] == 'enter':
                out.append({
                    'key': f'entry:{ticker}:{strategy}',
                    'kind': 'entry',
                    'state': 'enter',
                    'ticker': ticker,
                    'strategy': strategy,
                    'met': r['met'],
                    'total': r['total'],
                    'checks': r['checks'],
                })
    return out


def exit_alerts(analyzed: list) -> list:
    """Return exit alert dicts for analyzed positions with trim/sell verdicts.
    analyzed = [{'pos': pos_dict, 'data': analyze_position(pos)}]."""
    out = []
    for item in analyzed:
        pos = item['pos']
        data = item['data']
        status = data['verdict']['status']
        if status in ('trim', 'sell'):
            out.append({
                'key': f'exit:{pos["id"]}',
                'kind': 'exit',
                'state': status,
                'ticker': pos.get('ticker'),
                'strike': pos.get('strike'),
                'strategy': pos.get('strategy'),
                'status': status,
                'reasons': data['verdict'].get('reasons', []),
                'pnl_pct': data.get('pnl_pct'),
            })
    return out


def diff_alerts(current: list, stored: dict) -> list:
    """Return alerts whose state differs from the stored state (new/changed)."""
    return [a for a in current if stored.get(a['key']) != a['state']]


def current_state_map(current: list) -> dict:
    """Map each current alert key to its state string (to persist)."""
    return {a['key']: a['state'] for a in current}


def format_entry_msg(a: dict) -> str:
    passed = [c['label'] for c in a.get('checks', []) if c['passed']]
    head = f"🟢 ENTRY  {a['ticker']} {a['strategy']} ({a['met']}/{a['total']})"
    return head + "\n" + " · ".join(passed) if passed else head


def format_exit_msg(a: dict) -> str:
    is_sell = a['status'] == 'sell'
    emoji = '🔴' if is_sell else '🟡'
    label = 'SELL' if is_sell else 'TRIM'
    strike = a.get('strike')
    strike_str = f"${strike:.0f} " if isinstance(strike, (int, float)) else ''
    pnl = a.get('pnl_pct')
    pnl_str = f" · P/L {pnl * 100:+.0f}%" if pnl is not None else ''
    reasons = '; '.join(a.get('reasons') or []) or 'exit conditions'
    return f"{emoji} {label}  {a['ticker']} {strike_str}{a['strategy']}\n{reasons}{pnl_str}"
