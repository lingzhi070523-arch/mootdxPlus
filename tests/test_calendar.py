from datetime import date

import pandas as pd
import pytest

from mootdx.calendar import TradingCalendar
from mootdx.cache import TTLCache


def calendar():
    return TradingCalendar(pd.DataFrame({'date': ['2024-01-02', '2024-01-03', '2024-01-05']}))


def test_calendar_queries_and_boundaries():
    client = calendar()

    assert client.is_trading_day('2024-01-03') is True
    assert client.is_trading_day('2024-01-04') is False
    assert client.previous_trading_day('2024-01-04') == date(2024, 1, 3)
    assert client.next_trading_day('2024-01-04') == date(2024, 1, 5)
    assert client.trading_days('2024-01-03', '2024-01-05')['date'].tolist() == [date(2024, 1, 3), date(2024, 1, 5)]


def test_calendar_include_current_and_missing_boundaries():
    client = calendar()

    assert client.previous_trading_day('2024-01-03', include_current=True) == date(2024, 1, 3)
    assert client.next_trading_day('2024-01-03', include_current=True) == date(2024, 1, 3)
    with pytest.raises(ValueError, match='previous'):
        client.previous_trading_day('2024-01-02')
    with pytest.raises(ValueError, match='next'):
        client.next_trading_day('2024-01-05')


def test_calendar_loader_uses_ttl_cache():
    calls = []

    def loader():
        calls.append(True)
        return pd.DataFrame({'date': ['2024-01-02']})

    client = TradingCalendar(loader=loader, cache=TTLCache(60))
    client.is_trading_day('2024-01-02')
    client.is_trading_day('2024-01-02')

    assert len(calls) == 1


def test_calendar_rejects_invalid_data():
    with pytest.raises(ValueError, match='date'):
        TradingCalendar(pd.DataFrame({'day': ['2024-01-02']})).trading_days()
    with pytest.raises(ValueError, match='valid'):
        calendar().is_trading_day('not-a-date')
