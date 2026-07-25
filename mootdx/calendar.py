"""Trading-day helpers built on the existing mootdx holiday source."""

from datetime import date
from datetime import datetime
from typing import Callable
from typing import Optional

import pandas as pd

from mootdx.cache import TTLCache


class TradingCalendar:
    def __init__(self, days=None, loader: Optional[Callable[[], pd.DataFrame]] = None,
                 cache: Optional[TTLCache] = None, ttl=86400):
        if days is not None and loader is not None:
            raise ValueError('days and loader cannot be used together')
        self._provided_days = days
        self._loader = loader or self._default_loader
        self._cache = cache
        self._ttl = ttl

    @staticmethod
    def _default_loader():
        from mootdx.utils.holiday import holidays

        return holidays()

    @staticmethod
    def _normalize_date(value) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            return pd.Timestamp(value).date()
        except (TypeError, ValueError) as exc:
            raise ValueError('date must be a valid str, date, or datetime') from exc

    @classmethod
    def _normalize_days(cls, data) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError('trading-day loader must return a pandas DataFrame')
        if 'date' not in data.columns:
            raise ValueError("trading-day data must include a 'date' column")

        result = data.loc[:, ['date']].copy()
        result['date'] = pd.to_datetime(result['date'], errors='coerce').dt.date
        result = result.dropna().drop_duplicates().sort_values('date').reset_index(drop=True)
        return result

    def _days(self) -> pd.DataFrame:
        if self._provided_days is not None:
            return self._normalize_days(self._provided_days)

        if self._cache is None:
            return self._normalize_days(self._loader())

        return self._cache.get_or_set(
            ('trading-days',),
            lambda: self._normalize_days(self._loader()),
            ttl=self._ttl,
        ).copy()

    def trading_days(self, start=None, end=None) -> pd.DataFrame:
        result = self._days()
        if start is not None:
            result = result[result['date'] >= self._normalize_date(start)]
        if end is not None:
            result = result[result['date'] <= self._normalize_date(end)]
        return result.reset_index(drop=True)

    def is_trading_day(self, value) -> bool:
        target = self._normalize_date(value)
        return bool((self._days()['date'] == target).any())

    def previous_trading_day(self, value, include_current=False) -> date:
        target = self._normalize_date(value)
        days = self._days()
        eligible = days[days['date'] <= target] if include_current else days[days['date'] < target]
        if eligible.empty:
            raise ValueError('no previous trading day is available')
        return eligible.iloc[-1]['date']

    def next_trading_day(self, value, include_current=False) -> date:
        target = self._normalize_date(value)
        days = self._days()
        eligible = days[days['date'] >= target] if include_current else days[days['date'] > target]
        if eligible.empty:
            raise ValueError('no next trading day is available')
        return eligible.iloc[0]['date']
