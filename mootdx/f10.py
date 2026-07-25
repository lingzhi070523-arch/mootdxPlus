"""Structured access to the legacy TDX F10 category and content endpoints."""

from datetime import datetime
from typing import Iterable
from typing import Optional

import pandas as pd

from mootdx.cache import TTLCache


CATALOG_COLUMNS = ['code', 'section', 'filename', 'start', 'length']
SECTION_COLUMNS = ['code', 'section', 'content', 'source', 'fetched_at']


class F10Client:
    PROFILE_ALIASES = ('公司概况', '公司简介', '公司资料', '公司基本资料')
    TOPIC_ALIASES = ('核心题材', '所属概念', '概念题材', '热点题材', '题材概念')
    FINANCIAL_ALIASES = ('财务分析', '财务报表', '财务指标', '经营分析', '主要财务指标')

    def __init__(self, quotes, cache: Optional[TTLCache] = None, ttl=300):
        self.quotes = quotes
        self.cache = cache
        self.ttl = ttl

    @staticmethod
    def _empty(columns):
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _names(names: Optional[Iterable[str]]):
        if names is None:
            return None
        if isinstance(names, str):
            return (names,)
        return tuple(str(name) for name in names)

    def catalog(self, symbol):
        raw = self.quotes.F10C(symbol=symbol)
        if raw is None:
            return self._empty(CATALOG_COLUMNS)

        data = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        if data.empty:
            return self._empty(CATALOG_COLUMNS)
        if 'name' not in data.columns:
            return self._empty(CATALOG_COLUMNS)
        data = data.rename(columns={'name': 'section'})
        data['code'] = str(symbol).zfill(6)
        for column in CATALOG_COLUMNS:
            if column not in data.columns:
                data[column] = None
        return data[CATALOG_COLUMNS].reset_index(drop=True)

    def section(self, symbol, name):
        catalog = self.catalog(symbol)
        if name not in set(catalog['section']):
            raise ValueError(f'unknown F10 section: {name}')
        content = self.quotes.F10(symbol=symbol, name=name)
        return pd.DataFrame(
            [{
                'code': str(symbol).zfill(6),
                'section': name,
                'content': content,
                'source': 'tdx_f10',
                'fetched_at': datetime.utcnow().isoformat() + 'Z',
            }],
            columns=SECTION_COLUMNS,
        )

    def sections(self, symbol, names=None):
        names = self._names(names)
        cache_key = ('f10-sections', str(symbol).zfill(6), names)

        def load():
            catalog = self.catalog(symbol)
            selected = list(catalog['section']) if names is None else list(names)
            if names is not None:
                known = set(catalog['section'])
                unknown = [name for name in selected if name not in known]
                if unknown:
                    raise ValueError(f'unknown F10 section: {unknown[0]}')
            if not selected:
                return self._empty(SECTION_COLUMNS)
            return pd.concat([self.section(symbol, name) for name in selected], ignore_index=True)

        if self.cache is None:
            return load()
        return self.cache.get_or_set(cache_key, load, ttl=self.ttl).copy()

    @staticmethod
    def _matches(data, aliases):
        if data.empty:
            return data
        pattern = '|'.join(aliases)
        return data[data['section'].astype(str).str.contains(pattern, regex=True)].reset_index(drop=True)

    def _by_aliases(self, symbol, aliases):
        catalog = self.catalog(symbol)
        sections = self._matches(catalog, aliases)
        if sections.empty:
            return self._empty(SECTION_COLUMNS)
        return self.sections(symbol, names=sections['section'].tolist())

    def profile(self, symbol):
        return self._by_aliases(symbol, self.PROFILE_ALIASES)

    def topics(self, symbol):
        return self._by_aliases(symbol, self.TOPIC_ALIASES)

    def financials(self, symbol):
        return self._by_aliases(symbol, self.FINANCIAL_ALIASES)
