import pytest

from mootdx.cache import TTLCache
from mootdx.f10 import F10Client
from mootdx.quotes import StdQuotes


class FakeQuotes:
    def __init__(self):
        self.category_calls = 0
        self.content_calls = []

    def F10C(self, symbol):  # noqa: N802
        self.category_calls += 1
        return [
            {'name': '公司概况', 'filename': '001.txt', 'start': 0, 'length': 10},
            {'name': '核心题材', 'filename': '001.txt', 'start': 10, 'length': 10},
            {'name': '财务分析', 'filename': '001.txt', 'start': 20, 'length': 10},
        ]

    def F10(self, symbol, name):  # noqa: N802
        self.content_calls.append(name)
        return f'{name}内容'


def test_f10_catalog_sections_and_alias_helpers():
    quotes = FakeQuotes()
    client = F10Client(quotes)

    assert client.catalog('1')['section'].tolist() == ['公司概况', '核心题材', '财务分析']
    assert client.profile('000001')['content'].tolist() == ['公司概况内容']
    assert client.topics('000001')['section'].tolist() == ['核心题材']
    assert client.financials('000001')['section'].tolist() == ['财务分析']


def test_f10_sections_cache_and_unknown_section():
    quotes = FakeQuotes()
    client = F10Client(quotes, cache=TTLCache(60))

    first = client.sections('000001', names=['公司概况'])
    second = client.sections('000001', names=['公司概况'])
    assert first.equals(second)
    assert quotes.content_calls == ['公司概况']
    with pytest.raises(ValueError, match='unknown F10'):
        client.section('000001', '不存在')


def test_f10_alias_without_matching_section_returns_fixed_empty_frame():
    client = F10Client(FakeQuotes())
    client.PROFILE_ALIASES = ('未配置栏目',)

    result = client.profile('000001')
    assert result.empty
    assert result.columns.tolist() == ['code', 'section', 'content', 'source', 'fetched_at']


def test_std_quotes_f10_wrapper_uses_structured_client():
    class OfflineQuotes(StdQuotes):
        def __init__(self):
            self.client = None

        def F10C(self, symbol):  # noqa: N802
            return [{'name': '公司概况'}]

        def F10(self, symbol, name):  # noqa: N802
            return '公司概况内容'

    result = OfflineQuotes().f10_profile('000001')
    assert result['section'].tolist() == ['公司概况']
