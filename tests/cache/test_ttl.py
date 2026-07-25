import pytest

from mootdx.cache import TTLCache


def test_ttl_cache_hits_and_clears_specific_keys():
    cache = TTLCache(60)
    calls = []

    assert cache.get_or_set('quote', lambda: calls.append(1) or {'price': 10}) == {'price': 10}
    assert cache.get_or_set('quote', lambda: calls.append(2) or {'price': 11}) == {'price': 10}
    assert calls == [1]
    assert cache.cache_info()['hits'] == 1

    cache.clear('quote')
    assert cache.get('quote') is None


def test_ttl_cache_validates_inputs():
    with pytest.raises(ValueError, match='ttl'):
        TTLCache(0)
    with pytest.raises(TypeError, match='hashable'):
        TTLCache().get([])


def test_ttl_cache_expires_entries(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('mootdx.cache.timed.time.monotonic', lambda: clock[0])
    cache = TTLCache(2)
    cache.set('calendar', 1)

    clock[0] = 103.0
    assert cache.get('calendar') is None
    assert cache.cache_info()['misses'] == 1
