import functools
import threading
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Callable
from typing import cast
from typing import Dict
from typing import Hashable
from typing import Optional
from typing import Protocol
from typing import TypeVar

import pandas as pd

from .compat import ParamSpec
from .compat import TypeAlias

P = ParamSpec('P')

PandasFunc: TypeAlias = Callable[P, pd.DataFrame]
T = TypeVar('T')
_MISSING = object()


class LRUCacheWrapper(Protocol[P]):
    lifetime: timedelta
    expiration: datetime

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> pd.DataFrame:
        pass

    def clear(self):
        pass


def lru_cache(seconds: Optional[int] = None, maxsize: Optional[int] = None, typed: bool = True):
    """
    装饰器工厂，生成一个装饰器，该装饰器在内存中缓存熊猫数据帧（具有一定的过期时间），以便将来更快地检索.

    :param seconds: 保留缓存的秒数
    :param maxsize: 要存储在缓存中的最大项目数
    :param typed: 不同类型的参数是否将单独缓存
    :return: 在内存中缓存 pd.DataFrame 的装饰器（含过期时间）.
    """

    def decorator(func: PandasFunc) -> PandasFunc:
        lru_func = cast(LRUCacheWrapper, functools.lru_cache(maxsize=maxsize, typed=typed)(func))

        if not seconds:
            return lru_func

        lru_func.lifetime = timedelta(seconds=seconds)
        lru_func.expiration = datetime.now(timezone.utc) + lru_func.lifetime

        @functools.wraps(func)
        def retrieve_cache(*args: P.args, **kwargs: P.kwargs) -> pd.DataFrame:
            if datetime.now(timezone.utc) >= lru_func.expiration:
                lru_func.clear()
                lru_func.expiration = datetime.now(timezone.utc) + lru_func.lifetime

            return lru_func(*args, **kwargs)

        return cast(PandasFunc, retrieve_cache)

    return decorator


class TTLCache:
    """Small thread-safe in-memory cache for low-frequency data."""

    def __init__(self, default_ttl: Optional[float] = 300):
        self.default_ttl = self._validate_ttl(default_ttl)
        self._items: Dict[Hashable, tuple] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _validate_ttl(ttl: Optional[float]) -> Optional[float]:
        if ttl is None:
            return None
        ttl = float(ttl)
        if ttl <= 0:
            raise ValueError('ttl must be greater than zero or None')
        return ttl

    @staticmethod
    def _validate_key(key: Hashable) -> None:
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError('cache key must be hashable') from exc

    def get(self, key: Hashable, default=None):
        self._validate_key(key)
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key, _MISSING)
            if item is _MISSING:
                self._misses += 1
                return default

            expires_at, value = item
            if expires_at is not None and now >= expires_at:
                del self._items[key]
                self._misses += 1
                return default

            self._hits += 1
            return value

    def set(self, key: Hashable, value: T, ttl: Optional[float] = None) -> T:
        self._validate_key(key)
        ttl = self.default_ttl if ttl is None else self._validate_ttl(ttl)
        expires_at = None if ttl is None else time.monotonic() + ttl
        with self._lock:
            self._items[key] = (expires_at, value)
        return value

    def get_or_set(self, key: Hashable, loader: Callable[[], T], ttl: Optional[float] = None) -> T:
        value = self.get(key, _MISSING)
        if value is not _MISSING:
            return value

        return self.set(key, loader(), ttl=ttl)

    def clear(self, key: Optional[Hashable] = None) -> None:
        with self._lock:
            if key is None:
                self._items.clear()
                return
            self._validate_key(key)
            self._items.pop(key, None)

    def cache_info(self) -> dict:
        with self._lock:
            return {
                'size': len(self._items),
                'hits': self._hits,
                'misses': self._misses,
                'default_ttl': self.default_ttl,
            }
