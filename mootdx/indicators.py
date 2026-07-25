from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Union

import numpy as np
import pandas as pd


def _series(values) -> pd.Series:
    return pd.Series(values, dtype='float64')


def _sma(values, window):
    return _series(values).rolling(window=int(window), min_periods=1).mean()


def _ema(values, span):
    return _series(values).ewm(span=int(span), adjust=False).mean()


def _std(values, window):
    return _series(values).rolling(window=int(window), min_periods=1).std(ddof=0)


def _round(values):
    return pd.Series(values).round(2)


def _positive_integer(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f'{name} must be greater than zero')
    return value


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    defaults: dict[str, Union[int, float]]
    description: str
    func: Callable[..., tuple[pd.Series, ...]]


def _macd(close, SHORT=12, LONG=26, M=9):  # noqa: N803
    dif = _ema(close, SHORT) - _ema(close, LONG)
    dea = dif.ewm(span=int(M), adjust=False).mean()
    hist = (dif - dea) * 2
    return _round(dif), _round(dea), _round(hist)


def _kdj(close, high, low, N=9, M1=3, M2=3):  # noqa: N803
    close = _series(close)
    high = _series(high)
    low = _series(low)
    low_n = low.rolling(window=int(N), min_periods=1).min()
    high_n = high.rolling(window=int(N), min_periods=1).max()
    spread = high_n - low_n
    rsv = ((close - low_n) / spread.replace(0, np.nan) * 100).fillna(50)
    k = rsv.ewm(com=int(M1) - 1, adjust=False).mean()
    d = k.ewm(com=int(M2) - 1, adjust=False).mean()
    j = k * 3 - d * 2
    return _round(k), _round(d), _round(j)


def _rsi(close, N=24):  # noqa: N803
    close = _series(close)
    diff = close.diff().fillna(0)
    up = diff.clip(lower=0)
    down = (-diff.clip(upper=0))
    avg_up = up.ewm(alpha=1 / int(N), adjust=False).mean()
    avg_down = down.ewm(alpha=1 / int(N), adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)
    return (_round(rsi),)


def _boll(close, N=20, P=2):  # noqa: N803
    mid = _sma(close, N)
    std = _std(close, N)
    upper = mid + std * float(P)
    lower = mid - std * float(P)
    return _round(upper), _round(mid), _round(lower)


def _ma(close, N=5):  # noqa: N803
    return (_round(_sma(close, _positive_integer(N, 'N'))),)


def _ema_indicator(close, N=5):  # noqa: N803
    return (_round(_ema(close, _positive_integer(N, 'N'))),)


def _atr(high, low, close, N=14):  # noqa: N803
    high = _series(high)
    low = _series(low)
    previous_close = _series(close).shift(1)
    true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    return (_round(true_range.rolling(_positive_integer(N, 'N'), min_periods=1).mean()),)


def _dmi(high, low, close, N=14, M=6):  # noqa: N803
    high = _series(high)
    low = _series(low)
    close = _series(close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    previous_close = close.shift(1)
    tr = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    period = _positive_integer(N, 'N')
    tr_sum = tr.rolling(period, min_periods=1).sum().replace(0, np.nan)
    pdi = plus_dm.rolling(period, min_periods=1).sum() / tr_sum * 100
    mdi = minus_dm.rolling(period, min_periods=1).sum() / tr_sum * 100
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100).fillna(0)
    smooth = _positive_integer(M, 'M')
    adx = dx.rolling(smooth, min_periods=1).mean()
    adxr = (adx + adx.shift(smooth)).div(2).fillna(adx)
    return _round(pdi.fillna(0)), _round(mdi.fillna(0)), _round(adx), _round(adxr)


def _wr(close, high, low, N=14):  # noqa: N803
    high = _series(high)
    low = _series(low)
    close = _series(close)
    period = _positive_integer(N, 'N')
    highest = high.rolling(period, min_periods=1).max()
    lowest = low.rolling(period, min_periods=1).min()
    return (_round(((highest - close) / (highest - lowest).replace(0, np.nan) * 100).fillna(0)),)


def _cci(close, high, low, N=14):  # noqa: N803
    typical = (_series(high) + _series(low) + _series(close)) / 3
    period = _positive_integer(N, 'N')
    average = typical.rolling(period, min_periods=1).mean()
    deviation = typical.rolling(period, min_periods=1).apply(lambda values: np.abs(values - values.mean()).mean(), raw=True)
    return (_round(((typical - average) / (deviation * 0.015).replace(0, np.nan)).fillna(0)),)


def _obv(close, vol):
    direction = _series(close).diff().apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    return (_round((direction * _series(vol)).fillna(0).cumsum()),)


def _vr(close, vol, N=26):  # noqa: N803
    period = _positive_integer(N, 'N')
    volume = _series(vol)
    diff = _series(close).diff()
    up = volume.where(diff > 0, 0.0)
    down = volume.where(diff < 0, 0.0)
    flat = volume.where(diff == 0, 0.0)
    numerator = up.rolling(period, min_periods=1).sum() + flat.rolling(period, min_periods=1).sum() / 2
    denominator = down.rolling(period, min_periods=1).sum() + flat.rolling(period, min_periods=1).sum() / 2
    return (_round((numerator / denominator.replace(0, np.nan) * 100).fillna(100)),)


_REGISTRY = {
    'MACD': IndicatorSpec(
        'MACD',
        ('close',),
        ('MACD_DIF', 'MACD_DEA', 'MACD_HIST'),
        {'SHORT': 12, 'LONG': 26, 'M': 9},
        'MACD exponential moving average convergence divergence.',
        _macd,
    ),
    'KDJ': IndicatorSpec(
        'KDJ',
        ('close', 'high', 'low'),
        ('KDJ_K', 'KDJ_D', 'KDJ_J'),
        {'N': 9, 'M1': 3, 'M2': 3},
        'KDJ stochastic oscillator.',
        _kdj,
    ),
    'RSI': IndicatorSpec(
        'RSI',
        ('close',),
        ('RSI',),
        {'N': 24},
        'RSI relative strength index.',
        _rsi,
    ),
    'BOLL': IndicatorSpec(
        'BOLL',
        ('close',),
        ('BOLL_UPPER', 'BOLL_MID', 'BOLL_LOWER'),
        {'N': 20, 'P': 2},
        'Bollinger bands.',
        _boll,
    ),
    'MA': IndicatorSpec('MA', ('close',), ('MA',), {'N': 5}, 'Moving average.', _ma),
    'EMA': IndicatorSpec('EMA', ('close',), ('EMA',), {'N': 5}, 'Exponential moving average.', _ema_indicator),
    'ATR': IndicatorSpec('ATR', ('high', 'low', 'close'), ('ATR',), {'N': 14}, 'Average true range.', _atr),
    'DMI': IndicatorSpec(
        'DMI',
        ('high', 'low', 'close'),
        ('DMI_PDI', 'DMI_MDI', 'DMI_ADX', 'DMI_ADXR'),
        {'N': 14, 'M': 6},
        'Directional movement index.',
        _dmi,
    ),
    'WR': IndicatorSpec('WR', ('close', 'high', 'low'), ('WR',), {'N': 14}, 'Williams %R.', _wr),
    'CCI': IndicatorSpec('CCI', ('close', 'high', 'low'), ('CCI',), {'N': 14}, 'Commodity channel index.', _cci),
    'OBV': IndicatorSpec('OBV', ('close', 'vol'), ('OBV',), {}, 'On-balance volume.', _obv),
    'VR': IndicatorSpec('VR', ('close', 'vol'), ('VR',), {'N': 26}, 'Volume ratio.', _vr),
}


def list_indicators():
    return [
        {
            'name': spec.name,
            'inputs': list(spec.inputs),
            'outputs': list(spec.outputs),
            'default_params': dict(spec.defaults),
            'description': spec.description,
        }
        for spec in _REGISTRY.values()
    ]


def _compute_one(df, name, params=None):
    key = str(name).upper()
    if key not in _REGISTRY:
        raise ValueError(f'unknown indicator: {name}')

    spec = _REGISTRY[key]
    missing = [column for column in spec.inputs if column not in df.columns]
    if missing:
        raise ValueError(f'DataFrame missing columns for {key}: {missing}')

    kwargs = dict(spec.defaults)
    kwargs.update(params or {})
    arrays = spec.func(*(df[column].to_numpy(dtype='float64') for column in spec.inputs), **kwargs)
    return pd.DataFrame({column: value.to_numpy() for column, value in zip(spec.outputs, arrays)}, index=df.index)


def compute_indicators(df, indicators, params=None):
    if isinstance(indicators, str):
        indicators = [indicators]

    result = df.copy()
    params = params or {}
    parts = []
    for name in indicators:
        key = str(name).upper()
        parts.append(_compute_one(result, key, params.get(key) or params.get(name) or {}))

    if parts:
        result = pd.concat([result, *parts], axis=1)
    return result


def MACD(df, SHORT=12, LONG=26, M=9):  # noqa: N802,N803
    return compute_indicators(df, ['MACD'], {'MACD': {'SHORT': SHORT, 'LONG': LONG, 'M': M}})


def KDJ(df, N=9, M1=3, M2=3):  # noqa: N802,N803
    return compute_indicators(df, ['KDJ'], {'KDJ': {'N': N, 'M1': M1, 'M2': M2}})


def RSI(df, N=24):  # noqa: N802,N803
    return compute_indicators(df, ['RSI'], {'RSI': {'N': N}})


def BOLL(df, N=20, P=2):  # noqa: N802,N803
    return compute_indicators(df, ['BOLL'], {'BOLL': {'N': N, 'P': P}})


def MA(df, N=5):  # noqa: N802,N803
    return compute_indicators(df, ['MA'], {'MA': {'N': N}})


def EMA(df, N=5):  # noqa: N802,N803
    return compute_indicators(df, ['EMA'], {'EMA': {'N': N}})


def ATR(df, N=14):  # noqa: N802,N803
    return compute_indicators(df, ['ATR'], {'ATR': {'N': N}})


def DMI(df, N=14, M=6):  # noqa: N802,N803
    return compute_indicators(df, ['DMI'], {'DMI': {'N': N, 'M': M}})


def WR(df, N=14):  # noqa: N802,N803
    return compute_indicators(df, ['WR'], {'WR': {'N': N}})


def CCI(df, N=14):  # noqa: N802,N803
    return compute_indicators(df, ['CCI'], {'CCI': {'N': N}})


def OBV(df):  # noqa: N802,N803
    return compute_indicators(df, ['OBV'])


def VR(df, N=26):  # noqa: N802,N803
    return compute_indicators(df, ['VR'], {'VR': {'N': N}})
