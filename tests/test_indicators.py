import pandas as pd
import pytest

from mootdx.indicators import BOLL
from mootdx.indicators import ATR
from mootdx.indicators import CCI
from mootdx.indicators import DMI
from mootdx.indicators import EMA
from mootdx.indicators import KDJ
from mootdx.indicators import MA
from mootdx.indicators import MACD
from mootdx.indicators import OBV
from mootdx.indicators import RSI
from mootdx.indicators import VR
from mootdx.indicators import WR
from mootdx.indicators import compute_indicators
from mootdx.indicators import list_indicators


def sample_data():
    return pd.DataFrame(
        {
            'open': [10, 11, 12, 13, 14, 15],
            'high': [11, 12, 13, 14, 15, 16],
            'low': [9, 10, 11, 12, 13, 14],
            'close': [10, 11, 12, 13, 14, 15],
            'vol': [100, 110, 120, 130, 140, 150],
        }
    )


def test_indicator_functions_append_expected_columns():
    df = sample_data()

    assert {'MACD_DIF', 'MACD_DEA', 'MACD_HIST'}.issubset(MACD(df).columns)
    assert {'KDJ_K', 'KDJ_D', 'KDJ_J'}.issubset(KDJ(df).columns)
    assert {'RSI'}.issubset(RSI(df).columns)
    assert {'BOLL_UPPER', 'BOLL_MID', 'BOLL_LOWER'}.issubset(BOLL(df).columns)


def test_compute_indicators_merges_multiple_indicators():
    data = compute_indicators(sample_data(), ['MACD', 'RSI'], params={'RSI': {'N': 3}})

    assert len(data) == 6
    assert 'MACD_DIF' in data.columns
    assert 'RSI' in data.columns
    assert data['RSI'].notna().all()


def test_indicator_missing_columns_raise_value_error():
    with pytest.raises(ValueError, match='missing columns'):
        KDJ(pd.DataFrame({'close': [1, 2, 3]}))


def test_list_indicators_contains_first_batch():
    names = {item['name'] for item in list_indicators()}

    assert {'MACD', 'KDJ', 'RSI', 'BOLL', 'MA', 'EMA', 'ATR', 'DMI', 'WR', 'CCI', 'OBV', 'VR'} == names


def test_extended_indicators_append_expected_columns():
    df = sample_data()

    assert {'MA'}.issubset(MA(df).columns)
    assert {'EMA'}.issubset(EMA(df).columns)
    assert {'ATR'}.issubset(ATR(df).columns)
    assert {'DMI_PDI', 'DMI_MDI', 'DMI_ADX', 'DMI_ADXR'}.issubset(DMI(df).columns)
    assert {'WR'}.issubset(WR(df).columns)
    assert {'CCI'}.issubset(CCI(df).columns)
    assert {'OBV'}.issubset(OBV(df).columns)
    assert {'VR'}.issubset(VR(df).columns)


def test_extended_indicators_support_combined_computation_and_parameter_validation():
    data = compute_indicators(sample_data(), ['ATR', 'OBV', 'VR'], params={'VR': {'N': 3}})

    assert len(data) == 6
    assert data['OBV'].tolist() == [0.0, 110.0, 230.0, 360.0, 500.0, 650.0]
    assert data[['ATR', 'VR']].notna().all().all()
    with pytest.raises(ValueError, match='greater than zero'):
        MA(sample_data(), N=0)
