import json

import numpy as np
import pandas as pd
import pytest

from mootdx.serialization import to_csv
from mootdx.serialization import to_json
from mootdx.serialization import to_records


def sample_data():
    return pd.DataFrame({'date': [pd.Timestamp('2024-01-02')], 'price': [np.nan], 'name': ['平安银行']})


def test_records_are_json_compatible():
    records = to_records(sample_data())

    assert records == [{'date': '2024-01-02T00:00:00', 'price': None, 'name': '平安银行'}]
    assert json.loads(to_json(sample_data())) == records


def test_csv_uses_header_and_empty_values():
    value = to_csv(sample_data())

    assert value.startswith('date,price,name')
    assert ',,平安银行' in value


@pytest.mark.parametrize('func', [to_records, to_json, to_csv])
def test_serializers_reject_non_dataframes(func):
    with pytest.raises(TypeError, match='DataFrame'):
        func([])


def test_json_rejects_unsupported_orient():
    with pytest.raises(ValueError, match='orient'):
        to_json(sample_data(), orient='split')
