"""Stable DataFrame serialization helpers for command-line and API callers."""

import json
from datetime import date
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd


def _require_dataframe(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')


def _value(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _value(value.item())
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def to_records(data, include_index=False):
    """Convert a DataFrame into JSON-compatible records."""
    _require_dataframe(data)
    frame = data.reset_index() if include_index else data
    return [{str(key): _value(value) for key, value in row.items()} for row in frame.to_dict(orient='records')]


def to_json(data, orient='records', include_index=False):
    """Serialize a DataFrame as UTF-8 JSON without pandas-specific NaN values."""
    if orient != 'records':
        raise ValueError("only orient='records' is supported")
    return json.dumps(to_records(data, include_index=include_index), ensure_ascii=False, separators=(',', ':'))


def to_csv(data, include_index=False):
    """Serialize a DataFrame as UTF-8 CSV text."""
    _require_dataframe(data)
    output = StringIO()
    data.to_csv(output, index=include_index, na_rep='', lineterminator='\n')
    return output.getvalue()
