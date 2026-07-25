from __future__ import annotations

import pandas as pd


SHORTLINE_COLUMNS = [
    'market',
    'code',
    'beta_60',
    'rolling_pe',
    'free_float_shares_z',
    'prev_amount',
    'prev_seal_amount',
    'prior_seal_amount',
    'prev_open_volume',
    'prev_open_amount',
    'limit_board_days',
    'limit_up_count',
    'file_consecutive_limit_up_days',
    'file_year_limit_up_days',
    'free_float_market_value',
    'opening_turnover_z',
    'opening_prev_volume_ratio',
    'auction_prev_volume_ratio',
    'opening_prev_seal_ratio',
    'seal_to_float_ratio',
    'seal_prev_ratio',
    'limit_board_text',
    'current_consecutive_limit_up_days',
]

STAT_COLUMNS = [
    'beta_60',
    'rolling_pe',
    'free_float_shares_z',
    'prev_amount',
    'prev_seal_amount',
    'prior_seal_amount',
    'prev_open_volume',
    'prev_open_amount',
    'limit_board_days',
    'limit_up_count',
    'file_consecutive_limit_up_days',
    'file_year_limit_up_days',
]


def shortline_indicators(client, symbols, stats=None, batch_size=80, **kwargs):
    if not symbols:
        return pd.DataFrame(columns=SHORTLINE_COLUMNS)

    if isinstance(symbols, str):
        symbols = [symbols]

    quotes = client.quotes_batch(symbols, batch_size=batch_size, **kwargs)
    if quotes.empty:
        return pd.DataFrame(columns=SHORTLINE_COLUMNS)

    data = quotes.copy()
    data['code'] = data['code'].astype(str).str.zfill(6)
    stat_data = _prepare_stats(stats)
    if not stat_data.empty:
        data = data.merge(stat_data, on='code', how='left')

    for column in STAT_COLUMNS:
        if column not in data.columns:
            data[column] = None

    data['seal_amount'] = _number(data.get('bid1')) * _number(data.get('bid_vol1'))
    data['free_float_market_value'] = _number(data.get('price')) * _number(data.get('free_float_shares_z'))
    data['opening_turnover_z'] = _ratio(data.get('opening_volume'), data.get('free_float_shares_z'))
    data['opening_prev_volume_ratio'] = _ratio(data.get('opening_volume'), data.get('prev_open_volume'))
    data['auction_prev_volume_ratio'] = _ratio(data.get('auction_volume'), data.get('prev_open_volume'))
    data['opening_prev_seal_ratio'] = _ratio(data.get('opening_amount'), data.get('prev_seal_amount'))
    data['seal_to_float_ratio'] = _ratio(data.get('seal_amount'), data.get('free_float_market_value'))
    data['seal_prev_ratio'] = _ratio(data.get('seal_amount'), data.get('prev_seal_amount'))
    data['limit_board_text'] = data.apply(_limit_board_text, axis=1)
    data['current_consecutive_limit_up_days'] = data['file_consecutive_limit_up_days']

    return data[[column for column in SHORTLINE_COLUMNS]]


def _prepare_stats(stats):
    if stats is None:
        return pd.DataFrame(columns=['code'] + STAT_COLUMNS)

    data = pd.DataFrame(stats).copy()
    if data.empty:
        return pd.DataFrame(columns=['code'] + STAT_COLUMNS)
    if 'code' not in data.columns:
        raise ValueError("stats missing columns: ['code']")

    data['code'] = data['code'].astype(str).str.zfill(6)
    for column in STAT_COLUMNS:
        if column not in data.columns:
            data[column] = None
    return data[['code'] + STAT_COLUMNS]


def _number(value):
    return pd.to_numeric(value, errors='coerce')


def _ratio(numerator, denominator):
    result = _number(numerator) / _number(denominator)
    return result.replace([float('inf'), -float('inf')], pd.NA)


def _limit_board_text(row):
    days = row.get('limit_board_days')
    count = row.get('limit_up_count')
    if pd.isna(days) or pd.isna(count):
        return None
    return f'{int(days)}天{int(count)}板'
