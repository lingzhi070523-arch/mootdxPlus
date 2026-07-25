import pandas as pd

from mootdx.quotes import StdQuotes


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


class FakeShortlineClient:
    def quotes_batch(self, symbols, batch_size=80, **kwargs):
        assert symbols == ['000001', '600036']
        return pd.DataFrame(
            [
                {
                    'market': 0,
                    'code': '000001',
                    'price': 10.0,
                    'open': 9.8,
                    'amount': 120000,
                    'volume': 12000,
                    'bid1': 10.0,
                    'bid_vol1': 300,
                    'opening_volume': 200,
                    'opening_amount': 1960,
                    'auction_volume': 100,
                    'auction_amount': 980,
                },
                {
                    'market': 1,
                    'code': '600036',
                    'price': 20.0,
                    'open': 20.1,
                    'amount': 0,
                    'volume': 0,
                    'bid1': 20.0,
                    'bid_vol1': 0,
                },
            ]
        )


def make_client(fake_client):
    client = StdQuotes.__new__(StdQuotes)
    client.client = fake_client
    client.quotes_batch = fake_client.quotes_batch
    return client


def test_shortline_indicators_returns_fixed_columns_and_computed_fields():
    stats = pd.DataFrame(
        [
            {
                'code': '000001',
                'beta_60': 1.2,
                'rolling_pe': 8.5,
                'free_float_shares_z': 2.0,
                'prev_amount': 100000,
                'prev_seal_amount': 1000,
                'prior_seal_amount': 800,
                'prev_open_volume': 50,
                'prev_open_amount': 900,
                'limit_board_days': 5,
                'limit_up_count': 3,
                'file_consecutive_limit_up_days': 2,
                'file_year_limit_up_days': 7,
            }
        ]
    )
    client = make_client(FakeShortlineClient())

    data = client.shortline_indicators(['000001', '600036'], stats=stats)

    assert list(data.columns) == SHORTLINE_COLUMNS
    row = data[data['code'] == '000001'].iloc[0]
    assert row['free_float_market_value'] == 20.0
    assert row['opening_turnover_z'] == 100.0
    assert row['opening_prev_volume_ratio'] == 4.0
    assert row['auction_prev_volume_ratio'] == 2.0
    assert row['opening_prev_seal_ratio'] == 1.96
    assert row['seal_to_float_ratio'] == 150.0
    assert row['seal_prev_ratio'] == 3.0
    assert row['limit_board_text'] == '5天3板'
    assert row['current_consecutive_limit_up_days'] == 2


def test_shortline_indicators_empty_input_returns_fixed_empty_frame():
    client = make_client(FakeShortlineClient())

    data = client.shortline_indicators([])

    assert data.empty
    assert list(data.columns) == SHORTLINE_COLUMNS
