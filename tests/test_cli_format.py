import json

import pandas as pd
from click.testing import CliRunner

from mootdx.__main__ import quotes as quotes_command
from mootdx.__main__ import reader as reader_command
from mootdx.quotes import Quotes
from mootdx.reader import Reader


class QuoteClient:
    def bars(self, symbol, frequency):
        return pd.DataFrame({'code': [symbol], 'price': [10.0]})


class ReaderClient:
    def daily(self, symbol):
        return pd.DataFrame({'code': [symbol], 'close': [10.0]})


def test_quotes_command_serializes_json(monkeypatch):
    monkeypatch.setattr(Quotes, 'factory', lambda **kwargs: QuoteClient())

    result = CliRunner().invoke(quotes_command, ['--format', 'json'])

    assert result.exit_code == 0
    assert json.loads(result.output) == [{'code': '600000', 'price': 10.0}]


def test_reader_command_serializes_csv(monkeypatch):
    monkeypatch.setattr(Reader, 'factory', lambda **kwargs: ReaderClient())

    result = CliRunner().invoke(reader_command, ['--format', 'csv'])

    assert result.exit_code == 0
    assert result.output == 'code,close\n600000,10.0\n'
