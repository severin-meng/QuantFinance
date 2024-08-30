"""
Example 1: Fetching data along with technical indicators
"""

from decouple import config
import finnhub
import json
from pprint import pprint

# get the tokens for API access
api_key = config("FINNHUB_KEY")
finnhub_client = finnhub.Client(api_key=api_key)

# specify params
symbol = 'IBM'
interval = 'D'
start_date = 1609455600  # UNIX timestamp, decode via datetime.fromtimestamp(1609455600) = 01.01.2021 00:00
end_date = 1612047600  # UNIX timestamp, encode via datetime.strptime('2021-01-31', '%Y-%m-%d').timestamp()

ta = finnhub_client.technical_indicator(
    symbol=symbol,
    resolution=interval,
    _from=start_date,
    to=end_date,
    indicator='rsi',
    indicator_fields={"timeperiods": 5}
)
pprint(ta)
