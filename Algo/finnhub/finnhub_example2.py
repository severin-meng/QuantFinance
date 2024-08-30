"""
Example 1: Fetching data using finnhub-python plugin
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

# fetch the candle
response = finnhub_client.stock_candles(symbol, interval, start_date, end_date)

# quote
pprint(finnhub_client.quote("AAPL"))

# recommendation trends
pprint(finnhub_client.recommendation_trends("TSLA"))
