"""
Example 1: Fetching data using python requests module from finnhub
"""

from decouple import config
import finnhub
import json
from pprint import pprint
import requests

# get the tokens for API access
api_key = config("FINNHUB_KEY_KANNAN")

# specify params
symbol = 'IBM'
interval = 'D'
start_date = 1609455600  # UNIX timestamp, decode via datetime.fromtimestamp(1609455600) = 01.01.2021 00:00
end_date = 1612047600  # UNIX timestamp, encode via datetime.strptime('2021-01-31', '%Y-%m-%d').timestamp()

# specify the endpoints
base_url = 'https://finnhub.io/api/v1'
endpoint = '/stock/candle?'
query = f'symbol={symbol}&resolution={interval}&from={start_date}&to={end_date}&token={api_key}'

# get request
response = requests.get(base_url+endpoint+query)
pprint(response.json())


