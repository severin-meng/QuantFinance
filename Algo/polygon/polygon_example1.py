"""
Example 1: Fetching data using python requests module from polygon
see also https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier___timespan___from___to
"""

from decouple import config
import pandas as pd
from polygon import RESTClient
from pprint import pprint
import requests


# get the api keys
api_key = config("POLYGON_KEY")

# specify params
symbol = 'AAPL'
interval_multiplier = 1
timespan = 'minute'
start_date = '2023-01-09'  # YYYY-MM-DD
end_date = '2023-02-10'
adjust_to_stock_splits = True
sorting = 'asc'

# specify the endpoints
base_url = 'https://api.polygon.io/v2'
endpoint = '/aggs'
query = f'/ticker/{symbol}/range/{interval_multiplier}/{timespan}/{start_date}/{end_date}' \
        f'?adjusted={adjust_to_stock_splits}&sort={sorting}&apiKey={api_key}'

# get the requests
request_url = base_url + endpoint + query
print(request_url)
r = requests.get(request_url)
pprint(r.json())
