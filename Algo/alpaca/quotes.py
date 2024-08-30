from decouple import config
import pandas as pd
from datetime import datetime, date, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import requests

# get the tokens for API access
api_key = config("ALPACA_PAPER_KEY")
secret_key = config("ALPACA_PAPER_SECRET")

client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

# To get quote data for stocks, you will use request models by providing a StockLatestQuoteRequest object.
request_params = StockLatestQuoteRequest(
    symbol_or_symbols=["SPY", "GLD", "TLT"]
)

# retrieve dataframe of stock data
latest_quotes = client.get_stock_latest_quote(request_params)

print(latest_quotes)

"""
Old way of doing this via url call - quite ugly

base_url = "https://data.alpaca.markets/v2/stocks/bars"
base_url += "?symbols=AAPL&timeframe=1D&start=2023-10-01&end=2023-10-27&"
url = base_url + "limit=1000&adjustments=raw&feed=sip&sort=asc"
headers = {
    "accept": "apllication/json",
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key
}

response = requests.get(url=url, headers=headers)

print(response.text)
"""
