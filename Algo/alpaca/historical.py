from decouple import config
import pandas as pd
from datetime import datetime, date, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import requests

# get the tokens for API access
api_key = config("ALPACA_PAPER_KEY")
secret_key = config("ALPACA_PAPER_SECRET")

client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
end_date = datetime.today() - timedelta(days=1)
start_date = end_date - timedelta(days=100)

# To get data for stocks, you will use request models by providing a StockBarsRequest object.
# historical data
request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL", "SPY"],
    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
    start=start_date,
    end=end_date
)

# retrieve dataframe of stock data
bars = client.get_stock_bars(request_params).df
print(bars.loc['AAPL'])

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
