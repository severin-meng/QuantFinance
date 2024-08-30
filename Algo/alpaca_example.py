"""
Example 1: Fetching data using python requests module from alpaca
"""

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import CryptoBarsRequest


# get the tokens for API access
# no API key required for historical crypto data
client = CryptoHistoricalDataClient()

# specify params
symbol = 'BTC/USD'
interval = TimeFrame(15, TimeFrameUnit.Minute)  # 15 minute intervals
start_date = "2022-09-01"
end_date = "2022-09-04"

request_params = CryptoBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=interval,
    start = start_date,
    end=end_date)


# get request
btc_bars = client.get_crypto_bars(request_params)
df = btc_bars.df
print(df)
