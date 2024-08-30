# Examples: Fetching data using openbb module
# see also https://docs.openbb.co/platform

from decouple import config
import pandas as pd
from datetime import datetime
from openbb import obb

import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# set the keys for data api
d = {
    "finnhub": {"key": config("FINNHUB_KEY")},
    "polygon": {"key": config("POLYGON_KEY")},
}

obb.user.credentials.polygon_api_key = d["polygon"]["key"]

# openbb.keys.mykeys(show=False)

from openbb import obb

# Get the price of a stock
quote_data = obb.equity.price.quote(symbol="AAPL", provider="yfinance")
print(quote_data.to_df())
