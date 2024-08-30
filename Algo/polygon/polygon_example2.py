"""
Example 1: Fetching options data using REST API from polygon
"""

from decouple import config
import pandas as pd
from polygon import RESTClient
from pprint import pprint
import requests


# get the api keys
api_key = config("POLYGON_KEY")
client = RESTClient(api_key=api_key)

# list aggregates
aggs = []
for a in client.list_aggs(
    "O:SPY240820C00548000",
    1,
    "minute",
    "2024-08-01",
    "2024-08-15",
    limit=5000):
    aggs.append(a)

pprint(aggs)
