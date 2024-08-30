# websocket data streaming for trades

from decouple import config
from alpaca.data.live import StockDataStream
from datetime import datetime

# get the tokens for API access
api_key = config("ALPACA_PAPER_KEY")
secret_key = config("ALPACA_PAPER_SECRET")

client = StockDataStream(api_key=api_key, secret_key=secret_key)


async def trade_handler(data):
    print(f"{datetime.now()}: {data}")


# Sadly this does not work.


if __name__ == '__main__':
    print("Inside main")
    client.subscribe_trades(trade_handler, "AAPL")
    client.run()
    print("Running..")
