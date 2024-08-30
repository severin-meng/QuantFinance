# Live Trading Strategy Execution

from decouple import config
from datetime import datetime, timedelta, date

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from pprint import pprint

# get the keys
api_key = config("ALPACA_PAPER_KEY")
secret_key = config("ALPACA_PAPER_SECRET")

# instantiate the trading client
api = TradingClient(api_key=api_key, secret_key=secret_key, raw_data=True)

# clock
clock = api.get_clock()  # NY time

# get account details
account = api.get_account()
# get list of portfolio
portfolio = api.get_all_positions()

# Trading APP


class App:

    def run(self):
        quantity = 10
        symbol = 'AAPL'

        # retrieve data
        stock_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=1)

        if clock.get('is_open'):
            # get data
            data = stock_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start_date,
                end=end_date)).df

            # signal logic
            high = data['high']
            low = data['low']

            portfolio = api.get_all_positions()
            if len(portfolio) == 0:
                if high[-1] > high[-5]:
                    api.submit_order(MarketOrderRequest(symbol=symbol, qty=quantity, side=OrderSide.BUY,
                                                        time_in_force=TimeInForce.DAY))
                    print(f"Bought {quantity} shares of {symbol}.")
                elif low[-1] < low[-5]:
                    amount = api.get_open_position(symbol)
                    api.close_position(symbol_or_asset_id=symbol)
                    print(f"Closed {amount} shares of {symbol}")

            else:
                balance_change = float(account.get('equity')) - float(account.get('last_equity'))
                print(f"Portfolio Balance Change: {balance_change}")


if __name__ == "__main__":
    app = App()
    while clock.get('is_open'):
        app.run()
    else:
        print(f"Market is closed, Next Open at {clock.get('next_open')}, which is in {0}")
        balance_change = float(account.get('equity')) - float(account.get('last_equity'))
        print(f"Today's portfolio balance change: {balance_change}")