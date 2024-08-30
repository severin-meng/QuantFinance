# stream data from binance using websocket interface
# see payload.txt for explanation of data that is received
# importantly, the "m" field shows whether the buyer is the market maker
# that is, if m==True then this was a Sell, otherwise a Buy

from decouple import config
from binance import Client, ThreadedWebsocketManager

# Get keys
api_key = config("BINANCE_API_KEY")
secret_key = config("BINANCE_SECRET_KEY")


def main():
    symbol = 'ETHBTC'

    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=secret_key)
    twm.start()

    def on_message(msg):
        print(msg)

    twm.start_trade_socket(callback=on_message, symbol=symbol)
    twm.join()


if __name__ == '__main__':
    main()
