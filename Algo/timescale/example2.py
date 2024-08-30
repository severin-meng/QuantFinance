# stream data from binance using websocket interface for multiple tokens via multiplex socket
# see also https://python-binance.readthedocs.io/en/latest/websockets.html

from decouple import config
from binance import ThreadedWebsocketManager

# Get keys
api_key = config("BINANCE_API_KEY")
secret_key = config("BINANCE_SECRET_KEY")


def main():
    streams = ['ethbtc@trade', "btcusdt@trade"]

    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=secret_key)
    twm.start()

    def on_message(msg):
        print(msg)

    twm.start_multiplex_socket(callback=on_message, streams=streams)
    twm.join()


if __name__ == '__main__':
    main()
