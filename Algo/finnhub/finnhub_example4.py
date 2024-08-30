"""
Example 4: Streaming data with websocket
"""

from decouple import config
import json
import websocket
import finnhub

# get the tokens for the API access
api_key = config("FINNHUB_KEY_KANNAN")
socket = f"wss://ws.finnhub.io?token={api_key}"


def on_message(ws, message):  # ws = Websocket
    print(json.loads(message))
    # to get only the price:
    # msg = json.loads(message)
    # print(msg['data'][0]['p'])


def on_error(ws, error):
    print(f"Websocket error: {error}")


def on_close(ws, close_status_code, close_reason):
    print(f"### Websocket Close ###")


def on_open(ws):
    ws.send('{"type": "subscribe", "symbol": "BINANCE:BTCUSDT}')


if __name__ == '__main__':
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        socket,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.on_open = on_open
    ws.run_forever()
