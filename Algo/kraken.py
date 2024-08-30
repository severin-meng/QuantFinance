#!/usr/bin/env python3

# Import required libraries (Python not API)
import time
import sys
import json
import base64
import hashlib
import hmac
import urllib.request

# Configure API key (copy/paste from account management)
api_key_public = 'COPY/PASTE API PUBLIC KEY HERE'
api_key_private = 'COPY/PASTE API PRIVATE KEY HERE'

# Configure market/orders/trades
trade_symbol = 'XXBTZUSD'
trade_interval = 1  # OHLC interval in minutes
trade_size = 0.0001  # Trade volume in base currency
trade_leverage = 1

# Initial indicator/trade variables
trade_direction = 0
sma_values = [0.0, 0.0, 0.0]

# Infinite loop (can be exited via keyboard interrupt)
try:
    while True:
        # Retrieve OHLC data at specified interval
        print('Retrieving OHLC data ... ', end='')
        try:
            api_request = urllib.request.Request(
                'https://api.kraken.com/0/public/OHLC?pair=%(symbol)s&interval=%(interval)d' % {'symbol': trade_symbol,
                                                                                                'interval': trade_interval})
            api_request.add_header('User-Agent', 'Kraken trading bot example')
            api_response = urllib.request.urlopen(api_request).read().decode()
            api_data = json.loads(api_response)
        except Exception as error:
            print('Failed (%s)' % error)
        else:
            print('Done' if len(api_data['error']) == 0 else 'Error (%s)' % api_data['error'])

        # Calculate SMA (20 candles / closing price)
        print('Calculating SMA 20 ... ', end='')
        api_ohlc = api_data['result'][trade_symbol]
        api_ohlc_length = len(api_ohlc) - 1
        sma_temp = 0.0
        for count in range(1, 21):
            sma_temp += float(api_ohlc[api_ohlc_length - count][4])
        sma_temp = sma_temp / 20
        print('Done')

        # Update SMA values
        sma_values[2] = sma_values[1]
        sma_values[1] = sma_values[0]
        sma_values[0] = sma_temp
        if sma_values[2] == 0.0:
            print('Waiting %d seconds ... ' % (trade_interval * 60))
            time.sleep(trade_interval * 60)
            continue
        else:
            print('SMA 20 values ... %(sma2)f / %(sma1)f / %(sma0)f' % {'sma2': sma_values[2], 'sma1': sma_values[1],
                                                                        'sma0': sma_values[0]})

        # Trading decision (change in slope of SMA)
        print('Trading decision ... ', end='')
        if (sma_values[0] > sma_values[1]) and (sma_values[1] < sma_values[2]):
            make_trade = 1
            print('Long')
        elif (sma_values[0] < sma_values[1]) and (sma_values[1] > sma_values[2]):
            make_trade = -1
            print('Short')
        else:
            make_trade = 0
            print('No trade')

        # Place order/trade (if applicable)
        if make_trade != 0:
            print('Placing order/trade ... ', end='')
            try:
                api_path = '/0/private/AddOrder'
                api_nonce = str(int(time.time() * 1000))
                api_post = 'nonce=%(api_nonce)s&pair=%(symbol)s&type=%(direction)s&ordertype=market&volume=%(volume)f&leverage=%(leverage)s' % {
                    'api_nonce': api_nonce, 'symbol': trade_symbol, 'direction': 'buy' if make_trade == 1 else 'sell',
                    'volume': trade_size if trade_direction == 0 else trade_size * 2,
                    'leverage': str(trade_leverage) if trade_leverage > 0 else 'none'}

                api_sha256 = hashlib.sha256(api_nonce.encode('utf8') + api_post.encode('utf8'))
                api_hmac = hmac.new(base64.b64decode(api_key_private), api_path.encode('utf8') + api_sha256.digest(),
                                    hashlib.sha512)
                api_signature = base64.b64encode(api_hmac.digest())

                api_request = urllib.request.Request('https://api.kraken.com/0/private/AddOrder',
                                                     api_post.encode('utf8'))
                api_request.add_header('API-Key', api_key_public)
                api_request.add_header('API-Sign', api_signature)
                api_request.add_header('User-Agent', 'Kraken trading bot example')
                api_response = urllib.request.urlopen(api_request).read().decode()
                api_data = json.loads(api_response)
            except Exception as error:
                print('Failed (%s)' % error)
            else:
                trade_direction = make_trade
                print('Done (%s)' % api_response if len(api_data['error']) == 0 else 'Error (%s)' % api_data['error'])

        # Wait until next OHLC interval
        print('Waiting %d seconds ... ' % (trade_interval * 60))
        time.sleep(trade_interval * 60)
except KeyboardInterrupt:
    sys.exit(0)
except Exception as error:
    print('Error (%s)' % error)

sys.exit(1)