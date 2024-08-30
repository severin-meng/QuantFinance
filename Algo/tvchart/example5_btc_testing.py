from lightweight_charts import Chart

import pandas as pd
import pandas_ta as ta
from lightweight_charts import Chart


df = pd.read_csv(r'D:\Quant Finance\PyRepo\Algo\notebooks\data\btc_usd_test_long.csv', index_col=0, parse_dates=True)

atr_length = 14
multiplier = 2
high_length = 22
low_length = 22
chandelier = pd.DataFrame(df.ta.chandelier_exit(
        atr_length=atr_length,
        multiplier=multiplier,
        high_length=high_length,
        low_length=low_length)[f'CHDLREXTd_{high_length}_{low_length}_{atr_length}_{multiplier}']).dropna()

short_length = 12
long_length = 26
macd = pd.DataFrame(df.ta.macd(
        fast=short_length,
        slow=long_length
    ))# [f'MACDs_{short_length}_{long_length}_9']).dropna()

avg_length = 20
avg_volume = pd.DataFrame(ta.sma(close=df.Volume, length=avg_length), columns=[f'SMA_{avg_length}']).dropna()

print("DONE")


if __name__ == '__main__':
    chart = Chart(toolbox=True)
    line = chart.create_line(name=f'MACDs_{short_length}_{long_length}_9', color='red')
    # line2 = chart.create_line(name=f'SMA_{avg_length}', color='blue')

    chart.set(df)
    line.set(macd)
    # line2.set(avg_volume)

    chart.legend(True)
    chart.topbar.textbox('symbol', 'BTCUSD')
    chart.show(block=True)
