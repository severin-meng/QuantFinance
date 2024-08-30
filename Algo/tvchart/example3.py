# add lines or indicators

import pandas as pd
from lightweight_charts import Chart


def calculate_sma(df, period: int = 50):
    return pd.DataFrame({
        'time': df['date'],
        f'SMA_{period}': df['close'] .rolling(window=period).mean()
    }).dropna()


if __name__ == '__main__':
    chart = Chart(toolbox=True)

    line = chart.create_line(name="SMA_21", color='red')
    line2 = chart.create_line(name="SMA_65", color='green')

    df = pd.read_csv("ohlcv.csv")
    sma_df = calculate_sma(df, period=21)
    sma2df = calculate_sma(df, period=65)

    chart.set(df)
    line.set(sma_df)
    line2.set(sma2df)
    chart.legend(True)
    chart.topbar.textbox('symbol', 'AAPL')
    chart.show(block=True)
