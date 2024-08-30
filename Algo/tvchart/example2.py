# grid charts

import pandas as pd
import yfinance as yf
from lightweight_charts import Chart


if __name__ == '__main__':
    # get data
    aapl = yf.Ticker("AAPL")
    df = aapl.history(period='1y')

    goog = yf.Ticker("GOOG")
    df2 = goog.history(period='1y')

    msft = yf.Ticker("MSFT")
    df3 = msft.history(period='1y')

    nvda = yf.Ticker("NVDA")
    df4 = nvda.history(period='1y')

    # create 4 charts as in here: https://lightweight-charts-python.readthedocs.io/en/latest/examples/subchart.html
    chart = Chart(toolbox=True, inner_width=0.5, inner_height=0.5)
    chart2 = chart.create_subchart(position='right', width=0.5, height=0.5)
    chart3 = chart.create_subchart(position='left', width=0.5, height=0.5)
    chart4 = chart.create_subchart(position='right', width=0.5, height=0.5)

    chart.watermark('AAPL')
    chart2.watermark('GOOG')
    chart3.watermark('MSFT')
    chart4.watermark('NVDA')

    chart.set(df)
    chart.legend(True)

    chart2.set(df2)
    chart2.legend(True)

    chart3.set(df3)
    chart3.legend(True)

    chart4.set(df4)
    chart4.legend(True)

    chart.show(block=True)
