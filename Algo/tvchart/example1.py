import pandas as pd
from lightweight_charts import Chart


if __name__ == '__main__':
    chart = Chart(toolbox=True)
    # requires columns to have lowercase names open high low close
    # datetime cannot be an index, ahs to be a column
    df = pd.read_csv("ohlcv.csv")
    chart.set(df)
    chart.legend(True)
    chart.show(block=True)

