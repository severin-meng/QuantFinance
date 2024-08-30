# replicate tick data

import pandas as pd
from lightweight_charts import Chart
from time import sleep


if __name__ == '__main__':

    df1 = pd.read_csv('histdata.csv')
    df2 = pd.read_csv('tickdata.csv')

    chart = Chart()
    chart.set(df1)
    chart.show()  # no block

    for i, tick in df2.iterrows():
        chart.update_from_tick(tick)
        sleep(0.003)
