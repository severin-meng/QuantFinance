from decouple import config
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from pprint import pprint

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

api_key = config("ALPACA_PAPER_KEY")
secret_key = config("ALPACA_PAPER_SECRET")
db_password = config("POSTGRES_KEY")

client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
end_date = datetime.today() - timedelta(days=1)
start_date = end_date - timedelta(days=1000)


def insert_stock():
    # estimate db connection
    connection = psycopg2.connect(
        user="postgres",
        password=db_password,
        host="localhost",
        port=5432,
        dbname="nasdaq100"
    )

    # specify the cursor
    cursor = connection.cursor()

    # read data from the database
    cursor.execute('''select * from stock''')
    rows = cursor.fetchall()

    # query to insert into the database
    query = "INSERT INTO stock_price (stock_id, time, open, high, low, close, volume, vwap)" + \
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

    for row in rows:
        bars = client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=row[1],
                             timeframe=TimeFrame.Day,
                             start=start_date,
                             end=end_date)).df
        bars = bars.reset_index(level=[0, 1])
        bars = bars.drop(['symbol', 'trade_count'], axis=1)
        bars['timestamp'] = pd.to_datetime(bars['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d')))

        # insert records
        for index, bar in bars.iterrows():
            record_to_insert = (row[0],  # stock id in database
                                bar['timestamp'],
                                bar['open'],
                                bar['high'],
                                bar['low'],
                                bar['close'],
                                bar['volume'],
                                bar['vwap'])

            cursor.execute(query, record_to_insert)
            connection.commit()


if __name__ == '__main__':
    # get the keys
    api_key = config("ALPACA_PAPER_KEY")
    secret_key = config("ALPACA_PAPER_SECRET")
    db_password = config("POSTGRES_KEY")

    insert_stock()
    print("DONE")
