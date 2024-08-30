from decouple import config
import pandas as pd
import psycopg2
from datetime import datetime
from pprint import pprint


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

    # query to insert into the database
    query = "INSERT INTO stock (symbol, name)" + \
            "VALUES (%s, %s)"

    for symbol, company in zip(symbols, companies):
        record_to_insert = (symbol, company)
        cursor.execute(query, record_to_insert)
        connection.commit()


if __name__ == '__main__':
    db_password = config("POSTGRES_KEY")

    # get nasdaq 100 tickers using pandas from the wikipedia page
    symbols = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4].Ticker.to_list()
    companies = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4].Company.to_list()
    # insert_stock()
    print("DONE")
