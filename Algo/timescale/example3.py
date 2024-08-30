# store streaming (tick) data to the postgres db

from decouple import config
from binance import ThreadedWebsocketManager

from datetime import datetime
import psycopg2


# Get keys
api_key = config("BINANCE_API_KEY")
secret_key = config("BINANCE_SECRET_KEY")
db_password = config("POSTGRES_KEY")


def main():

    # estimate db connection
    connection = psycopg2.connect(
        user="postgres",
        password=db_password,
        host="localhost",
        port=5432,
        dbname="postgres"
    )

    # specify the cursor
    cursor = connection.cursor()

    # token streams
    streams = ['ethbtc@trade', "btcusdt@trade", "adabtc@trade"]

    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=secret_key)
    twm.start()

    insert_query = """INSERT INTO binance_data (time, symbol, price, quantity, maker) 
                    VALUES (%s, %s, %s, %s, %s)"""

    def on_message(msg):
        # store incoming data in database
        # datatype is a dictionary. the real data is inside a dictionary with the "data" key, but it is nested.
        # print(msg)
        try:
            msg = msg["data"]
        except KeyError as e:
            print(f"Key Error occurred: {str(msg)}")
            # this kills the websocket connection -> have to re establish.
            return

        # query to insert into the database
        time_stamp = datetime.fromtimestamp(msg["T"]/1000)  # timestamp in message is epoch in milliseconds
        record_to_insert = (time_stamp, msg["s"], msg["p"], msg["q"], msg["m"])
        cursor.execute(insert_query, record_to_insert)
        connection.commit()

    twm.start_multiplex_socket(callback=on_message, streams=streams)
    twm.join()


if __name__ == '__main__':
    main()
