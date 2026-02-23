## Импорт CSV в таблицу

import pandas as pd
from db_connection import DBConnection

class DataImport:
    @staticmethod
    def import_csv_to_db(csv_path):
        df = pd.read_csv(csv_path)

        conn = DBConnection.connect()
        cursor = conn.cursor()

        for _, row in df.iterrows():
            cursor.execute("""
            INSERT INTO shipments (sku, price, number_sold, revenue, shipping_time, shipping_cost, transport_mode, route)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row['SKU'],
                float(row['Price']),
                int(row['Number of products sold']),
                float(row['Revenue generated']),
                int(row['Shipping times']),
                float(row['Shipping costs']),
                row['Transportation modes'],
                row['Routes']
            ))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"CSV '{csv_path}' imported successfully!")