## Модуль создания таблиц

from db_connection import DBConnection

class DBSetup:
    @staticmethod
    def create_table():
        conn = DBConnection.connect()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS shipments (
            sku TEXT,
            price NUMERIC,
            number_sold INTEGER,
            revenue NUMERIC,
            shipping_time INTEGER,
            shipping_cost NUMERIC,
            transport_mode TEXT,
            route TEXT
        );
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("Table 'shipments' created successfully!")