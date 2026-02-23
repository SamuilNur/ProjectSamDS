## Модуль запросов SQL

from db_connection import DBConnection
import pandas as pd

class Queries:

    @staticmethod
    def get_all_shipments():
        conn = DBConnection.connect()
        df = pd.read_sql('SELECT * FROM shipments;', conn)
        conn.close()
        return df

    @staticmethod
    def delayed_shipments(days=5):
        conn = DBConnection.connect()
        df = pd.read_sql(f'''
            SELECT *
            FROM shipments
            WHERE shipping_time > {days};
        ''', conn)
        conn.close()
        return df

    @staticmethod
    def total_revenue():
        conn = DBConnection.connect()
        df = pd.read_sql('''
            SELECT SUM(revenue) AS total_revenue
            FROM shipments;
        ''', conn)
        conn.close()
        return df

    @staticmethod
    def orders_by_route():
        conn = DBConnection.connect()
        df = pd.read_sql('''
            SELECT route AS route_name, COUNT(*) AS total_orders
            FROM shipments
            GROUP BY route
            ORDER BY total_orders DESC;
        ''', conn)
        conn.close()
        return df

    @staticmethod
    def revenue_by_transport_mode():
        conn = DBConnection.connect()
        df = pd.read_sql('''
            SELECT transport_mode, SUM(revenue) AS total_revenue
            FROM shipments
            GROUP BY transport_mode
            ORDER BY total_revenue DESC;
        ''', conn)
        conn.close()
        return df