## Модуль подключения к БД (Через Neon.tech)

import psycopg2

class DBConnection:
    @staticmethod
    def connect():
        connection = psycopg2.connect(
            host="ep-shy-tree-aie7syym-pooler.c-4.us-east-1.aws.neon.tech",
            database="neondb",
            user="neondb_owner",
            password="npg_2PSxN3kTwObA",
            port=5432,
            sslmode="require"
        )
        return connection