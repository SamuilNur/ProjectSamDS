## Визуализация

import matplotlib.pyplot as plt

class Visualization:

    @staticmethod
    def plot_orders_by_country(df):
        grouped = df.groupby("customer_country").size().reset_index(name="total_orders")
        plt.figure(figsize=(10,5))
        plt.bar(grouped["customer_country"], grouped["total_orders"])
        plt.xticks(rotation=45)
        plt.title("Orders by Country")
        plt.show()

    @staticmethod
    def plot_revenue(df):
        plt.figure(figsize=(10,5))
        plt.plot(df['order_date'], df['revenue'])
        plt.title("Revenue over time")
        plt.xticks(rotation=45)
        plt.show()