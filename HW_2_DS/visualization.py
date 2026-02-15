import matplotlib.pyplot as plt 
import seaborn as sns

class Visualization:
    def __init__(self):
      self.plots = []

    def add_histogram(self, df, column):
        fig = plt.figure()
        sns.histplot(df[column], kde=True)
        plt.title(f"Histogram of {column}")
        self.plots.append(fig)
        plt.show()

    def add_line_plot(self, df, x, y):
        fig = plt.figure()
        sns.lineplot(data=df, x=x, y=y)
        plt.title(f"{y} over {x}")
        self.plots.append(fig)
        plt.show()

    def add_scatter_plot(self, df, x, y):
        fig = plt.figure()
        sns.scatterplot(data=df, x=x, y=y)
        plt.title(f"{y} vs {x}")
        self.plots.append(fig)
        plt.show()

    def remove_last_plot(self):
        if self.plots:
            fig = self.plots.pop()
            plt.close(fig)    