import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df


def basic_info(df):

    print("Dataset shape:", df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nInfo:")
    print(df.info())


def describe_data(df):

    return df.describe()