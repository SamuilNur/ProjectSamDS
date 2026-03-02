## load data

import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Данные загружены: {df.shape}")
    return df

def basic_statistics(df):
    display(df.describe())
    print(df.info())

def class_distribution(df, target_col='Class'):
    print(df[target_col].value_counts())
    print(round(df[target_col].value_counts(normalize=True)*100,2))