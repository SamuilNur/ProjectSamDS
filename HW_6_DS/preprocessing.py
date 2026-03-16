import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical(df):

    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:

        df[col] = le.fit_transform(df[col].astype(str))

    return df



def fill_missing(df):

    df = df.fillna(df.median(numeric_only=True))

    return df