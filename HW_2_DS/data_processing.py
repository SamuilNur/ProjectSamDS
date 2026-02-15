import pandas as pd

class DataProcessing:

    @staticmethod
    def count_missing(df):
        return df.isnull().sum()

    @staticmethod
    def missing_report(df):
        missing = df.isnull().sum()
        percent = (missing / len(df)) * 100
        return pd.DataFrame({
            "Missing values": missing,
            "Percent (%)": percent
        })

    @staticmethod
    def fill_missing(df, strategy="mean"):
        df = df.copy()
        for col in df.select_dtypes(include="number").columns:
            if strategy == "mean":
                 df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                 df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                 df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    @staticmethod
    def to_datetime(df,column):
        df[column] = pd.to_datetime(df[column])
        return df       