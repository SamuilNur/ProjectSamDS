import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


def train_arima(series):

    model = ARIMA(series, order=(2,1,2))

    model_fit = model.fit()

    return model_fit



def forecast_arima(model, steps):

    forecast = model.forecast(steps=steps)

    return forecast



def train_prophet(series):

    df_prophet = pd.DataFrame({
        "ds": range(len(series)),
        "y": series.values
    })

    model = Prophet()

    model.fit(df_prophet)

    return model, df_prophet



def forecast_prophet(model, df_prophet, steps):

    future = model.make_future_dataframe(
        periods=steps,
        freq="D"
    )

    forecast = model.predict(future)

    return forecast