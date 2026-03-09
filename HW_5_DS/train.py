## Обучаеv все модели, делает предсказания и возвращает DataFrame с метриками.

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_and_evaluate(models, preprocessor, X_train, X_test, y_train, y_test):
    """
    Обучает все модели, делает предсказания и возвращает DataFrame с метриками.
    """
    results = []

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    return pd.DataFrame(results)