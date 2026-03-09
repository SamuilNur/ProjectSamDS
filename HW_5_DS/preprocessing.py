## Создаем ColumnTransformer для числовых и категориальных признаков.

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def create_preprocessor(X):

    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    num_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ])

    return preprocessor