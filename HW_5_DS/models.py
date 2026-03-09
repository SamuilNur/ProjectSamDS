## Возвращаем словарь моделей для сравнения.

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(random_state=42)
    }
    return models