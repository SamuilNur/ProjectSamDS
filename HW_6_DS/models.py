from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


def train_models(X_train, y_train):

    models = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["LinearRegression"] = lr

    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    models["DecisionTree"] = dt

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    svr = SVR()
    svr.fit(X_train, y_train)
    models["SVR"] = svr

    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)
    models["GradientBoosting"] = gb

    return models