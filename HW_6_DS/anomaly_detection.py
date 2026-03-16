from sklearn.ensemble import IsolationForest


def detect_anomalies(X):

    iso = IsolationForest(contamination=0.05)

    labels = iso.fit_predict(X)

    return labels