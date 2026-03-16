import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_predictions(y_test, predictions):

    plt.figure(figsize=(10,6))

    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")

    plt.legend()

    plt.title("Prediction vs Actual")

    plt.show()



def plot_model_errors(names, errors):

    plt.figure(figsize=(8,5))

    plt.bar(names, errors)

    plt.title("Model MAE Comparison")

    plt.ylabel("MAE")

    plt.xticks(rotation=30)

    plt.show()



def plot_feature_importance(model, feature_names):

    importances = model.feature_importances_

    plt.figure(figsize=(8,6))

    plt.barh(feature_names, importances)

    plt.title("Feature Importance")

    plt.show()



def plot_anomalies(series, labels):

    plt.figure(figsize=(12,6))

    plt.plot(series.values, label="Normal")

    anomalies = series[labels == -1]

    plt.scatter(
        anomalies.index,
        anomalies.values,
        color="red",
        label="Anomaly"
    )

    plt.legend()

    plt.title("Detected Anomalies")

    plt.show()