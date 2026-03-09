## Делаем визуализацию

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_distributions(df):
    """
    Распределения числовых признаков.
    """
    num_features = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_features:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f'Distribution of {col}')
        plt.show()

def plot_correlation_heatmap(df):
    """
    Корреляции между числовыми признаками.
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_categorical_vs_target(df, target):
    """
    Boxplots категориальных признаков vs target
    """
    cat_features = df.select_dtypes(include=["object"]).columns
    for col in cat_features:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=col, y=target, data=df)
        plt.title(f'{col} vs {target}')
        plt.xticks(rotation=45)
        plt.show()

def plot_prediction_vs_real(y_true, y_pred, model_name="Model"):
    """
    Scatter plot реальных vs предсказанных значений
    """
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.show()

def plot_model_comparison(results_df, metric="RMSE"):
    """
    Сравнение моделей по выбранной метрике
    """
    plt.figure(figsize=(8,5))
    sns.barplot(x="Model", y=metric, data=results_df.sort_values(metric))
    plt.title(f"Model Comparison: {metric}")
    plt.xticks(rotation=45)
    plt.show()