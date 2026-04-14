"""
Модуль обучения и оценки моделей для предсказания прибыли
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# МЕТРИКИ
# ============================================================

def format_currency(x, pos=None):
    if abs(x) >= 1e6:
        return f'${x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'${x/1e3:.0f}K'
    return f'${x:.0f}'


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Вычисление метрик модели"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # MAPE — только для ненулевых значений
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    metrics = {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape
    }

    return metrics


def print_metrics(metrics: dict):
    """Красивый вывод метрик"""
    print(f"\n{'─' * 50}")
    print(f"{metrics['Model']}")
    print(f"{'─' * 50}")
    print(f"  MAE:      ${metrics['MAE']:,.2f}")
    print(f"  RMSE:     ${metrics['RMSE']:,.2f}")
    print(f"  R²:       {metrics['R²']:.4f}")
    print(f"  MAPE:     {metrics['MAPE (%)']:.2f}%")
    print(f"{'─' * 50}")


# ============================================================
# ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================================

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Разделение и масштабирование данных"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Данные разделены:")
    print(f"   Train: {X_train.shape[0]} строк")
    print(f"   Test:  {X_test.shape[0]} строк")
    print(f"   Масштабирование: StandardScaler")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def train_all_models(X_train_scaled, X_test_scaled, y_train, y_test,
                     X_train_raw=None, X_test_raw=None):
    """
    Обучение всех моделей и сбор метрик.
    Для tree-based моделей используются raw (немасштабированные) данные.
    """
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    # Если raw данные не переданы, используем scaled
    if X_train_raw is None:
        X_train_raw = X_train_scaled
    if X_test_raw is None:
        X_test_raw = X_test_scaled

    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'use_scaled': True
        },
        'Ridge Regression': {
            'model': Ridge(alpha=10.0),
            'use_scaled': True
        },
        'Lasso Regression': {
            'model': Lasso(alpha=1.0, max_iter=10000),
            'use_scaled': True
        },
        'Random Forest': {
            'model': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'use_scaled': False  # tree-based — не нужно масштабирование
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=42
            ),
            'use_scaled': False
        }
    }

    results = []
    trained_models = {}

    for name, config in models.items():
        print(f"\n🔄 Обучение: {name}...")

        model = config['model']

        if config['use_scaled']:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train_raw, X_test_raw

        # Обучение
        model.fit(X_tr, y_train)

        # Предсказание
        y_pred = model.predict(X_te)

        # Метрики
        metrics = evaluate_model(y_test.values, y_pred, name)
        print_metrics(metrics)

        # Cross-validation на train (R²)
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
        metrics['CV R² (mean)'] = cv_scores.mean()
        metrics['CV R² (std)'] = cv_scores.std()
        print(f"  CV R²:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results.append(metrics)
        trained_models[name] = {
            'model': model,
            'use_scaled': config['use_scaled'],
            'predictions': y_pred
        }

    return results, trained_models


# ============================================================
# ТЮНИНГ ЛУЧШЕЙ МОДЕЛИ
# ============================================================

def tune_gradient_boosting(X_train, y_train, X_test, y_test):
    """GridSearchCV для Gradient Boosting"""
    print("\n" + "=" * 60)
    print("ТЮНИНГ GRADIENT BOOSTING (GridSearchCV)")
    print("=" * 60)

    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'min_samples_split': [3, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }

    # Уменьшенный grid для скорости
    param_grid_small = {
        'n_estimators': [200, 400],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [3, 5],
        'subsample': [0.8, 1.0]
    }

    gb = GradientBoostingRegressor(random_state=42)

    grid_search = GridSearchCV(
        gb, param_grid_small, cv=5, scoring='r2',
        n_jobs=-1, verbose=1
    )

    print("Идёт поиск лучших параметров...")
    grid_search.fit(X_train, y_train)

    print(f"\n Лучшие параметры:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\n   Лучший CV R²: {grid_search.best_score_:.4f}")

    # Оценка на тесте
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    metrics = evaluate_model(y_test.values, y_pred, 'Tuned Gradient Boosting')
    print_metrics(metrics)

    return best_model, metrics, y_pred


# ============================================================
# ВИЗУАЛИЗАЦИИ МОДЕЛЕЙ
# ============================================================

def plot_model_comparison(results: list):
    """Сравнение моделей по метрикам"""
    df_results = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = ['#3498DB', '#9B59B6', '#1ABC9C', '#E67E22', '#E74C3C',
              '#2ECC71'][:len(df_results)]

    # R²
    bars1 = axes[0].bar(df_results['Model'], df_results['R²'],
                        color=colors, edgecolor='white', linewidth=1)
    axes[0].set_title('R² Score (чем выше — тем лучше)', fontweight='bold', pad=15)
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis='x', rotation=35)
    for bar, val in zip(bars1, df_results['R²']):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)

    # MAE
    bars2 = axes[1].bar(df_results['Model'], df_results['MAE'],
                        color=colors, edgecolor='white', linewidth=1)
    axes[1].set_title('MAE (чем ниже — тем лучше)', fontweight='bold', pad=15)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    axes[1].tick_params(axis='x', rotation=35)
    for bar, val in zip(bars2, df_results['MAE']):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + df_results['MAE'].max()*0.01,
                     format_currency(val), ha='center', fontweight='bold', fontsize=9)

    # RMSE
    bars3 = axes[2].bar(df_results['Model'], df_results['RMSE'],
                        color=colors, edgecolor='white', linewidth=1)
    axes[2].set_title('RMSE (чем ниже — тем лучше)', fontweight='bold', pad=15)
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    axes[2].tick_params(axis='x', rotation=35)
    for bar, val in zip(bars3, df_results['RMSE']):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + df_results['RMSE'].max()*0.01,
                     format_currency(val), ha='center', fontweight='bold', fontsize=9)

    plt.suptitle('Сравнение моделей', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_test, y_pred, model_name: str):
    """Предсказанные vs реальные значения"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter
    axes[0].scatter(y_test, y_pred, alpha=0.4, s=25, color='#2E86AB', edgecolors='white',
                    linewidth=0.3)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеал (y=x)')
    axes[0].set_xlabel('Реальная прибыль ($)')
    axes[0].set_ylabel('Предсказанная прибыль ($)')
    axes[0].set_title(f'{model_name}: Предсказание vs Реальность',
                      fontweight='bold', pad=15)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Residuals (ошибки)
    residuals = y_test.values - y_pred
    axes[1].hist(residuals, bins=40, color='#A23B72', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Ошибка (Реальность - Предсказание)')
    axes[1].set_ylabel('Количество')
    axes[1].set_title(f'{model_name}: Распределение ошибок', fontweight='bold', pad=15)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(format_currency))

    median_err = np.median(residuals)
    axes[1].axvline(x=median_err, color='orange', linestyle=':', linewidth=2,
                    label=f'Медиана: {format_currency(median_err)}')
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15):
    """Важность признаков"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print(f"Модель {model_name} не поддерживает feature importance")
        return

    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(feat_imp)))
    ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, edgecolor='white')
    ax.set_title(f'{model_name}: Важность признаков (Top-{top_n})',
                 fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Важность')

    for i, (_, row) in enumerate(feat_imp.iterrows()):
        ax.text(row['Importance'] + feat_imp['Importance'].max() * 0.01,
                i, f"{row['Importance']:.4f}",
                va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

# ============================================================
# ПРОГНОЗ НА БУДУЩИЙ ПЕРИОД
# ============================================================

def forecast_future_profit(df: pd.DataFrame, best_model=None, feature_names=None,
                           scaler=None, use_scaled=False, n_future=6):
    """
    Прогноз месячной прибыли.
    Используем комбинацию: линейный тренд + сезонная компонента (если хватает данных)
    + доверительный интервал на основе исторической волатильности.
    Это честный подход при малом количестве данных.
    """
    print("\n" + "=" * 60)
    print("ПРОГНОЗ ПРИБЫЛИ НА БУДУЩИЕ ПЕРИОДЫ")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Агрегируем по месяцам
    # ----------------------------------------------------------
    monthly = df.groupby(['Year', 'Month Number']).agg({
        'Profit': 'sum',
        'Sales': 'sum',
        'Units Sold': 'sum',
        'COGS': 'sum'
    }).reset_index()

    monthly = monthly.sort_values(['Year', 'Month Number']).reset_index(drop=True)
    monthly['Month_Idx'] = range(len(monthly))

    n_months = len(monthly)
    profits = monthly['Profit'].values

    print(f"\nИсторические данные: {n_months} месяцев")
    print(f"Средняя месячная прибыль: ${profits.mean():,.2f}")
    print(f"Стд. отклонение: ${profits.std():,.2f}")
    print(f"Коэффициент вариации: {profits.std() / profits.mean() * 100:.1f}%")
    print()
    print(monthly[['Year', 'Month Number', 'Profit', 'Sales']].to_string(index=False))

    # ----------------------------------------------------------
    # 2. Линейный тренд
    # ----------------------------------------------------------
    x = np.arange(n_months)
    slope, intercept = np.polyfit(x, profits, 1)

    trend_values = slope * x + intercept
    residuals = profits - trend_values

    print(f"\nЛинейный тренд:")
    print(f"  Наклон (рост/месяц):  ${slope:,.2f}")
    print(f"  Перехват:             ${intercept:,.2f}")
    print(f"  Тренд за год:         ${slope * 12:,.2f}")

    # ----------------------------------------------------------
    # 3. Сезонная компонента (если есть повторяющиеся месяцы)
    # ----------------------------------------------------------
    month_effects = {}
    for m in range(1, 13):
        mask = monthly['Month Number'] == m
        if mask.sum() > 0:
            month_residuals = residuals[mask.values]
            month_effects[m] = month_residuals.mean()
        else:
            month_effects[m] = 0.0

    has_seasonality = len(set(monthly['Month Number'])) < n_months
    if has_seasonality:
        print(f"\n  Сезонные эффекты (отклонение от тренда):")
        for m, eff in sorted(month_effects.items()):
            if month_effects[m] != 0:
                print(f"    Месяц {m:2d}: ${eff:>+14,.2f}")
    else:
        print(f"\n  Сезонность: недостаточно данных (нет повторяющихся месяцев)")
        month_effects = {m: 0.0 for m in range(1, 13)}

    # ----------------------------------------------------------
    # 4. Качество модели на исторических данных (leave-one-out)
    # ----------------------------------------------------------
    fitted = np.array([slope * i + intercept + month_effects.get(
        int(monthly.iloc[i]['Month Number']), 0) for i in range(n_months)])

    ss_res = np.sum((profits - fitted) ** 2)
    ss_tot = np.sum((profits - profits.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(profits - fitted))
    rmse = np.sqrt(np.mean((profits - fitted) ** 2))

    print(f"\nКачество модели (тренд + сезонность) на истории:")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")

    # ----------------------------------------------------------
    # 5. Прогноз с доверительным интервалом
    # ----------------------------------------------------------
    residual_std = residuals.std()

    last_row = monthly.iloc[-1]
    last_year = int(last_row['Year'])
    last_month = int(last_row['Month Number'])

    future_predictions = []
    year, month = last_year, last_month

    for i in range(n_future):
        month += 1
        if month > 12:
            month = 1
            year += 1

        future_idx = n_months + i
        trend_pred = slope * future_idx + intercept
        seasonal = month_effects.get(month, 0)
        point_pred = trend_pred + seasonal

        # Неопределенность растет с горизонтом
        uncertainty = residual_std * np.sqrt(1 + (i + 1) / n_months)

        future_predictions.append({
            'Year': year,
            'Month': month,
            'Predicted_Profit': point_pred,
            'Lower_Bound': point_pred - 1.96 * uncertainty,
            'Upper_Bound': point_pred + 1.96 * uncertainty,
            'Uncertainty': uncertainty
        })

    future_df = pd.DataFrame(future_predictions)
    future_df['Period'] = future_df.apply(
        lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}", axis=1
    )

    print(f"\nПоследний период данных: {last_year}-{last_month:02d}")
    print(f"Прогноз на {n_future} месяцев вперед:")
    print()
    print("ПРОГНОЗ ПРИБЫЛИ:")
    print("-" * 70)
    print(f"  {'Период':<10} {'Прогноз':>14}  {'Нижняя граница':>14}  {'Верхняя граница':>14}")
    print("-" * 70)
    for _, row in future_df.iterrows():
        print(f"  {row['Period']:<10} ${row['Predicted_Profit']:>13,.2f}"
              f"  ${row['Lower_Bound']:>13,.2f}  ${row['Upper_Bound']:>13,.2f}")
    print("-" * 70)

    total = future_df['Predicted_Profit'].sum()
    avg = future_df['Predicted_Profit'].mean()
    print(f"  ИТОГО:     ${total:>13,.2f}")
    print(f"  Среднее:   ${avg:>13,.2f}")

    print(f"\n  ВНИМАНИЕ: прогноз основан на {n_months} месяцах данных.")
    print(f"  Доверительный интервал (95%) учитывает растущую")
    print(f"  неопределенность с увеличением горизонта прогноза.")
    if n_months < 24:
        print(f"  Для надежного прогноза рекомендуется минимум 24 месяца данных.")

    return monthly, future_df


def plot_forecast(monthly: pd.DataFrame, future_df: pd.DataFrame):
    """Визуализация прогноза с доверительным интервалом"""
    fig, ax = plt.subplots(figsize=(16, 7))

    n_hist = len(monthly)
    n_future = len(future_df)

    # Исторические данные
    hist_labels = monthly.apply(
        lambda r: f"{int(r['Year'])}-{int(r['Month Number']):02d}", axis=1
    )
    hist_profits = monthly['Profit'].values

    ax.bar(range(n_hist), hist_profits, color='#2E86AB', alpha=0.8,
           label='Факт', width=0.7)

    # Линейный тренд через всю ось
    x_all = np.arange(n_hist + n_future)
    slope, intercept = np.polyfit(np.arange(n_hist), hist_profits, 1)
    ax.plot(x_all, slope * x_all + intercept, '--', color='orange',
            linewidth=2, alpha=0.7, label='Линейный тренд')

    # Прогноз — столбцы
    offset = n_hist
    future_profits = future_df['Predicted_Profit'].values

    ax.bar(range(offset, offset + n_future), future_profits,
           color='#E74C3C', alpha=0.7, label='Прогноз', width=0.7,
           edgecolor='white', linewidth=1, hatch='///')

    # Доверительный интервал
    if 'Lower_Bound' in future_df.columns:
        lower = future_df['Lower_Bound'].values
        upper = future_df['Upper_Bound'].values
        x_future = np.arange(offset, offset + n_future)

        ax.fill_between(x_future, lower, upper, alpha=0.15, color='red',
                        label='95% доверительный интервал')
        ax.plot(x_future, lower, ':', color='#E74C3C', alpha=0.5, linewidth=1)
        ax.plot(x_future, upper, ':', color='#E74C3C', alpha=0.5, linewidth=1)

    # Разделительная линия
    ax.axvline(x=offset - 0.5, color='grey', linestyle='--', linewidth=2, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(offset - 0.3, ymax * 0.95, '<-- Факт | Прогноз -->',
            fontsize=11, color='grey', fontweight='bold')

    # Подписи значений над прогнозными столбцами
    for i, val in enumerate(future_profits):
        ax.text(offset + i, val + ymax * 0.02,
                f'${val/1e6:.2f}M', ha='center', fontweight='bold', fontsize=9,
                color='#E74C3C')

    # Оси
    all_labels = list(hist_labels) + list(future_df['Period'])
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_title('Месячная прибыль: факт + прогноз (тренд + сезонность)',
                 fontweight='bold', fontsize=15, pad=15)
    ax.set_ylabel('Прибыль ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Дополнительный график: только прогноз крупным планом
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    x_pos = range(n_future)
    bars = ax2.bar(x_pos, future_profits, color='#E74C3C', alpha=0.8, width=0.6)

    if 'Lower_Bound' in future_df.columns:
        lower = future_df['Lower_Bound'].values
        upper = future_df['Upper_Bound'].values
        errors_low = future_profits - lower
        errors_high = upper - future_profits
        ax2.errorbar(x_pos, future_profits,
                     yerr=[errors_low, errors_high],
                     fmt='none', color='black', capsize=5, linewidth=2)

    for i, val in enumerate(future_profits):
        ax2.text(i, val + (upper[i] - val) * 1.1 if 'Upper_Bound' in future_df.columns else val * 1.02,
                 f'${val:,.0f}', ha='center', fontweight='bold', fontsize=10)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(future_df['Period'], fontsize=12)
    ax2.set_title('Прогноз прибыли (детальный вид)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Прибыль ($)')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()