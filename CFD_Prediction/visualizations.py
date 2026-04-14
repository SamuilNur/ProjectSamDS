"""
Модуль визуализаций для Financials dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Настройка стиля
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Цветовая палитра
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#2ECC71',
    'danger': '#E74C3C',
    'dark': '#2C3E50',
    'segments': ['#2E86AB', '#A23B72', '#F18F01', '#2ECC71', '#E74C3C'],
    'products': ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'],
}


def format_currency(x, pos=None):
    """Форматирование чисел в валюту"""
    if abs(x) >= 1e6:
        return f'${x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'${x/1e3:.0f}K'
    return f'${x:.0f}'


def plot_sales_profit_by_segment(df: pd.DataFrame):
    """1. Продажи и прибыль по сегментам"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Агрегация
    seg_data = df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).sort_values('Sales', ascending=True)
    
    # Продажи по сегментам
    bars1 = axes[0].barh(seg_data.index, seg_data['Sales'], color=COLORS['segments'][:len(seg_data)])
    axes[0].set_title('Общие продажи по сегментам', fontweight='bold', pad=15)
    axes[0].set_xlabel('Продажи ($)')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    for bar, val in zip(bars1, seg_data['Sales']):
        axes[0].text(val + seg_data['Sales'].max() * 0.01, bar.get_y() + bar.get_height()/2,
                     format_currency(val), va='center', fontweight='bold', fontsize=10)
    
    # Прибыль по сегментам
    colors_profit = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in seg_data['Profit']]
    bars2 = axes[1].barh(seg_data.index, seg_data['Profit'], color=colors_profit)
    axes[1].set_title('Общая прибыль по сегментам', fontweight='bold', pad=15)
    axes[1].set_xlabel('Прибыль ($)')
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    for bar, val in zip(bars2, seg_data['Profit']):
        axes[1].text(val + seg_data['Profit'].max() * 0.01, bar.get_y() + bar.get_height()/2,
                     format_currency(val), va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_monthly_trend(df: pd.DataFrame):
    """2. Тренд продаж и прибыли по месяцам"""
    monthly = df.groupby('YearMonth').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    monthly['YearMonth'] = monthly['YearMonth'].astype(str)
    
    fig, ax1 = plt.subplots(figsize=(16, 6))
    
    x = range(len(monthly))
    
    # Продажи — столбцы
    bars = ax1.bar(x, monthly['Sales'], alpha=0.3, color=COLORS['primary'], label='Продажи', width=0.6)
    ax1.set_ylabel('Продажи ($)', color=COLORS['primary'])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    
    # Прибыль — линия
    ax2 = ax1.twinx()
    ax2.plot(x, monthly['Profit'], color=COLORS['danger'], linewidth=2.5, marker='o',
             markersize=8, label='Прибыль', zorder=5)
    ax2.set_ylabel('Прибыль ($)', color=COLORS['danger'])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    ax2.tick_params(axis='y', labelcolor=COLORS['danger'])
    
    # Оформление
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly['YearMonth'], rotation=45, ha='right')
    ax1.set_title('Динамика продаж и прибыли по месяцам', fontweight='bold', fontsize=16, pad=15)
    
    # Объединённая легенда
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_profit_by_product(df: pd.DataFrame):
    """3. Прибыль по продуктам"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    prod_data = df.groupby('Product').agg({
        'Profit': 'sum',
        'Sales': 'sum'
    }).sort_values('Profit', ascending=False)
    
    # Прибыль
    colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in prod_data['Profit']]
    axes[0].bar(prod_data.index, prod_data['Profit'], color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title('Прибыль по продуктам', fontweight='bold', pad=15)
    axes[0].set_ylabel('Прибыль ($)')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, (prod, val) in enumerate(prod_data['Profit'].items()):
        axes[0].text(i, val + prod_data['Profit'].max() * 0.02, format_currency(val),
                     ha='center', fontweight='bold', fontsize=9)
    
    # Маржинальность (Profit/Sales)
    margin = (prod_data['Profit'] / prod_data['Sales'] * 100).sort_values(ascending=True)
    bars = axes[1].barh(margin.index, margin.values, color=COLORS['products'][:len(margin)])
    axes[1].set_title('Маржинальность по продуктам (%)', fontweight='bold', pad=15)
    axes[1].set_xlabel('Маржа (%)')
    
    for bar, val in zip(bars, margin.values):
        axes[1].text(val + 0.3, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_country_analysis(df: pd.DataFrame):
    """4. Анализ по странам"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    country_data = df.groupby('Country').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).sort_values('Profit', ascending=True)
    
    # Продажи vs Прибыль по странам
    y = range(len(country_data))
    h = 0.35
    
    axes[0].barh([i - h/2 for i in y], country_data['Sales'], h, label='Продажи',
                 color=COLORS['primary'], alpha=0.8)
    axes[0].barh([i + h/2 for i in y], country_data['Profit'], h, label='Прибыль',
                 color=COLORS['success'], alpha=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(country_data.index)
    axes[0].set_title('Продажи vs Прибыль по странам', fontweight='bold', pad=15)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    axes[0].legend()
    
    # Pie chart — доля прибыли
    profit_positive = country_data['Profit'].clip(lower=0)
    axes[1].pie(profit_positive, labels=country_data.index, autopct='%1.1f%%',
                colors=COLORS['products'][:len(country_data)], startangle=90,
                textprops={'fontsize': 10})
    axes[1].set_title('Доля прибыли по странам', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.show()


def plot_discount_impact(df: pd.DataFrame):
    """5. Влияние скидок на прибыль"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    disc_data = df.groupby('Discount Band').agg({
        'Profit': ['sum', 'mean'],
        'Sales': 'sum'
    }).reset_index()
    disc_data.columns = ['Discount Band', 'Total Profit', 'Avg Profit', 'Total Sales']
    disc_data['Margin'] = disc_data['Total Profit'] / disc_data['Total Sales'] * 100
    
    # Средняя прибыль по скидкам
    order = ['None', 'Low', 'Medium', 'High']
    disc_ordered = disc_data.set_index('Discount Band').reindex(order).reset_index()
    
    colors = [COLORS['success'], '#82E0AA', COLORS['accent'], COLORS['danger']]
    axes[0].bar(disc_ordered['Discount Band'], disc_ordered['Avg Profit'], color=colors,
                edgecolor='white', linewidth=1)
    axes[0].set_title('Средняя прибыль по уровню скидки', fontweight='bold', pad=15)
    axes[0].set_ylabel('Средняя прибыль ($)')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    
    for i, val in enumerate(disc_ordered['Avg Profit']):
        axes[0].text(i, val + disc_ordered['Avg Profit'].max() * 0.02,
                     format_currency(val), ha='center', fontweight='bold')
    
    # Маржинальность по скидкам
    axes[1].bar(disc_ordered['Discount Band'], disc_ordered['Margin'], color=colors,
                edgecolor='white', linewidth=1)
    axes[1].set_title('Маржинальность по уровню скидки (%)', fontweight='bold', pad=15)
    axes[1].set_ylabel('Маржа (%)')
    
    for i, val in enumerate(disc_ordered['Margin']):
        axes[1].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame):
    """6. Корреляционная матрица"""
    numeric_cols = ['Units Sold', 'Manufacturing Price', 'Sale Price',
                    'Gross Sales', 'Discounts', 'Sales', 'COGS', 'Profit']
    existing_cols = [c for c in numeric_cols if c in df.columns]
    
    corr = df[existing_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Корреляция'})
    
    ax.set_title('🔗 Корреляционная матрица числовых признаков',
                 fontweight='bold', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()


def plot_sales_profit_scatter(df: pd.DataFrame):
    """7. Scatter plot: Sales vs Profit с разбивкой по сегментам"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    segments = df['Segment'].unique()
    for i, seg in enumerate(segments):
        mask = df['Segment'] == seg
        ax.scatter(df.loc[mask, 'Sales'], df.loc[mask, 'Profit'],
                   alpha=0.5, s=30, label=seg, color=COLORS['segments'][i % len(COLORS['segments'])])
    
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Продажи ($)')
    ax.set_ylabel('Прибыль ($)')
    ax.set_title('Продажи vs Прибыль (по сегментам)', fontweight='bold', fontsize=14, pad=15)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    ax.legend(title='Сегмент', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_full_eda(df: pd.DataFrame):
    """Запуск всех визуализаций EDA"""
    print("ВИЗУАЛИЗАЦИЯ ДАННЫХ")
    print("=" * 60)
    
    plot_sales_profit_by_segment(df)
    plot_monthly_trend(df)
    plot_profit_by_product(df)
    plot_country_analysis(df)
    plot_discount_impact(df)
    plot_correlation_heatmap(df)
    plot_sales_profit_scatter(df)