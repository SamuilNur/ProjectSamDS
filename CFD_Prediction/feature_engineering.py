"""
Модуль Feature Engineering для предсказания прибыли
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создание новых признаков"""
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df_fe = df.copy()
    
    # 1. Маржинальность
    df_fe['Profit_Margin'] = np.where(
        df_fe['Sales'] != 0,
        df_fe['Profit'] / df_fe['Sales'] * 100,
        0
    )
    print("Profit_Margin (маржинальность)")
    
    # 2. Коэффициент скидки
    df_fe['Discount_Rate'] = np.where(
        df_fe['Gross Sales'] != 0,
        df_fe['Discounts'] / df_fe['Gross Sales'] * 100,
        0
    )
    print("Discount_Rate (процент скидки)")
    
    # 3. Стоимость за единицу (COGS per unit)
    df_fe['COGS_per_Unit'] = np.where(
        df_fe['Units Sold'] != 0,
        df_fe['COGS'] / df_fe['Units Sold'],
        0
    )
    print("COGS_per_Unit")
    
    # 4. Profit per unit
    df_fe['Profit_per_Unit'] = np.where(
        df_fe['Units Sold'] != 0,
        df_fe['Profit'] / df_fe['Units Sold'],
        0
    )
    print("Profit_per_Unit")
    
    # 5. Price markup (наценка)
    df_fe['Price_Markup'] = np.where(
        df_fe['Manufacturing Price'] != 0,
        df_fe['Sale Price'] / df_fe['Manufacturing Price'],
        0
    )
    print("Price_Markup (наценка)")
    
    # 6. Флаг: убыточная сделка
    df_fe['Is_Loss'] = (df_fe['Profit'] < 0).astype(int)
    print("Is_Loss (убыточная сделка)")
    
    # 7. Размер сделки (категория)
    df_fe['Deal_Size'] = pd.qcut(df_fe['Sales'], q=4, labels=['Small', 'Medium', 'Large', 'XLarge'])
    print("Deal_Size (размер сделки)")
    
    print(f"\n Итого признаков: {df_fe.shape[1]}")
    
    return df_fe


def encode_categoricals(df: pd.DataFrame) -> tuple:
    """
    Кодирование категориальных переменных для моделей
    Возвращает: (df_encoded, label_encoders_dict)
    """
    print("\n КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ")
    print("=" * 60)
    
    df_enc = df.copy()
    label_encoders = {}
    
    cat_cols = ['Segment', 'Country', 'Product', 'Discount Band', 'Deal_Size']
    
    for col in cat_cols:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[f'{col}_encoded'] = le.fit_transform(df_enc[col].astype(str))
            label_encoders[col] = le
            print(f"{col}: {len(le.classes_)} категорий -> {list(le.classes_)}")
    
    return df_enc, label_encoders


def prepare_model_data(df: pd.DataFrame, target: str = 'Profit') -> tuple:
    """
    Подготовка данных для модели
    Возвращает: (X, y, feature_names)
    """
    print("\n ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ")
    print("=" * 60)
    
    feature_cols = [
        'Units Sold', 'Manufacturing Price', 'Sale Price',
        'Gross Sales', 'Discounts', 'COGS',
        'Month Number', 'Year',
        'Discount_Rate', 'COGS_per_Unit', 'Price_Markup',
        'Segment_encoded', 'Country_encoded', 'Product_encoded',
        'Discount Band_encoded', 'Quarter'
    ]
    
    # Берём только существующие колонки
    existing_features = [c for c in feature_cols if c in df.columns]
    
    X = df[existing_features].copy()
    y = df[target].copy()
    
    # Заполняем NaN
    X = X.fillna(0)
    y = y.fillna(0)
    
    print(f"  Признаки ({len(existing_features)}): {existing_features}")
    print(f"  Целевая переменная: {target}")
    print(f"  Размер X: {X.shape}")
    print(f"  Размер y: {y.shape}")
    
    return X, y, existing_features