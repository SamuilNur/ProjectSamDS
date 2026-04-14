"""
Модуль загрузки и предобработки данных Financials.csv
"""

import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Загрузка сырых данных из CSV"""
    df = pd.read_csv(filepath)
    return df


def inspect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверка и исправление имён колонок:
    - Убираем пробелы в начале/конце
    - Находим ошибку: колонка ' Sales ' имеет двойной пробел в начале ('  Sales')
    """
    print("=" * 60)
    print("ПРОВЕРКА ИМЁН КОЛОНОК")
    print("=" * 60)
    
    print("\n Оригинальные имена колонок:")
    for i, col in enumerate(df.columns):
        # Показываем скрытые пробелы
        print(f"  [{i}] '{col}' | repr: {repr(col)}")
    
    # Ищем ошибку — двойной пробел перед 'Sales'
    errors_found = []
    for col in df.columns:
        if col != col.strip():
            errors_found.append((col, col.strip()))
        # Проверяем двойные пробелы
        if '  ' in col:
            errors_found.append((col, f"Двойной пробел обнаружен!"))
    
    if errors_found:
        print("\n НАЙДЕНЫ ОШИБКИ В ИМЕНАХ КОЛОНОК:")
        for original, issue in errors_found:
            print(f"  '{original}' -> {issue}")
    
    # Исправляем: strip + замена множественных пробелов
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    
    print("\n Исправленные имена колонок:")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] '{col}'")
    
    return df


def clean_currency_column(series: pd.Series) -> pd.Series:
    """
    Очистка одного столбца от валютных символов:
    - Убирает '$'
    - Убирает '-' (означает ноль)
    - Убирает запятые из чисел
    - Обрабатывает отрицательные значения в формате $(xxx)
    - Убирает пробелы
    """
    if series.dtype == 'object':
        cleaned = series.astype(str).str.strip()
        
        # Обработка отрицательных значений типа "$(1,234.56)"
        # Сначала помечаем их
        cleaned = cleaned.str.replace(r'\$\(', '-', regex=True)
        cleaned = cleaned.str.replace(r'\)', '', regex=True)
        
        # Убираем знак доллара
        cleaned = cleaned.str.replace('$', '', regex=False)
        
        # Убираем " - " (означает ноль)
        cleaned = cleaned.str.replace(r'^\s*-\s*$', '0', regex=True)
        
        # Убираем запятые из чисел
        cleaned = cleaned.str.replace(',', '', regex=False)
        
        # Убираем оставшиеся пробелы
        cleaned = cleaned.str.strip()
        
        # Конвертируем в числовой тип
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        return cleaned
    return series


def clean_all_currency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет очистку ко всем столбцам с символом '$'
    """
    print("\n" + "=" * 60)
    print("ОЧИСТКА ВАЛЮТНЫХ СТОЛБЦОВ")
    print("=" * 60)
    
    currency_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Проверяем, содержит ли колонка знак '$'
            sample = df[col].astype(str).head(20)
            if sample.str.contains(r'\$', regex=True).any():
                currency_columns.append(col)
    
    print(f"\n Столбцы с символом '$': {currency_columns}")
    
    for col in currency_columns:
        print(f"\n  Обработка: '{col}'")
        print(f"    До:    {df[col].iloc[0]} (тип: {df[col].dtype})")
        df[col] = clean_currency_column(df[col])
        print(f"    После: {df[col].iloc[0]} (тип: {df[col].dtype})")
    
    return df


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка пробелов в категориальных столбцах"""
    print("\n" + "=" * 60)
    print("ОЧИСТКА КАТЕГОРИАЛЬНЫХ СТОЛБЦОВ")
    print("=" * 60)
    
    cat_cols = ['Segment', 'Country', 'Product', 'Discount Band', 'Month Name']
    
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            print(f" '{col}' — уникальные значения: {df[col].unique()}")
    
    return df


def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Парсинг колонки Date в datetime"""
    print("\n" + "=" * 60)
    print("ПАРСИНГ ДАТ")
    print("=" * 60)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        print(f"  Date: от {df['Date'].min()} до {df['Date'].max()}")
        
        # Создаем дополнительные временные признаки
        df['Quarter'] = df['Date'].dt.quarter
        df['YearMonth'] = df['Date'].dt.to_period('M')
        print(f"  Добавлены: Quarter, YearMonth")
    
    return df


def full_preprocessing(filepath: str) -> pd.DataFrame:
    """
    Полный пайплайн предобработки данных
    """
    print(" ЗАПУСК ПОЛНОЙ ПРЕДОБРАБОТКИ")
    print("=" * 60)
    
    # 1. Загрузка
    df = load_raw_data(filepath)
    print(f" Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
    
    # 2. Исправление имен колонок
    df = inspect_columns(df)
    
    # 3. Очистка категориальных столбцов
    df = clean_categorical_columns(df)
    
    # 4. Очистка валютных столбцов (убираем $, -, запятые)
    df = clean_all_currency_columns(df)
    
    # 5. Парсинг дат
    df = parse_date_column(df)
    
    # 6. Финальная проверка
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ПРОВЕРКА")
    print("=" * 60)
    print(f"\n Размер: {df.shape}")
    print(f"\n Типы данных:")
    print(df.dtypes)
    print(f"\n Пропущенные значения:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "  Нет пропусков!")
    
    return df