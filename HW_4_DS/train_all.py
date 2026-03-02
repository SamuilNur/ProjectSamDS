## train_all.py

from models import get_classifiers

def train_all(X_train, y_train):
    classifiers = get_classifiers()
    for name, model in classifiers.items():
        print(f"Обучаем {name}...")
        model.fit(X_train, y_train)
    return classifiers