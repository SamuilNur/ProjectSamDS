Домашняя работа №4

- В работе использован датасет Credit Card Fraud Detection (284 807 транзакций, 31 признак) с сильно несбалансированным целевым классом (0 — 99.83%, 1 — 0.17%).
- Данные были масштабированы и разделены на train/test. Обучено 5 классификаторов: Decision Tree, Extra Trees, KNN, CatBoost и LightGBM.
- Для CatBoost и LightGBM проведён быстрый RandomizedSearchCV, после чего модели оценены по метрикам Accuracy, Precision, Recall, F1 и ROC-AUC.
- Лучший результат показал CatBoost (ROC-AUC = 0.9776). Построены ROC-кривые и анализ важности признаков для наглядного сравнения моделей.