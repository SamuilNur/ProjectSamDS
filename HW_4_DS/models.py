## models

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

def get_classifiers():
    classifiers = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    return classifiers