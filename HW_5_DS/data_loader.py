 ## Загружаем Student Performance Dataset из файла. Возвращаем DataFrame.

import pandas as pd

def load_data(path="Student_performance_data_.csv"):
    df = pd.read_csv(path)
    return df