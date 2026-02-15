import pandas as pd 
import requests

class DataLoader:
    @staticmethod
    def load_csv(path: str):
        return pd.read_csv(path)

    @staticmethod
    def load_json(path: str):
        return pd.read_json(path)

    @staticmethod
    def load_api(url: str):
        responce = requests.get(url)
        responce.raise_for_status()
        data = responce.json()
        return pd.DataFrame(data)