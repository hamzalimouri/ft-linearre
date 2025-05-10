import json
import sys

import pandas as pd


def normalize_mileage(mileage, X_min=None, X_max=None):
    try:
        if X_min is None or X_max is None:
            X_min, X_max = get_normalization_params()
        return (mileage - X_min) / (X_max - X_min)  # Normalize the data
    except Exception:
        raise


def load_params(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)


def load_data(data_path):
    try:
        return pd.read_csv(data_path)
    except Exception:
        raise


def get_normalization_params(data_path='data.csv'):
    try:
        df = load_data(data_path)
        X = df['km']
        return X.min(), X.max()
    except Exception:
        raise