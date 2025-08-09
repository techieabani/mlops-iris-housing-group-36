from sklearn.datasets import load_iris, fetch_california_housing
import pandas as pd
import os

def load_iris_data():
    data = load_iris(as_frame=True)
    df = data.frame
    df['target'] = df['target'].astype(str)
    return df

def load_housing_data():
    csv_path = os.path.join('data','raw','housing.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        data = fetch_california_housing(as_frame=True)
        df = data.frame
    return df
