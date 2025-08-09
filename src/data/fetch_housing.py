from sklearn.datasets import fetch_california_housing
import pandas as pd, os

os.makedirs('data/raw', exist_ok=True)
data = fetch_california_housing(as_frame=True)
df = data.frame
csv_path = os.path.join('data','raw','housing.csv')
df.to_csv(csv_path, index=False)
print(f"Wrote housing data to {csv_path}")