import os, logging
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from src.data.data_loader import load_housing_data

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/housing_train.log', level=logging.INFO)

df = load_housing_data()
# Ensure the housing target column is named 'MedHouseVal' for compatibility with sklearn fetch
if 'MedHouseVal' not in df.columns and 'MedHouseVal' in df.columns:
    target = 'MedHouseVal'
else:
    # try to detect target by common names
    target = 'MedHouseVal' if 'MedHouseVal' in df.columns else df.columns[-1]

X, y = df.drop(target, axis=1), df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

models = {
    'linear': LinearRegression(),
    'tree': DecisionTreeRegressor(random_state=42)
}

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        mlflow.log_param('model', name)
        mlflow.log_metric('mse', mse)
        mlflow.sklearn.log_model(model, artifact_path='model')
        logging.info(f"{name} -> MSE: {mse}")
        print(f"{name} -> MSE: {mse}")