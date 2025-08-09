import os
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from src.data.data_loader import load_housing_data

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/housing_train.log', level=logging.INFO)

# Load dataset
df = load_housing_data()

# Detect target column
target = 'MedHouseVal' if 'MedHouseVal' in df.columns else df.columns[-1]

X, y = df.drop(target, axis=1), df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

models = {
    'linear': LinearRegression(),
    'tree': DecisionTreeRegressor(random_state=42)
}

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))

best_rmse = float('inf')
best_model_name = None
best_run_id = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        rmse = sqrt(mse)
        
        mlflow.log_param('model', name)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('rmse', rmse)
        mlflow.sklearn.log_model(model, artifact_path='model')
        
        logging.info(f"{name} -> MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        print(f"{name} -> MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_run_id = run.info.run_id

# Register the best model
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    model_name = "HousingRegressor"
    
    try:
        client = mlflow.tracking.MlflowClient()
        # Create registered model if doesn't exist
        if not any(m.name == model_name for m in client.list_registered_models()):
            client.create_registered_model(model_name)
        # Create a new model version
        client.create_model_version(name=model_name, source=model_uri, run_id=best_run_id)
        print(f"Registered best model '{best_model_name}' with RMSE={best_rmse:.4f} as '{model_name}'")
    except Exception as e:
        print(f"Failed to register model: {e}")