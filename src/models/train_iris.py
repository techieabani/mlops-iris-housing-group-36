import os
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data.data_loader import load_iris_data

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/iris_train.log', level=logging.INFO)

# Load dataset
df = load_iris_data()
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

models = {
    'logistic': LogisticRegression(max_iter=200),
    'rf': RandomForestClassifier(random_state=42)
}

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))

best_acc = 0
best_model_name = None
best_run_id = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        mlflow.log_param('model', name)
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(model, artifact_path='model')
        
        logging.info(f"{name} -> Accuracy: {acc:.4f}")
        print(f"{name} -> Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_run_id = run.info.run_id

# Register the best model in MLflow Model Registry
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = "IrisClassifier"
    try:
        client = mlflow.tracking.MlflowClient()
        # Create registered model if it doesn't exist
        if not any(m.name == registered_model_name for m in client.list_registered_models()):
            client.create_registered_model(registered_model_name)
        # Create a new model version
        client.create_model_version(name=registered_model_name, source=model_uri, run_id=best_run_id)
        print(f"Registered best model '{best_model_name}' with accuracy={best_acc:.4f} as '{registered_model_name}'")
    except Exception as e:
        print(f"Failed to register model: {e}")
