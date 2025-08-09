import os, logging
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data.data_loader import load_iris_data

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/iris_train.log', level=logging.INFO)

df = load_iris_data()
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

models = {
    'logistic': LogisticRegression(max_iter=200),
    'rf': RandomForestClassifier(random_state=42)
}

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_param('model', name)
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(model, artifact_path='model')
        logging.info(f"{name} -> accuracy: {acc}")
        print(f"{name} -> Accuracy: {acc}")