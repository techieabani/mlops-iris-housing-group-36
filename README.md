# Iris & California Housing MLOps Template

This repository is a starter template for an MLOps pipeline featuring:
- Iris classification (Logistic Regression, RandomForest)
- California Housing regression (Linear Regression, Decision Tree)
- MLflow experiment tracking (server included in docker-compose)
- FastAPI model serving with Pydantic validation
- Docker & docker-compose for local dev and deployment
- GitHub Actions workflow for CI / Docker Hub push
- DVC stub for dataset versioning (housing dataset)
- Logging and basic metrics endpoint (Prometheus compatible)

## Quickstart (local)

1. Build & start services:
```bash
docker-compose up --build
```

2. MLflow UI: http://localhost:5000  
3. FastAPI docs: http://localhost:8000/docs

## Notes
- Replace placeholders like `<DOCKERHUB_USER>` and `<RUN_ID>` where appropriate.
- Configure GitHub Secrets for Docker Hub push (`DOCKER_USERNAME`, `DOCKER_PASSWORD`).
