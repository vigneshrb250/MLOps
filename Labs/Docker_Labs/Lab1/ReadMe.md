# Docker Lab 1 - Submission Notes (Vignesh)

This fork contains modifications from the original Docker Lab 1:

* Replaced the **Iris dataset** with the **Wine Recognition dataset** (178 samples, 13 features, 3 classes)
* Switched model from `RandomForestClassifier` to `GradientBoostingClassifier` with tuned hyperparameters:
  * `n_estimators=150`, `learning_rate=0.1`, `max_depth=4`
* Added model evaluation output during training:
  * Accuracy score, classification report, and confusion matrix
* Saves model metadata (metrics, feature names, hyperparameters) to `model_metadata.json`
* Added a **Flask REST API** for serving predictions on port 5000:
  * `GET /` — API info
  * `GET /health` — Health check
  * `GET /metadata` — Model metadata
  * `POST /predict` — Make predictions with JSON body
* Updated Dockerfile:
  * Used `python:3.10-slim` base image for smaller image size
  * Optimized layer caching by copying `requirements.txt` before source code
  * Exposed port 5000

## How to Run
```bash
docker build -t lab1:v1 .
docker run -p 5000:5000 lab1:v1
docker save lab1:v1 > my_image.tar
```

The container trains the model on startup and automatically begins serving predictions via the Flask API.