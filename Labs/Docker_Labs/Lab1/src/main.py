from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import numpy as np
from flask import Flask, request, jsonify


def train_model():
    """Train a Gradient Boosting Classifier on the Wine dataset."""
    
    # Load the Wine dataset (instead of Iris)
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names.tolist()

    print(f"Dataset: Wine Recognition")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {target_names}")
    print(f"Feature names: {feature_names}")
    print("-" * 50)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train a Gradient Boosting classifier (instead of Random Forest)
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model: GradientBoostingClassifier")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Save the model
    joblib.dump(model, 'wine_model.pkl')
    print("\nModel saved to wine_model.pkl")

    # Save metadata (metrics, feature info) as JSON
    metadata = {
        "model_type": "GradientBoostingClassifier",
        "dataset": "Wine Recognition",
        "accuracy": round(accuracy, 4),
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "feature_names": list(feature_names),
        "target_names": target_names,
        "hyperparameters": {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 4
        }
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to model_metadata.json")

    return model, metadata


def serve_model():
    """Start a Flask API to serve predictions."""
    app = Flask(__name__)
    model = joblib.load('wine_model.pkl')
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)

    @app.route('/')
    def home():
        return jsonify({
            "service": "Wine Quality Prediction API",
            "model": metadata["model_type"],
            "accuracy": metadata["accuracy"],
            "endpoints": {
                "/": "API info",
                "/predict": "POST with JSON body {'features': [13 float values]}",
                "/health": "Health check",
                "/metadata": "Model metadata"
            }
        })

    @app.route('/health')
    def health():
        return jsonify({"status": "healthy"})

    @app.route('/metadata')
    def get_metadata():
        return jsonify(metadata)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            return jsonify({
                "prediction": int(prediction),
                "class_name": metadata["target_names"][prediction],
                "probabilities": {
                    name: round(float(prob), 4)
                    for name, prob in zip(metadata["target_names"], probabilities)
                }
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    print("\nStarting Flask API on port 5000...")
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'serve':
        serve_model()
    else:
        model, metadata = train_model()
        print("\nThe model training was successful")
        # Auto-start serving after training
        serve_model()