import os, pickle, json, argparse
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score
import sys
sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

    try:
        model = joblib.load(f"model_{timestamp}_rf_model.joblib")
    except:
        raise ValueError("Failed to load trained model")

    try:
        with open("data/data.pickle", "rb") as f:
            X = pickle.load(f)
        with open("data/target.pickle", "rb") as f:
            y = pickle.load(f)
    except:
        raise ValueError("Failed to load data")

    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated_model.fit(X, y)

    y_predict = calibrated_model.predict(X)
    y_proba = calibrated_model.predict_proba(X)

    print(f"Calibrated Accuracy: {accuracy_score(y, y_predict):.4f}")
    print(f"Calibrated F1 Score: {f1_score(y, y_predict, average='weighted'):.4f}")
    print(f"Sample probabilities (first 3 rows):\n{np.round(y_proba[:3], 3)}")

    calibrated_filename = f"model_{timestamp}_calibrated.joblib"
    joblib.dump(calibrated_model, calibrated_filename)
    print(f"Calibrated model saved: {calibrated_filename}")