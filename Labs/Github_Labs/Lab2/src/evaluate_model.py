import pickle, os, json
from sklearn.metrics import f1_score, accuracy_score, classification_report
import joblib, sys
import argparse
sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

    model_version = f"model_{timestamp}_rf_model"
    try:
        model = joblib.load(f"{model_version}.joblib")
    except:
        raise ValueError("Failed to load the latest model")

    try:
        with open("data/data.pickle", "rb") as f:
            X = pickle.load(f)
        with open("data/target.pickle", "rb") as f:
            y = pickle.load(f)
    except:
        raise ValueError("Failed to load data")

    y_predict = model.predict(X)

    metrics = {
        "F1_Score_Weighted": f1_score(y, y_predict, average="weighted"),
        "Accuracy": accuracy_score(y, y_predict),
        "Classification_Report": classification_report(y, y_predict, output_dict=True)
    }

    os.makedirs("metrics", exist_ok=True)
    with open(f"{timestamp}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"F1 Score: {metrics['F1_Score_Weighted']:.4f}")