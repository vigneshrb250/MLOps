import mlflow, datetime, os, pickle
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import sys, argparse
sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    df = pd.read_csv("data/sleep_health_lifestyle.csv")
    df.drop(columns=["Person ID"], errors="ignore", inplace=True)
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("No Disorder")

    if "Blood Pressure" in df.columns:
        df[["BP_Systolic", "BP_Diastolic"]] = (
            df["Blood Pressure"].str.split("/", expand=True).astype(float)
        )
        df.drop(columns=["Blood Pressure"], inplace=True)

    for col in df.select_dtypes(include="object").columns:
        if col != "Sleep Disorder":
            df[col] = LabelEncoder().fit_transform(df[col])

    le_target = LabelEncoder()
    df["Sleep Disorder"] = le_target.fit_transform(df["Sleep Disorder"])

    X = df.drop(columns=["Sleep Disorder"]).values
    y = df["Sleep Disorder"].values

    os.makedirs("data", exist_ok=True)
    with open("data/data.pickle", "wb") as f:
        pickle.dump(X, f)
    with open("data/target.pickle", "wb") as f:
        pickle.dump(y, f)
    np.save("data/target_classes.npy", le_target.classes_)

    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Sleep Health and Lifestyle"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):
        mlflow.log_params({
            "dataset_name": dataset_name,
            "number of datapoints": X.shape[0],
            "number of dimensions": X.shape[1]
        })

        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42))
        ])

        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__class_weight": ["balanced"]
        }

        print("Running GridSearchCV...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1
        )
        grid_search.fit(train_X, train_y)
        forest = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        mlflow.log_params(grid_search.best_params_)

        y_predict = forest.predict(test_X)
        mlflow.log_metrics({
            "Accuracy": accuracy_score(test_y, y_predict),
            "F1_Score": f1_score(test_y, y_predict, average="weighted")
        })

        os.makedirs("models", exist_ok=True)
        model_filename = f"model_{timestamp}_rf_model.joblib"
        dump(forest, model_filename)
        print(f"Model saved: {model_filename}")