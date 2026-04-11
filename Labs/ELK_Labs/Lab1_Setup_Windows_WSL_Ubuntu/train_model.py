import logging
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)

# ─────────────────────────────────────────────
# SECTION 1: Configure logging
# Each log line will look like:
#   2024-01-01 12:00:00,000 - INFO - F1 Score: 0.8900
# This format is what logstash.conf expects to parse.
# ─────────────────────────────────────────────
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ─────────────────────────────────────────────
# SECTION 2: Load the Heart Disease dataset
# X = all feature columns (age, chol, bp, etc.)
# y = target column (1 = disease present, 0 = not present)
# ─────────────────────────────────────────────
df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)   # features: everything except target
y = df['target']                 # label: what we want to predict

logging.info("Dataset: Heart Disease (Cleveland UCI)")
logging.info(f"Total samples: {len(df)}")
logging.info(f"Total features: {X.shape[1]}")
logging.info(f"Class distribution - Disease: {sum(y==1)}, No Disease: {sum(y==0)}")

# Split into 80% train, 20% test — same split used for every run
# random_state=42 ensures the split is identical across runs
# so differences in metrics are purely due to n_estimators, not data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logging.info(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ─────────────────────────────────────────────
# SECTION 3: Hyperparameter sweep over n_estimators
# n_estimators = number of trees in the Random Forest
# More trees → generally better accuracy, but slower training
# We sweep across 6 values to observe this tradeoff
# ─────────────────────────────────────────────
n_estimators_values = [10, 50, 100, 200, 300, 500]

for n in n_estimators_values:
    logging.info(f"--- Starting run: n_estimators={n} ---")
    logging.info(f"n_estimators: {n}")

    # ─────────────────────────────────────────
    # SECTION 4: Train the model
    # random_state=42 ensures each forest is built consistently
    # n_jobs=-1 uses all CPU cores to speed up training
    # ─────────────────────────────────────────
    start_time = time.time()

    model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    training_time = round(time.time() - start_time, 4)
    logging.info(f"Training Time: {training_time}")

    # ─────────────────────────────────────────
    # SECTION 5: Evaluate the model
    # ─────────────────────────────────────────
    predictions = model.predict(X_test)

    accuracy  = round(accuracy_score(y_test, predictions), 4)
    f1        = round(f1_score(y_test, predictions), 4)
    precision = round(precision_score(y_test, predictions), 4)
    recall    = round(recall_score(y_test, predictions), 4)

    # Confusion matrix for binary classification gives us a 2x2 matrix:
    #         Predicted No   Predicted Yes
    # Actual No  [ TN           FP ]
    # Actual Yes [ FN           TP ]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    # FPR = how often we wrongly predict disease when there is none
    # FNR = how often we miss disease when it actually exists (dangerous in medical context)
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0

    # ─────────────────────────────────────────
    # SECTION 6: Log all metrics
    # These exact strings are what logstash.conf will parse using grok patterns
    # The format "MetricName: value" makes grok extraction straightforward
    # ─────────────────────────────────────────
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"True Positive: {tp}")
    logging.info(f"True Negative: {tn}")
    logging.info(f"False Positive: {fp}")
    logging.info(f"False Negative: {fn}")
    logging.info(f"False Positive Rate: {fpr}")
    logging.info(f"False Negative Rate: {fnr}")

    # Summary line — all metrics on one line so Logstash creates a single
    # document per run with all fields populated. This is what Kibana plots.
    logging.info(
        f"RUN_SUMMARY n_estimators={n} training_time={training_time} "
        f"accuracy={accuracy} f1_score={f1} precision={precision} "
        f"recall={recall} fpr={fpr} fnr={fnr} "
        f"tp={tp} tn={tn} fp={fp} fn={fn}"
    )
    logging.info(f"--- Run complete: n_estimators={n} ---")

logging.info("All runs completed.")
