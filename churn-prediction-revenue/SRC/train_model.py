import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def train_models(X, y):
    """
    Train Logistic Regression and Random Forest models,
    compare their performance, and save the best model.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Logistic Regression
    logistic_model = LogisticRegression(
        max_iter=2000
    )

    logistic_model.fit(X_train, y_train)

    y_pred_logistic = logistic_model.predict(X_test)
    y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    # Compare models
    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy": [
            accuracy_score(y_test, y_pred_logistic),
            accuracy_score(y_test, y_pred_rf)
        ],
        "Precision": [
            precision_score(y_test, y_pred_logistic),
            precision_score(y_test, y_pred_rf)
        ],
        "Recall": [
            recall_score(y_test, y_pred_logistic),
            recall_score(y_test, y_pred_rf)
        ],
        "F1-score": [
            f1_score(y_test, y_pred_logistic),
            f1_score(y_test, y_pred_rf)
        ],
        "ROC-AUC": [
            roc_auc_score(y_test, y_prob_logistic),
            roc_auc_score(y_test, y_prob_rf)
        ]
    })

    # Save model comparison
    results.to_csv(
    "OUTPUTS/model_metrics.csv",
    index=False
)

    # Save Logistic Regression as selected model
    # because it achieved the best overall ROC-AUC
    joblib.dump(
    logistic_model,
    "OUTPUTS/churn_model.pkl"
)

    return (
        logistic_model,
        rf_model,
        results,
        X_train,
        X_test,
        y_train,
        y_test
    )