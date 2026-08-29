import pandas as pd

# Load dataset
df = pd.read_csv("../DATA/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Show first rows
print("\nFIRST 5 ROWS:")
print(df.head())

# Show dataset size
print("\nDATASET SHAPE:")
print(df.shape)

# Show column names
print("\nCOLUMNS:")
print(df.columns.tolist())

# Show basic information
print("\nDATASET INFO:")
print(df.info())

# Show missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())





















# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

print("\n--- DATA CLEANING ---")

# Check duplicate rows
print("\nDuplicate rows before cleaning:")
print(df.duplicated().sum())

# Convert TotalCharges to numeric
# Invalid/blank values become NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset shape after cleaning:")
print(df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum().sum())

print("\nRemaining duplicate rows:")
print(df.duplicated().sum())










# --------------------------------------------------
# PREPARE DATA FOR MACHINE LEARNING
# --------------------------------------------------

print("\n--- PREPARING DATA FOR MACHINE LEARNING ---")

# Convert target variable Churn: Yes/No -> 1/0
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn rate:")
print(df["Churn"].value_counts(normalize=True))

# Save customer IDs separately for later business analysis
customer_ids = df["customerID"].copy()

# Remove customerID because it is only an identifier
df_model = df.drop(columns=["customerID"])

# Convert categorical variables using one-hot encoding
df_encoded = pd.get_dummies(
    df_model,
    drop_first=True
)

print("\nEncoded dataset shape:")
print(df_encoded.shape)

print("\nFirst encoded columns:")
print(df_encoded.columns[:20].tolist())

# Separate features and target
X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


















# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("\n--- TRAIN / TEST SPLIT ---")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

# --------------------------------------------------
# LOGISTIC REGRESSION
# --------------------------------------------------

print("\n--- LOGISTIC REGRESSION ---")

logistic_model = LogisticRegression(
    max_iter=2000
)

logistic_model.fit(X_train, y_train)

# Predictions
y_pred_logistic = logistic_model.predict(X_test)

# Probability of churn
y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]

# Metrics
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic)
recall_logistic = recall_score(y_test, y_pred_logistic)
f1_logistic = f1_score(y_test, y_pred_logistic)
roc_auc_logistic = roc_auc_score(y_test, y_prob_logistic)

print("\nAccuracy:", round(accuracy_logistic, 4))
print("Precision:", round(precision_logistic, 4))
print("Recall:", round(recall_logistic, 4))
print("F1-score:", round(f1_logistic, 4))
print("ROC-AUC:", round(roc_auc_logistic, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))














# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

print("\n--- RANDOM FOREST ---")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Probability of churn
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\nAccuracy:", round(accuracy_rf, 4))
print("Precision:", round(precision_rf, 4))
print("Recall:", round(recall_rf, 4))
print("F1-score:", round(f1_rf, 4))
print("ROC-AUC:", round(roc_auc_rf, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_logistic,
        accuracy_rf
    ],
    "Precision": [
        precision_logistic,
        precision_rf
    ],
    "Recall": [
        recall_logistic,
        recall_rf
    ],
    "F1-score": [
        f1_logistic,
        f1_rf
    ],
    "ROC-AUC": [
        roc_auc_logistic,
        roc_auc_rf
    ]
})

print("\n--- MODEL COMPARISON ---")
print(results.round(4))

# Save comparison
results.to_csv(
    "../OUTPUTS/model_metrics.csv",
    index=False
)

print("\nModel comparison saved to:")
print("../OUTPUTS/model_metrics.csv")









# --------------------------------------------------
# FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# --------------------------------------------------

print("\n--- ERROR COST ANALYSIS ---")

# Logistic Regression confusion matrix
tn_log, fp_log, fn_log, tp_log = confusion_matrix(
    y_test,
    y_pred_logistic
).ravel()

print("\nLogistic Regression:")
print("True Negatives:", tn_log)
print("False Positives:", fp_log)
print("False Negatives:", fn_log)
print("True Positives:", tp_log)

# Random Forest confusion matrix
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(
    y_test,
    y_pred_rf
).ravel()

print("\nRandom Forest:")
print("True Negatives:", tn_rf)
print("False Positives:", fp_rf)
print("False Negatives:", fn_rf)
print("True Positives:", tp_rf)

# --------------------------------------------------
# BUSINESS INTERPRETATION
# --------------------------------------------------

print("\n--- BUSINESS INTERPRETATION ---")

print(
    "A False Positive means the company may contact or "
    "offer a discount to a customer who would not have churned."
)

print(
    "A False Negative means the company fails to identify "
    "a customer who actually leaves, which may result in lost revenue."
)

print(
    "\nFor this project, False Negatives are considered more costly."
)

print(
    "Therefore, Recall is an important business metric because "
    "higher Recall means identifying more actual churners."
)

print(
    "\nRandom Forest has higher Recall, while Logistic Regression "
    "has better overall performance and ROC-AUC."
)
















# --------------------------------------------------
# FINANCIAL VALUE AND REVENUE RISK
# --------------------------------------------------

print("\n--- FINANCIAL RISK ANALYSIS ---")

# Use Logistic Regression probabilities
# It has the best overall ROC-AUC in our comparison
all_churn_probabilities = logistic_model.predict_proba(X)[:, 1]

# Create business results dataframe
business_results = pd.DataFrame({
    "CustomerID": customer_ids.values,
    "ChurnProbability": all_churn_probabilities,
    "MonthlyCharges": df["MonthlyCharges"].values
})

# Estimated customer value for the next 12 months
business_results["CustomerValue"] = (
    business_results["MonthlyCharges"] * 12
)

# Financial risk score
business_results["EstimatedLoss"] = (
    business_results["ChurnProbability"]
    * business_results["CustomerValue"]
)

# Create priority levels
business_results["Priority"] = pd.cut(
    business_results["EstimatedLoss"],
    bins=[
        -float("inf"),
        business_results["EstimatedLoss"].quantile(0.50),
        business_results["EstimatedLoss"].quantile(0.80),
        float("inf")
    ],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

# Sort customers by highest financial risk
business_results = business_results.sort_values(
    by="EstimatedLoss",
    ascending=False
)

print("\nTop 10 customers by financial risk:")
print(
    business_results[
        [
            "CustomerID",
            "ChurnProbability",
            "MonthlyCharges",
            "CustomerValue",
            "EstimatedLoss",
            "Priority"
        ]
    ].head(10).round(2)
)

# Total revenue at risk
total_revenue_at_risk = business_results["EstimatedLoss"].sum()

print("\nTotal estimated revenue at risk:")
print(round(total_revenue_at_risk, 2))

# Save final customer ranking
business_results.to_csv(
    "../OUTPUTS/customers_risk_ranking.csv",
    index=False
)

print("\nCustomer ranking saved to:")
print("../OUTPUTS/customers_risk_ranking.csv")












# --------------------------------------------------
# EXPLORATORY DATA ANALYSIS AND VISUALIZATIONS
# --------------------------------------------------

import matplotlib.pyplot as plt
import os

print("\n--- CREATING VISUALIZATIONS ---")

# Create figures folder
os.makedirs("../OUTPUTS/figures", exist_ok=True)

# 1. Churn rate
plt.figure(figsize=(6, 4))
df["Churn"].value_counts().sort_index().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks([0, 1], ["No Churn", "Churn"], rotation=0)
plt.tight_layout()
plt.savefig("../OUTPUTS/figures/churn_distribution.png")
plt.close()

# 2. Churn by contract
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
)

contract_churn.plot(kind="bar", figsize=(8, 5))
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.legend(["No Churn", "Churn"])
plt.tight_layout()
plt.savefig("../OUTPUTS/figures/churn_by_contract.png")
plt.close()

# 3. Monthly charges by churn
plt.figure(figsize=(8, 5))
df[df["Churn"] == 0]["MonthlyCharges"].hist(
    bins=30,
    alpha=0.6,
    label="No Churn"
)
df[df["Churn"] == 1]["MonthlyCharges"].hist(
    bins=30,
    alpha=0.6,
    label="Churn"
)
plt.title("Monthly Charges by Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.savefig("../OUTPUTS/figures/monthly_charges_by_churn.png")
plt.close()

# 4. Tenure by churn
plt.figure(figsize=(8, 5))
df[df["Churn"] == 0]["tenure"].hist(
    bins=30,
    alpha=0.6,
    label="No Churn"
)
df[df["Churn"] == 1]["tenure"].hist(
    bins=30,
    alpha=0.6,
    label="Churn"
)
plt.title("Customer Tenure by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.savefig("../OUTPUTS/figures/tenure_by_churn.png")
plt.close()

# 5. Confusion Matrix - Logistic Regression
cm = confusion_matrix(y_test, y_pred_logistic)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig("../OUTPUTS/figures/confusion_matrix_logistic.png")
plt.close()

# 6. Churn probability distribution
plt.figure(figsize=(8, 5))
plt.hist(
    all_churn_probabilities,
    bins=30
)
plt.title("Distribution of Churn Probabilities")
plt.xlabel("Churn Probability")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("../OUTPUTS/figures/churn_probability_distribution.png")
plt.close()

# 7. Top 10 customers by estimated financial loss
top10 = business_results.head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top10["CustomerID"],
    top10["EstimatedLoss"]
)
plt.title("Top 10 Customers by Revenue at Risk")
plt.xlabel("Estimated Loss")
plt.ylabel("Customer ID")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../OUTPUTS/figures/top10_revenue_at_risk.png")
plt.close()

print("\nVisualizations saved in:")
print("../OUTPUTS/figures/")




























# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

print("\n--- FEATURE IMPORTANCE ---")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Save feature importance
feature_importance.to_csv(
    "../OUTPUTS/feature_importance.csv",
    index=False
)

# Plot top 10 features
top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.title("Top 10 Most Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../OUTPUTS/figures/feature_importance.png")
plt.close()

# --------------------------------------------------
# BUSINESS RECOMMENDATIONS
# --------------------------------------------------

print("\n--- BUSINESS RECOMMENDATIONS ---")

print(
    "High risk + High value: priority retention offer."
)

print(
    "High risk + Low value: low-cost automated retention campaign."
)

print(
    "Low risk + High value: loyalty program or proactive engagement."
)

print(
    "Low risk + Low value: no immediate intervention required."
)

print(
    "\nThe company should prioritize customers with the highest "
    "EstimatedLoss rather than only the highest churn probability."
)