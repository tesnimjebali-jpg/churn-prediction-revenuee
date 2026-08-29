from SRC.preprocessing import load_and_clean_data, prepare_features
from SRC.train_model import train_models
from SRC.revenue_analysis import calculate_revenue_risk


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

df = load_and_clean_data(
    "DATA/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

X, y, customer_ids = prepare_features(df)


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

(
    logistic_model,
    rf_model,
    results,
    X_train,
    X_test,
    y_train,
    y_test
) = train_models(X, y)


# --------------------------------------------------
# FINANCIAL RISK ANALYSIS
# --------------------------------------------------

business_results, total_revenue_at_risk = calculate_revenue_risk(
    df,
    logistic_model,
    X,
    customer_ids
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n--- PROJECT COMPLETED SUCCESSFULLY ---")

print("\nModel Comparison:")
print(results.round(4))

print("\nTop 10 Customers by Revenue Risk:")
print(
    business_results[
        [
            "CustomerID",
            "ChurnProbability",
            "CustomerValue",
            "EstimatedLoss",
            "Priority"
        ]
    ].head(10).round(2)
)

print("\nTotal Revenue at Risk:")
print(round(total_revenue_at_risk, 2))

print("\nSaved files:")
print("OUTPUTS/model_metrics.csv")
print("OUTPUTS/customers_risk_ranking.csv")
print("OUTPUTS/churn_model.pkl")