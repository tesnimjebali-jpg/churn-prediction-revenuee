import pandas as pd


def calculate_revenue_risk(
    df,
    model,
    X,
    customer_ids
):
    """
    Calculate customer financial value,
    churn probability, estimated loss,
    and priority.
    """

    # Predict churn probabilities
    churn_probabilities = model.predict_proba(X)[:, 1]

    # Create business results dataframe
    business_results = pd.DataFrame({
        "CustomerID": customer_ids.values,
        "ChurnProbability": churn_probabilities,
        "MonthlyCharges": df["MonthlyCharges"].values
    })

    # Estimate customer value over 12 months
    business_results["CustomerValue"] = (
        business_results["MonthlyCharges"] * 12
    )

    # Calculate expected financial loss
    business_results["EstimatedLoss"] = (
        business_results["ChurnProbability"]
        * business_results["CustomerValue"]
    )

    # Assign priority levels
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

    # Rank customers by financial risk
    business_results = business_results.sort_values(
        by="EstimatedLoss",
        ascending=False
    )

    # Calculate total revenue at risk
    total_revenue_at_risk = (
        business_results["EstimatedLoss"].sum()
    )

    # Save customer ranking
    business_results.to_csv(
    "OUTPUTS/customers_risk_ranking.csv",
    index=False
)

    return business_results, total_revenue_at_risk