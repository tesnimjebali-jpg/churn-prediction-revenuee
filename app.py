import streamlit as st
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    customers = pd.read_csv("OUTPUTS/customers_risk_ranking.csv")
    metrics = pd.read_csv("OUTPUTS/model_metrics.csv")
    importance = pd.read_csv("OUTPUTS/feature_importance.csv")

    return customers, metrics, importance


customers, metrics, importance = load_data()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Churn Prediction & Revenue Impact")

st.write(
    "Dashboard for identifying customers at risk of churn "
    "and estimating the associated financial impact."
)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

total_customers = len(customers)

high_risk_customers = (
    customers["Priority"] == "High"
).sum()

total_revenue_at_risk = customers["EstimatedLoss"].sum()

average_churn_probability = (
    customers["ChurnProbability"].mean()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "High Priority Customers",
    f"{high_risk_customers:,}"
)

col3.metric(
    "Revenue at Risk",
    f"${total_revenue_at_risk:,.2f}"
)

col4.metric(
    "Average Churn Probability",
    f"{average_churn_probability:.1%}"
)

st.divider()

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.header("🤖 Model Performance")

st.dataframe(
    metrics.round(4),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# CUSTOMERS AT RISK
# --------------------------------------------------

st.header("⚠️ Customers at Risk")

priority_filter = st.selectbox(
    "Select priority level",
    [
        "All",
        "High",
        "Medium",
        "Low"
    ]
)

if priority_filter == "All":
    filtered_customers = customers
else:
    filtered_customers = customers[
        customers["Priority"] == priority_filter
    ]

st.dataframe(
    filtered_customers[
        [
            "CustomerID",
            "ChurnProbability",
            "MonthlyCharges",
            "CustomerValue",
            "EstimatedLoss",
            "Priority"
        ]
    ].head(50),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# TOP CUSTOMERS BY REVENUE RISK
# --------------------------------------------------

st.header("💰 Top Customers by Revenue at Risk")

top_n = st.slider(
    "Number of customers",
    min_value=5,
    max_value=30,
    value=10
)

top_customers = customers.head(top_n)

chart_data = (
    top_customers[
        [
            "CustomerID",
            "EstimatedLoss"
        ]
    ]
    .set_index("CustomerID")
)

st.bar_chart(chart_data)

st.divider()

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.header("🔍 Most Important Churn Factors")

top_features = importance.head(10)

feature_chart = (
    top_features[
        [
            "Feature",
            "Importance"
        ]
    ]
    .set_index("Feature")
)

st.bar_chart(feature_chart)

st.divider()

# --------------------------------------------------
# BUSINESS RECOMMENDATIONS
# --------------------------------------------------

st.header("💡 Business Recommendations")

st.markdown(
    """
### High Risk + High Value
Offer a priority retention package.

### High Risk + Low Value
Use an automated low-cost retention campaign.

### Low Risk + High Value
Include the customer in a loyalty or proactive engagement program.

### Low Risk + Low Value
No immediate intervention is required.
"""
)

st.info(
    "Customers should be prioritized using financial risk, "
    "not only churn probability."
)