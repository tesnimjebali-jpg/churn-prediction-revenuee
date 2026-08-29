import pandas as pd


def load_and_clean_data(filepath):
    """
    Load and clean the Telco Customer Churn dataset.
    """

    df = pd.read_csv(filepath)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert target variable
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    return df


def prepare_features(df):
    """
    Prepare features and target for Machine Learning.
    """

    # Save IDs for business analysis
    customer_ids = df["customerID"].copy()

    # Remove customer identifier
    df_model = df.drop(columns=["customerID"])

    # Encode categorical variables
    df_encoded = pd.get_dummies(
        df_model,
        drop_first=True
    )

    # Separate features and target
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    return X, y, customer_ids