# Customer Churn Prediction & Revenue Impact Analysis

An end-to-end machine learning project that predicts telecommunications customer churn, estimates the associated revenue at risk, and prioritizes customers for targeted retention actions.

The project goes beyond traditional churn classification by translating model predictions into measurable business impact. Each customer is assigned an estimated financial risk, enabling decision-makers to focus retention resources on customers whose departure would create the greatest potential loss.

---

## Project Overview

Customer churn is a major challenge in the telecommunications industry. Losing customers does not only reduce the customer base—it also creates a direct financial impact through lost recurring revenue.

This project addresses two key business questions:

1. Which customers are most likely to churn?
2. How much revenue could the company lose if those customers leave?

To answer these questions, the project combines:

* Data cleaning and preprocessing
* Exploratory data analysis
* Machine learning classification
* Model evaluation and comparison
* False-positive and false-negative analysis
* Customer financial value estimation
* Revenue-at-risk calculation
* Customer prioritization
* Business recommendations
* An interactive Streamlit dashboard

---

## Business Objective

The objective is to build a decision-support system that helps a telecommunications company:

* Identify customers with a high probability of churn
* Estimate the potential revenue associated with each customer
* Calculate the expected financial loss from churn
* Rank customers according to business priority
* Select appropriate retention strategies
* Allocate retention resources more effectively

Instead of prioritizing customers based only on churn probability, the project uses both churn risk and customer financial value.

---

## Dataset

The project uses the **IBM Telco Customer Churn dataset**, which contains demographic, account, subscription, service, and billing information for telecommunications customers.

The dataset includes variables such as:

* Customer ID
* Gender
* Senior citizen status
* Partner and dependent status
* Customer tenure
* Phone and internet services
* Online security and backup
* Device protection
* Technical support
* Streaming services
* Contract type
* Payment method
* Monthly charges
* Total charges
* Churn status

After data cleaning, the final dataset contains:

| Dataset characteristic                   | Value |
| ---------------------------------------- | ----: |
| Customers                                | 7,032 |
| Original variables                       |    21 |
| Machine learning features after encoding |    30 |

---

## Project Workflow

The project follows an end-to-end data science workflow:

```text
Raw Customer Data
        ↓
Data Cleaning and Validation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering and Encoding
        ↓
Model Training and Evaluation
        ↓
Churn Probability Estimation
        ↓
Customer Financial Value Calculation
        ↓
Revenue-at-Risk Estimation
        ↓
Customer Prioritization
        ↓
Business Recommendations and Dashboard
```

---

## Data Preprocessing

The preprocessing pipeline includes:

* Loading and inspecting the raw dataset
* Detecting missing values
* Converting `TotalCharges` to a numeric variable
* Removing records with unresolved missing values
* Checking and removing duplicate records
* Converting the `Churn` target from `Yes/No` to `1/0`
* Preserving customer IDs for the final business output
* Excluding customer IDs from model training
* Encoding categorical variables using one-hot encoding
* Separating the features and target variable
* Creating training and testing datasets

The final machine learning dataset contains **7,032 observations and 30 input features**.

---

## Exploratory Data Analysis

Exploratory analysis was performed to identify customer characteristics associated with churn.

The analysis includes:

* Overall churn distribution
* Churn rate by contract type
* Monthly charges by churn status
* Customer tenure by churn status
* Churn probability distribution
* Model confusion matrix
* Feature importance analysis
* Customers with the highest estimated revenue at risk

Generated visualizations are stored in:

```text
OUTPUTS/figures/
```

---

## Machine Learning Models

Two classification algorithms were trained and evaluated:

* Logistic Regression
* Random Forest

### Model Performance

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.8038 |    0.6476 | 0.5749 |   0.6091 |  0.8363 |
| Random Forest       |   0.7683 |    0.5545 | 0.6524 |   0.5995 |  0.8209 |

### Model Interpretation

**Logistic Regression** achieved the strongest overall performance, with the highest:

* Accuracy
* Precision
* F1-score
* ROC-AUC

**Random Forest** achieved a higher recall, meaning it identified a larger proportion of customers who actually churned.

This comparison illustrates an important business trade-off:

* Logistic Regression provides better overall predictive performance.
* Random Forest detects more actual churners.

Because failing to identify a real churner can lead to lost revenue, recall is an important metric for this business problem.

---

## False Positives and False Negatives

Understanding prediction errors is essential when connecting a machine learning model to business decisions.

### False Positive

A false positive occurs when the model predicts that a customer will churn, but the customer actually stays.

Potential consequence:

* The company may offer an unnecessary discount or retention incentive.
* Marketing resources may be assigned to a customer who was not planning to leave.

### False Negative

A false negative occurs when the model predicts that a customer will stay, but the customer actually churns.

Potential consequence:

* The company does not intervene.
* The customer leaves without receiving a retention offer.
* The company loses the revenue associated with that customer.

For this project, false negatives are considered more costly because they represent missed opportunities to prevent revenue loss.

---

## Customer Financial Value

The estimated annual financial value of a customer is calculated as:

```text
Customer Value = Monthly Charges × 12
```

This represents the expected revenue generated by the customer over the next 12 months.

This is a simplified business estimate and does not include acquisition costs, service costs, discount rates, or customer lifetime value beyond one year.

---

## Financial Risk Score

A churn probability alone does not fully represent the business impact of losing a customer.

For example, a customer with a moderate churn probability and high monthly charges may represent a greater financial risk than a customer with a very high churn probability but low monthly charges.

The project therefore calculates an expected financial loss:

```text
Estimated Loss = Churn Probability × Customer Value
```

Where:

```text
Customer Value = Monthly Charges × 12
```

In the project outputs, the financial risk score is stored as `EstimatedLoss`.

This calculation transforms the model output from:

> Who is most likely to churn?

into:

> Which customers represent the greatest potential financial loss?

---

## Revenue at Risk

The total estimated revenue at risk identified by the project is:

## **$1,679,422.41**

This amount represents the sum of the probability-weighted annual revenue associated with the evaluated customers:

```text
Total Revenue at Risk = Σ Estimated Loss
```

It should be interpreted as an expected-risk estimate rather than a guaranteed future loss.

---

## Customer Prioritization

Customers are ranked according to their estimated financial loss and assigned one of three priority levels:

* High
* Medium
* Low

The final ranking is stored in:

```text
OUTPUTS/customers_risk_ranking.csv
```

The output contains:

* Customer ID
* Churn probability
* Monthly charges
* Estimated annual customer value
* Estimated financial loss
* Priority level

This prioritized list can help retention teams decide which customers should be contacted first.

---

## Business Recommendations

### High Churn Risk and High Customer Value

These customers should receive immediate and personalized attention.

Recommended actions:

* Personalized retention discounts
* Contract renewal incentives
* Premium customer support
* Direct contact from a retention specialist
* Service-package reviews

### High Churn Risk and Low Customer Value

These customers may be targeted through scalable, low-cost campaigns.

Recommended actions:

* Automated email campaigns
* Promotional offers
* Discount codes
* SMS campaigns
* Self-service retention options

### Low Churn Risk and High Customer Value

These customers should be included in proactive loyalty and engagement programs.

Recommended actions:

* Loyalty benefits
* Personalized communication
* Premium customer service
* Satisfaction surveys
* Early access to new services

### Low Churn Risk and Low Customer Value

No immediate retention intervention is required.

These customers can remain in standard communication and monitoring programs.

---

## Interactive Dashboard

A Streamlit dashboard was developed to make the analysis accessible to business users.

The dashboard displays:

* Total number of customers
* Number of high-priority customers
* Total estimated revenue at risk
* Average churn probability
* Model performance metrics
* Customer risk ranking
* Priority-level filters
* Customers with the highest revenue at risk
* Feature importance
* Business recommendations

### Dashboard Preview

Add a screenshot of the running dashboard to `OUTPUTS/figures/` and display it here:

```markdown
![Streamlit Dashboard](OUTPUTS/figures/dashboard_preview.png)
```

---

## Project Structure

```text
churn-prediction-revenue/
│
├── DATA/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── NOTEBOOKS/
│   └── churn_analysis.py
│
├── OUTPUTS/
│   ├── customers_risk_ranking.csv
│   ├── model_metrics.csv
│   ├── feature_importance.csv
│   │
│   └── figures/
│       ├── churn_distribution.png
│       ├── churn_by_contract.png
│       ├── monthly_charges_by_churn.png
│       ├── tenure_by_churn.png
│       ├── confusion_matrix_logistic.png
│       ├── churn_probability_distribution.png
│       ├── top10_revenue_at_risk.png
│       └── feature_importance.png
│
├── SRC/
│   └── app.py
│
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/churn-prediction-revenue.git
cd churn-prediction-revenue
```

Replace `your-username` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Running the Analysis

From the main project directory, run:

```bash
python NOTEBOOKS/churn_analysis.py
```

The analysis script performs:

* Data loading and validation
* Data cleaning and encoding
* Train-test splitting
* Logistic Regression training
* Random Forest training
* Model comparison
* Prediction-error analysis
* Churn probability estimation
* Financial risk calculation
* Customer prioritization
* Feature importance analysis
* Visualization generation
* Business output generation

---

## Launching the Dashboard

From the main project directory, run:

```bash
python -m streamlit run SRC/app.py
```

Streamlit should automatically open the dashboard in your browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## Main Outputs

| Output                               | Description                                            |
| ------------------------------------ | ------------------------------------------------------ |
| `OUTPUTS/model_metrics.csv`          | Performance metrics for the trained models             |
| `OUTPUTS/customers_risk_ranking.csv` | Customer-level churn risk and financial prioritization |
| `OUTPUTS/feature_importance.csv`     | Variables contributing to churn predictions            |
| `OUTPUTS/figures/`                   | Generated charts and model visualizations              |

---

## Technologies Used

* Python
* pandas
* NumPy
* Matplotlib
* scikit-learn
* Streamlit
* Joblib
* Jupyter

---

## Key Skills Demonstrated

This project demonstrates practical experience in:

* Data cleaning and preparation
* Exploratory data analysis
* Supervised machine learning
* Classification model evaluation
* Business-oriented metric selection
* Prediction-error analysis
* Feature importance interpretation
* Financial risk estimation
* Customer segmentation and prioritization
* Data visualization
* Interactive dashboard development
* Translating analytical results into business recommendations

---

## Potential Improvements

Future versions of the project could include:

* Hyperparameter optimization
* Cross-validation
* Class-imbalance techniques
* Decision-threshold optimization
* Cost-sensitive classification
* Additional models such as XGBoost or LightGBM
* Customer lifetime value estimation
* Retention campaign cost simulation
* Model explainability using SHAP
* Automated model retraining
* Cloud deployment of the Streamlit dashboard

---

## Conclusion

This project demonstrates how machine learning can be connected directly to business decision-making.

Rather than producing only a binary churn prediction, the solution:

* Estimates the probability that each customer will churn
* Calculates the annual financial value associated with each customer
* Measures probability-weighted revenue at risk
* Creates a prioritized list for retention actions
* Provides recommendations based on customer risk and value
* Makes the results accessible through an interactive dashboard

The final solution delivers both **predictive insight** and **financial decision support**, helping the company focus its retention efforts where they can generate the greatest business value.

---

## Author

**Tesnim Jebali**
Applied Mathematics Student — Data Science Specialization
