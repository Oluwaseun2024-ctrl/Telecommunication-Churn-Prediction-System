# CUSTOMER CHURN PREDICTION SYSTEM
This project builds a machine learning model to predict customer churn using behavioral, financial, and engagement features.  It identifies key churn drivers and provides insights to support targeted retention strategies and improve customer lifetime value.

## TABLE OF CONTENT
- Business objective
- Business Context
- Problem Statement
- Dataset Description
- Data Dictionary
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Encoding, Feature Selection, Train Test Split & Scaling
- Model Development and Model Evaluation/Performance
- Key Insights
- Business Recommendations
- Final Executive Summary
- Deployment
- How to Run the Project

## BUSINESS OBJECTIVE
The objective of this project is to build a robust churn prediction model that proactively identifies customers at risk of leaving. This enables the business to take data-driven retention actions, minimize revenue loss, and maximize customer lifetime value.

## BUSINESS CONTEXT
This project is situated within the telecommunications industry, where customer retention is a critical driver of profitability and long-term growth. Telecom companies operate in highly competitive markets with low switching costs, making customer churn a persistent challenge.

Customer churn directly impacts revenue, increases customer acquisition costs, and reduces overall customer lifetime value. Retaining existing customers is significantly more cost-effective than acquiring new ones, making churn prediction a high-priority business problem.

By solving this problem, the business can proactively identify at-risk customers, implement targeted retention strategies, and optimize marketing efforts. This leads to reduced revenue loss, improved customer satisfaction, and stronger competitive positioning.

## PROBLEM STATEMENT
This project focuses on predicting customer churn, specifically identifying whether a telecom customer is likely to leave the service or remain.

The target variable is **churn**, a binary outcome where:
- 1 indicates a customer who has churned
- 0 indicates a customer who has retained their service

This is a **supervised binary classification problem**, where the goal is to learn patterns from historical customer data to accurately classify future customers as churn or non-churn.

## DATASET DESCRIPTION
The dataset used in this project is a telecom customer churn dataset, designed to simulate real-world customer behaviour in a subscription-based service environment. It contains a mix of demographic, service usage, billing, and customer interaction features.

The dataset consists of **1,000 rows and 13 columns**, with each row representing an individual customer and each column capturing specific attributes such as tenure, monthly charges, service subscriptions, and customer satisfaction.

This dataset is structured to support churn prediction by providing both behavioural and transactional data, enabling the model to learn patterns associated with customer retention and attrition.

## DATA DICTIONARY
| Column | Description | Type |
|---|---|---|
| customer_id | Unique identifier assigned to each customer | String |
| tenure_months | Number of months the customer has remained subscribed | Integer |
| monthly_charges | Customer's monthly subscription fee | Float |
| total_charges | Total amount paid by the customer throughout the subscription period | Float |
| contract_type | Customer's subscription contract duration (Monthly, One-year, Two-year) | Categorical |
| internet_service | Type of internet service subscribed to (DSL, Fiber, None) | Categorical |
| tech_support | Indicates whether the customer has technical support service | Binary |
| streaming_service | Indicates whether the customer uses streaming services | Binary |
| payment_method | Customer's preferred payment channel | Categorical |
| paperless_billing | Indicates whether the customer uses paperless billing | Binary |
| satisfaction_score | Customer satisfaction rating on a scale of 1–5 | Integer |
| num_support_tickets | Number of support complaints raised by the customer | Integer |
| churn | Indicates whether the customer discontinued the service (target variable) | Binary |

## DATA CLEANING & PREPROCESSING
**1. Standardized Column Names**

Standardized column names by removing spaces, converting to lowercase, and replacing spaces with underscores for consistency. This ensures uniform formatting, making the dataset easier to work with and reducing errors during analysis and modelling.

```PYTHON
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

df.columns
```

**2. Columns Check**

Checked for missing values across all columns to assess data quality. Identified missing entries in total_charges and satisfaction_score, which require handling before modelling.

```PYTHON
#Checking Missing Values
df.isna().sum()
```

| Column | Missing Values |
|---|---|
| customer_id | 0 |
| tenure_months | 0 |
| monthly_charges | 0 |
| total_charges | 100 |
| contract_type | 0 |
| internet_service | 0 |
| tech_support | 0 |
| streaming_service | 0 |
| payment_method | 0 |
| paperless_billing | 0 |
| satisfaction_score | 100 |
| num_support_tickets | 0 |
| churn | 0 |

**3. Fix total_charges**

Cleaned and validated the total_charges column by handling blanks, converting to numeric, and treating negative values as missing based on business logic. Missing values were then imputed using the median to account for the skewed distribution and preserve data integrity.

```PYTHON
#Step A: Replace blanks
df['total_charges'] = df['total_charges'].replace(" ", np.nan)

#Step B: Convert to numeric
df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

#Step C: Investigate invalid values
df['total_charges'].describe()
```

| Statistic | Value |
|---|---:|
| Count | 4,900 |
| Mean | 2,681.93 |
| Standard Deviation | 1,855.74 |
| Minimum | -79.20 |
| 25th Percentile (Q1) | 1,157.52 |
| Median (Q2) | 2,360.55 |
| 75th Percentile (Q3) | 3,928.41 |
| Maximum | 8,380.40 |

```PYTHON
#Step D: Fix negatives (business logic)
#In telecom: Total charges should NEVER be negative
df.loc[df['total_charges'] < 0, 'total_charges'] = np.nan
```

What this means logically? “If total_charges is less than 0, treat it as missing data.”

In a telecom churn dataset: Total_charges represents money paid by a customer. Negative charges don’t make business sense (unless it's a refund system, which is rare and should be explicitly modelled)

So negative values are:
- Data entry errors
- ETL issues
- Corrupt records

Why convert to NaN instead of deleting? I am preserving the row but marking the value as invalid so I can later:
- Impute (mean, median, model-based)
- Drop selectively
- Investigate anomalies
Equivalent plain-English translation: “Find all customers with negative total charges and mark those values as missing because they are invalid.”

```PYTHON
#Checking Missing Values again
df.isna().sum()
```

| Column | Missing Values |
|---|---:|
| customer_id | 0 |
| tenure_months | 0 |
| monthly_charges | 0 |
| total_charges | 113 |
| contract_type | 0 |
| internet_service | 0 |
| tech_support | 0 |
| streaming_service | 0 |
| payment_method | 0 |
| paperless_billing | 0 |
| satisfaction_score | 100 |
| num_support_tickets | 0 |
| churn | 0 |

```PYTHON
#Handle missing values
df['total_charges'].hist()
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Total%20Charge%20Distribution.png)

```PYTHON
#Because the distribution above is skewed, I used median to replace the missing values
df['total_charges'].fillna(df['total_charges'].median(), inplace=True)
```

**4. Handled Missing satisfaction_score**

Handled missing values in satisfaction_score by imputing with the median, as it is an ordinal feature (1–5 scale). This ensures consistency while preserving the inherent ranking structure of the data.

```PYTHON
#Checking the distribution on satisfaction score
df['satisfaction_score'].hist()
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Distribution%20of%20Satisfaction%20Score.png)

```PYTHON
#Since it's ordinal (1–5):
#I Imputed with median
df['satisfaction_score'].fillna(df['satisfaction_score'].median(), inplace=True)
```

```PYTHON
#Checked missing values afters fixing them
df.isna().sum()
```

| Column | Missing Values |
|---|---:|
| customer_id | 0 |
| tenure_months | 0 |
| monthly_charges | 0 |
| total_charges | 0 |
| contract_type | 0 |
| internet_service | 0 |
| tech_support | 0 |
| streaming_service | 0 |
| payment_method | 0 |
| paperless_billing | 0 |
| satisfaction_score | 0 |
| num_support_tickets | 0 |
| churn | 0 |

**5. Fix Data Types**

Converted binary columns to integer format and categorical columns to appropriate category types. This ensures correct data representation and improves efficiency for analysis and model training.

```PYTHON
#Convert binary columns properly
binary_cols = ['tech_support', 'streaming_service', 'paperless_billing', 'churn']
for col in binary_cols:
    df[col] = df[col].astype(int)
```

```PYTHON
#Convert categorical columns
categorical_cols = ['contract_type', 'internet_service', 'payment_method']
for col in categorical_cols:
    df[col] = df[col].astype('category')
```

**6. Validate Data Integrity**

Validated data integrity by checking for duplicate records in the dataset. No duplicate entries were found, confirming data consistency.

```PYTHON
#Check duplicates
df.duplicated().sum()
#There were no duplicates
```

**7. Validate logical relationships**

Validated logical relationships by removing records with unrealistic values, such as zero or negative tenure and charges. This ensures the dataset aligns with real-world business rules and improves model reliability.

```PYTHON
#Check unrealistic values
# tenure should be >= 1
df = df[df['tenure_months'] >= 1]
# charges should be positive
df = df[df['monthly_charges'] > 0]
```

**8. Outlier Detection**
Performed outlier detection by visualizing the distribution of monthly_charges using a histogram. This helps identify extreme values and understand the overall spread of the data before modelling.

```PYTHON
import matplotlib.pyplot as plt
df['monthly_charges'].hist(bins=30)
plt.title("Monthly Charges Distribution")
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Monthly%20Charge%20Distribution.png)
