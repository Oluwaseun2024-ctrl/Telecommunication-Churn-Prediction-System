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

## EXPLORATORY DATA ANALYSIS (EDA)
**1. Target Variable Analysis (Churn)**

Analysed the distribution of the target variable (churn) to understand class balance. Results show that ~23% of customers churned while ~77% were retained, indicating a moderately imbalanced dataset. The dataset shows a noticeable churn rate (~23%), highlighting a meaningful business risk and the need for targeted retention strategies.

```PYTHON
df['churn'].value_counts(normalize=True)
```
| Churn | Proportion |
|---|---:|
| 0 | 0.7694 |
| 1 | 0.2306 |

```PYTHON
import matplotlib.pyplot as plt
df['churn'].value_counts().plot(kind='bar')
plt.title("Churn Distribution")
plt.xticks(rotation=0)
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Churn%20Distribution.png)

**2. Numerical Features vs Churn**

Analysed numerical features against churn to identify key behavioural differences between retained and churned customers. Churned customers tend to have shorter tenure, higher monthly charges, lower satisfaction scores, and more support tickets, indicating dissatisfaction and higher service friction. 

Customers with high monthly costs and frequent support interactions are significantly more likely to churn, making them critical targets for retention strategies.

```PYTHON
#Grouped statistics
df.groupby('churn')[[
    'tenure_months',
    'monthly_charges',
    'total_charges',
    'satisfaction_score',
    'num_support_tickets'
]].mean()
```
| Churn | Average Tenure (Months) | Average Monthly Charges | Average Total Charges | Average Satisfaction Score | Average Support Tickets |
|---|---:|---:|---:|---:|---:|
| 0 | 37.40 | 73.36 | 2703.48 | 3.17 | 1.36 |
| 1 | 30.70 | 84.64 | 2610.04 | 2.44 | 1.99 |

```PYTHON
#Visual distributions
import seaborn as sns
num_cols = [
    'tenure_months', 'monthly_charges',
    'total_charges', 'satisfaction_score',
    'num_support_tickets'
]
for col in num_cols:
    plt.figure()
    sns.boxplot(x='churn', y=col, data=df)
    plt.title(f"{col} vs Churn")
    plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Monthly%20Charges%20vs%20Churn.png)
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Num%20support%20tickets%20vs%20Churn.png)
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Satisfaction%20score%20vs%20Churn.png)
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Tenure%20vs%20Churn.png)
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Total%20Charges%20vs%20Churn.png)

**3. Categorical Features vs Churn**

Analysed categorical features against churn to uncover patterns in customer behavior. Customers on month-to-month contracts and fiber optic services show higher churn rates, while longer contracts show stronger retention; payment method has a weaker impact, with only slight increases in churn for manual payment options.
Contract type and service usage are the strongest categorical drivers of churn, while payment method plays a minimal role.

```PYTHON
#Contract Type
pd.crosstab(df['contract_type'], df['churn'], normalize='index')
```
| Contract Type | Churn = 0 | Churn = 1 |
|---|---:|---:|
| Month-to-month | 0.725437 | 0.274563 |
| One year | 0.829641 | 0.170359 |
| Two year | 0.841446 | 0.158554 |

```PYTHON
sns.countplot(x='contract_type', hue='churn', data=df)
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Contract%20type%20vs%20Churn.png)

```PYTHON
#Internet Service
pd.crosstab(df['internet_service'], df['churn'], normalize='index')
```
| Internet Service | Churn = 0 | Churn = 1 |
|---|---:|---:|
| DSL | 0.841470 | 0.158530 |
| Fiber optic | 0.694708 | 0.305292 |
| No | 0.864865 | 0.135135 |

```PYTHON
sns.countplot(x='internet_service', hue='churn', data=df)
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Internet%20service%20vs%20Churn.png)

```PYTHON
#Payment Method
pd.crosstab(df['payment_method'], df['churn'], normalize='index')
```
| Payment Method | Churn = 0 | Churn = 1 |
|---|---:|---:|
| Bank transfer | 0.777417 | 0.222583 |
| Credit card | 0.779715 | 0.220285 |
| Electronic check | 0.764419 | 0.235581 |
| Mailed check | 0.756270 | 0.243730 |

```PYTHON
sns.countplot(x='payment_method', hue='churn', data=df)
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Payment%20Method%20Vs%20Churn.png)

**4. Binary Features**

Analyzed binary features against churn to identify their impact on customer behavior. Customers without tech support show significantly higher churn, while streaming service and paperless billing have minimal influence.

Lack of tech support is a strong churn indicator, highlighting the importance of customer support in retention.

```PYTHON
binary_cols = ['tech_support', 'streaming_service', 'paperless_billing']
for col in binary_cols:
    print(pd.crosstab(df[col], df['churn'], normalize='index'))
```
| Tech Support | Churn = 0 | Churn = 1 |
|---|---:|---:|
| 0 | 0.718026 | 0.281974 |
| 1 | 0.845126 | 0.154874 |

| Streaming Service | Churn = 0 | Churn = 1 |
|---|---:|---:|
| 0 | 0.764128 | 0.235872 |
| 1 | 0.774433 | 0.225567 |

| Paperless Billing | Churn = 0 | Churn = 1 |
|---|---:|---:|
| 0 | 0.778135 | 0.221865 |
| 1 | 0.765457 | 0.234543 |

```PYTHON
#Visualizing the above
for col in binary_cols:
    sns.countplot(x=col, hue='churn', data=df)
    plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Paperless%20Billing%20vs%20Churn.png)
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Streaming%20services%20Vs%20Churn.png)
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Tech%20Supoort%20Vs%20Churn.png)

## FEATURE ENGINEERING
**1. Customer Lifetime Value (CLV proxy)**

Created a proxy for Customer Lifetime Value (CLV) by calculating the average monthly value per customer using total charges and tenure. This feature helps capture customer value over time and can improve churn prediction by highlighting high- vs low-value customers.

```PYTHON
df['avg_monthly_value'] = df['total_charges'] / df['tenure_months']
```

**2. Tenure Segmentation**

Segmented customers into tenure groups (new, mid, long) based on their subscription duration. This helps capture customer lifecycle stages and improves the model’s ability to detect churn patterns across different tenure levels.

```PYTHON
def tenure_group(x):
    if x <= 12:
        return 'new'
    elif x <= 36:
        return 'mid'
    else:
        return 'long'
df['tenure_group'] = df['tenure_months'].apply(tenure_group)
```

**3. High Risk Customer Flag**

Created a high-risk customer flag by identifying customers with low satisfaction scores and high support ticket counts. This feature helps the model quickly capture customers with strong churn signals based on negative experience indicators.

```PYTHON
df['high_risk'] = (
    (df['satisfaction_score'] <= 2) &
    (df['num_support_tickets'] >= 3)
).astype(int)
```

**4. Charge Intensity**

Created a charge intensity feature by measuring total charges relative to customer support interactions. This helps capture how much value a customer generates per issue raised, providing insight into cost vs service experience.

```PYTHON
df['charges_per_ticket'] = df['total_charges'] / (df['num_support_tickets'] + 1)
```

**5. Engagement Score (Composite Feature)**

Created an engagement score by combining tenure, satisfaction, and support interactions into a single composite metric. This feature captures overall customer engagement, helping the model distinguish between highly engaged and at-risk customers.

```PYTHON
df['engagement_score'] = (
    df['tenure_months'] * 0.4 +
    df['satisfaction_score'] * 0.3 -
    df['num_support_tickets'] * 0.3
)
```

## ENCODING, FEATURE SELECTION, TRAIN TEST SPLIT & SCALING

**Encoding Categorical Variables**

Converted categorical variables into numerical format using One-Hot Encoding to make them suitable for machine learning algorithms. Most models cannot interpret non-numeric data directly, so categorical features such as contract type, internet service, payment method, and tenure group were transformed into binary indicator variables. 

The **drop_first=True** parameter was applied to avoid the dummy variable trap (multicollinearity), ensuring that redundant features are removed and the model remains stable and interpretable.

```PYTHON
# One-Hot Encoding
df = pd.get_dummies(df, columns=[
    'contract_type',
    'internet_service',
    'payment_method',
    'tenure_group'
], drop_first=True)
```

**Feature Selection Prep**

Prepared the dataset for modeling by removing non-informative features and defining input and target variables. The **customer_id** column was dropped as it serves only as a unique identifier and does not contribute to predicting customer churn. Including such features can introduce noise and negatively impact model performance.

The dataset was then split into features (X) and target (y), where X contains all independent variables and y represents the dependent variable (churn). This separation is essential for training machine learning models, as it clearly defines the inputs used for prediction and the outcome to be predicted.

```PYTHON
# Dropped unnecessary columns
df = df.drop(columns=['customer_id'])

# Define X and y
x = df.drop('churn', axis=1)
y = df['churn']
```

**Train-Test Split**

Split the dataset into training and testing sets to evaluate model performance on unseen data. An 80/20 split was used, where 80% of the data was allocated for training and 20% for testing. This ensures sufficient data for model learning while retaining a representative portion for evaluation.

The stratify=y parameter was applied to preserve the original distribution of the target variable (churn) in both training and testing sets. This is particularly important for classification problems to prevent class imbalance from skewing results.

A fixed random_state=42 was used to ensure reproducibility, allowing consistent results across multiple runs.

After splitting, the training feature set (x_train) was inspected to confirm that all engineered and encoded features were correctly included, ensuring the dataset is properly structured for model training.

```PYTHON
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

**Inspected columns in x_train**
- tenure_months
- monthly_charges
- total_charges
- tech_support
- streaming_service
- paperless_billing
- satisfaction_score
- num_support_tickets
- avg_monthly_value
- high_risk
- charges_per_ticket
- engagement_score
- contract_type_One year
- contract_type_Two year
- internet_service_Fiber optic
- internet_service_No
- payment_method_Credit card
- payment_method_Electronic check
- payment_method_Mailed check
- tenure_group_mid
- tenure_group_new

**Scaling**

Standardized numerical features to ensure consistent scale across all input variables. Selected continuous variables such as tenure, charges, satisfaction score, and engineered features were scaled using **StandardScaler**, which transforms data to have a mean of 0 and a standard deviation of 1. 

The scaler was fitted only on the training data and then applied to both training and testing sets. This prevents data leakage, ensuring that information from the test set does not influence the training process.

Feature scaling is particularly important for models that rely on distance or gradient-based optimization, as it ensures that no single feature disproportionately influences the model due to its scale.

```PYTHON
# Columns to scale
num_cols = [
    'tenure_months',
    'monthly_charges',
    'total_charges',
    'satisfaction_score',
    'num_support_tickets',
    'avg_monthly_value',
    'charges_per_ticket',
    'engagement_score'
]

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])
```

## MODEL DEVELOPMENT AND EVALAUTION
Developed and evaluated models with consideration for class imbalance in the target variable.

The dataset exhibits an imbalanced distribution of approximately 77% non-churn and 23% churn cases. In such scenarios, accuracy alone can be misleading, as a model may achieve high accuracy by simply predicting the majority class.

To address this, greater emphasis was placed on evaluation metrics that better capture model performance for the minority class (churn). In particular, recall for class 1 (churn) was prioritized, as it measures the model’s ability to correctly identify customers who are likely to churn.

Additional metrics including precision, F1-score, and ROC-AUC were also used to provide a balanced assessment of performance. Precision evaluates the correctness of churn predictions, F1-score balances precision and recall, and ROC-AUC measures the model’s ability to distinguish between classes across different thresholds.

This multi-metric evaluation approach ensures a more reliable and business-relevant assessment of model effectiveness, especially in identifying at-risk customers.

**Logistic Regression (Baseline Model)**

Evaluated the Logistic Regression model using multiple performance metrics to account for class imbalance.

The model achieved an accuracy of 71%, indicating moderate overall performance. However, given the imbalanced nature of the dataset, greater emphasis was placed on recall and F1-score for the churn class (class 1).

The model demonstrated a strong recall of 71% for churn, meaning it successfully identified a large proportion of customers who actually churned. This is valuable from a business perspective, as capturing potential churners is a priority.

However, precision for the churn class was relatively low at 42%, indicating a higher number of false positives, where non-churning customers were incorrectly classified as churners. This trade-off suggests the model is more aggressive in predicting churn.

The F1-score of 0.53 reflects a balance between precision and recall, while the ROC-AUC score of 0.79 indicates good overall ability to distinguish between churn and non-churn customers.

The confusion matrix further highlights this behavior, showing that while many churn cases were correctly identified, there is a notable number of false positives.

Overall, the Logistic Regression model performs reasonably well, particularly in identifying churners, but could benefit from further tuning or imbalance handling techniques to improve precision without significantly reducing recall.

```PYTHON
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
log_model.fit(x_train, y_train)
```

**Evaluating Logistic Model**

```PYTHON
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    f1_score
)

# Predictions
y_pred_log = log_model.predict(x_test)
y_prob_log = log_model.predict_proba(x_test)[:, 1]

print("===== LOGISTIC REGRESSION EVALUATION =====\n")
# Accuracy
acc = accuracy_score(y_test, y_pred_log)
print(f"Accuracy: {acc:.4f}")

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob_log)
print(f"ROC-AUC: {roc_auc:.4f}")

# F1 Score
f1 = f1_score(y_test, y_pred_log)
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_log)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_log))
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Logistics%20Regression.png)

**Random Forest Model**

Evaluated the Random Forest model using multiple performance metrics with attention to class imbalance.

The model achieved an accuracy of 72.4%, slightly higher than Logistic Regression. However, accuracy alone is not sufficient due to the imbalanced dataset.

For the churn class (class 1), the model showed a recall of 46%, which is significantly lower than Logistic Regression. This indicates that the Random Forest model failed to identify a substantial portion of actual churners, which is a critical limitation in this context.

Precision for the churn class was 41%, similar to Logistic Regression, indicating that when the model predicts churn, it is correct less than half the time.

The F1-score of 0.44 reflects a weaker balance between precision and recall compared to Logistic Regression. Additionally, the ROC-AUC score of 0.76 suggests a lower overall ability to distinguish between churn and non-churn customers.

The confusion matrix shows fewer false positives compared to Logistic Regression, but at the cost of a higher number of false negatives, meaning more churners were missed.

Overall, while Random Forest provides slightly better overall accuracy, it underperforms in identifying churners. Given the business objective of minimizing missed churn cases, this model is less suitable without further tuning or imbalance handling.

```PYTHON
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(x_train, y_train)
```

**Evaluate Random Forest**

```PYTHON
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    f1_score
)

# Predictions
y_pred_rf = rf_model.predict(x_test)
y_prob_rf = rf_model.predict_proba(x_test)[:, 1]

print("===== RANDOM FOREST EVALUATION =====\n")

# Accuracy
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {acc_rf:.4f}")

# ROC-AUC
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"ROC-AUC: {roc_auc_rf:.4f}")

# F1 Score
f1_rf = f1_score(y_test, y_pred_rf)
print(f"F1 Score: {f1_rf:.4f}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:\n", cm_rf)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf))
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/Random%20Forest.png)

**XGBoost (High-Performance Model)**

Evaluated the XGBoost model using multiple performance metrics with consideration for class imbalance.

The model achieved an accuracy of 71.3%, which is comparable to Logistic Regression but slightly lower than Random Forest.

For the churn class (class 1), the model achieved a recall of 56%, indicating a moderate ability to identify customers who churn. This performance is better than Random Forest but still lower than Logistic Regression.

Precision for the churn class was 41%, similar to the other models, suggesting that false positives remain a challenge.

The F1-score of 0.47 indicates a more balanced performance between precision and recall compared to Random Forest, though still below Logistic Regression. The ROC-AUC score of 0.76 reflects a reasonable but not outstanding ability to distinguish between churn and non-churn customers.

The confusion matrix shows a balanced trade-off between false positives and false negatives, performing better than Random Forest in detecting churners but not as effectively as Logistic Regression.

Overall, XGBoost provides a middle-ground performance, improving recall over Random Forest while maintaining a more balanced classification. However, it does not outperform Logistic Regression in identifying churners, which remains the primary business objective.

```PYTHON
#Compute imbalance ratio
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(x_train, y_train)
```

**Evaluate XGBoost**

```PYTHON
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    f1_score
)

# Predictions
y_pred_xgb = xgb_model.predict(x_test)
y_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]

print("===== XGBOOST EVALUATION =====\n")

# Accuracy
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {acc_xgb:.4f}")

# ROC-AUC
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
print(f"ROC-AUC: {roc_auc_xgb:.4f}")

# F1 Score
f1_xgb = f1_score(y_test, y_pred_xgb)
print(f"F1 Score: {f1_xgb:.4f}")

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\nConfusion Matrix:\n", cm_xgb)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_xgb))
```
![](https://github.com/Oluwaseun2024-ctrl/Telecommunication-Churn-Prediction-System/blob/main/XGBoost.png)

**MODEL EVALUATION**

**Model Results**

**ROC-AUC**
- Logistic: 0.7939 ✅ (Best)
- Random Forest: 0.7597
- XGBoost: 0.7544

**PR-AUC (Imbalance-sensitive)**
- Logistic: 0.5215 ✅ (Best)
- Random Forest: 0.4980
- XGBoost: 0.4552

**Key Observation**

A key observation from the model evaluation is that the simplest model, Logistic Regression, outperformed more complex models such as Random Forest and XGBoost. This is not an anomaly but provides important insight into the structure of the data.

One reason for this outcome is that the dataset appears to exhibit near-linear separability. Core features such as tenure, satisfaction score, and monthly charges show strong linear relationships with customer churn, making them well-suited for a linear model.

Additionally, feature engineering played a significant role in enhancing model performance. Engineered features such as engagement score, high-risk flag, and charges per ticket effectively capture underlying patterns in a linear form, reducing the need for complex nonlinear modeling.

In contrast, tree-based models like Random Forest and XGBoost are designed to capture complex interactions and nonlinearities. In this case, their added complexity may not be necessary and can lead to slight overfitting, where the model learns noise rather than meaningful patterns.

Overall, this result indicates that the problem is relatively well-structured and can be effectively addressed using a simpler, more interpretable model. Logistic Regression not only performs better in identifying churners but also offers greater transparency, making it more suitable for business decision-making.

**Model Performance — Business Meaning**

**ROC-AUC ≈ 0.79**

Model can distinguish churn vs non-churn ~79% of the time

**PR-AUC ≈ 0.52**

Strong performance given imbalance

Interpretation: “The model is reliable enough to drive targeted retention campaigns.”


## KEY INSIGHTS
**1. What my model Actually Learned.**

The Logistic Regression model provides interpretable insights into the key drivers of customer churn. Based on the learned relationships between features and the target variable, the model identifies clear patterns that distinguish churners from retained customers.

Customers are more likely to churn when they exhibit characteristics such as low tenure (new customers), low satisfaction scores, a high number of support tickets, higher monthly charges, and lower overall engagement. These factors indicate dissatisfaction, higher perceived cost, and weaker attachment to the service.

Conversely, customers are more likely to remain when they have longer tenure, higher satisfaction levels, fewer support issues, and more stable engagement with the service. These attributes reflect stronger customer relationships and a more positive overall experience.

These insights align well with business intuition, reinforcing the validity of the model and highlighting actionable areas such as improving customer satisfaction, reducing service issues, and strengthening early-stage customer engagement to reduce churn.

**2. Most Important Drivers of Churn (From Features)**

Identified key drivers of customer churn using engineered features that capture behavioral and experiential signals.

The **high_risk** feature emerged as the strongest indicator of churn. It flags customers with very low satisfaction scores (≤ 2) combined with a high number of support tickets (≥ 3). This combination reflects customers who are both dissatisfied and repeatedly experiencing issues, making them highly likely to churn.

The **engagement_score** serves as a composite metric combining tenure, satisfaction, and support interactions. Lower engagement scores indicate weaker relationships with the company, signaling a higher likelihood of churn.

The **charges_per_ticket** feature captures the “cost-to-frustration” ratio by measuring how much a customer pays relative to how often they require support. Customers who incur high charges while frequently needing assistance are more likely to feel dissatisfied, increasing churn risk.

The **avg_monthly_value** acts as a proxy for perceived value versus cost. Customers with higher spending levels tend to have higher expectations; if these expectations are not met through service quality or experience, they are more prone to churn.

Overall, these engineered features provide meaningful, interpretable signals that enhance the model’s ability to detect at-risk customers and offer actionable insights for improving retention strategies.

**3. Customer Segments Discovered**

Segmented customers into distinct groups based on behavioral patterns and churn risk, enabling targeted retention strategies.

**Segment 1: New and Frustrated Users (Highest Risk)**

Customers in this segment have low tenure (≤ 12 months), high support ticket frequency, and low satisfaction scores. These users are in the early stages of their journey but are already experiencing dissatisfaction, making them the most likely to churn. Immediate intervention is critical for retention.

**Segment 2: High-Paying but Unsatisfied Customers**

These customers have high monthly charges combined with low satisfaction levels and moderate tenure. Despite their financial value, unmet expectations increase their likelihood of churn. This segment represents a high business risk due to potential revenue loss.

**Segment 3: Silent Risk Customers**

Customers in this group have moderate tenure and declining engagement scores but do not actively report many issues. Their dissatisfaction may not be immediately visible, making them a hidden churn risk. Proactive monitoring and engagement are necessary to prevent future churn.

**Segment 4: Loyal Customers (Low Risk)**

This segment includes customers with long tenure, high satisfaction, and minimal support interactions. They demonstrate strong engagement and stability, making them least likely to churn. These customers are valuable for long-term retention and potential upselling opportunities.

Overall, this segmentation provides a structured understanding of customer behavior, allowing the business to tailor retention strategies based on risk level and customer profile.


## BUSINESS RECOMMENDATIONS

Developed targeted business strategies based on identified churn drivers and customer segments to improve retention and customer satisfaction.

**Action 1: Early Retention Program**

Focused on customers with tenure of less than 12 months, who are at the highest risk of early churn. Strategies include enhancing the onboarding experience, providing welcome offers, and conducting proactive check-ins to ensure a smooth initial customer journey and build early engagement.

**Action 2: Support Escalation System**

Triggered when customers generate three or more support tickets, indicating repeated issues. These customers should be automatically flagged and prioritized for faster resolution. Providing dedicated support and compensation where necessary can help reduce frustration and prevent churn.

**Action 3: Satisfaction Recovery Program**

Targets customers with very low satisfaction scores (≤ 2). This involves conducting follow-up surveys, initiating direct customer callbacks, and offering personalized solutions or loyalty incentives to rebuild trust and improve customer experience.

**Action 4: High-Value Customer Protection**

Focuses on customers with high monthly charges who represent significant revenue. These customers should receive premium support services and, where possible, dedicated account management to ensure their expectations are consistently met.

Overall, these actions align closely with model insights and customer segmentation, enabling a proactive and data-driven approach to churn reduction while maximizing customer lifetime value.


## FINAL EXECUTIVE SUMMARY
Addressed the business problem of customer churn, which leads to significant revenue loss and increased customer acquisition costs.

A machine learning solution was developed to predict churn using a combination of behavioral, financial, and engagement-based features. The model enables early identification of at-risk customers, allowing for proactive intervention.

Logistic Regression emerged as the best-performing model, achieving a ROC-AUC score of 0.79 and a PR-AUC of 0.52, demonstrating strong effectiveness in handling imbalanced data and accurately identifying churners.

Key drivers of churn were identified as low customer satisfaction, high support ticket volume, low tenure, and poor overall engagement. These factors highlight critical areas where customer experience improvements can reduce churn risk.

From a business perspective, the model provides actionable value by enabling early detection of high-risk customers, supporting targeted retention strategies, and ultimately reducing churn while improving customer lifetime value.

Overall, this project demonstrates how data-driven insights and predictive modeling can be leveraged to enhance customer retention and drive long-term business performance.


## DEPLOYMENT
Deployed the trained churn prediction model using a Streamlit web application to enable interactive and user-friendly access.

The deployment integrates the trained Logistic Regression model with preprocessing components, including feature scaling and encoding, ensuring consistency between training and real-time predictions.

Users can input customer details through the interface, and the application processes the data, applies the same transformations used during model training, and generates a churn prediction along with the associated probability.

This deployment allows non-technical stakeholders to easily assess customer churn risk in real time, supporting data-driven decision-making and proactive retention strategies.

Overall, the Streamlit application serves as a lightweight and effective solution for operationalizing the machine learning model in a business environment.

```PYTHON
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD FILES
# =========================
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction")

# =========================
# USER INPUT
# =========================
st.sidebar.header("Customer Input")

tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 1.0, 200.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges", 1.0, 10000.0, 500.0)
satisfaction = st.sidebar.slider("Satisfaction Score", 1, 5, 3)
tickets = st.sidebar.slider("Support Tickets", 0, 20, 2)

tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
streaming = st.sidebar.selectbox("Streaming Service", ["Yes", "No"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

contract = st.sidebar.selectbox("Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox("Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.sidebar.selectbox("Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# =========================
# CREATE DATAFRAME
# =========================
df = pd.DataFrame([{
    "tenure_months": tenure,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "satisfaction_score": satisfaction,
    "num_support_tickets": tickets,
    "tech_support": 1 if tech_support == "Yes" else 0,
    "streaming_service": 1 if streaming == "Yes" else 0,
    "paperless_billing": 1 if paperless == "Yes" else 0,
    "contract_type": contract,
    "internet_service": internet,
    "payment_method": payment
}])

# =========================
# FEATURE ENGINEERING (MATCH NOTEBOOK)
# =========================

# Avg monthly value
df["avg_monthly_value"] = df["total_charges"] / df["tenure_months"]

# Charges per ticket (NOTE: uses total_charges, not monthly)
df["charges_per_ticket"] = df["total_charges"] / (df["num_support_tickets"] + 1)

# Engagement score (exact formula)
df["engagement_score"] = (
    df["tenure_months"] * 0.4 +
    df["satisfaction_score"] * 0.3 -
    df["num_support_tickets"] * 0.3
)

# High risk (IMPORTANT: threshold is 3, not 5)
df["high_risk"] = (
    (df["satisfaction_score"] <= 2) &
    (df["num_support_tickets"] >= 3)
).astype(int)

# Tenure group (exact labels)
def tenure_group(x):
    if x <= 12:
        return "new"
    elif x <= 36:
        return "mid"
    else:
        return "long"

df["tenure_group"] = df["tenure_months"].apply(tenure_group)

# =========================
# ONE-HOT ENCODING (EXACT)
# =========================
df = pd.get_dummies(df, columns=[
    "contract_type",
    "internet_service",
    "payment_method",
    "tenure_group"
], drop_first=True)

# =========================
# ALIGN COLUMNS
# =========================
for col in feature_columns:
    if col not in df:
        df[col] = 0

df = df[feature_columns]

# =========================
# SCALE NUMERICAL ONLY
# =========================
num_cols = [
    'tenure_months',
    'monthly_charges',
    'total_charges',
    'satisfaction_score',
    'num_support_tickets',
    'avg_monthly_value',
    'charges_per_ticket',
    'engagement_score'
]

df_num = df[num_cols]
df_cat = df.drop(columns=num_cols)

df_num_scaled = scaler.transform(df_num)
df_num_scaled = pd.DataFrame(df_num_scaled, columns=num_cols)

df_final = pd.concat([df_num_scaled, df_cat.reset_index(drop=True)], axis=1)
df_final = df_final[feature_columns]

# =========================
# PREDICT
# =========================
prediction = model.predict(df_final)[0]
probability = model.predict_proba(df_final)[0][1]

# =========================
# OUTPUT
# =========================
st.subheader("Prediction Result")

if prediction == 1:
    st.error("⚠️ Customer WILL churn")
else:
    st.success("✅ Customer will NOT churn")

st.write(f"### Churn Probability: {probability:.2%}")

# =========================
# BUSINESS INTERPRETATION
# =========================
if probability > 0.7:
    st.warning("🔴 High Risk — Immediate action needed")
elif probability > 0.4:
    st.info("🟠 Medium Risk — Monitor closely")
else:
    st.success("🟢 Low Risk — Stable customer")
```

## HOW TO RUN THE PROJECT
The Telecommunication Customer Churn Prediction System is deployed using Streamlit Cloud and can be accessed directly through the application link below:

**Application Link:**

```
telecommunication-churn-prediction-system.streamlit.app
```

Users can open the link in any web browser to interact with the churn prediction application. No local installation or additional setup is required.

The application allows users to input customer information and receive a churn prediction result generated by the trained machine learning model.
