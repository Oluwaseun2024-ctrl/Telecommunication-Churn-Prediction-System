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