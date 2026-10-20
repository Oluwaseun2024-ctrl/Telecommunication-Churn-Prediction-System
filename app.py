import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 2. FILE PATHS AND MODEL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

THRESHOLD = 0.45

NUM_COLS = [
    "monthly_charges",
    "total_charges",
    "satisfaction_score",
    "avg_monthly_value",
    "num_support_tickets",
    "charges_per_ticket",
    "engagement_score",
    "tenure_months"
]

ENGINEERED_FEATURES = [
    "avg_monthly_value",
    "charges_per_ticket",
    "engagement_score",
    "high_risk"
]

CATEGORICAL_PREFIXES = [
    "contract_type",
    "internet_service",
    "payment_method",
    "tenure_group"
]


# ============================================================
# 3. LOAD THE SAVED ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, scaler, feature_columns


st.title("📊 Customer Churn Prediction System")

st.write(
    "Enter a customer's information to estimate the probability "
    "that the customer will churn."
)

st.caption(
    "Model: Logistic Regression | "
    f"Classification threshold: {THRESHOLD:.2f}"
)

try:
    model, scaler, feature_columns = load_artifacts()

except FileNotFoundError as error:
    st.error(
        f"Required model file not found: {error.filename}. "
        "Place all three model files in the same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load model artifacts: {error}")
    st.stop()


feature_columns = list(feature_columns)

if not hasattr(model, "predict_proba"):
    st.error("The saved model does not support predict_proba().")
    st.stop()

if not hasattr(scaler, "n_features_in_"):
    st.error("The saved scaler has not been fitted.")
    st.stop()

if len(NUM_COLS) != scaler.n_features_in_:
    st.error(
        "NUM_COLS does not match the number of columns "
        "used to fit the saved scaler."
    )
    st.stop()

missing_numeric_columns = [
    col for col in NUM_COLS if col not in feature_columns
]

if missing_numeric_columns:
    st.error(
        f"Missing numerical model columns: {missing_numeric_columns}"
    )
    st.stop()


# ============================================================
# 4. IDENTIFY CATEGORICAL OPTIONS
# ============================================================

def get_category_options(prefix):
    options = []

    for column in feature_columns:
        if column.startswith(prefix + "_"):
            category = column[len(prefix) + 1:]
            options.append(category)

    return options


category_options = {
    prefix: get_category_options(prefix)
    for prefix in CATEGORICAL_PREFIXES
}

missing_categories = [
    prefix for prefix, options in category_options.items()
    if not options
]

if missing_categories:
    st.error(
        "Could not identify categories for: "
        f"{missing_categories}. Check feature_columns.pkl."
    )
    st.stop()


# ============================================================
# 5. IDENTIFY ADDITIONAL FEATURES
# ============================================================

def is_encoded_column(column):
    return any(
        column.startswith(prefix + "_")
        for prefix in CATEGORICAL_PREFIXES
    )


handled_columns = (
    set(NUM_COLS)
    | set(ENGINEERED_FEATURES)
    | {
        f"{prefix}_{category}"
        for prefix, options in category_options.items()
        for category in options
    }
)

# These include the original binary columns, such as
# tech_support, streaming_service, and paperless_billing.
extra_features = [
    col for col in feature_columns
    if col not in handled_columns
    and not is_encoded_column(col)
    and col not in ["customer_id", "churn"]
]


# ============================================================
# 6. CUSTOMER INPUT FORM
# ============================================================

st.subheader("Customer Information")

with st.form("churn_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        tenure_months = st.number_input(
            "Tenure (months)",
            min_value=1,
            max_value=120,
            value=12,
            step=1
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.01,
            value=75.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=900.0,
            step=10.0
        )

        satisfaction_score = st.number_input(
            "Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        )

    with col2:
        num_support_tickets = st.number_input(
            "Number of Support Tickets",
            min_value=0,
            max_value=1000,
            value=2,
            step=1
        )

        contract_type = st.selectbox(
            "Contract Type",
            options=category_options["contract_type"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            options=category_options["internet_service"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            options=category_options["payment_method"]
        )

    # --------------------------------------------------------
    # BINARY FEATURES: YES = 1, NO = 0
    # --------------------------------------------------------

    st.markdown("### Additional Customer Services")

    binary_features = [
        "tech_support",
        "streaming_service",
        "paperless_billing"
    ]

    binary_values = {}

    binary_labels = {
        "tech_support": "Tech Support",
        "streaming_service": "Streaming Service",
        "paperless_billing": "Paperless Billing"
    }

    for feature in binary_features:

        # Only display features that exist in the trained model.
        if feature in feature_columns:

            answer = st.selectbox(
                binary_labels[feature],
                options=["Yes", "No"],
                key=feature
            )

            binary_values[feature] = (
                1 if answer == "Yes" else 0
            )

    # --------------------------------------------------------
    # OTHER FEATURES PRESENT IN THE MODEL
    # --------------------------------------------------------

    extra_values = {}

    remaining_features = [
        feature for feature in extra_features
        if feature not in binary_features
    ]

    if remaining_features:

        st.markdown("### Other Model Features")

        st.caption(
            "These additional predictors are present in your "
            "saved model. Enter their values using the same "
            "encoding and definitions as the training dataset."
        )

        extra_columns = st.columns(2)

        for index, feature in enumerate(remaining_features):

            with extra_columns[index % 2]:

                extra_values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=0.0,
                    key=f"extra_{feature}"
                )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

if submitted:

    avg_monthly_value = (
        total_charges / tenure_months
    )

    charges_per_ticket = (
        monthly_charges / (num_support_tickets + 1)
    )

    engagement_score = (
        tenure_months * 0.4
        + satisfaction_score * 0.3
        - num_support_tickets * 0.3
    )

    high_risk = int(
        satisfaction_score <= 2
        and num_support_tickets >= 3
    )

    if tenure_months <= 12:
        tenure_group = "new"
    elif tenure_months <= 36:
        tenure_group = "mid"
    else:
        tenure_group = "old"


    # ========================================================
    # 8. CREATE MODEL INPUT
    # ========================================================

    input_df = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    input_values = {
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "satisfaction_score": satisfaction_score,
        "num_support_tickets": num_support_tickets,
        "tenure_months": tenure_months,
        "avg_monthly_value": avg_monthly_value,
        "charges_per_ticket": charges_per_ticket,
        "engagement_score": engagement_score,
        "high_risk": high_risk
    }

    # Add original numerical and engineered features.
    for column, value in input_values.items():
        if column in input_df.columns:
            input_df.loc[0, column] = value

    # Add binary features directly as 1 or 0.
    # Do not one-hot encode these columns.
    for column, value in binary_values.items():
        input_df.loc[0, column] = value

    # Add any other original predictors.
    for column, value in extra_values.items():
        if column in input_df.columns:
            input_df.loc[0, column] = value

    # Set the selected one-hot encoded categories.
    selected_categories = {
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "tenure_group": tenure_group
    }

    for prefix, category in selected_categories.items():

        encoded_column = f"{prefix}_{category}"

        if encoded_column not in input_df.columns:
            st.error(
                f"Expected feature '{encoded_column}' was not found "
                "in feature_columns.pkl."
            )
            st.stop()

        input_df.loc[0, encoded_column] = 1


    # ========================================================
    # 9. SCALE NUMERICAL FEATURES ONLY
    # ========================================================

    try:
        input_df.loc[:, NUM_COLS] = scaler.transform(
            input_df[NUM_COLS]
        )

    except Exception as error:
        st.error(f"Scaling failed: {error}")
        st.stop()

    # Preserve the exact feature order used during training.
    input_df = input_df[feature_columns]


    # ========================================================
    # 10. PREDICTION
    # ========================================================

    try:
        probabilities = model.predict_proba(input_df)[0]

        class_labels = list(model.classes_)

        if 1 not in class_labels:
            st.error(
                "Class 1 was not found in the model. "
                "Check your churn target encoding."
            )
            st.stop()

        churn_index = class_labels.index(1)

        churn_probability = float(
            probabilities[churn_index]
        )

        prediction = int(
            churn_probability >= THRESHOLD
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()


    # ========================================================
    # 11. DISPLAY RESULTS
    # ========================================================

    st.divider()
    st.subheader("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

    with result_col2:
        st.metric(
            "Classification Threshold",
            f"{THRESHOLD:.0%}"
        )

    st.progress(churn_probability)

    if prediction == 1:

        st.error(
            "⚠️ High Churn Risk: The model predicts that "
            "this customer is likely to churn."
        )

        st.write(
            "Recommended action: Review customer satisfaction, "
            "support history, contract type, and service experience. "
            "Consider a targeted retention intervention."
        )

    else:

        st.success(
            "✅ Lower Churn Risk: The customer is not classified "
            "as likely to churn at the selected threshold."
        )

        st.write(
            "Recommended action: Continue monitoring customer "
            "engagement and satisfaction."
        )

    st.caption(
        "This is a model-based estimate, not a guarantee of "
        "future customer behaviour."
    )

    # --------------------------------------------------------
    # ENGINEERED FEATURES
    # --------------------------------------------------------

    with st.expander("View Engineered Features"):

        engineered_df = pd.DataFrame({
            "Feature": [
                "Average Monthly Value",
                "Charges Per Ticket",
                "Engagement Score",
                "High Risk Flag",
                "Tenure Group"
            ],
            "Value": [
                round(avg_monthly_value, 3),
                round(charges_per_ticket, 3),
                round(engagement_score, 3),
                high_risk,
                tenure_group
            ]
        })

        st.dataframe(
            engineered_df,
            hide_index=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PROCESSED MODEL INPUT
    # --------------------------------------------------------

    with st.expander("View Processed Model Input"):

        st.dataframe(
            input_df,
            hide_index=True,
            use_container_width=True
        )