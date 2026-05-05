import streamlit as st
import pandas as pd
import joblib

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Income Prediction System",
    page_icon="💼",
    layout="centered"
)

# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
bundle = joblib.load("xgb_pipeline_v2.pkl")
model = bundle['model']   # your original pipeline
threshold = bundle['threshold']

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("💼 Income Prediction System")
st.markdown("Predict whether a person's income exceeds **$50K/year**")

st.markdown("---")

# ─────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────
st.subheader("📋 Enter Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    education_years = st.slider("Education Years", 1, 16, 10)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)

with col2:
    weekly_hours = st.slider("Weekly Work Hours", 1, 100, 40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status",
                                 ["Married", "Single", "Divorced", "Widowed"])
    employment_type = st.selectbox("Employment Type",
                                  ["Private", "Self-emp", "Government"])

job_role = st.selectbox("Job Role", [
    "Executive_Managerial",
    "Professional_Specialized",
    "Sales_Business_Development",
    "Administrative_Clerical",
    "Skilled_Trades_Repair",
    "General_Service"
])

relationship = st.selectbox("Relationship",
                           ["Husband", "Wife", "Own-child", "Not-in-family"])

race = st.selectbox("Race",
                    ["White", "Black", "Asian-Pac-Islander", "Other"])

country = st.selectbox("Country",
                       ["United-States", "India", "Other"])

education_group = st.selectbox("Education Group",
                              ["High_School", "College", "Undergraduate", "Postgraduate"])

# ─────────────────────────────────────────
# Convert inputs
# ─────────────────────────────────────────
gender = 1 if gender == "Male" else 0

input_df = pd.DataFrame([{
    'Age': age,
    'Employment_Type': employment_type,
    'Education_Years': education_years,
    'Marital_Status': marital_status,
    'Job_Role': job_role,
    'Relationship': relationship,
    'Race': race,
    'Gender': gender,
    'Capital_Gain': capital_gain,
    'Capital_Loss': capital_loss,
    'Weekly_Work_Hours': weekly_hours,
    'Country': country,
    'Education_Group': education_group
}])

# ─────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────
if st.button("🚀 Predict Income"):

    proba = model.predict_proba(input_df)[0][1]
    prediction = ">50K" if proba >= threshold else "<=50K"

    st.markdown("---")

    # Result
    if prediction == ">50K":
        st.success("💰 High Income (>50K)")
    else:
        st.warning("📉 Lower Income (<=50K)")

    # Better probability display (fix for 1.00 issue)
    st.metric("Probability of High Income", f"{proba:.2%}")

    # Progress bar
    st.progress(int(proba * 100))

    # Threshold info
    st.caption(f"Decision Threshold: {threshold}")

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown("---")
st.caption("🚀 Model: Tuned XGBoost | Threshold Optimized")