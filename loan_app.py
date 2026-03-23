import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# ------------------ LOAD MODEL ------------------ #
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# ------------------ CUSTOM CSS ------------------ #
st.markdown("""
<style>
.stButton button {
    width: 100%;
    background-color: #0E6FFF;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------ #
st.title("🏦 Smart Loan Approval System")
st.markdown("### Enter Applicant Details")
st.info("📌 All monetary values should be entered in Naira (₦)")

# ------------------ PERSONAL INFO ------------------ #
st.markdown("## 👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ------------------ FINANCIAL INFO ------------------ #
st.markdown("## 💰 Financial Information")

col3, col4 = st.columns(2)

with col3:
    income = st.number_input("Applicant Monthly Income (₦)", min_value=0.0)
    coincome = st.number_input("Coapplicant Monthly Income (₦)", min_value=0.0)

with col4:
    loan_amount = st.number_input("Loan Amount Requested (₦)", min_value=0.0)
    loan_term = st.selectbox("Loan Term (Months)", [12, 36, 60, 120, 180, 240, 360])

# ------------------ METRICS ------------------ #
st.markdown("## 📊 Summary")

total_income = income + coincome

m1, m2 = st.columns(2)
m1.metric("Total Monthly Income", f"₦{total_income:,.0f}")
m2.metric("Loan Requested", f"₦{loan_amount:,.0f}")

# ------------------ ENCODING ------------------ #
gender = 1 if gender == "Female" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# ------------------ VALIDATION ------------------ #
def validate_inputs(income, coincome, loan_amount):
    total_income = income + coincome
    
    if total_income == 0:
        return False, "Income cannot be zero."

    if loan_amount > (total_income * 50):
        return False, "Loan amount too high compared to income."

    return True, ""

# ------------------ PREDICTION ------------------ #
st.markdown("---")

if st.button("🔍 Predict Loan Status"):

    is_valid, message = validate_inputs(income, coincome, loan_amount)

    if not is_valid:
        st.warning(f"⚠️ {message}")
    else:
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': income,
            'CoapplicantIncome': coincome,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }])

        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

        prediction = model.predict(input_data)

        try:
            prob = model.predict_proba(input_data)[0][1]
        except:
            prob = None

        st.markdown("## 🧾 Result")

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        if prob is not None:
            st.progress(int(prob * 100))
            st.write(f"Approval Probability: {prob:.2%}")