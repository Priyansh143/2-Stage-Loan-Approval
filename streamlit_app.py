import streamlit as st
import yaml
import requests
API_URL = "http://127.0.0.1:8000/api/predict/"

with open("backend/config.yaml") as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="Loan Approval", layout="centered")
default = config['ui']['default_inputs']


st.markdown(
    """
    <h1 style='text-align: center;'>Loan Approval System</h1>
    <p style='text-align: center; color: gray;'>
        Two-Stage Machine Learning Decision Engine
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Model info")
try :
    expected_features = config["input_features"]
    st.sidebar.write("Classifier expects:")
    st.sidebar.code(expected_features)
except Exception:
    st.sidebar.write("Classifier feature names unavailable")

st.markdown(
    """
    ### How this system works

    This application evaluates a loan request using a **two-stage machine learning approach**:

    - **Stage 1:** A classification model estimates the probability of loan approval  
    - **Stage 2:** If approved, a regression model estimates the eligible loan amount  

    The models are trained on historical loan data and served through a dedicated backend service.
    """
)

st.markdown("### Applicant Details")

with st.container():
    st.markdown("#### Personal & Employment Information")

    col1, col2 = st.columns(2)
    with col1:
        no_of_dependents = st.number_input("Number of Dependents",
                                           value=int(default['no_of_dependents']),
                                           key = "no_of_dependents")
        education = st.selectbox(
            "Education Level",
            ["Graduate", "Not Graduate"],
            index=0 if default['education'] == "Graduate" else 1,
            key = "education"
        )
        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"],
            index=0 if default['self_employed'] == "Yes" else 1,
            key = "self_employed"
        )

    with col2:
        income_annum = st.number_input("Annual Income",
                                       value=float(default['income_annum']),
                                       key = "income_annum")
        loan_amount = st.number_input("Requested Loan Amount", 
                                      value=float(default['loan_amount']),
                                      key = "loan_amount")
        loan_term = st.number_input("Loan Term (Years)", 
                                    value=int(default['loan_term']),
                                    key = "loan_term")


st.markdown("#### Financial & Asset Information")

col3, col4 = st.columns(2)
with col3:
    cibil_score = st.number_input("CIBIL Score", 
                                  value=int(default['cibil_score']),
                                  key = "cibil_score")
    residential_assets_value = st.number_input("Residential Assets Value", 
                                               value=float(default['residential_assets_value']),
                                               key = "residential_assets_value")
    commercial_assets_value = st.number_input("Commercial Assets Value", 
                                              value=float(default['commercial_assets_value']),
                                              key = "commercial_assets_value")

with col4:
    luxury_assets_value = st.number_input("Luxury Assets Value", 
                                          value=float(default['luxury_assets_value']),
                                          key = "luxury_assets_value")
    bank_asset_value = st.number_input("Bank Assets Value", 
                                       value=float(default['bank_asset_value']),
                                       key = "bank_asset_value")



def build_applicant_from_state():
    return {
        "no_of_dependents": st.session_state.no_of_dependents,
        "education": st.session_state.education,
        "self_employed": st.session_state.self_employed,
        "income_annum": st.session_state.income_annum,
        "loan_amount": st.session_state.loan_amount,
        "loan_term": st.session_state.loan_term,
        "cibil_score": st.session_state.cibil_score,
        "residential_assets_value": st.session_state.residential_assets_value,
        "commercial_assets_value": st.session_state.commercial_assets_value,
        "luxury_assets_value": st.session_state.luxury_assets_value,
        "bank_asset_value": st.session_state.bank_asset_value,
    }



st.markdown("---")
st.markdown("### Prediction Result")

if st.button("Run Loan Assessment"):
    with st.spinner("Evaluating application..."):
        applicant = build_applicant_from_state()
        response = requests.post(API_URL, json=applicant)

    if response.status_code == 200:
        res = response.json()

        with st.container():
            st.metric(
                label="Approval Probability",
                value=f"{res['approval_prob'] * 100:.2f}%"
            )

            if res["approve"] == 1:
                st.success("Loan Approved")
                st.markdown(
                    f"""
                    **Estimated Eligible Loan Amount:**  
                    ₹ {res['reg_pred']:,.2f}
                    """
                )
            else:
                st.error("Loan Rejected")
                
    elif response.status_code == 400:
        error_msg = response.json().get("error", "Invalid input")
        st.warning(error_msg)
    
    else:
        st.error("Backend error. Please try again.")

