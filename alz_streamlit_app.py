import streamlit as st
import pandas as pd 
import joblib

# Load the alzheimer_prediction trained model (make sure the .pkl file is in the same folder!)
model = joblib.load("alzheimer_prediction_model.pkl")

st.set_page_config(page_title="Alzheimer's Risk Prediction", page_icon="🧠", layout="centered")

# Set the title of the app
st.title("Alzheimer's Risk Prediction (Questionnaire-Based)")
st.markdown("This app predicts the risk of Alzheimer's disease based on a questionnaire. Please Fill in the patient's details below to estimate their dementia risk.")

#Form inputs
age = st.number_input("Age", min_value=40, max_value=100, value=65)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.slider("Years of Education", 0, 25, 12)
ses = st.slider("Socioeconomic Status (1 = low, 5 = high)", 1, 5, 3)
mmse = st.slider("MMSE ( Mini Mental State Examination) Score (0–30)", 0, 30, 27)
cdr = st.slider("CDR (Clinical Dementia Rating)", 0.0, 3.0, 0.0, step=0.5)
etiv = st.number_input("eTIV (Estimated Total Intracranial Volume)", min_value=1000, max_value=2000, value=1500)
nwbv = st.slider("nWBV (Normalized Whole Brain Volume)", 0.60, 0.90, 0.75)
asf = st.number_input("ASF (Atlas Scaling Factor)", min_value=0.80, max_value=1.60, value=1.2)

# Convert gender to numeric
gender_value = 0 if gender == "Male" else 1

# Predict button
if st.button("Predict Dementia Risk"):
    input_data = pd.DataFrame([{
        "M/F": gender_value,
        "Age": age,
        "EDUC": education,
        "SES": ses,
        "MMSE": mmse,
        "CDR": cdr,
        "eTIV": etiv,
        "nWBV": nwbv,
        "ASF": asf
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High risk: The patient is likely to be demented.")
    else:
        st.success("✅ Low risk: The patient is likely non-demented.")

    st.markdown("### 📊 Input Summary")
    st.dataframe(input_data.T)
