import streamlit as st
import numpy as np
import joblib

# load model and scaler
model = joblib.load("rf_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detection Demo")

st.write("Enter transaction details to get fraud probability (demo app).")

time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0, step=1.0)
amount = st.number_input("Amount", min_value=0.0, value=1.0, step=1.0)

st.write("PCA features (V1 - V28). Use small values between -5 and 5 for demo.")

v1  = st.number_input("V1",  value=0.0, step=0.1)
v2  = st.number_input("V2",  value=0.0, step=0.1)
v3  = st.number_input("V3",  value=0.0, step=0.1)
v4  = st.number_input("V4",  value=0.0, step=0.1)
v5  = st.number_input("V5",  value=0.0, step=0.1)
v6  = st.number_input("V6",  value=0.0, step=0.1)
v7  = st.number_input("V7",  value=0.0, step=0.1)
v8  = st.number_input("V8",  value=0.0, step=0.1)
v9  = st.number_input("V9",  value=0.0, step=0.1)
v10 = st.number_input("V10", value=0.0, step=0.1)
v11 = st.number_input("V11", value=0.0, step=0.1)
v12 = st.number_input("V12", value=0.0, step=0.1)
v13 = st.number_input("V13", value=0.0, step=0.1)
v14 = st.number_input("V14", value=0.0, step=0.1)
v15 = st.number_input("V15", value=0.0, step=0.1)
v16 = st.number_input("V16", value=0.0, step=0.1)
v17 = st.number_input("V17", value=0.0, step=0.1)
v18 = st.number_input("V18", value=0.0, step=0.1)
v19 = st.number_input("V19", value=0.0, step=0.1)
v20 = st.number_input("V20", value=0.0, step=0.1)
v21 = st.number_input("V21", value=0.0, step=0.1)
v22 = st.number_input("V22", value=0.0, step=0.1)
v23 = st.number_input("V23", value=0.0, step=0.1)
v24 = st.number_input("V24", value=0.0, step=0.1)
v25 = st.number_input("V25", value=0.0, step=0.1)
v26 = st.number_input("V26", value=0.0, step=0.1)
v27 = st.number_input("V27", value=0.0, step=0.1)
v28 = st.number_input("V28", value=0.0, step=0.1)

if st.button("Predict"):
    input_data = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, amount]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.write(f"Fraud probability: **{prob:.3f}**")

    if pred == 1:
        st.error("Model prediction: FRAUDULENT transaction")
    else:
        st.success("Model prediction: NORMAL transaction")
