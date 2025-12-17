import streamlit as st
import numpy as np
import joblib

model = joblib.load("rf_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detection Demo")

st.write("Enter a few simple details about the transaction. The model will estimate the fraud probability.")

# --- Simple user inputs ---

time = st.slider("Time since first transaction (in seconds)", min_value=0, max_value=172800, value=0, step=60)
amount = st.number_input("Transaction amount", min_value=0.0, value=100.0, step=10.0)

st.subheader("Risk scores (0 = low, 10 = high)")
unusual_spend = st.slider("How unusual is this spending compared to customer’s normal behaviour?", 0, 10, 3)
location_risk = st.slider("Is the location risky or far from usual places?", 0, 10, 2)
merchant_risk = st.slider("Merchant risk level (online / unknown website etc.)", 0, 10, 2)

st.caption("These scores are just for demo and are mapped internally to hidden PCA features (V1–V28).")

if st.button("Predict"):
    # map simple scores to a few PCA-like features
    # start with all zeros
    V = np.zeros(28)

    # very rough mapping just for demo
    V[0] = (unusual_spend - 5) / 2.0   # V1
    V[1] = (location_risk - 5) / 2.0   # V2
    V[2] = (merchant_risk - 5) / 2.0   # V3

    # build full feature vector in same order as training: Time, V1..V28, Amount
    input_data = np.array([[time,
                            V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8], V[9],
                            V[10], V[11], V[12], V[13], V[14], V[15], V[16], V[17], V[18], V[19],
                            V[20], V[21], V[22], V[23], V[24], V[25], V[26], V[27],
                            amount]])

    # scale with same scaler
    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.write(f"Estimated fraud probability: **{prob:.3f}**")

    if pred == 1:
        st.error("Model prediction: FRAUDULENT transaction")
        st.write("💡 This transaction looks suspicious based on the provided risk scores and amount.")
    else:
        st.success("Model prediction: NORMAL transaction")
        st.write("✅ This transaction does not look risky based on the current inputs.")
