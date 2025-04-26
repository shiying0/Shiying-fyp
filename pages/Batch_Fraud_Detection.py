import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json
import sys
import os
from streamlit_lottie import st_lottie

def render():
    # === Load model and scaler === 
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler_rf.pkl')

    # === Streamlit Layout ===
    # Display the animation
    with open("images/Animation - 1745418315197.json", "r") as f:
        lottie_animation = json.load(f)
    
    st_lottie(lottie_animation, height=300)

    st.title(" Blockchain Fraud Detection ")
    st.write("---")
    st.write("Upload your blockchain transaction dataset (.csv) to detect fraudulent transactions.")

    # === File Upload ===
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success("✅ File successfully uploaded!")

            st.write("### File Preview")
            st.dataframe(user_df)

            # Feature Engineering
            user_df['out_malicious_to_total_btc'] = user_df['out_malicious'] / (user_df['total_btc'] + 1e-6)
            user_df['log_total_btc'] = np.log1p(user_df['total_btc'])
            user_df['out_malicious_in_btc_interaction'] = user_df['out_malicious'] * user_df['in_btc']
            user_df['net_btc_flow'] = user_df['in_btc'] - user_df['out_btc']

            # Select and reorder features used in training
            features = [
                'in_btc', 'out_btc', 'total_btc', 'out_malicious','indegree','outdegree',
                'out_malicious_to_total_btc', 'log_total_btc',
                'out_malicious_in_btc_interaction', 'net_btc_flow'
            ]
            user_input = user_df[features]
            threshold = 0.75

            # === Scale and Predict ===
            scaled_input = scaler.transform(user_input)
            fraud_probas = model.predict_proba(scaled_input)[:, 1]
            fraud_preds = (fraud_probas >= threshold).astype(int)

            # Add predictions to dataframe
            user_df["Fraud Probability"] = fraud_probas
            user_df["Prediction"] = np.where(fraud_preds == 1, "Fraud", "Non-Fraud")

            #  Toast popup to confirm prediction completed
            st.toast("Fraud Prediction is Completed!", icon="✅")

            if uploaded_file is not None:
                st.session_state['processed_data'] = user_df
                
            st.write("---")
            # === Results Section ===
            st.write("### Fraudulent Transaction Prediction Results:")

            # === Fraud Highlight Section ===
            st.write("### 🚨 Detected Fraudulent Transactions")

            fraud_df = user_df[user_df["Prediction"] == "Fraud"]
            
            # === Lottie Animation for Visual Feedback ===
            if len(fraud_df) > 0:
                # Lottie JSON animation
                with open("images/Animation (fraud).json", "r") as f:
                    animation_fraud = json.load(f)
                st_lottie(animation_fraud, height=250)                
                st.error("⚠️ Fraudulent transactions found in the dataset ⚠️")

            else:
                with open("images/Animation (non-fraud).json", "r") as f:
                    animation_safe = json.load(f)
                st_lottie(animation_safe, height=250)
                st.success("✅ No fraudulent transactions detected.")
                
            st.write(f"Total Fraudulent Transactions Detected: **{len(fraud_df)}**")
            st.dataframe(fraud_df)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.write("")  
        st.info("""Please upload a CSV file to begin fraud detection.""")
        st.caption(""" Make sure that your (.csv) file has the required columns:'in_btc', 'out_btc', 'total_btc', 'indegree', 'outdegree', 'out_malicious'""")
        
    st.write("---")
    st.subheader("Want to Explore Fraud Patterns?")

    if st.button("Go to Fraud Insights Page"):
        st.success("Please click on ' Visualization' in the left sidebar to explore the visual analytics.")
        st.balloons()