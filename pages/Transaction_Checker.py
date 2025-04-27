import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import sys
import os
from streamlit_lottie import st_lottie

def render():
# === Load the best model and scaler ===
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler_rf.pkl')
    threshold = 0.75  

    # === Streamlit Layout ===
    # Display the animation
    with open("images/Animation - 1745401555152.json", "r") as f:
        lottie_animation = json.load(f)
    st_lottie(lottie_animation, height=300)
    
    st.write("---")
    st.title("Blockchain (BTC) Fraud Prediction")
    st.write(" ")
    with st.expander("🔍 How Fraud is Detected (Click to expand)", expanded=False):
        st.markdown("""
        <style>
        /* Light mode (default) */
        .feature-box {
            background-color: #f0f8ff; /* Light blue */
            padding: 16px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #87ceeb; /* Sky blue */
            color: #000000; /* Black text */
        }

        .pattern-box {
            background-color: #fffaf0; /* Light orange */
            padding: 16px;
            border-radius: 10px;
            border-left: 4px solid #ffd700; /* Gold */
            color: #000000; /* Black text */
        }

        .fraud-table {
            width: 100%;
            border-collapse: collapse;
            font-family: "Segoe UI", sans-serif;
            font-size: 15px;
            margin-top: 10px;
            color: #000000; /* Black text */
        }
        .fraud-table thead {
            background-color: #ffe4e1; /* Light pink */
            text-align: left;
        }
        .fraud-table th, .fraud-table td {
            padding: 12px 16px;
        }
        .fraud-table tbody tr {
            border-bottom: 1px solid #e0e0e0;
        }
        .fraud-table tbody tr:hover {
            background-color: #f5f5f5;
        }

        /* Dark mode overrides */
        @media (prefers-color-scheme: dark) {
            .feature-box {
                background-color: #1e293b; /* Dark blue-gray */
                border-left: 4px solid #38bdf8; /* Light blue */
                color: #ffffff; /* White text */
            }

            .pattern-box {
                background-color: #3b2f2f; /* Dark reddish brown */
                border-left: 4px solid #facc15; /* Yellow */
                color: #ffffff; /* White text */
            }

            .fraud-table {
                color: #ffffff; /* White text */
            }
            .fraud-table thead {
                background-color: #4b5563; /* Darker header */
            }
            .fraud-table tbody tr {
                border-bottom: 1px solid #6b7280;
            }
            .fraud-table tbody tr:hover {
                background-color: #374151; /* Hover darker */
            }
        }
        </style>

        <div class="feature-box">
        <h4>🧠 Feature Summary</h4>
        <ul>
            <li><b>indegree</b>: Number of input transactions feeding into <code>tx_hash</code></li>
            <li><b>outdegree</b>: Number of output transactions from <code>tx_hash</code></li>
            <li><b>in_btc</b>: Total BTC received by this transaction</li>
            <li><b>out_btc</b>: Total BTC sent from this transaction</li>
            <li><b>total_btc</b>: Net flow = <code>in_btc - out_btc</code></li>
        </ul>
        </div>

        <div class="pattern-box">
        <h4>🚩 Suspicious Transaction Patterns</h4>
        <table class="fraud-table">
            <thead>
            <tr>
                <th>Pattern</th>
                <th>What It Could Mean</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <td>High <code>indegree</code></td>
                <td>Fund aggregation from many sources (e.g., laundering or mixer)</td>
            </tr>
            <tr>
                <td>High <code>outdegree</code></td>
                <td>Splitting funds to many addresses (e.g., scam payout or dusting)</td>
            </tr>
            <tr>
                <td>Large <code>in_btc</code></td>
                <td>Sudden inflow — possibly from phishing, scam, or hack</td>
            </tr>
            <tr>
                <td>Large <code>out_btc</code></td>
                <td>Draining funds quickly — potential wallet compromise</td>
            </tr>
            <tr>
                <td>High <code>total_btc</code> (positive)</td>
                <td>Funds received but not sent — bait wallet or abnormal hoarding</td>
            </tr>
            <tr>
                <td>High <code>total_btc</code> (negative)</td>
                <td>Unusual outflow — possible exit scam or fraud pattern</td>
            </tr>
            </tbody>
        </table>
        </div>

        <p style="font-size: 14px; margin-top: 1rem; color: gray;">
        💡 These behavioral signals help explain the decision behind fraud prediction.
        </p>
        """, unsafe_allow_html=True)

    def get_fraud_insight(indegree, outdegree, in_btc, out_btc, total_btc):
        insights = []

        if indegree >= 15:
            insights.append("📥 High number of incoming sources-- This transaction receives BTC from many addresses. It might be gathering funds — a common tactic in laundering.")

        if outdegree >= 15:
            insights.append("📤 High number of outgoing targets-- This transaction sends BTC to many wallets. This could mean it's trying to split and hide funds.")

        if in_btc > 2:
            insights.append("💰 Large amount received-- The incoming BTC is unusually high, which might suggest scam funds or hacked wallet activity.")

        if out_btc > 2:
            insights.append("💸 Large amount sent-- This transaction sends a lot of BTC, which might be part of a draining attack.")

        if total_btc >= 3:
            insights.append("📊 High BTC flow-- This transaction involves a large total amount of BTC, which could indicate a high-value transfer, whale wallet activity, or institutional movement.")

        if not insights:
            insights.append("✅ This transaction has no obvious red flags based on the individual features, but was flagged due to overall behavior detected by our model.")

        return insights

    # --- User Input Prediction ---
    st.write(" ")
    st.write(" ")
    st.subheader("Check Your Own Transaction!")
    st.write("Enter the Bitcoin transaction attributes below to predict if it's fraudulent or not.")
    def out_malicious(user_in_btc, user_out_btc, user_total_btc, user_indegree):
            if (
                user_total_btc > 4 or 
                user_in_btc > 2 or 
                user_out_btc > 2 or 
                user_indegree > 10
            ):
                return 1  
            else:
                return 0  
            
    user_in_btc = st.number_input(
        "Amount of BTC received:",
        min_value=0.00001,
        value=0.01,
        step=0.00001,
        format="%.8f"  
    )
    user_out_btc = st.number_input(
        "Amount of BTC sent:",
        min_value=0.00001,
        value=0.01,
        step=0.00001,
        format="%.8f"
    )
    user_total_btc = st.number_input(
        "Number of BTC flowing in and out:",
        min_value=0.00001,
        value=0.01,
        step=0.00001,
        format="%.8f"
    )
    user_indegree = st.number_input(
        "Number of Input into this BTC transactions:",
        min_value=1,
        value=1,
        step=1,
    )
    user_outdegree = st.number_input(
        "Number of Output into this BTC transactions:",
        min_value=1,
        value=1,
        step=1,
    )
    user_out_malicious = out_malicious(
        user_in_btc, user_out_btc, user_total_btc, user_indegree,
    )

    # Create input array
    user_input = np.array([[user_in_btc, user_out_btc, user_total_btc, user_indegree, user_outdegree,user_out_malicious]])

    st.write(" ")
    button_col1, button_col2, button_col3 = st.columns([3, 1, 3])

    with button_col2:
        predict_clicked = st.button("🔍 Predict", key="center_predict")
    st.write("---")
    if predict_clicked:

        st.toast("Prediction complete!")
        # Feature engineering
        user_log_total_btc = np.log1p(user_total_btc)
        user_net_btc_flow = user_in_btc - user_out_btc
        user_out_malicious_to_total_btc = user_out_malicious / (user_total_btc + 1e-6)
        user_out_malicious_in_btc_interaction = user_out_malicious * user_in_btc

        # Final input vector
        additional_features = np.array([
            user_out_malicious_to_total_btc,
            user_out_malicious_in_btc_interaction,
            user_log_total_btc,
            user_net_btc_flow
        ]).reshape(1, -1)
        user_input = np.hstack((user_input, additional_features))

        # Predict
        user_scaled = scaler.transform(user_input)
        user_proba = model.predict_proba(user_scaled)[0, 1]
        user_pred = 1 if user_proba >= 0.75 else 0

        # Display result
        if user_pred == 1:
            
            with open("images/Animation (fraud).json", "r") as f:
                animation_fraud = json.load(f)
            st_lottie(animation_fraud, height=300)

            # Fraud prediction message
            st.error("🚨 Prediction: Fraudulent Transaction Detected")

            # Display Fraud Probability here
            bg_color = "#006400" if user_proba < 0.74 else "#8B0000"
            st.markdown(
                f"""
                <div style='background-color: {bg_color}; color: white; padding: 10px; border-radius: 5px; width: 50%; margin-top: -10px; margin-bottom: 20px;'>
                    <strong>Fraud Probability:</strong> {user_proba:.5f}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("---")
            st.markdown("### 🔎 Reason for Flagging (Suspicious Behaviours)")

            # Get reasons
            flagged_reasons = get_fraud_insight(
                indegree=user_indegree,
                outdegree=user_outdegree,
                in_btc=user_in_btc,
                out_btc=user_out_btc,
                total_btc=user_total_btc
            )

            # Style for each insight box
            st.markdown("""
            <style>
            /* Light mode (default) */
            .insight-card {
                background-color: #f9fafb; /* very light gray */
                border-left: 4px solid #ff6b6b; /* soft red */
                padding: 14px 18px;
                border-radius: 10px;
                margin-bottom: 12px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 15px;
                line-height: 1.6;
                color: #000000; /* black text */
            }

            /* Dark mode overrides */
            @media (prefers-color-scheme: dark) {
                .insight-card {
                    background-color: #1f2937; /* dark gray-blue */
                    border-left: 4px solid #fb7185; /* bright pink-red */
                    color: #ffffff; /* white text */
                }
            }
            </style>
            """, unsafe_allow_html=True)

            for reason in flagged_reasons:
                st.markdown(f'<div class="insight-card">{reason}</div>', unsafe_allow_html=True)

            st.write(" ")
            st.markdown("### ⚠️Suggested Next Steps⚠️ ")

            # Styling: Boxed layout 
            st.markdown("""
            <style>
            .next-steps-box {
                background-color: #f9fafb;
                border-left: 4px solid #fbbc05;
                padding: 20px 25px;
                border-radius: 10px;
                margin-top: 25px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            }

            .next-steps-title {
                display: flex;
                align-items: center;
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 15px;
                gap: 12px;
                color: #333;
            }

            .next-steps-list {
                font-size: 15px;
                line-height: 1.7;
                padding-left: 20px;
                color: #444;
            }
            </style>
            """, unsafe_allow_html=True)
           
            # HTML layout for the boxed suggestion panel
            st.markdown("""
            <div class="next-steps-box">
                <div class="next-steps-title">
                </div>
                <div class="next-steps-list">
                    <ul>
                        <li><b>Investigate</b> the transaction and its connected addresses.</li>
                        <li><b>Flag or freeze</b> the associated wallet for review.</li>
                        <li><b>Log</b> this case in the fraud database.</li>
                        <li><b>Notify</b> the fraud monitoring team.</li>
                        <li><b>Trace</b> the transaction path using blockchain explorer.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)            

        else:
            with open("images/Animation (non-fraud).json", "r") as f:
                animation_safe = json.load(f)
            st_lottie(animation_safe, height=300)
            st.success("✅ Prediction: Non-Fraudulent Transaction!")

            # show probability in green box for legit transactions
            bg_color = "#006400"
            st.markdown(
                f"""
                <div style='background-color: {bg_color}; color: white; padding: 10px; border-radius: 5px; width: 50%; margin-top: -10px; margin-bottom: 20px;'>
                    <strong>Fraud Probability:</strong> {user_proba:.5f}
                </div>
                """,
                unsafe_allow_html=True
            )


