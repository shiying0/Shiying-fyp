import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from streamlit_lottie import st_lottie

# Load models and scalers
models = {
    "XG Boost Model": {"model": joblib.load('xgb_model.pkl'), "scaler": joblib.load('scaler_xgb.pkl')},

    # bad model
    "XG Boost Model2": {"model": joblib.load('best_xgb_model.pkl'), "scaler": joblib.load('scaler_xgb.pkl')},

    "Random Forest Model": {"model": joblib.load('rf_model.pkl'), "scaler": joblib.load('scaler_rf.pkl')},
    "Random Forest Model 2": {"model": joblib.load('best_rf_model.pkl'), "scaler": joblib.load('scaler_rf.pkl')},

    # "Light GBM Model2": {"model": joblib.load('lgbm_final_best_model_optimal_threshold.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
    # good - Light GBM Model4
    "Light GBM Model": {"model": joblib.load('lgbm_model.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
    # no good 
    # "Light GBM Model5": {"model": joblib.load('lgbm_model_5.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
    
    "Ensemble Model": {"model": joblib.load('ensemble_model.pkl'), "scaler": joblib.load('scaler_ensemble.pkl')},
}

# Load test data
# X_test = pd.read_csv('X_test.csv')
# y_test = pd.read_csv('y_test.csv')['out_and_tx_malicious']

X_test = pd.read_csv('X_val.csv')
y_test = pd.read_csv('y_val.csv')['out_and_tx_malicious']
 
df_path = "C:/Users/Enduser/OneDrive - Asia Pacific University/uni/Y3S2/fyp/Model_trial/btc_trial_dataset2.csv"
data = pd.read_csv(df_path)

# Streamlit Layout
st.title("Blockchain (BTC) Fraud Prediction")
st.header("Multi-Model Comparison")

st.write("Explore and compare predictions from multiple fraud detection models.")
st.write("---")

# # Navigation to Visualizations Page
# st.subheader("Explore More")
# st.write("Check out fraud pattern visualizations:")
# if st.button("Go to Visualizations"):
# st.info("Please select 'visualizations' from the sidebar on the left to view fraud pattern visualizations.")

st.subheader("Bitcoin Fraud Detection:")
st.subheader("1. Model Selection")
model_name = st.selectbox("Select a model:", list(models.keys()))

st.write("")  
st.subheader("2. Select Threshold")
threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.5, step=0.05)
st.info("Lower thresholds increase recall (catch more fraud cases), while higher thresholds increase precision (reduce false positives).")

#  Range-based Filtering
st.write("")  
st.subheader("3. Filter by Feature Range")

# Reset index
X_test_reset = X_test.reset_index(drop=True)

# Restrict filtering to specific numeric columns
allowed_cols = ['in_btc', 'out_btc', 'total_btc', 'indegree', 'outdegree']
filter_col = st.selectbox("Select a column:", allowed_cols)

# Adjust the range according to the data range in the dataset
min_val = float(X_test[filter_col].min())
max_val = float(X_test[filter_col].max())
range_values = st.slider(f"Select range for {filter_col}", min_val, max_val, (min_val, max_val))

# Apply range filter
filtered_df = X_test_reset[(X_test_reset[filter_col] >= range_values[0]) & (X_test_reset[filter_col] <= range_values[1])]
st.write("")  
st.write("")  
st.subheader(f'**Filtered Samples: {len(filtered_df)} found**')

# Calculate total pages for filtered samples
if len(filtered_df) > 0:
    total_pages = (len(filtered_df) - 1) // 20 + 1
    st.write(f"*{total_pages} pages*")
    
if len(filtered_df) == 0:
    st.warning("⚠️ No data matched your filter.⚠️ Please adjust the range.")
else:
    # Pagination controls
    page_size = 20
    total_pages = (len(filtered_df) - 1) // page_size + 1
    page_number = st.number_input("Insert Page Number", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    paginated_df = filtered_df.iloc[start_idx:end_idx]

    # Show current page of data
    st.dataframe(paginated_df)

    # Row selection from visible rows
    selected_index = st.selectbox("Select a row index to analyze (based on data above):", paginated_df.index)

    # Display the selected sample
    selected_sample = X_test.loc[selected_index]
    st.write("")  
    st.write("### Selected Row:")
    st.dataframe(selected_sample.to_frame().T)

# === Run prediction using the selected model and threshold ===
# Get selected model and scaler
selected_model = models[model_name]["model"]
scaler = models[model_name]["scaler"]

# Get selected model and scaler
X_test_scaled = scaler.transform(X_test) if scaler else X_test.to_numpy()

# Define index based on the selected row in the sidebar
index = selected_index
st.write("---") 
st.subheader(f"Prediction for {model_name}")
sample = X_test_scaled[index].reshape(1, -1)
proba = selected_model.predict_proba(sample)[0, 1]
pred = 1 if proba >= threshold else 0

# Prediction for Selected Sample
# Determine background color based on probability
bg_color = "#006400" if proba < 0.8 else "#8B0000"

st.markdown(
    f"<div style='background-color: {bg_color}; color: white; padding: 10px; border-radius: 5px; width: 50%;'>"
    f"<strong>Fraud Probability:</strong> {proba:.5f}"
    f"</div>",
    unsafe_allow_html=True
)
st.write("")  

# insert animation 
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        st.error("Failed to load animation. The response is not valid JSON.")
        return None

# Check prediction and display animation
if pred == 1:
    
    animation_fraud = load_lottie_url("https://lottie.host/f5e996ee-2c05-4f8a-aba2-bcfd2f34e032/Qm6EyGASgT.json")
    st_lottie(animation_fraud, height=300)
    st.error(f"🚨 **Prediction: Fraudulent Transaction! (Threshold {threshold})**")
    
else:

    animation_fraud1 = load_lottie_url("https://lottie.host/a53f8268-bf8a-40ea-adae-71f9fd7cdc1d/AW8OKxBbgd.json")
    st_lottie(animation_fraud1, height=300)
    st.success(f"✅ **Prediction: Non-Fraudulent Transaction! (Threshold {threshold})**")

# st.write(f"**True Label:** {'Fraud' if y_test[index] == 1 else 'Non-Fraud'}")

# # Display sample features
# if st.checkbox("Show Sample Features"):
#     st.write("**Input Features:**")
#     st.dataframe(pd.DataFrame(X_test.iloc[index]).T)

# Test Set Performance for Selected Model
st.write("")  
st.subheader(f"Model Performance - {model_name}")
if st.button("Evaluate Selected Model"):
    y_test_proba = selected_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    # Display metrics in styled cards
    st.markdown(
        f"""
        <div style="display: flex; gap: 20px;">
            <div style="background-color: #ADD8E6; color: black; padding: 10px; border-radius: 5px; text-align: center; width: 150px;">
                <strong>ROC AUC Score</strong><br>{roc_auc_score(y_test, y_test_proba):.4f}
            </div>
            <div style="background-color: #87CEEB; color: black; padding: 10px; border-radius: 5px; text-align: center; width: 150px;">
                <strong>Recall</strong><br>{recall_score(y_test, y_test_pred):.4f}
            </div>
            <div style="background-color: #B0E0E6; color: black; padding: 10px; border-radius: 5px; text-align: center; width: 150px;">
                <strong>F1-Score</strong><br>{f1_score(y_test, y_test_pred):.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("")  
    st.write("**Classification Report:**")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

    precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    st.sidebar.write(f"Optimal Threshold (F1-Score): {optimal_threshold:.2f}")

# Compare All Models
st.subheader("Compare All Models")
if st.button("Show Model Comparison"):
    comparison = {}
    for name, model_dict in models.items():
        model = model_dict["model"]  
        X_scaled = model_dict["scaler"].transform(X_test) if model_dict["scaler"] else X_test.to_numpy()
        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        comparison[name] = {
            "ROC AUC": roc_auc_score(y_test, y_proba),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        }
    st.write("**Performance Comparison (Threshold = {:.2f}):**".format(threshold))
    st.table(pd.DataFrame(comparison).T)


# Model Predictions for Selected Sample
st.subheader("Model Predictions for Selected Sample")
sample_probas = {}
for name, model_dict in models.items():
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    X_scaled = scaler.transform(X_test) if scaler else X_test.to_numpy()
    sample_proba = model.predict_proba(X_scaled[index].reshape(1, -1))[0, 1]
    sample_probas[name] = sample_proba

st.write("**Fraud Probabilities for Selected Sample:**")
st.table(pd.DataFrame(sample_probas, index=["Fraud Probability"]).T)

#########################################################################################################
# --- User Input Prediction ---
st.subheader("Check Your Own Transaction!")

st.write("Enter the transaction attributes below to predict if it's fraudulent or not.")

# Input fields for user-defined features
user_in_btc = st.number_input("Amount of BTC received:", min_value=0.0, value=0.1, step=0.1)
st.caption("Amount of BTC received.")

user_out_btc = st.number_input("Amount of BTC sent:", min_value=0.0, value=0.1, step=0.1)
st.caption("Amount of BTC sent.")

user_total_btc = st.number_input("Number of BTC flowing in and out:", min_value=0.0, value=0.2, step=0.1)
st.caption("Total amount of BTC in the transaction.")

user_indegree = st.number_input("Number of Input BTC transactions:", min_value=0.0, value=0.1, step=0.1)
user_outdegree = st.number_input("Number of Output BTC transactions: ", min_value=0.0, value=0.1, step=0.1)
user_out_malicious = st.selectbox("‘1’ for the transaction is the output from a malicious transaction and ‘0’ is non-malicious (0 or 1)", options=[0, 1], index=0)

# Create input array
user_input = np.array([[user_in_btc, user_out_btc, user_total_btc, user_indegree, user_outdegree, user_out_malicious]])

# Add engineered features to match the training feature set
user_out_malicious_to_total_btc = user_out_malicious / (user_total_btc + 1e-6)
user_log_total_btc = np.log1p(user_total_btc)
user_out_malicious_in_btc_interaction = user_out_malicious * user_in_btc
user_net_btc_flow = user_in_btc - user_out_btc

# Combine all features into the input array
user_input = np.hstack((user_input, [[
    user_out_malicious_to_total_btc,
    user_log_total_btc,
    user_out_malicious_in_btc_interaction,
    user_net_btc_flow
]]))

# Scale input using selected model's scaler
user_scaled = scaler.transform(user_input) if scaler else user_input

# Predict probability and label
user_proba = selected_model.predict_proba(user_scaled)[0, 1]
user_pred = 1 if user_proba >= threshold else 0

# Prediction for Selected Sample
# Determine background color based on probability

bg_color = "#006400" if proba < 0.8 else "#8B0000"
st.write("")  
st.markdown(
    f"<div style='background-color: {bg_color}; color: white; padding: 10px; border-radius: 5px; width: 50%;'>"
    f"<strong>Fraud Probability:</strong> {proba:.5f}"
    f"</div>",
    unsafe_allow_html=True
)
st.write("")  

if user_pred == 1:
    animation_fraud = load_lottie_url("https://lottie.host/f5e996ee-2c05-4f8a-aba2-bcfd2f34e032/Qm6EyGASgT.json")
    st_lottie(animation_fraud, height=300)
    st.error(f"🚨 **Prediction: Fraudulent Transaction! (Threshold {threshold})**")
else:
    animation_fraud1 = load_lottie_url("https://lottie.host/a53f8268-bf8a-40ea-adae-71f9fd7cdc1d/AW8OKxBbgd.json")
    st_lottie(animation_fraud1, height=300)
    st.success(f"✅ **Prediction: Non-Fraudulent Transaction! (Threshold {threshold})**")

#########################################################################################################
st.write("---")
# Display Fraudulent Samples
st.subheader("Fraudulent Samples")
if st.button("Show Fraudulent Samples"):
    # Predict probabilities for the entire test set
    y_test_proba = selected_model.predict_proba(X_test_scaled)[:, 1]
    # Convert probabilities to binary predictions using the threshold
    y_test_pred = (y_test_proba >= threshold).astype(int)
    # Filter the test dataset for fraudulent samples
    fraudulent_samples = X_test[y_test_pred == 1]
    # Display the number of fraudulent samples and their details
    st.write(f"**Number of Fraudulent Samples Identified:** {len(fraudulent_samples)}")
    st.dataframe(fraudulent_samples)

# Footer
st.write("---")


# Fraud Pattern Visualizations
# st.subheader("Fraud Pattern Visualizations")
# viz_option = st.selectbox("Select Visualization", [
#     "Transaction Volumes (Boxplot)",
#     "out_malicious Prevalence (Bar Chart)",
#     "indegree Distribution (Boxplot)",
#     "total_btc vs. indegree (Scatter)"
# ])

# sns.set(style="whitegrid")
# if viz_option == "Transaction Volumes (Boxplot)":
#     fig, axes = plt.subplots(1, 3, figsize=(12, 6))
#     for i, col in enumerate(['total_btc', 'in_btc', 'out_btc']):
#         sns.boxplot(x='out_and_tx_malicious', y=col, data=data, showfliers=False, ax=axes[i])
#         axes[i].set_title(f'{col} by Fraud Status')
#         axes[i].set_xlabel('Fraud (1) vs Non-Fraud (0)')
#         axes[i].set_ylabel(col)
#     plt.tight_layout()
#     st.pyplot(fig)

# elif viz_option == "out_malicious Prevalence (Bar Chart)":
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.countplot(x='out_malicious', hue='out_and_tx_malicious', data=data, ax=ax)
#     ax.set_title('Prevalence of out_malicious by Fraud Status')
#     ax.set_xlabel('out_malicious (0 or 1)')
#     ax.set_ylabel('Count')
#     ax.legend(title='Fraud Status', labels=['Non-Fraud (0)', 'Fraud (1)'])
#     st.pyplot(fig)

# elif viz_option == "indegree Distribution (Boxplot)":
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.boxplot(x='out_and_tx_malicious', y='indegree', data=data, showfliers=False, ax=ax)
#     ax.set_title('indegree by Fraud Status')
#     ax.set_xlabel('Fraud (1) vs Non-Fraud (0)')
#     ax.set_ylabel('indegree')
#     st.pyplot(fig)

# elif viz_option == "total_btc vs. indegree (Scatter)":
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.scatterplot(x='total_btc', y='indegree', hue='out_and_tx_malicious', data=data, alpha=0.5, ax=ax)
#     ax.set_title('total_btc vs. indegree by Fraud Status')
#     ax.set_xlabel('total_btc')
#     ax.set_ylabel('indegree')
#     ax.set_xscale('log')  # Log scale for wide range
#     ax.legend(title='Fraud Status', labels=['Non-Fraud (0)', 'Fraud (1)'])
#     st.pyplot(fig)


# st.write("Built with Streamlit by [Your Name] | Multi-Model Fraud Detection | April 2025")

# # Load all five models and scalers with consistent keys
# models = {
#     "XG Boost Model": {"model": joblib.load('xgb_model.pkl'), "scaler": joblib.load('scaler_xgb.pkl')},

#     # bad model
#     "XG Boost Model2": {"model": joblib.load('best_xgb_model.pkl'), "scaler": joblib.load('scaler_xgb.pkl')},

#     "Random Forest Model": {"model": joblib.load('rf_model.pkl'), "scaler": joblib.load('scaler_rf.pkl')},
#     "Random Forest Model 2": {"model": joblib.load('best_rf_model.pkl'), "scaler": joblib.load('scaler_rf.pkl')},

#     # "Light GBM Model2": {"model": joblib.load('lgbm_final_best_model_optimal_threshold.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
#     # good - Light GBM Model4
#     "Light GBM Model4": {"model": joblib.load('lgbm_model.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
#     # no good 
#     "Light GBM Model5": {"model": joblib.load('lgbm_model_5.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
    

#     "Ensemble Model": {"model": joblib.load('ensemble_model.pkl'), "scaler": joblib.load('scaler_ensemble.pkl')},
# }

# # Load test data
# X_test = pd.read_csv('X_test.csv')
# y_test = pd.read_csv('y_test.csv')['out_and_tx_malicious']
# data = pd.concat([X_test, y_test], axis=1)

# # Reapply preprocessing (to match training)
# log_features = ['indegree', 'outdegree', 'in_btc', 'out_btc', 'total_btc']
# data[log_features] = np.log1p(data[log_features])

# def add_features(df):
#     df['out_malicious_to_total_btc'] = df['out_malicious'] / (df['total_btc'] + 1e-6)
#     df['log_total_btc'] = np.log1p(df['total_btc'])
#     df['out_malicious_in_btc_interaction'] = df['out_malicious'] * df['in_btc']
#     df['net_btc_flow'] = df['in_btc'] - df['out_btc']
#     return df

# data_fe = add_features(data)
# selected_features = [
#     'in_btc', 'out_btc', 'total_btc', 'out_malicious',
#     'out_malicious_to_total_btc', 'log_total_btc',
#     'out_malicious_in_btc_interaction', 'net_btc_flow'
# ]
# data_final = data_fe[selected_features + ['out_and_tx_malicious']]

# # Streamlit App Layout
# st.title("Fraud Prediction Dashboard - Multi-Model Comparison")
# st.write("Explore and compare predictions from multiple fraud detection models.")

# # Sidebar for User Inputs
# st.sidebar.header("Prediction Settings")
# model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
# index = st.sidebar.slider("Select Test Sample Index", 0, len(X_test) - 1, 0)
# threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.5, step=0.05)

# # Prepare data for selected model
# selected_model = models[model_name]["model"]
# scaler = models[model_name]["scaler"]
# X_test_final = data_final[selected_features]
# X_test_scaled = scaler.transform(X_test_final) if scaler else X_test_final.to_numpy()

# # Prediction for Selected Sample
# st.subheader(f"Prediction for {model_name}")
# sample = X_test_scaled[index].reshape(1, -1)
# proba = selected_model.predict_proba(sample)[0, 1]
# pred = 1 if proba >= threshold else 0

# st.write(f"**Sample Index:** {index}")
# st.write(f"**Fraud Probability:** {proba:.5f}")
# st.write(f"**Prediction (Threshold {threshold}):** {'Fraud' if pred == 1 else 'Non-Fraud'}")
# st.write(f"**True Label:** {'Fraud' if y_test[index] == 1 else 'Non-Fraud'}")

# # Test Set Performance for Selected Model
# st.subheader(f"Test Set Performance - {model_name}")
# if st.button("Evaluate Selected Model"):
#     y_test_proba = selected_model.predict_proba(X_test_scaled)[:, 1]
#     y_test_pred = (y_test_proba >= threshold).astype(int)
#     st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_test_proba):.4f}")
#     st.table(pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).transpose())

# # Fraud Pattern Visualizations
# st.subheader("Fraud Pattern Visualizations")
# viz_option = st.selectbox("Select Visualization", [
#     "Transaction Volumes (Boxplot)",
#     "out_malicious Prevalence (Bar Chart)",
#     "indegree Distribution (Boxplot)",
#     "total_btc vs. indegree (Scatter)"
# ])

# sns.set(style="whitegrid")
# if viz_option == "Transaction Volumes (Boxplot)":
#     fig, axes = plt.subplots(1, 3, figsize=(12, 6))
#     for i, col in enumerate(['total_btc', 'in_btc', 'out_btc']):
#         sns.boxplot(x='out_and_tx_malicious', y=col, data=data, showfliers=False, ax=axes[i])
#         axes[i].set_title(f'{col} by Fraud Status')
#         axes[i].set_xlabel('Fraud (1) vs Non-Fraud (0)')
#         axes[i].set_ylabel(col)
#     plt.tight_layout()
#     st.pyplot(fig)

# elif viz_option == "out_malicious Prevalence (Bar Chart)":
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.countplot(x='out_malicious', hue='out_and_tx_malicious', data=data, ax=ax)
#     ax.set_title('Prevalence of out_malicious by Fraud Status')
#     ax.set_xlabel('out_malicious (0 or 1)')
#     ax.set_ylabel('Count')
#     ax.legend(title='Fraud Status', labels=['Non-Fraud (0)', 'Fraud (1)'])
#     st.pyplot(fig)

# elif viz_option == "indegree Distribution (Boxplot)":
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.boxplot(x='out_and_tx_malicious', y='indegree', data=data, showfliers=False, ax=ax)
#     ax.set_title('indegree by Fraud Status')
#     ax.set_xlabel('Fraud (1) vs Non-Fraud (0)')
#     ax.set_ylabel('indegree')
#     st.pyplot(fig)

# elif viz_option == "total_btc vs. indegree (Scatter)":
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.scatterplot(x='total_btc', y='indegree', hue='out_and_tx_malicious', data=data, alpha=0.5, ax=ax)
#     ax.set_title('total_btc vs. indegree by Fraud Status')
#     ax.set_xlabel('total_btc')
#     ax.set_ylabel('indegree')
#     ax.set_xscale('log')  # Log scale for wide range
#     ax.legend(title='Fraud Status', labels=['Non-Fraud (0)', 'Fraud (1)'])
#     st.pyplot(fig)

# # Footer
# st.write("---")
# st.write("Built with Streamlit by [Your Name] | Multi-Model Fraud Detection | April 2025")
