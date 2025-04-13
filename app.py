import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import json
import requests
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Use a direct .json animation file
# animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_tno6cg2w.json")
# st_lottie(animation, height=300)

#best
animation = load_lottie_url("https://lottie.host/08bbdffd-d390-4ecf-91ea-9fd2a0582994/VkGHfNVGVv.json")
st_lottie(animation, height=300)

# animation = load_lottie_url("	https://assets1.lottiefiles.com/packages/lf20_kkflmtur.json")
# st_lottie(animation, height=600)

# animation = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_49rdyysj.json")
# st_lottie(animation, height=300)

# ---- Function to Get Live Crypto Prices ----
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_crypto_prices():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ethereum,solana",
        "vs_currencies": "usd",
        "include_24hr_change": "true"
    }
    response = requests.get(url, params=params)
    return response.json()

# ---- Call the Function ----
prices = get_crypto_prices()

# ---- Display Section ----
# Display current date and day
current_date = datetime.now().strftime("%A, %d %B %Y")
st.write(f"{current_date}")
st.header(" Today's Crypto Prices (USD)")
col1, col2, col3 = st.columns(3)
 
# Crypto icons
btc_icon = "https://cryptologos.cc/logos/bitcoin-btc-logo.png?v=026"  
eth_icon = "https://cryptologos.cc/logos/ethereum-eth-logo.png?v=026"  
sol_icon = "https://cryptologos.cc/logos/solana-sol-logo.png?v=026"

# Bitcoin
with col1:
    st.image(btc_icon, width=40)
    btc_price = prices["bitcoin"]["usd"]
    btc_change = prices["bitcoin"]["usd_24h_change"]
    st.metric("**Bitcoin (BTC)**", f"${btc_price:,.0f}", f"{btc_change:.2f}%")

# Ethereum
with col2:
    st.image(eth_icon, width=40)
    eth_price = prices["ethereum"]["usd"]
    eth_change = prices["ethereum"]["usd_24h_change"]
    st.metric("**Ethereum (ETH)**", f"${eth_price:,.0f}", f"{eth_change:.2f}%")

# Solana
with col3:
    st.image(sol_icon, width=40)
    sol_price = prices["solana"]["usd"]
    sol_change = prices["solana"]["usd_24h_change"]
    st.metric("**Solana (SOL)**", f"${sol_price:,.2f}", f"{sol_change:.2f}%")

st.caption(" \nMarket activity is often linked to fraud trends in blockchain networks. (data fetched from CoinGecko)")

 # Add a gap
st.markdown("<br>", unsafe_allow_html=True) 

def get_historical_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "30",
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "prices" not in data:
        st.error("Error: 'prices' key not found in the API response.")
        return pd.DataFrame()  # Return an empty DataFrame to avoid further errors
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# Fetch historical data
df = get_historical_data()

# Plot the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["price"], mode="lines", name="BTC Price"))
fig.update_layout(title="Bitcoin Price Over the Last 30 Days", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
st.subheader("Bitcoin Price")
st.plotly_chart(fig)
st.write("---")

#################################################################################################

st.markdown("## Bitcoin Network Activity Dashboard")
st.caption("Gain insights into Bitcoin's on-chain behavior. Useful for analyzing patterns that may relate to fraudulent activities.")
st.caption("(data fetched from Blockchain.info)")
st.write("Explore the following metrics:")

@st.cache_data(ttl=600)
def fetch_blockchain_data(chart_type):
    url = f"https://api.blockchain.info/charts/{chart_type}?timespan=30days&format=json&cors=true"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['values'])
    df['x'] = pd.to_datetime(df['x'], unit='s')
    df.rename(columns={'x': 'Date', 'y': chart_type}, inplace=True)
    return df

# --- Fetch all needed data ---
tx_count_df = fetch_blockchain_data("n-transactions")
output_volume_df = fetch_blockchain_data("output-volume")
mempool_df = fetch_blockchain_data("mempool-size")
block_size_df = fetch_blockchain_data("avg-block-size")

# ---- Compute average transaction value (BTC) ----
avg_tx_value_df = pd.DataFrame()
avg_tx_value_df['Date'] = tx_count_df['Date']
avg_tx_value_df['AvgTxValue'] = output_volume_df['output-volume'] / tx_count_df['n-transactions']

#  Number of Transactions Per Day
fig1 = px.line(tx_count_df, x='Date', y='n-transactions', title=' Number of Transactions Per Day')
fig1.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="Transactions(USD)")
st.plotly_chart(fig1)
st.caption("A sudden spike in the number of transactions could indicate fraudulent activities like spam attacks or money laundering")

#  Average Transaction Value
fig2 = px.line(avg_tx_value_df, x='Date', y='AvgTxValue', title=' Avg. Transaction Value (BTC)')
fig2.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="BTC")
st.plotly_chart(fig2)
st.caption("A sudden increase in transaction volume may indicate a surge in illicit activities or market manipulation")

#  Mempool Size
fig3 = px.area(mempool_df, x='Date', y='mempool-size', title=' Mempool Size (Pending Transactions)')
fig3.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="Transactions in Mempool")
st.plotly_chart(fig3)
st.caption("A congested mempool (pending transactions) could indicate network manipulation or spam attacks")

#  Average Block Size
fig4 = px.bar(block_size_df, x='Date', y='avg-block-size', title=' Average Block Size (MB)')
fig4.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="MB")
st.plotly_chart(fig4)
st.caption("Unusually large or sstmall block sizes might be linked to abnormal transaction patterns")

# ---- Get BTC daily transaction volume ----
def get_btc_tx_volume():
    url = "https://api.blockchain.info/charts/output-volume?timespan=7days&format=json"
    res = requests.get(url)
    data = res.json()
    return pd.DataFrame(data['values'])

volume_df = get_btc_tx_volume()
volume_df['Date'] = pd.to_datetime(volume_df['x'], unit='s')
volume_df['BTC Volume (B)'] = volume_df['y'] / 1e9

fig5 = px.line(volume_df, x='Date', y='BTC Volume (B)', title=' BTC Output Volume (Last 7 Days)')
fig5.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="BTC Volume (B)")
st.plotly_chart(fig5)
st.caption("Large transaction volumes might be associated with suspicious activities, such as transferring large sums of cryptocurrency to obfuscate the origin of funds")

##########################################################################################################
# can delete
# Load test data

df_path = "C:/Users/Enduser/OneDrive - Asia Pacific University/uni/Y3S2/fyp/Model_trial/btc_trial_dataset2.csv"
dataset = pd.read_csv(df_path)

# data = pd.read_csv('X_test.csv')
# y_test = pd.read_csv('y_test.csv')['out_and_tx_malicious']
# data = pd.concat([data, y_test], axis=1)

# Streamlit App Layout
st.title("Blockchain Transaction Explorer")
st.write("Browse a sample of blockchain transactions.")

# Filters
st.subheader("Filter Transactions")
fraud_filter = st.selectbox("Fraud Status", ["All", "Fraud (1)", "Non-Fraud (0)"])
btc_range = st.slider("Total BTC Range", float(dataset['total_btc'].min()), float(dataset['total_btc'].max()), (0.0, 1000.0))

# Apply Filters
filtered_data = dataset.copy()
if fraud_filter != "All":
    filtered_data = filtered_data[filtered_data['out_and_tx_malicious'] == (1 if fraud_filter == "Fraud (1)" else 0)]
filtered_data = filtered_data[(filtered_data['total_btc'] >= btc_range[0]) & (filtered_data['total_btc'] <= btc_range[1])]

# Display Data
st.subheader(f"Filtered Transactions ({len(filtered_data)})")
st.dataframe(filtered_data.head(10))  # Show first 10 rows

# Navigation Instructions
st.subheader("Explore More")
st.write("Use the sidebar to navigate to:")
st.markdown("- **Fraud Detection**: Predict fraud on transactions.")
st.markdown("- **Visualizations**: Visualize fraud patterns.")

# Footer
st.write("---")
# st.write("Built with Streamlit by [Your Name] | Blockchain Fraud Detection | April 2025")



# import streamlit as st
# import pandas as pd

# # Load full dataset for summary (or use test data)
# data = pd.read_csv("C:/Users/Enduser/OneDrive - Asia Pacific University/uni/Y3S2/fyp/Model_trial/btc_trial_dataset2.csv")  # Adjust path if needed

# # Calculate summary stats
# total_transactions = len(data)
# fraud_count = data['out_and_tx_malicious'].sum()
# fraud_percent = (fraud_count / total_transactions) * 100
# avg_total_btc = data['total_btc'].mean()
# max_total_btc = data['total_btc'].max()

# # Streamlit App Layout
# st.title("Blockchain Fraud Detection System")
# st.write("Welcome to the Blockchain Fraud Detection System. Explore transaction analysis and fraud patterns.")

# # Summary Stats
# st.subheader("Dataset Overview")
# st.write(f"**Total Transactions:** {total_transactions:,}")
# st.write(f"**Fraudulent Transactions:** {fraud_count:,} ({fraud_percent:.2f}%)")
# st.write(f"**Average Total BTC per Transaction:** {avg_total_btc:,.2f}")
# st.write(f"**Maximum Total BTC in a Transaction:** {max_total_btc:,.2f}")

# # Navigation Instructions
# st.subheader("Explore More")
# st.write("Use the sidebar to navigate to:")
# st.markdown("- **Fraud Detection**: Predict fraud in blockchain transactions.")
# st.markdown("- **Visualizations**: Analyze fraud patterns visually.")

# # Footer
# st.write("---")
# st.write("Built with Streamlit by [Your Name] | Blockchain Fraud Detection | April 2025")

#################################################################
# CRYPTO PRICE DASHBOARD

# import streamlit as st
# import requests
# import pandas as pd

# # Function to fetch crypto prices from CoinGecko API
# def fetch_crypto_prices():
#     url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,binancecoin,ripple,cardano&vs_currencies=usd"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         st.error("Failed to fetch crypto prices. Please try again later.")
#         return {}

# # Streamlit App Layout
# st.title("Cryptocurrency Price Dashboard")
# st.write("Real-time prices of top cryptocurrencies (updated every minute).")

# # Fetch and display prices
# if st.button("Refresh Prices"):
#     prices = fetch_crypto_prices()
#     if prices:
#         crypto_data = {
#             "Cryptocurrency": ["Bitcoin (BTC)", "Ethereum (ETH)", "Binance Coin (BNB)", "Ripple (XRP)", "Cardano (ADA)"],
#             "Price (USD)": [
#                 prices.get("bitcoin", {}).get("usd", "N/A"),
#                 prices.get("ethereum", {}).get("usd", "N/A"),
#                 prices.get("binancecoin", {}).get("usd", "N/A"),
#                 prices.get("ripple", {}).get("usd", "N/A"),
#                 prices.get("cardano", {}).get("usd", "N/A")
#             ]
#         }
#         st.table(pd.DataFrame(crypto_data))

# # Navigation Instructions
# st.subheader("Explore More")
# st.write("Use the sidebar to navigate to:")
# st.markdown("- **Fraud Detection**: Analyze blockchain fraud predictions.")
# st.markdown("- **Visualizations**: View fraud pattern visualizations.")

# # Footer
# st.write("---")
# st.write("Built with Streamlit by [Your Name] | Crypto Price & Fraud Detection | April 2025")






################################################################

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score, precision_recall_curve

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score
# import matplotlib.pyplot as plt
# import seaborn as sns


# # Load models and scalers
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
# # X_test = pd.read_csv('X_test.csv')
# # y_test = pd.read_csv('y_test.csv')['out_and_tx_malicious']

# X_test = pd.read_csv('X_val.csv')
# y_test = pd.read_csv('y_val.csv')['out_and_tx_malicious']
 

 
# df_path = "C:/Users/Enduser/OneDrive - Asia Pacific University/uni/Y3S2/fyp/Model_trial/btc_trial_dataset2.csv"
# data = pd.read_csv(df_path)

# # Streamlit Layout
# st.title("Fraud Prediction Dashboard - Multi-Model Comparison")
# st.write("Explore and compare predictions from multiple fraud detection models.")

# # Navigation to Visualizations Page
# st.subheader("Explore More")
# st.write("Check out fraud pattern visualizations:")
# if st.button("Go to Visualizations"):
#     st.info("Please select 'visualizations' from the sidebar on the left to view fraud pattern visualizations.")

# # Sidebar for User Inputs
# st.sidebar.header("Prediction Settings")
# model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
# index = st.sidebar.slider("Select Test Sample Index", 0, len(X_test) - 1, 0)
# threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.5, step=0.05)
# st.sidebar.info("Lower thresholds increase recall (catch more fraud cases), while higher thresholds increase precision (reduce false positives).")

# # Get selected model and scaler
# selected_model = models[model_name]["model"]
# scaler = models[model_name]["scaler"]
# X_test_scaled = scaler.transform(X_test) if scaler else X_test.to_numpy()

# st.subheader(f"Prediction for {model_name}")
# sample = X_test_scaled[index].reshape(1, -1)
# proba = selected_model.predict_proba(sample)[0, 1]
# pred = 1 if proba >= threshold else 0

# # Prediction for Selected Sample
# st.write(f"**Sample Index:** {index}")
# st.write(f"**Fraud Probability:** {proba:.5f}")
# # st.write(f"**Prediction (Threshold {threshold}):** {'Fraud' if pred == 1 else 'Non-Fraud'}")

# if pred == 1:
#     st.error(f"🚨 **Prediction: Fraudulent Transaction! (Threshold {threshold})**")
# else:
#     st.success(f"✅ **Prediction: Non-Fraudulent Transaction! (Threshold {threshold})**")
# st.write(f"**True Label:** {'Fraud' if y_test[index] == 1 else 'Non-Fraud'}")

# # Display sample features
# if st.checkbox("Show Sample Features"):
#     st.write("**Input Features:**")
#     st.dataframe(pd.DataFrame(X_test.iloc[index]).T)

# # Test Set Performance for Selected Model
# st.subheader(f"Test Set Performance - {model_name}")
# if st.button("Evaluate Selected Model"):
#     y_test_proba = selected_model.predict_proba(X_test_scaled)[:, 1]
#     y_test_pred = (y_test_proba >= threshold).astype(int)
    
#     st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_test_proba):.4f}")
#     st.write(f"**Recall:** {recall_score(y_test, y_test_pred):.4f}")
#     st.write(f"**F1-Score:** {f1_score(y_test, y_test_pred):.4f}")
    
#     st.write("**Classification Report:**")
#     report = classification_report(y_test, y_test_pred, output_dict=True)
#     st.table(pd.DataFrame(report).transpose())

#     precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
#     f1_scores = 2 * (precision * recall) / (precision + recall)
#     optimal_threshold = thresholds[np.argmax(f1_scores)]
#     st.sidebar.write(f"Optimal Threshold (F1-Score): {optimal_threshold:.2f}")

# # Compare All Models
# st.subheader("Compare All Models")
# if st.button("Show Model Comparison"):
#     comparison = {}
#     for name, model_dict in models.items():
#         model = model_dict["model"]  # Use consistent "model" key
#         X_scaled = model_dict["scaler"].transform(X_test) if model_dict["scaler"] else X_test.to_numpy()
#         y_proba = model.predict_proba(X_scaled)[:, 1]
#         y_pred = (y_proba >= threshold).astype(int)
#         comparison[name] = {
#             "ROC AUC": roc_auc_score(y_test, y_proba),
#             "Recall": recall_score(y_test, y_pred),
#             "F1-Score": f1_score(y_test, y_pred)
#         }
#     st.write("**Performance Comparison (Threshold = {:.2f}):**".format(threshold))
#     st.table(pd.DataFrame(comparison).T)


# # Model Predictions for Selected Sample
# st.subheader("Model Predictions for Selected Sample")
# sample_probas = {}
# for name, model_dict in models.items():
#     model = model_dict["model"]
#     scaler = model_dict["scaler"]
#     X_scaled = scaler.transform(X_test) if scaler else X_test.to_numpy()
#     sample_proba = model.predict_proba(X_scaled[index].reshape(1, -1))[0, 1]
#     sample_probas[name] = sample_proba

# st.write("**Fraud Probabilities for Selected Sample:**")
# st.table(pd.DataFrame(sample_probas, index=["Fraud Probability"]).T)

# # Display Fraudulent Samples
# st.subheader("Fraudulent Samples")
# if st.button("Show Fraudulent Samples"):
#     # Predict probabilities for the entire test set
#     y_test_proba = selected_model.predict_proba(X_test_scaled)[:, 1]
#     # Convert probabilities to binary predictions using the threshold
#     y_test_pred = (y_test_proba >= threshold).astype(int)
#     # Filter the test dataset for fraudulent samples
#     fraudulent_samples = X_test[y_test_pred == 1]
#     # Display the number of fraudulent samples and their details
#     st.write(f"**Number of Fraudulent Samples Identified:** {len(fraudulent_samples)}")
#     st.dataframe(fraudulent_samples)

# # Fraud Pattern Visualizations
# # st.subheader("Fraud Pattern Visualizations")
# # viz_option = st.selectbox("Select Visualization", [
# #     "Transaction Volumes (Boxplot)",
# #     "out_malicious Prevalence (Bar Chart)",
# #     "indegree Distribution (Boxplot)",
# #     "total_btc vs. indegree (Scatter)"
# # ])

# # sns.set(style="whitegrid")
# # if viz_option == "Transaction Volumes (Boxplot)":
# #     fig, axes = plt.subplots(1, 3, figsize=(12, 6))
# #     for i, col in enumerate(['total_btc', 'in_btc', 'out_btc']):
# #         sns.boxplot(x='out_and_tx_malicious', y=col, data=data, showfliers=False, ax=axes[i])
# #         axes[i].set_title(f'{col} by Fraud Status')
# #         axes[i].set_xlabel('Fraud (1) vs Non-Fraud (0)')
# #         axes[i].set_ylabel(col)
# #     plt.tight_layout()
# #     st.pyplot(fig)

# # elif viz_option == "out_malicious Prevalence (Bar Chart)":
# #     fig, ax = plt.subplots(figsize=(8, 6))
# #     sns.countplot(x='out_malicious', hue='out_and_tx_malicious', data=data, ax=ax)
# #     ax.set_title('Prevalence of out_malicious by Fraud Status')
# #     ax.set_xlabel('out_malicious (0 or 1)')
# #     ax.set_ylabel('Count')
# #     ax.legend(title='Fraud Status', labels=['Non-Fraud (0)', 'Fraud (1)'])
# #     st.pyplot(fig)

# # elif viz_option == "indegree Distribution (Boxplot)":
# #     fig, ax = plt.subplots(figsize=(8, 6))
# #     sns.boxplot(x='out_and_tx_malicious', y='indegree', data=data, showfliers=False, ax=ax)
# #     ax.set_title('indegree by Fraud Status')
# #     ax.set_xlabel('Fraud (1) vs Non-Fraud (0)')
# #     ax.set_ylabel('indegree')
# #     st.pyplot(fig)

# # elif viz_option == "total_btc vs. indegree (Scatter)":
# #     fig, ax = plt.subplots(figsize=(10, 6))
# #     sns.scatterplot(x='total_btc', y='indegree', hue='out_and_tx_malicious', data=data, alpha=0.5, ax=ax)
# #     ax.set_title('total_btc vs. indegree by Fraud Status')
# #     ax.set_xlabel('total_btc')
# #     ax.set_ylabel('indegree')
# #     ax.set_xscale('log')  # Log scale for wide range
# #     ax.legend(title='Fraud Status', labels=['Non-Fraud (0)', 'Fraud (1)'])
# #     st.pyplot(fig)

# # Footer
# st.write("---")
# st.write("Built with Streamlit by [Your Name] | Multi-Model Fraud Detection | April 2025")

# # # Load all five models and scalers with consistent keys
# # models = {
# #     "XG Boost Model": {"model": joblib.load('xgb_model.pkl'), "scaler": joblib.load('scaler_xgb.pkl')},

# #     # bad model
# #     "XG Boost Model2": {"model": joblib.load('best_xgb_model.pkl'), "scaler": joblib.load('scaler_xgb.pkl')},

# #     "Random Forest Model": {"model": joblib.load('rf_model.pkl'), "scaler": joblib.load('scaler_rf.pkl')},
# #     "Random Forest Model 2": {"model": joblib.load('best_rf_model.pkl'), "scaler": joblib.load('scaler_rf.pkl')},

# #     # "Light GBM Model2": {"model": joblib.load('lgbm_final_best_model_optimal_threshold.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
# #     # good - Light GBM Model4
# #     "Light GBM Model4": {"model": joblib.load('lgbm_model.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
# #     # no good 
# #     "Light GBM Model5": {"model": joblib.load('lgbm_model_5.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
    

# #     "Ensemble Model": {"model": joblib.load('ensemble_model.pkl'), "scaler": joblib.load('scaler_ensemble.pkl')},
# # }

# # # Load test data
# # X_test = pd.read_csv('X_test.csv')
# # y_test = pd.read_csv('y_test.csv')['out_and_tx_malicious']
# # data = pd.concat([X_test, y_test], axis=1)

# # # Reapply preprocessing (to match training)
# # log_features = ['indegree', 'outdegree', 'in_btc', 'out_btc', 'total_btc']
# # data[log_features] = np.log1p(data[log_features])

# # def add_features(df):
# #     df['out_malicious_to_total_btc'] = df['out_malicious'] / (df['total_btc'] + 1e-6)
# #     df['log_total_btc'] = np.log1p(df['total_btc'])
# #     df['out_malicious_in_btc_interaction'] = df['out_malicious'] * df['in_btc']
# #     df['net_btc_flow'] = df['in_btc'] - df['out_btc']
# #     return df

# # data_fe = add_features(data)
# # selected_features = [
# #     'in_btc', 'out_btc', 'total_btc', 'out_malicious',
# #     'out_malicious_to_total_btc', 'log_total_btc',
# #     'out_malicious_in_btc_interaction', 'net_btc_flow'
# # ]
# # data_final = data_fe[selected_features + ['out_and_tx_malicious']]

# # # Streamlit App Layout
# # st.title("Fraud Prediction Dashboard - Multi-Model Comparison")
# # st.write("Explore and compare predictions from multiple fraud detection models.")

# # # Sidebar for User Inputs
# # st.sidebar.header("Prediction Settings")
# # model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
# # index = st.sidebar.slider("Select Test Sample Index", 0, len(X_test) - 1, 0)
# # threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.5, step=0.05)

# # # Prepare data for selected model
# # selected_model = models[model_name]["model"]
# # scaler = models[model_name]["scaler"]
# # X_test_final = data_final[selected_features]
# # X_test_scaled = scaler.transform(X_test_final) if scaler else X_test_final.to_numpy()

# # # Prediction for Selected Sample
# # st.subheader(f"Prediction for {model_name}")
# # sample = X_test_scaled[index].reshape(1, -1)
# # proba = selected_model.predict_proba(sample)[0, 1]
# # pred = 1 if proba >= threshold else 0

# # st.write(f"**Sample Index:** {index}")
# # st.write(f"**Fraud Probability:** {proba:.5f}")
# # st.write(f"**Prediction (Threshold {threshold}):** {'Fraud' if pred == 1 else 'Non-Fraud'}")
# # st.write(f"**True Label:** {'Fraud' if y_test[index] == 1 else 'Non-Fraud'}")

# # # Test Set Performance for Selected Model
# # st.subheader(f"Test Set Performance - {model_name}")
# # if st.button("Evaluate Selected Model"):
# #     y_test_proba = selected_model.predict_proba(X_test_scaled)[:, 1]
# #     y_test_pred = (y_test_proba >= threshold).astype(int)
# #     st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_test_proba):.4f}")
# #     st.table(pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).transpose())

# # # Fraud Pattern Visualizations
# # st.subheader("Fraud Pattern Visualizations")
# # viz_option = st.selectbox("Select Visualization", [
# #     "Transaction Volumes (Boxplot)",
# #     "out_malicious Prevalence (Bar Chart)",
# #     "indegree Distribution (Boxplot)",
# #     "total_btc vs. indegree (Scatter)"
# # ])

# # sns.set(style="whitegrid")
# # if viz_option == "Transaction Volumes (Boxplot)":
# #     fig, axes = plt.subplots(1, 3, figsize=(12, 6))
# #     for i, col in enumerate(['total_btc', 'in_btc', 'out_btc']):
# #         sns.boxplot(x='out_and_tx_malicious', y=col, data=data, showfliers=False, ax=axes[i])
# #         axes[i].set_title(f'{col} by Fraud Status')
# #         axes[i].set_xlabel('Fraud (1) vs Non-Fraud (0)')
# #         axes[i].set_ylabel(col)
# #     plt.tight_layout()
# #     st.pyplot(fig)

# # elif viz_option == "out_malicious Prevalence (Bar Chart)":
# #     fig, ax = plt.subplots(figsize=(8, 6))
# #     sns.countplot(x='out_malicious', hue='out_and_tx_malicious', data=data, ax=ax)
# #     ax.set_title('Prevalence of out_malicious by Fraud Status')
# #     ax.set_xlabel('out_malicious (0 or 1)')
# #     ax.set_ylabel('Count')
# #     ax.legend(title='Fraud Status', labels=['Non-Fraud (0)', 'Fraud (1)'])
# #     st.pyplot(fig)

# # elif viz_option == "indegree Distribution (Boxplot)":
# #     fig, ax = plt.subplots(figsize=(8, 6))
# #     sns.boxplot(x='out_and_tx_malicious', y='indegree', data=data, showfliers=False, ax=ax)
# #     ax.set_title('indegree by Fraud Status')
# #     ax.set_xlabel('Fraud (1) vs Non-Fraud (0)')
# #     ax.set_ylabel('indegree')
# #     st.pyplot(fig)

# # elif viz_option == "total_btc vs. indegree (Scatter)":
# #     fig, ax = plt.subplots(figsize=(10, 6))
# #     sns.scatterplot(x='total_btc', y='indegree', hue='out_and_tx_malicious', data=data, alpha=0.5, ax=ax)
# #     ax.set_title('total_btc vs. indegree by Fraud Status')
# #     ax.set_xlabel('total_btc')
# #     ax.set_ylabel('indegree')
# #     ax.set_xscale('log')  # Log scale for wide range
# #     ax.legend(title='Fraud Status', labels=['Non-Fraud (0)', 'Fraud (1)'])
# #     st.pyplot(fig)

# # # Footer
# # st.write("---")
# # st.write("Built with Streamlit by [Your Name] | Multi-Model Fraud Detection | April 2025")
