import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import json
import requests
from datetime import datetime
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

from pages import Batch_Fraud_Detection as batch
from pages import Transaction_Checker as checker
from pages import Fraud_Insights as insights

# Set page configuration
st.set_page_config(page_title="Fraud Detection", layout="wide")

# Add parent directory to sys.path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import display_custom_sidebar

# Display the sidebar and get the function to close the content area
close_content = display_custom_sidebar()

# Get the selected page from session state
page = st.session_state.get("selected_page", "dashboard")

# Main content based on the selected page
if page == "dashboard":

    # Main Dashboard Content
    st.markdown("<h1 style='text-align: center;'>Welcome to the Blockchain Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <h5>A blockchain fraud detection platform built to help users, analysts, and investigators identify and analyze suspicious activities within Bitcoin transaction data using the Random Forest Machine Learning Model.</h5>
            <p>Upload transaction datasets, run fraud analysis, and visualize insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("---")

    with open("images/Animation (home_page).json", "r") as f:
        lottie_animation = json.load(f)
    st_lottie(lottie_animation, height=300)

    # Call Coingecko API to get Live Crypto Prices
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

    prices = get_crypto_prices()

    # Display current date and day
    current_date = datetime.now().strftime("%A, %d %B %Y")
    st.write(f"{current_date}")
    st.header("Today's Crypto Prices (USD)")
    col1, col2, col3 = st.columns(3)

    # Crypto icons
    btc_icon = "images/bitcoin-btc-logo.png"  
    eth_icon = "images/ethereum-eth-logo.png"  
    sol_icon = "images/solana-sol-logo.png"  

    # Set the logo size
    btc_icon_size = 120  
    eth_icon_size = 120  
    sol_icon_size = 120  

    # Crypto prices
    # Bitcoin
    with col1:
        st.image(btc_icon, width=btc_icon_size)
        btc_price = prices["bitcoin"]["usd"]
        btc_change = prices["bitcoin"]["usd_24h_change"]
        st.metric("**Bitcoin (BTC)**", f"${btc_price:,.0f}", f"{btc_change:.2f}%")

    # Ethereum
    with col2:
        st.image(eth_icon, width=eth_icon_size)
        eth_price = prices["ethereum"]["usd"]
        eth_change = prices["ethereum"]["usd_24h_change"]
        st.metric("**Ethereum (ETH)**", f"${eth_price:,.0f}", f"{eth_change:.2f}%")

    # Solana
    with col3:
        st.image(sol_icon, width=sol_icon_size)
        sol_price = prices["solana"]["usd"]
        sol_change = prices["solana"]["usd_24h_change"]
        st.metric("**Solana (SOL)**", f"${sol_price:.2f}", f"{sol_change:.2f}%")

    st.caption(" \nMarket activity is often linked to fraud trends in blockchain networks. (data fetched from CoinGecko)")
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
            return pd.DataFrame()  
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

    # Fetch data
    tx_count_df = fetch_blockchain_data("n-transactions")
    output_volume_df = fetch_blockchain_data("output-volume")
    mempool_df = fetch_blockchain_data("mempool-size")
    block_size_df = fetch_blockchain_data("avg-block-size")

    # Compute average transaction value (BTC)
    avg_tx_value_df = pd.DataFrame()
    avg_tx_value_df['Date'] = tx_count_df['Date']
    avg_tx_value_df['AvgTxValue'] = output_volume_df['output-volume'] / tx_count_df['n-transactions']

    # Number of Transactions Per Day
    fig1 = px.line(tx_count_df, x='Date', y='n-transactions', title='Number of Transactions Per Day')
    fig1.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="Transactions(USD)")
    st.plotly_chart(fig1)
    st.caption("A sudden spike in the number of transactions could indicate fraudulent activities like spam attacks or money laundering")

    # Average Transaction Value
    fig2 = px.line(avg_tx_value_df, x='Date', y='AvgTxValue', title='Avg. Transaction Value (BTC)')
    fig2.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="BTC")
    st.plotly_chart(fig2)
    st.caption("A sudden increase in transaction volume may indicate a surge in illicit activities or market manipulation")

    # Mempool Size
    fig3 = px.area(mempool_df, x='Date', y='mempool-size', title='Mempool Size (Pending Transactions)')
    fig3.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="Transactions in Mempool")
    st.plotly_chart(fig3)
    st.caption("A congested mempool (pending transactions) could indicate network manipulation or spam attacks")

    # Average Block Size
    fig4 = px.bar(block_size_df, x='Date', y='avg-block-size', title='Average Block Size (MB)')
    fig4.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="MB")
    st.plotly_chart(fig4)
    st.caption("Unusually large or small block sizes might be linked to abnormal transaction patterns")

    # Get BTC daily transaction volume
    def get_btc_tx_volume():
        url = "https://api.blockchain.info/charts/output-volume?timespan=7days&format=json"
        res = requests.get(url)
        data = res.json()
        return pd.DataFrame(data['values'])

    volume_df = get_btc_tx_volume()
    volume_df['Date'] = pd.to_datetime(volume_df['x'], unit='s')
    volume_df['BTC Volume (B)'] = volume_df['y'] / 1e9

    fig5 = px.line(volume_df, x='Date', y='BTC Volume (B)', title='BTC Output Volume (Last 7 Days)')
    fig5.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="BTC Volume (B)")
    st.plotly_chart(fig5)
    st.caption("Large transaction volumes might be associated with suspicious activities, such as transferring large sums of cryptocurrency to obfuscate the origin of funds")
    st.markdown("### ")
    st.markdown("---")
    st.markdown(
        """
        <section style="
            background: #e0f2fe;
            padding: 30px 25px;
            border-radius: 12px;
            color: #1e3a8a;
            font-family: 'Poppins', sans-serif;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
            margin-top: 40px;
        ">
            <h3 style="margin-bottom: 20px; font-weight: 600; color: #0f172a;"> Explore More Features</h3>
            <p style="font-size: 15px; line-height: 1.6; margin-bottom: 20px;">
                Use the sidebar to access different tools. 
            </p>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="background-color: #bae6fd; padding: 12px 16px; border-radius: 8px;">
                    <strong>📤 Batch Fraud Detection</strong><br>
                    Upload .csv files to scan for suspicious activity across multiple transactions.
                </div>
                <div style="background-color: #bae6fd; padding: 12px 16px; border-radius: 8px;">
                    <strong>🔍 Transaction Checker</strong><br>
                    Input and analyze a single blockchain transaction for fraud likelihood.
                </div>
                <div style="background-color: #bae6fd; padding: 12px 16px; border-radius: 8px;">
                    <strong>📈 Fraud Insights</strong><br>
                    Visualize fraud patterns.
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True
    )
elif page == "batch_fraud_detection":
    batch.render() 

elif page == "transaction_checker":
    checker.render() 

elif page == "fraud_insights":
    insights.render()

else:
    st.error("Page not found.")

# Footer: clean and centered
st.markdown("### ")
st.markdown("---", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; padding: 15px 0; color: #888; font-size: 13px;">
        Built with  <strong>Streamlit</strong> <br>
        <span style="color:#bbb;">Tan Shi Ying • Final Year Project</span><br>
        <span style="color:#aaa;"> April 2025 |  Blockchain Fraud Detection System</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Close the main content area
close_content()