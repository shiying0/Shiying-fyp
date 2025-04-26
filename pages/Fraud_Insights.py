import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
import sys
import os
import json
from streamlit_lottie import st_lottie

def render():
        
    with open("images/Animation (fraud_insights).json", "r") as f:
        lottie_animation = json.load(f)
    st_lottie(lottie_animation, height=300)
    st.title("Blockchain Fraud Pattern Visualizations")
    st.write("---")

    # === Step 1: Choose data source ===
    data_option = st.radio("Select data source:", [
        "Use data from Fraud Detection page",
        "Upload a new CSV file"
    ])

    # hold the final dataframe
    data = None  

    if data_option == "Use data from Fraud Detection page":
        if 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
            st.success("✅ Using data from Fraud Detection page.")
        else:
            st.warning("⚠️ No data found from Fraud Detection. Please upload a file there first.")
    elif data_option == "Upload a new CSV file":
        uploaded_file = st.file_uploader("Upload your CSV file for visualization", type=["csv"])
        st.caption(""" Make sure that your (.csv) file has the required columns:
        'in_btc', 'out_btc', 'total_btc', 'indegree', 'outdegree', 'out_malicious'
        """)
        st.markdown(" ") 

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully.")
                st.markdown(" ") 
                st.dataframe(data.head())  # Display the first few rows of the dataframe
                st.caption("Preview of the uploaded data.")
                st.markdown("---")  

            except FileNotFoundError:
                st.error("❌ File not found. Please check the file path.")      
            except pd.errors.ParserError:
                st.error("❌ Error parsing the file. Please check the format.")
            except pd.errors.EmptyDataError:
                st.error("❌ The uploaded file is empty.")  
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

    # === Step 2: Proceed with visualization if data is loaded ===
    if data is not None:
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    
        required_columns = {'in_btc', 'out_btc', 'total_btc', 'indegree', 'outdegree', 'out_malicious'}
        available_columns = set(data.columns)

        # Check for required columns
        missing_columns = required_columns - available_columns

        if missing_columns:
            st.error(f"❌ The uploaded file is missing required columns: {', '.join(missing_columns)}")
            st.stop()

        # Safe default columns (only those that exist in the file)
        default_cols = [col for col in required_columns if col in numeric_cols]
        selected_cols = st.multiselect("Select attributes for analysis", numeric_cols, default=default_cols)

        if len(selected_cols) < 2:
            st.warning("Please select at least two numeric columns.")
        else:
            viz_option = st.selectbox("Choose visualization type", [
                "Boxplot by Prediction",
                "Histogram",
                "Fraud Distribution (Pie Chart)",
                "Scatter Plot",
                "Feature Importance (Correlation)",
                "Scatter Matrix",
                "Joint Density Plot (2 Features)"
            ])

            label_col = "Prediction" if "Prediction" in data.columns else "out_malicious"

            if viz_option == "Boxplot by Prediction":
                selected_feature = st.selectbox("Select feature for boxplot", selected_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=label_col, y=selected_feature, data=data, ax=ax)
                st.pyplot(fig)
                st.caption("Boxplot shows the distribution of the selected feature by fraud status.")

            elif viz_option == "Histogram":
                st.subheader("Feature Histogram")
                selected_hist = st.selectbox("Select feature to plot histogram", selected_cols, key="hist")
                fig = px.histogram(data, x=selected_hist, color=label_col, barmode='overlay', nbins=30)
                st.plotly_chart(fig)
                st.caption("Histogram shows the distribution of the selected feature.")

            elif viz_option == "Scatter Plot":
                x_axis = st.selectbox("X-axis", selected_cols)
                y_axis = st.selectbox("Y-axis", selected_cols, index=1 if len(selected_cols) > 1 else 0)
                fig = px.scatter(data, x=x_axis, y=y_axis, color=label_col,
                                title=f"{x_axis} vs {y_axis} by {label_col}")
                st.plotly_chart(fig)
                st.caption("Scatter plot shows the relationship between two selected features.")

            elif viz_option == "Histogram":
                st.subheader("Feature Histogram")
                selected_hist = st.selectbox("Select feature to plot histogram", selected_cols, key="hist")
                fig = px.histogram(data, x=selected_hist, color=label_col, barmode='overlay', nbins=30)
                st.plotly_chart(fig)
                st.caption("Histogram shows the distribution of the selected feature.")

            elif viz_option == "Fraud Distribution (Pie Chart)":
                st.subheader("Fraud vs Non-Fraud Distribution")
                fraud_counts = data[label_col].value_counts()
                fig = px.pie(values=fraud_counts, names=fraud_counts.index, title="Fraud Label Distribution")
                st.plotly_chart(fig)
                st.caption("Pie chart shows the distribution of fraud and non-fraud cases.")
            
            elif viz_option == "Feature Importance (Correlation)":
                if label_col == "Prediction":
                    label_map = {"Fraud": 1, "Non-Fraud": 0}
                    data["PredictionLabel"] = data["Prediction"].map(label_map)
                    corr_values = data[selected_cols].corrwith(data["PredictionLabel"]).abs().sort_values(ascending=False)
                else:
                    corr_values = data[selected_cols].corrwith(data[label_col]).abs().sort_values(ascending=False)
                fig = px.bar(corr_values, orientation='h', title="Feature Importance via Correlation")
                st.plotly_chart(fig)
                st.caption("Feature importance based on correlation with the target variable.")

            elif viz_option == "Scatter Matrix":
                fig = px.scatter_matrix(data, dimensions=selected_cols, color=label_col)
                st.plotly_chart(fig)
                st.caption("Scatter matrix shows pairwise relationships between selected features.")
            
            
