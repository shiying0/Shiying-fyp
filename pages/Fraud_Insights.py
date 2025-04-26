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
        numeric_cols = [col for col in data.select_dtypes(include=np.number).columns if col != "out_malicious"]
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
                "Fraud Distribution (Pie Chart)",
                "2D Scatter Plot",
                "Boxplot by Prediction",
                "Histogram",
                "Feature Importance (Correlation)",
                "Scatter Matrix",
                "Histogram/KDE"
            ])

            label_col = "Prediction" if "Prediction" in data.columns else "out_malicious"

            if viz_option == "Boxplot by Prediction":
                selected_feature = st.selectbox("Select feature for boxplot", selected_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=label_col, y=selected_feature, data=data, ax=ax)
                st.pyplot(fig)
                st.caption("Boxplot shows the distribution of the selected feature by fraud status.")        

            elif viz_option == "Histogram":
                st.subheader("Feature Histogram (Normalized)")

                selected_hist = st.selectbox("Select feature to plot histogram", selected_cols, key="hist")

                fig = px.histogram(
                    data,
                    x=selected_hist,
                    color=label_col,
                    nbins=30,
                    histnorm="probability density",  
                    color_discrete_map={
                        "Non-Fraud": "#1f77b4",
                        "Fraud": "#d62728"
                    },
                    opacity=0.7
                )

                fig.update_layout(
                    title=f"Normalized Histogram of {selected_hist} by Fraud Status",
                    xaxis_title=selected_hist,
                    yaxis_title="Density",
                    legend_title="Fraud Status",
                    bargap=0.05,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    dragmode="zoom"  # Zoomable
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption("Normalized histogram showing fraud and non-fraud distributions. Fraud cases are now visible even with class imbalance.")


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

            elif viz_option == "2D Scatter Plot":
                st.subheader("2D Scatter Plot (Interactive - Zoomable)")

                x_feature = st.selectbox("Select X-axis feature", selected_cols, key="2dscatter_x")
                y_feature = st.selectbox("Select Y-axis feature", selected_cols, index=1 if len(selected_cols) > 1 else 0, key="2dscatter_y")

                #  Ensure label column is string for proper color mapping
                data = data.copy()
                if data[label_col].dtype != "object":
                    data[label_col] = data[label_col].map({0: "Non-Fraud", 1: "Fraud"})

                fig = px.scatter(
                    data,
                    x=x_feature,
                    y=y_feature,
                    color=label_col,
                    color_discrete_map={
                        "Non-Fraud": "#1f77b4",
                        "Fraud": "#d62728"
                    },
                    title=f"{x_feature} vs {y_feature} by {label_col}",
                    opacity=0.8,
                    log_x=True,
                    log_y=True,
                    height=600
                )

                fig.update_layout(
                    xaxis_title=x_feature,
                    yaxis_title=y_feature,
                    legend_title="Fraud Label",
                    dragmode='zoom',
                    margin=dict(l=40, r=40, t=40, b=40),
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption("Interactive scatter plot. You can zoom, pan, and hover to explore fraud patterns across two features (optimized for light mode).")

            elif viz_option == "Histogram/KDE":
                st.subheader("Histogram / KDE Plot by Fraud Status")

                selected_feature = st.selectbox("Select feature for histogram/KDE plot", selected_cols, key="hist_kde")

                df_viz = data.copy()

                # Step 1: Clip extreme outliers first
                upper_clip = df_viz[selected_feature].quantile(0.99)
                df_viz[selected_feature] = df_viz[selected_feature].clip(upper=upper_clip)

                # Step 2: IMPORTANT! Remove rows where selected_feature <= 0 (for log scale safety)
                df_viz = df_viz[df_viz[selected_feature] > 0]

                # Step 3: Drop any NaN or inf remaining
                df_viz = df_viz[np.isfinite(df_viz[selected_feature])]
                df_viz = df_viz.dropna(subset=[selected_feature])

                # Step 4: Make sure label is mapped properly
                if df_viz[label_col].dtype != "object":
                    df_viz[label_col] = df_viz[label_col].map({0: "Non-Fraud", 1: "Fraud"})

                # Step 5: Plot safely
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(
                    data=df_viz,
                    x=selected_feature,
                    hue=label_col,
                    bins=50,
                    kde=True,
                    log_scale=True,
                    stat="density",  # ✅ normalize so fraud curve becomes visible
                    common_norm=False,  # ✅ treat each label separately
                    palette={"Non-Fraud": "#1f77b4", "Fraud": "#d62728"},
                    alpha=0.7
                )


                plt.title(f"Histogram and KDE of {selected_feature} by Fraud Status (log scale)")
                plt.xlabel(selected_feature)
                plt.ylabel("Count")
                plt.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)

                st.caption("Histogram shows the distribution of selected feature, with Kernel Density Estimate overlayed. Log-scale applied. Zeros and negatives removed for safe plotting.")
                