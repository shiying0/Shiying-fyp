import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
from sklearn.decomposition import PCA

# # Load the full dataset (or test data if sufficient)

# # Alternatively, use test data: 
# # X_test = pd.read_csv('../X_test.csv')
# # y_test = pd.read_csv('../y_test.csv')['out_and_tx_malicious']
# # data = pd.concat([X_test, y_test], axis=1)

df_path = "C:/Users/Enduser/OneDrive - Asia Pacific University/uni/Y3S2/fyp/Model_trial/btc_trial_dataset2.csv"
data = pd.read_csv(df_path)

# # Streamlit Page Layout
# st.title("Bitcoin Fraud Pattern Visualizations")
# st.write("Analyze fraud patterns within the blockchain transactions data collected.")

# # Visualization Selection
# st.subheader("Select Visualization")
# viz_option = st.selectbox("Choose a Visualization", [
#     "Transaction Volumes (Boxplot)",
#     "out_malicious Prevalence (Pie Chart)",
#     "indegree Distribution (Boxplot)",
#     "total_btc vs. indegree (Scatter)",
#     "Fraudulent Data Correlation (Heatmap)"
# ])

# # Set plot style
# sns.set(style="whitegrid")

# # Generate Visualizations
# if viz_option == "Transaction Volumes (Boxplot)":
#     fig, axes = plt.subplots(1, 3, figsize=(12, 6))
#     for i, col in enumerate(['total_btc', 'in_btc', 'out_btc']):
#         sns.boxplot(x='out_and_tx_malicious', y=col, data=data, showfliers=False, ax=axes[i])
#         axes[i].set_title(f'{col} by Fraud Status')
#         axes[i].set_xlabel('Fraud (1) vs Non-Fraud (0)')
#         axes[i].set_ylabel(col)
#     plt.tight_layout()
#     st.pyplot(fig)

# elif viz_option == "out_malicious Prevalence (Pie Chart)":
#     fig, ax = plt.subplots(figsize=(35, 70))  # Adjusted figure size for better readability
#     fraud_counts = data.groupby(['out_malicious', 'out_and_tx_malicious']).size().unstack(fill_value=0)
#     fraud_counts = fraud_counts.sum(axis=1)  # Aggregate counts across fraud statuses
#     fraud_counts.plot.pie(
#         autopct='%1.1f%%', 
#         labels=['Non-Malici ous (0)', 'Malicious (1)'], 
#         ax=ax, 
#         startangle=90, 
#         colors=['#66ff66', '#ff6666'],  
#         textprops={'fontsize': 50}  # Increased font size for better readability
#     )
#     ax.set_ylabel('') 
#     ax.set_title('Prevalence of out_malicious', fontsize=14)  # Increased title font size
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

# elif viz_option == "Fraudulent Data Correlation (Heatmap)":
#     fraud_data = data[data['out_and_tx_malicious'] == 1]  # Filter fraudulent data
#     corr_matrix = fraud_data.select_dtypes(include=[np.number]).corr()  # Compute correlation matrix for numeric columns only
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#     ax.set_title('Correlation Heatmap for Fraudulent Data')
#     st.pyplot(fig)

# ######################################################################################

# from sklearn.decomposition import PCA
# import shap
# import plotly.express as px
# # Define the columns to plot
# cols_to_plot = ['in_btc', 'out_btc', 'total_btc', 'out_malicious', 'indegree','outdegree']  # Add relevant column names here

# # --- B. Correlation Heatmap (Fraud Only) ---
# st.subheader(" Correlation Heatmap (Fraud Cases)")
# fraud_df = data[data['out_and_tx_malicious'] == 1]
# if len(fraud_df) > 2:
#     plt.figure(figsize=(8, 5))
#     sns.heatmap(fraud_df[cols_to_plot].corr(), cmap='coolwarm', annot=True)
#     st.pyplot(plt.gcf())
# else:
#     st.info("Not enough fraud data to generate a correlation heatmap.")

# # --- C. PCA Projection ---
# st.subheader(" PCA Projection of Feature Space")
# pca = PCA(n_components=2)
# components = pca.fit_transform(data[cols_to_plot])
# pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
# pca_df['out_and_tx_malicious'] = data['out_and_tx_malicious'].values

# fig = px.scatter(pca_df, x='PC1', y='PC2', color='out_and_tx_malicious',
#                  title="PCA Clustering: Fraud vs Normal")
# st.plotly_chart(fig, use_container_width=True)

# explained = pca.explained_variance_ratio_
# st.write(f"PC1 explains {explained[0]*100:.2f}% of variance")
# st.write(f"PC2 explains {explained[1]*100:.2f}% of variance")
# with st.expander("PCA Interpretation"):
#     st.markdown(f"""
#     - **PC1 (Principal Component 1)** explains **{pca.explained_variance_ratio_[0]*100:.2f}%** of the variance.
#     - **PC2 (Principal Component 2)** explains only **{pca.explained_variance_ratio_[1]*100:.2f}%**.
#     - Most of the variance in the dataset is driven by PC1, making it highly informative.
#     - Fraud patterns may be distinguishable along PC1 based on feature combinations like transaction value, block size, etc.
#     """)
# # Display feature loadings for PC1 and PC2
# with st.expander("Feature Contributions to PCA"):
#     feature_loadings = pd.DataFrame(pca.components_, columns=cols_to_plot, index=['PC1', 'PC2'])
#     st.write("Feature Loadings for PC1 and PC2:")
#     st.dataframe(feature_loadings)
    
# # --- D. Feature Importance via SHAP (Simulated) ---
# st.subheader(" Feature Importance (SHAP Style)")

# importances = data[cols_to_plot].corrwith(data['out_and_tx_malicious']).abs().sort_values(ascending=False)
# fig = px.bar(importances, orientation='h', title="Feature Influence on Fraud Predictions")
# fig.update_layout(yaxis_title="Feature", xaxis_title="|Correlation with Prediction|")
# st.plotly_chart(fig, use_container_width=True)

# # --- E. Interactive Scatter Matrix ---
# st.subheader(" Scatter Matrix")
# fig = px.scatter_matrix(data, dimensions=cols_to_plot, color="out_and_tx_malicious")
# st.plotly_chart(fig, use_container_width=True)



# # Footer
# st.write("---")
# # st.write("Built with Streamlit by [Your Name] | Fraud Pattern Analysis | April 2025")
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Use a direct .json animation file
# animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_tno6cg2w.json")
# st_lottie(animation, height=300)

animation = load_lottie_url("https://lottie.host/d6154002-a76b-4c0f-872d-9a95dab89e8b/SC5egFNZb5.json")
st_lottie(animation, height=300)

st.title("Bitcoin Fraud Pattern Visualizations")
st.write("Analyze fraud patterns interactively within the blockchain transactions data.")

# Column selection
default_cols = ['in_btc', 'out_btc', 'total_btc', 'indegree', 'outdegree', 'out_malicious']
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
selected_cols = st.multiselect("Select Columns for Analysis", numeric_cols, default=default_cols)

if len(selected_cols) < 2:
    st.warning("Please select at least two columns for meaningful visualizations.")
else:
    viz_option = st.selectbox("Choose Visualization Type", [
        "Boxplot by Fraud Status",
        "Scatter Plot",
        "Correlation Heatmap",
        "PCA Projection",
        "Feature Importance (Correlation)",
        "Scatter Matrix"
    ])

    if viz_option == "Boxplot by Fraud Status":
        st.subheader("Boxplot Distribution by Fraud Status")
        selected_feature = st.selectbox("Select Feature for Boxplot", selected_cols)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='out_and_tx_malicious', y=selected_feature, data=data, ax=ax, showfliers=False)
        ax.set_title(f'{selected_feature} by Fraud Status')
        st.pyplot(fig)

    elif viz_option == "Scatter Plot":
        st.subheader("Scatter Plot")
        x_axis = st.selectbox("X-axis", selected_cols)
        y_axis = st.selectbox("Y-axis", selected_cols, index=1 if len(selected_cols) > 1 else 0)
        fig = px.scatter(data, x=x_axis, y=y_axis, color="out_and_tx_malicious",
                         title=f"{x_axis} vs {y_axis} by Fraud Status", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Correlation Heatmap":
        st.subheader("Correlation Heatmap (Fraud Cases Only)")
        fraud_data = data[data['out_and_tx_malicious'] == 1]
        if len(fraud_data) >= 3:
            corr = fraud_data[selected_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough fraud data to generate correlation heatmap.")

    elif viz_option == "PCA Projection":
        st.subheader("PCA Projection")
        pca = PCA(n_components=2)
        components = pca.fit_transform(data[selected_cols])
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        pca_df["Fraud"] = data["out_and_tx_malicious"]

        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Fraud", title="PCA Projection of Selected Features")
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance")
        st.write(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.2f}% of variance")

        with st.expander("Feature Loadings"):
            loadings = pd.DataFrame(pca.components_, columns=selected_cols, index=["PC1", "PC2"])
            st.dataframe(loadings)

    elif viz_option == "Feature Importance (Correlation)":
        st.subheader("Feature Correlation with Fraud Status")
        corr_values = data[selected_cols].corrwith(data["out_and_tx_malicious"]).abs().sort_values(ascending=False)
        fig = px.bar(corr_values, orientation='h', title="Feature Importance via Correlation")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Scatter Matrix":
        st.subheader("Scatter Matrix (Fraud vs Non-Fraud)")
        fig = px.scatter_matrix(data, dimensions=selected_cols, color="out_and_tx_malicious")
        st.plotly_chart(fig, use_container_width=True)
