# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score, precision_recall_curve

# # Load models and scalers
# models = {
#     "XG Boost Model": {"model": joblib.load('xgb_model.pkl'), "scaler": joblib.load('scaler_xgb.pkl')},
#     "XG Boost Model2": {"model": joblib.load('best_xgb_model.pkl'), "scaler": joblib.load('scaler_xgb.pkl')},
#     "Random Forest Model": {"model": joblib.load('rf_model.pkl'), "scaler": joblib.load('scaler_rf.pkl')},
#     "Random Forest Model 2": {"model": joblib.load('best_rf_model.pkl'), "scaler": joblib.load('scaler_rf.pkl')},
#     "Light GBM Model1": {"model": joblib.load('lgbm_final_best_model.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
#     "Light GBM Model4": {"model": joblib.load('lgbm_model.pkl'), "scaler": joblib.load('scaler_lgbm.pkl')},
# }

# # Load test data
# X_test = pd.read_csv('X_test.csv')
# y_test = pd.read_csv('y_test.csv')['out_and_tx_malicious']

# # Streamlit Layout
# st.title("🚀 Fraud Prediction Dashboard")
# st.write("Compare predictions from multiple fraud detection models.")

# # Sidebar
# st.sidebar.header("⚙️ Prediction Settings")
# model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
# index = st.sidebar.slider("Select Test Sample Index", 0, len(X_test) - 1, 0)
# threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.5, step=0.05)
# st.sidebar.info("Lower thresholds increase recall (catch more fraud cases), while higher thresholds increase precision (reduce false positives).")

# # Get selected model and scaler
# selected_model = models[model_name]["model"]
# scaler = models[model_name]["scaler"]
# X_test_scaled = scaler.transform(X_test) if scaler else X_test.to_numpy()

# # Prediction
# st.subheader(f"🔍 Prediction for {model_name}")
# sample = X_test_scaled[index].reshape(1, -1)
# proba = selected_model.predict_proba(sample)[0, 1]
# pred = 1 if proba >= threshold else 0

# st.write(f"**Sample Index:** {index}")
# st.write(f"**Fraud Probability:** {proba:.5f}")
# if pred == 1:
#     st.error(f"🚨 **Prediction: Fraudulent Transaction! (Threshold {threshold})**")
# else:
#     st.success(f"✅ **Prediction: Non-Fraudulent Transaction! (Threshold {threshold})**")
# st.write(f"**True Label:** {'Fraud' if y_test[index] == 1 else 'Non-Fraud'}")

# # Display Features
# if st.checkbox("Show Sample Features"):
#     st.write("**Input Features:**")
#     st.dataframe(pd.DataFrame(X_test.iloc[index]).T)

# # Model Evaluation
# st.subheader(f"📊 Model Performance - {model_name}")
# if st.button("Evaluate Model"):
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
# st.subheader("📈 Model Comparison")
# if st.button("Compare Models"):
#     comparison = {}
#     for name, model_dict in models.items():
#         model = model_dict["model"]
#         X_scaled = model_dict["scaler"].transform(X_test) if model_dict["scaler"] else X_test.to_numpy()
#         y_proba = model.predict_proba(X_scaled)[:, 1]
#         y_pred = (y_proba >= threshold).astype(int)
#         comparison[name] = {
#             "ROC AUC": roc_auc_score(y_test, y_proba),
#             "Recall": recall_score(y_test, y_pred),
#             "F1-Score": f1_score(y_test, y_pred)
#         }
#     st.write(f"**Performance at Threshold = {threshold:.2f}:**")
#     st.table(pd.DataFrame(comparison).T)

# # Model Predictions for Selected Sample
# st.subheader("🔬 Model Predictions for Selected Sample")
# sample_probas = {}
# for name, model_dict in models.items():
#     model = model_dict["model"]
#     scaler = model_dict["scaler"]
#     X_scaled = scaler.transform(X_test) if scaler else X_test.to_numpy()
#     sample_proba = model.predict_proba(X_scaled[index].reshape(1, -1))[0, 1]
#     sample_probas[name] = sample_proba

# st.write("**Fraud Probabilities for Selected Sample:**")
# st.table(pd.DataFrame(sample_probas, index=["Fraud Probability"]).T)

# # Fraudulent Samples
# st.subheader("🚔 Fraudulent Samples")
# if st.button("Show Fraudulent Samples"):
#     y_test_proba = selected_model.predict_proba(X_test_scaled)[:, 1]
#     y_test_pred = (y_test_proba >= threshold).astype(int)
#     fraudulent_samples = X_test[y_test_pred == 1]
#     st.write(f"**Fraudulent Samples Identified:** {len(fraudulent_samples)}")
#     st.dataframe(fraudulent_samples)

# # Footer
# st.write("---")
# st.write("🔗 Built with Streamlit | Multi-Model Fraud Detection Dashboard | April 2025")
