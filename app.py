import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from io import StringIO

st.set_page_config(page_title="Multi-Feature ML Trainer", layout="wide")
st.title("📊 Multi-Feature ML Training & Evaluation Tool")
st.write("Upload or paste data, select up to 5 features, and compare Random Forest vs. a 3-Layer DNN.")

# 1. Data Input Section
data_source = st.radio("Choose data input method:", ("Upload CSV", "Paste Data (CSV format)"))

df = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    pasted_data = st.text_area(
        "Paste your CSV data here (include headers):", 
        placeholder="Feature1,Feature2,Feature3,Feature4,Feature5,Target\n1,2,3,4,5,10\n2,3,4,5,6,15",
        height=150
    )
    if pasted_data:
        df = pd.read_csv(StringIO(pasted_data))

# 2. Configuration & Model Execution
if df is not None:
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(5))
    
    columns = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        # Limit feature selection up to 5 items
        features = st.multiselect(
            "Select Features (X) - Up to 5 columns:", 
            options=columns, 
            max_selections=5
        )
    with col2:
        target = st.selectbox("Select Target Variable (Y):", columns)
    
    # Algorithm Choice
    algo = st.selectbox("Choose Machine Learning Algorithm:", ("Random Forest", "Deep Neural Network (DNN)"))
    
    # Requirements Check
    if not features:
        st.warning("Please select at least one feature column to begin.")
    elif target in features:
        st.error("The Target Variable (Y) cannot be selected as a Feature (X).")
    else:
        if st.button("🚀 Train Model & Evaluate"):
            X = df[features].values
            y = df[target].values
            
            # 80/20 Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_test_split=0.20, random_state=42
            )
            
            # Initialize Selected Algorithm
            if algo == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                # 3 hidden layers with 64 nodes each
                model = MLPRegressor(
                    hidden_layer_sizes=(64, 64, 64), 
                    max_iter=1000, 
                    random_state=42
                )
            
            # Train the Model
            with st.spinner(f"Training {algo}..."):
                model.fit(X_train, y_train)
            
            # Make Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate MSE Metrics
            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            
            # Display Metrics
            st.subheader("📈 Model Performance Metrics")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric(label="Training MSE", value=f"{mse_train:.4f}")
            m_col2.metric(label="Testing MSE", value=f"{mse_test:.4f}")
            
            # 3. Plotting Results (Actual vs. Predicted Scatter Plots)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Perfect prediction reference line calculations
            all_min = min(y_train.min(), y_test.min())
            all_max = max(y_train.max(), y_test.max())
            
            # Train Scatter Plot
            ax1.scatter(y_train, y_train_pred, color="#1f77b4", alpha=0.7, label="Predictions")
            ax1.plot([all_min, all_max], [all_min, all_max], 'r--', lw=2, label="Perfect Fit")
            ax1.set_title(f"Train Set Evaluation (MSE: {mse_train:.3f})")
            ax1.set_xlabel("Actual Target Values")
            ax1.set_ylabel("Predicted Target Values")
            ax1.legend()
            ax1.grid(True, linestyle=":", alpha=0.6)
            
            # Test Scatter Plot
            ax2.scatter(y_test, y_test_pred, color="#ff7f0e", alpha=0.7, label="Predictions")
            ax2.plot([all_min, all_max], [all_min, all_max], 'r--', lw=2, label="Perfect Fit")
            ax2.set_title(f"Test Set Evaluation (MSE: {mse_test:.3f})")
            ax2.set_xlabel("Actual Target Values")
            ax2.set_ylabel("Predicted Target Values")
            ax2.legend()
            ax2.grid(True, linestyle=":", alpha=0.6)
            
            st.pyplot(fig)
            st.success("App executed successfully! Run again with different settings or files anytime.")
