import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.title("📊 Instant ML Training & Plotting Tool")
st.write("Upload your CSV file, pick an algorithm, and view the fitted results instantly.")

# 1. Data Input Options
data_source = st.radio("Choose data input method:", ("Upload CSV", "Paste Data (CSV format)"))

df = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    pasted_data = st.text_area("Paste your CSV data here (include headers):", placeholder="X,Y\n1,2\n2,4\n3,5")
    if pasted_data:
        from io import StringIO
        df = pd.read_csv(StringIO(pasted_data))

# 2. ML Processing & Plotting
if df is not None:
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())
    
    # Let user select X and Y columns
    columns = df.columns.tolist()
    x_col = st.selectbox("Select Target Predictor (X):", columns)
    y_col = st.selectbox("Select Target Variable (Y):", columns)
    
    # Model Selection
    algo = st.selectbox("Choose Machine Learning Algorithm:", ("Linear Regression", "Random Forest"))
    
    if st.button("🚀 Train Model & Plot"):
        X = df[[x_col]].values
        y = df[y_col].values
        
        # Fit Model
        if algo == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=50)
            
        model.fit(X, y)
        
        # Create a smooth line for plotting predictions
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        
        # 3. Plotting the Results
        fig, ax = plt.subplots()
        ax.scatter(X, y, color="blue", label="Actual Data")
        ax.plot(x_line, y_line, color="red", linewidth=2, label=f"Fitted {algo}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        
        st.subheader("📈 Model Results")
        st.pyplot(fig)
        st.success(f"Model trained successfully! R² Score: {model.score(X, y):.4f}")
