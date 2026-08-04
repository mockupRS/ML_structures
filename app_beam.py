import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Beam Deflection ML App", layout="wide")
st.title("🏗️ Cantilever Beam Deflection: Theory vs. Machine Learning")
st.write("Generate structural data using Beam Theory, train a machine learning model, and compare their curves.")

# --- SIDEBAR: Physics Controls & Parameters ---
st.sidebar.header("🎯 1. Beam Settings")
L = st.sidebar.number_input("Beam Length (L) [m]:", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
P = st.sidebar.number_input("Concentrated Tip Load (P) [N]:", min_value=1.0, max_value=100000.0, value=5000.0, step=500.0)

st.sidebar.header("📊 2. Synthetic Data Generation Range")
ei_min = st.sidebar.number_input("Minimum EI [N·m²]:", min_value=1e4, max_value=1e12, value=1e6, format="%e")
ei_max = st.sidebar.number_input("Maximum EI [N·m²]:", min_value=1e4, max_value=1e12, value=5e6, format="%e")
num_samples = st.sidebar.slider("Number of synthetic data points to generate:", min_value=200, max_value=5000, value=1000, step=100)

st.sidebar.header("🤖 3. ML Model Configuration")
algo = st.sidebar.selectbox("Select ML Algorithm:", ("Random Forest", "Deep Neural Network (DNN)"))

# --- MAIN PAGE: Data Engine ---
if ei_min >= ei_max:
    st.error("Error: Minimum EI must be strictly less than Maximum EI.")
else:
    # 1. Generate Dataset randomly distributed across bounds
    np.random.seed(42)
    rand_ei = np.random.uniform(ei_min, ei_max, num_samples)
    rand_x = np.random.uniform(0, L, num_samples)
    
    # Calculate target variable (vertical displacement) using Euler-Bernoulli theory
    target_displacement = (P * (rand_x**2) * (3 * L - rand_x)) / (6 * rand_ei)
    
    # Put into a DataFrame for ML processing
    df = pd.DataFrame({
        'EI': rand_ei,
        'x_location': rand_x,
        'vertical_displacement': target_displacement
    })
    
    st.subheader("📋 Generated Dataset Preview")
    st.dataframe(df.head(5))
    
    # 2. Train / Test Split (80% Train, 20% Test)
    X = df[['EI', 'x_location']].values
    y = df['vertical_displacement'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # 3. Model Initialization (Neural Networks require scaling for convergence!)
    if algo == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        # Pipeline pairs StandardScaler with 3 Hidden Layers (64 nodes each)
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 64, 64), max_iter=1500, random_state=42)
        )
        
    if st.button("🚀 Train Model & Generate Level Plots"):
        with st.spinner(f"Fitting {algo} model to physics data..."):
            model.fit(X_train, y_train)
        st.success(f"{algo} training complete!")
        
        # --- 4. Plotting Level Curves ---
        st.subheader("📈 Multi-Level Plot: Beam Theory vs. ML Predictions")
        
        # Select 4 evenly spaced discrete EI profiles to showcase as level curves
        ei_levels = np.linspace(ei_min, ei_max, 4)
        x_grid = np.linspace(0, L, 100)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Assign distinct color maps for contrast
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(ei_levels)))
        
        for idx, ei_val in enumerate(ei_levels):
            # Calculate Pure Analytical Physics Values
            y_theory = (P * (x_grid**2) * (3 * L - x_grid)) / (6 * ei_val)
            
            # Predict through ML model
            # Prepare mock input dataset grid: [[EI, x1], [EI, x2], ...]
            ml_input = np.column_stack((np.full_like(x_grid, ei_val), x_grid))
            y_ml_pred = model.predict(ml_input)
            
            # Format label info using scientific notation
            ei_label = f"{ei_val:.2e}"
            
            # Plot solid lines for true beam theory
            ax.plot(x_grid, y_theory, '-', color=colors[idx], linewidth=2, 
                    label=f"Theory (EI={ei_label})")
            
            # Plot dotted lines for ML model replication
            ax.plot(x_grid, y_ml_pred, ':', color=colors[idx], linewidth=2.5, 
                    label=f"ML Pred ({algo})")
        
        ax.set_xlabel("x location along the beam [m]", fontsize=11)
        ax.set_ylabel("Vertical Displacement [m]", fontsize=11)
        ax.set_title(f"Displacement Profile Level Curves (Load P = {P} N)", fontsize=13, fontweight='bold')
        ax.invert_yaxis() # Invert axis so displacement points downwards visually
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        st.pyplot(fig)
