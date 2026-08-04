import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Simplified Beam Deflection ML", layout="wide")
st.title("🏗️ Cantilever Beam Deflection: Theory vs. Machine Learning")
st.write("Hello TADAA2 colleagues! This app intends to show you a practical application of ML for structural problems.")
st.write("What does it do?/n It trains a ML model on physics-generated data and compare predictions against exact beam theory.")
st.write("How does it work?/n Choose the amount of input data to train the ML model and the algorithm type.")
st.write("What should I look for in the resulting plot?/n Check results accuracy as a function of amount of training data and choosen algorithm.")

# --- Locked Internal Constants (Hidden from User) ---
L = 10.0       # Beam Length (m)
P = 5000.0     # Concentrated Tip Load (N)
ei_min = 1e6   # Minimum EI (N·m²)
ei_max = 5e6   # Maximum EI (N·m²)

# --- SIDEBAR: Simple Controls ---
st.sidebar.header("🎛️ App Controls")

# Control 1: Dataset Size
num_samples = st.sidebar.slider(
    "Number of synthetic data points to generate:", 
    min_value=200, 
    max_value=5000, 
    value=1000, 
    step=100
)

# Control 2: Algorithm Choice
algo = st.sidebar.selectbox(
    "Select ML Algorithm:", 
    ("Random Forest", "Deep Neural Network (DNN)")
)

# --- DATA GENERATION ENGINE ---
np.random.seed(42)
rand_ei = np.random.uniform(ei_min, ei_max, num_samples)
rand_x = np.random.uniform(0, L, num_samples)

# Calculate exact displacement targets using Euler-Bernoulli theory
target_displacement = (P * (rand_x**2) * (3 * L - rand_x)) / (6 * rand_ei)

# Pack variables into DataFrame
df = pd.DataFrame({
    'EI': rand_ei,
    'x_location': rand_x,
    'vertical_displacement': target_displacement
})

st.subheader("📋 Generated Dataset Preview")
st.dataframe(df.head(5))

# Prepare ML Data splits
X = df[['EI', 'x_location']].values
y = df['vertical_displacement'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize Model Selection
if algo == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64, 64), max_iter=1500, random_state=42)
    )

# --- EXECUTION BUTTON ---
if st.button("🚀 Train Model & Generate Rainbow Level Plots"):
    with st.spinner(f"Fitting {algo} model to physics data..."):
        model.fit(X_train, y_train)
    st.success(f"{algo} training complete!")
    
    # --- PLOTTING LEVEL CURVES ---
    st.subheader("📈 Multi-Level Plot: Beam Theory vs. ML Predictions")
    
    # Create 4 evenly distributed EI curves across the predefined range
    ei_levels = np.linspace(ei_min, ei_max, 4)
    x_grid = np.linspace(0, L, 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Apply a vibrant rainbow colormap for high-contrast visibility
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ei_levels)))
    
    for idx, ei_val in enumerate(ei_levels):
        # 1. Evaluate Pure Theoretical Curve
        y_theory = (P * (x_grid**2) * (3 * L - x_grid)) / (6 * ei_val)
        
        # 2. Evaluate ML Prediction Curve
        ml_input = np.column_stack((np.full_like(x_grid, ei_val), x_grid))
        y_ml_pred = model.predict(ml_input)
        
        ei_label = f"{ei_val:.2e}"
        
        # Plot continuous solid line for true physics theory
        ax.plot(
            x_grid, y_theory, '-', 
            color=colors[idx], linewidth=2.5, 
            label=f"Theory (EI={ei_label})"
        )
        
        # Plot highly distinct dotted line for the ML model
        ax.plot(
            x_grid, y_ml_pred, ':', 
            color=colors[idx], linewidth=3.5, 
            label=f"ML Pred ({algo})"
        )
    
    ax.set_xlabel("x location along the beam [m]", fontsize=11)
    ax.set_ylabel("Vertical Displacement [m]", fontsize=11)
    ax.set_title(f"Displacement Profile Level Curves (Locked Load P = {P} N)", fontsize=13, fontweight='bold')
    
    # Invert the Y-axis so displacement projects downward like a real bending beam
    ax.invert_yaxis()
    
    # Place legend cleanly outside the main graph boundaries
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    st.pyplot(fig)
