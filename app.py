import streamlit as st
import pandas as pd
import joblib
import os

# --- PATH CONFIGURATION ---
# This ensures Streamlit finds the 'model' folder relative to this app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# --- CACHED DATA LOADING ---
@st.cache_data
def load_comparison_matrix():
    matrix_path = os.path.join(MODEL_DIR, 'comparison_matrix.csv')
    if os.path.exists(matrix_path):
        return pd.read_csv(matrix_path)
    return None

def load_model_file(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

# --- UI SETUP ---
st.title("🛡️ Fraud Detection: 6-Model Benchmark")
st.sidebar.header("Navigation")
choice = st.sidebar.radio("View", ["Performance Matrix", "Live Prediction"])

# --- PAGE 1: PERFORMANCE MATRIX ---
if choice == "Performance Matrix":
    st.subheader("Model Comparison (Accuracy, AUC, Score, Precision, Recall, F1)")
    
    df = load_comparison_matrix()
    if df is not None:
        # Highlight the best performing model in each category
        st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Plotly/Streamlit Bar Chart for F1-Score
        st.bar_chart(data=df, x='Model', y='F1_Score')
    else:
        st.error(f"Could not find comparison_matrix.csv in {MODEL_DIR}")
        st.info("Check if your 'train_models.py' has finished running and created the file.")

# --- PAGE 2: LIVE PREDICTION ---
else:
    st.subheader("Predict with a Specific Model")
    
    selected_model_name = st.selectbox("Select Model", 
        ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "Random_Forest", "XGBoost"])
    
    # Constructing the expected filename (e.g., 'xgboost_model.pkl')
    file_to_load = f"{selected_model_name.lower()}_model.pkl"
    model = load_model_file(file_to_load)
    
    if model:
        st.success(f"Loaded {selected_model_name} successfully!")
        # Add your input fields here (V1-V28) as shown in the previous example
    else:
        st.warning(f"Model file {file_to_load} not found in {MODEL_DIR}")