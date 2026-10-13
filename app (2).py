
import streamlit as st
import pickle
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Breast Cancer Classification App", layout="centered")

st.title("Breast Cancer Classification App")
st.write("This application uses a K-Nearest Neighbors (KNN) model to classify clinical samples as Healthy Controls or Patients based on 9 biomarkers.")

# 1. Load the model and scaler
@st.cache_resource
def load_artifacts():
    with open('best_knn_model.pkl', 'rb') as mf:
        model = pickle.load(mf)
    with open('scaler.pkl', 'rb') as sf:
        scaler = pickle.load(sf)
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# 2. Setup inputs for the 9 clinical biomarkers
st.header("Input Biomarkers")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=48)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=23.5, step=0.1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=10, max_value=300, value=70)

with col2:
    insulin = st.number_input("Insulin (µU/mL)", min_value=0.1, max_value=100.0, value=2.7, step=0.1)
    homa = st.number_input("HOMA-IR", min_value=0.1, max_value=50.0, value=0.47, step=0.01)
    leptin = st.number_input("Leptin (ng/mL)", min_value=1.0, max_value=150.0, value=8.8, step=0.1)

with col3:
    adiponectin = st.number_input("Adiponectin (µg/mL)", min_value=1.0, max_value=100.0, value=9.7, step=0.1)
    resistin = st.number_input("Resistin (ng/mL)", min_value=1.0, max_value=150.0, value=8.0, step=0.1)
    mcp_1 = st.number_input("MCP-1 (pg/mL)", min_value=10.0, max_value=2000.0, value=417.1, step=1.0)

# 3. Predict button
if st.button("Predict Classification", type="primary"):
    # Assemble inputs into a 2D array
    raw_input = np.array([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1]])
    
    # Scale the inputs
    scaled_input = scaler.transform(raw_input)
    
    # Predict
    prediction = model.predict(scaled_input)[0]
    
    # Display result
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("Class 1: Healthy Control")
    elif prediction == 2:
        st.error("Class 2: Breast Cancer Patient")
