import streamlit as st
import numpy as np
import os

# Import your Custom Engine
from src.layers import Layer_Dense
from src.activations import Activation_ReLU, Activation_Softmax

# --- 1. SETTINGS & CONSTANTS ---
# These match your training data exactly
SCALERS = {
    'fee_mean': 30000, 'fee_std': 20000,
    'att_mean': 70,    'att_std': 17,
    'kcse_mean': 9.5,  'kcse_std': 1.7
}

MODEL_PATH = 'data/model_parameters.npz'

# --- 2. LOAD MODEL FUNCTION ---
# We use @st.cache_resource so the model loads once and stays in memory (faster!)
@st.cache_resource
def load_brain():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found! Please run train_model.py first.")
        return None, None, None, None

    data = np.load(MODEL_PATH)
    
    # Rebuild Architecture (Must match train_model.py)
    dense1 = Layer_Dense(5, 64)
    act1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 2)
    act2 = Activation_Softmax()
    
    # Inject Weights
    dense1.weights = data['d1_weights']
    dense1.biases = data['d1_biases']
    dense2.weights = data['d2_weights']
    dense2.biases = data['d2_biases']
    
    return dense1, act1, dense2, act2

# Load the model
d1, act1, d2, act2 = load_brain()

# --- 3. THE WEBSITE LAYOUT ---
st.set_page_config(page_title="Dropout Predictor", page_icon="🎓")

st.title("🎓 Student Dropout Predictor")
st.write("Enter student details below to assess their dropout risk using the Neural Network.")

st.sidebar.header("Student Details")
st.sidebar.write("Adjust the sliders to simulate a student.")

# INPUTS (The Sliders & Dropdowns)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
dept = st.sidebar.selectbox("Department", ["Engineering", "Business", "Computer Science", "Education"])

# Number Inputs
fees = st.sidebar.number_input("Fee Balance (KES)", min_value=0, max_value=200000, value=30000, step=5000)
attendance = st.sidebar.slider("Attendance Rate (%)", 0, 100, 70)
kcse = st.sidebar.slider("KCSE Points (Mean Grade)", 1, 12, 10)

# --- 4. PREDICTION LOGIC ---
if st.sidebar.button("Predict Risk", type="primary"):
    
    # 1. Convert Text to Numbers
    g_val = 0 if gender == "Male" else 1
    
    dept_map = {"Engineering": 0, "Business": 1, "Computer Science": 2, "Education": 3}
    d_val = dept_map[dept]
    
    # 2. Scale the Numbers (Math Logic)
    f_scaled = (fees - SCALERS['fee_mean']) / SCALERS['fee_std']
    a_scaled = (attendance - SCALERS['att_mean']) / SCALERS['att_std']
    k_scaled = (kcse - SCALERS['kcse_mean']) / SCALERS['kcse_std']
    
    # 3. Create Input Array
    input_data = np.array([[g_val, d_val, f_scaled, a_scaled, k_scaled]])
    
    # 4. Run the Neural Network
    d1.forward(input_data)
    act1.forward(d1.output)
    d2.forward(act1.output)
    act2.forward(d2.output)
    
    # 5. Get Results
    confidence = act2.output[0] # e.g. [0.2, 0.8]
    prediction = np.argmax(confidence) # 0 = Graduate, 1 = Dropout
    
    # --- 5. DISPLAY RESULTS ---
    st.divider()
    
    col1, col2 = st.columns(2)
    
    if prediction == 1:
        with col1:
            st.error("⚠️ HIGH RISK")
            st.write("This student is likely to **DROPOUT**.")
        with col2:
            st.metric(label="Confidence Level", value=f"{confidence[1]*100:.1f}%")
        
        st.progress(int(confidence[1]*100))
        st.warning("💡 **Recommendation:** Arrange a meeting with the Finance Office and Student Counselor.")
        
    else:
        with col1:
            st.success("✅ LOW RISK")
            st.write("This student is likely to **GRADUATE**.")
        with col2:
            st.metric(label="Confidence Level", value=f"{confidence[0]*100:.1f}%")
            
        st.progress(int(confidence[0]*100))
        st.balloons()