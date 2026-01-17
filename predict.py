import numpy as np
import sys
from src.layers import Layer_Dense
from src.activations import Activation_ReLU, Activation_Softmax

# --- 1. SETTINGS & CONSTANTS ---
# We need the SAME mean/std we used during training to scale new inputs correctly.
# (Derived from your create_data.py logic)
SCALERS = {
    'fee_mean': 30000, 'fee_std': 20000,
    'att_mean': 70,    'att_std': 17,
    'kcse_mean': 9.5,  'kcse_std': 1.7
}

MODEL_PATH = 'data/model_parameters.npz'

def load_model():
    """Recreates the model and loads saved weights."""
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Load the file
    try:
        data = np.load(MODEL_PATH)
    except FileNotFoundError:
        print("ERROR: Model file not found! Run train_model.py first.")
        sys.exit()

    # 2. Rebuild the Architecture (Must match train_model.py exactly!)
    # Input (5) -> Hidden (64)
    dense1 = Layer_Dense(n_inputs=5, n_neurons=64)
    activation1 = Activation_ReLU()
    
    # Hidden (64) -> Output (2)
    dense2 = Layer_Dense(n_inputs=64, n_neurons=2)
    activation2 = Activation_Softmax() # We use Softmax for final probability

    # 3. Inject the Saved Weights
    dense1.weights = data['d1_weights']
    dense1.biases = data['d1_biases']
    dense2.weights = data['d2_weights']
    dense2.biases = data['d2_biases']

    print("Model loaded successfully!")
    return dense1, activation1, dense2, activation2

def get_user_input():
    """Asks the user for student details via the terminal."""
    print("\n--- ENTER STUDENT DETAILS ---")
    
    # 1. Gender
    gender = input("Gender (Male/Female): ").strip().title()
    gender_val = 0 if gender == 'Male' else 1
    
    # 2. Department
    print("Departments: 0=Engineering, 1=Business, 2=CompSci, 3=Education")
    dept_val = int(input("Department Code (0-3): "))
    
    # 3. Numeric Data
    fees = float(input("Fee Balance (KES): "))
    attendance = float(input("Attendance (%): "))
    kcse = float(input("KCSE Points (1-12): "))
    
    return gender_val, dept_val, fees, attendance, kcse

def preprocess_input(g, d, f, a, k):
    """Scales the raw input to match what the AI expects."""
    # Scaling the numbers using (Value - Mean) / Std
    f_scaled = (f - SCALERS['fee_mean']) / SCALERS['fee_std']
    a_scaled = (a - SCALERS['att_mean']) / SCALERS['att_std']
    k_scaled = (k - SCALERS['kcse_mean']) / SCALERS['kcse_std']
    
    # Combine into a single row of data
    # Shape must be (1, 5) -> One student, 5 features
    input_array = np.array([[g, d, f_scaled, a_scaled, k_scaled]])
    return input_array

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load the Brain
    d1, act1, d2, act2 = load_model()
    
    while True:
        # 2. Get Data
        g, d, f, a, k = get_user_input()
        
        # 3. Process
        processed_data = preprocess_input(g, d, f, a, k)
        
        # 4. Predict (Forward Pass)
        d1.forward(processed_data)
        act1.forward(d1.output)
        d2.forward(act1.output)
        act2.forward(d2.output)
        
        # 5. Interpret Results
        confidence = act2.output[0] # e.g., [0.1, 0.9]
        prediction_index = np.argmax(confidence) # 0 or 1
        
        print("\n--- PREDICTION RESULT ---")
        if prediction_index == 1:
            print(f"⚠️  RISK STATUS: HIGH DROPOUT RISK")
            print(f"Confidence: {confidence[1]*100:.2f}%")
        else:
            print(f"✅  RISK STATUS: Safe (Likely to Graduate)")
            print(f"Confidence: {confidence[0]*100:.2f}%")
            
        # Ask to continue
        if input("\nCheck another student? (y/n): ").lower() != 'y':
            break