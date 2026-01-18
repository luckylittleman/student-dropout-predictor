import numpy as np
import pandas as pd
import os

# Settings
SAMPLES = 1000
FILENAME = "data/student_data.csv"

# Set a seed so we get the same random numbers every time
np.random.seed(42)

print(f"Generating data for {SAMPLES} students...")

# --- 1. Generate Random Base Data ---

# Student IDs (e.g., 1001 to 2000)
ids = range(1001, 1001 + SAMPLES)

# Gender (Male/Female)
genders = np.random.choice(['Male', 'Female'], SAMPLES)

# Department (4 Options)
depts = np.random.choice(['Engineering', 'Business', 'Computer Science', 'Education'], SAMPLES)

# Fee Balance (KES) - Normal distribution around 30k
fee_balances = np.random.normal(30000, 20000, SAMPLES)
fee_balances = np.clip(fee_balances, 0, 100000).astype(int)

# Attendance Rate (40% to 100%)
attendance = np.random.uniform(40, 100, SAMPLES).round(1)

# KCSE Points (7 to 12, where 12 is an A)
kcse_points = np.random.randint(7, 13, SAMPLES)

# --- 2. The Logic (Creating the Patterns) ---
dropout_status = []

for i in range(SAMPLES):
    # Start with a base risk score
    risk_score = 0
    
    # RULE 1: Money Problems (High Fee Balance)
    if fee_balances[i] > 60000:
        risk_score += 60
        
    # RULE 2: Low Attendance
    if attendance[i] < 70:
        risk_score += 60
        
    # RULE 3: Academic Struggle (Engineering + Low KCSE)
    if depts[i] == 'Engineering' and kcse_points[i] < 9:
        risk_score += 20
        
    # Add some randomness
    risk_score += np.random.randint(-10, 10)
    
    # Final Decision (Risk > 50 = Dropout)
    if risk_score > 50:
        dropout_status.append(1) # DROPOUT
    else:
        dropout_status.append(0) # GRADUATE

# --- 3. Save to CSV ---
df = pd.DataFrame({
    'Student_ID': ids,
    'Gender': genders,
    'Department': depts,
    'Fee_Balance_KES': fee_balances,
    'Attendance_Pct': attendance,
    'KCSE_Points': kcse_points,
    'Dropout': dropout_status
})

# Ensure the 'data' folder exists
if not os.path.exists('data'):
    os.makedirs('data')

# Save file
df.to_csv(FILENAME, index=False)

print(f"SUCCESS! Created '{FILENAME}' with {SAMPLES} rows.")
print("------------------------------------------------")
