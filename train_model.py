#import necessary libraries and the brain from src
import sys
import numpy as np
import pandas as pd
from src.activations import Activation_Softmax,Activation_ReLU
from src.layers import Layer_Dropout,Layer_Dense
from src.loss import Loss,Loss_CategoricalCrossentropy,Activation_Softmax_Loss_CategoricalCrossentropy
from src.optimizers import Optimizer_SGD,Optimizer_Adam


#load the dataset
df=pd.read_csv('Data/student_data.csv')
print(df.head())

#data processing
#1.drop the student id column since it wont be used
df=df.drop("Student_ID",axis=1)

#2.encode male and female to 0 and 1
df["Gender"]=df["Gender"].map({"Male":0,"Female":1})

#3.encode departments
df["Department"] = df["Department"].astype('category').cat.codes

#Scaling numbers
col_to_scale=["Fee_Balance_KES", "Attendance_Pct", "KCSE_Points"]
for col in col_to_scale:
  mean_val=df[col].mean()
  std_val=df[col].std()
  df[col]=(df[col]-mean_val)/std_val



#split X and Y
#y=dropout column(the answer)
#x=all other columns
x=df.drop("Dropout",axis=1).values
y=df["Dropout"].values

x=x.astype(np.float64)
y=y.astype('int')

#Initiate the network(define the model)
dense1=Layer_Dense(n_inputs=5,n_neurons=64)
activation1=Activation_ReLU()

dense2=Layer_Dense(n_inputs=64,n_neurons=2)


loss=Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer=Optimizer_Adam(learning_rate=0.005,decay=5e-7)

#Training loop

print("Starting training...")

for epoch in range(1001):
    # 1. Forward Pass
    # Use 'X' (capital) because that's what we defined earlier
    dense1.forward(x)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    # Calculate Loss (Data Loss + Regularization loss if any)
    # The output of dense2 is our 'predictions'
    data_loss = loss.forward(dense2.output, y)

    # Calculate Accuracy (Optional but good to see!)
    # axis=1 finds the index of the highest confidence (0 or 1)
    predictions = np.argmax(loss.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Print status every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, ' +
              f'Acc: {accuracy:.3f}, ' +
              f'Loss: {data_loss:.3f}, ' +
              f'LR: {optimizer.current_learning_rate}')

    # 2. Backward Pass (The Learning)
    loss.backward(loss.output, y)
    dense2.backward(loss.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # 3. Optimization (Updating the Weights)
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

print("Training Complete!")

# --- 4. MANUAL TESTING (See the Brain Work) ---
print("\n--- Testing the Model ---")

# Let's create a 'New Student' who is High Risk:
# Gender: Male (0), Dept: Engineering (0), Fees: 80000, Att: 60%, KCSE: 8
new_student = np.array([[0, 0, 80000, 60, 8]])

# WE MUST SCALE THE DATA (Just like we did for training!)
# We use the same mean/std from before (roughly)
# Fees (mean=30k, std=20k) -> (80000 - 30000)/20000 = 2.5
# Att (mean=70, std=17) -> (60 - 70)/17 = -0.58
# KCSE (mean=9.5, std=1.7) -> (8 - 9.5)/1.7 = -0.88

# Manual normalized values based on our data generation logic:
# Gender(0), Dept(0), Fees(2.5), Att(-0.6), KCSE(-0.9)
processed_student = np.array([[0, 0, 2.5, -0.6, -0.9]], dtype=np.float32)

# Run the Forward Pass
dense1.forward(processed_student)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss.forward(dense2.output, np.array([0])) # Dummy target

# Get the probability
confidences = loss.output[0] # The softmax output
prediction = np.argmax(confidences)

print(f"Raw Input: Fees=80k, Attendance=60%")
print(f"Model Prediction: {'DROPOUT' if prediction == 1 else 'GRADUATE'}")
print(f"Confidence: {confidences[prediction]*100:.2f}%")

# --- 5. SAVE THE MODEL (The Freeze) ---
print("\nSaving the trained model...")

# We save the weights and biases from both layers
# .npz is a standard numpy file format
outfile = 'data/model_parameters.npz'

np.savez(outfile, 
         d1_weights=dense1.weights, 
         d1_biases=dense1.biases, 
         d2_weights=dense2.weights, 
         d2_biases=dense2.biases)

print(f"Success! Model parameters saved to '{outfile}'")

