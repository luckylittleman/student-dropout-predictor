import numpy as np
from src.activations import Activation_Softmax

# Common Loss Class
class Loss:
    def calculate(self, output, y):
        """
        Calculates the data and regularization losses
        given model output and ground truth values.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        """
        y_pred: The output from the Softmax layer (probabilities)
        y_true: The actual class labels (can be one-hot or simple integers)
        """
        samples = len(y_pred)

        # 1. Clip data to prevent division by 0 (log(0) = -infinity)
        # We clip from 1e-7 to (1 - 1e-7)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 2. Handle different target formats
        if len(y_true.shape) == 1:
            # SPARSE TARGETS (e.g., [0, 1, 1])
            # We use "fancy indexing" to grab the probability of the correct class only
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            # ONE-HOT TARGETS (e.g., [[1,0,0], [0,1,0]])
            # We multiply pred * true and sum the rows
            correct_confidences = np.sum(
                y_pred_clipped * y_true, 
                axis=1
            )

        # 3. Calculate Negative Log Likelihood
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Combined Softmax + Loss for faster Backward Pass
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # 1. Output layer activation function
        self.activation.forward(inputs)
        # We must save the output so the backward pass can access it
        self.output = self.activation.output
        
        # 2. Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """
        Calculates the gradient of the combined Softmax + Loss.
        Math Shortcut: Gradient = Predicted_Probabilities - True_Labels
        """
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # We copy dvalues (which are the outputs from Softmax)
        self.dinputs = dvalues.copy()
        
        # Calculate gradient
        # Subtract 1 from the correct class probabilities
        # (This implements: Predicted - True)
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient by number of samples so loss doesn't scale with batch size
        self.dinputs = self.dinputs / samples