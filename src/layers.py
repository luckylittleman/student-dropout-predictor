
import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self,dvalues):
        #gradients on parameters (Weights and Biases)
        # transpose inputs to align shapes:(Inputs^Tdot gradients)
        self.dweights=np.dot(self.inputs.T,dvalues)
        #sum the gradients along the rows(axis 0)for biases
        self.dbiases=np.sum(dvalues,axis=0,keepdims=True)

        #2.gradient on values(to pass to the previous layer)
        #Transpose weights to align shapes:(Gradients dot Weights^T)
        self.dinputs=np.dot(dvalues,self.weights.T)

    

class Layer_Dropout:
    def __init__(self, rate):
        # Rate is how many neurons we turn off (e.g., 0.1 = 10%)
        # We store 1 - rate (e.g., 0.9) because we multiply by what we KEEP
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        # Generate a "Mask" of 1s and 0s
        # If rate is 0.1, approx 10% of this mask will be 0
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to inputs
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # Gradient is passed through the mask
        self.dinputs = dvalues * self.binary_mask