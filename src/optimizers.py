import numpy as np
class Optimizer_SGD:
    #initialize a learning rate of 1.0
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self,layers):
       #update the weights and biases for a given layer using the gradients calculated during backprop.
       #weights=weights - learning_rate * dweights
       layers.weights +=-self.learning_rate * layers.dweights

       #bias=bias - learning_rate * dbias
       layers.biases +=-self.learning_rate * layers.dbiases

class Optimizer_Adam:
    # Initialize hyperparameters
    # learning_rate: How fast we learn (usually 0.001 for Adam)
    # decay: Lowers learning rate over time
    # epsilon: Prevents division by zero
    # beta_1: The "Momentum" setting (history of direction)
    # beta_2: The "Cache" setting (history of magnitude)
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # The Main Logic
    def update_params(self, layer):
        
        # If layer doesn't have cache/momentums, create them (filled with zeros)
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update Momentums (Speed) - Beta 1
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases

        # Update Cache (Friction) - Beta 2
        # We square the gradients (dweights**2) to see how big the steps are
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dbiases**2

        # Get "Corrected" Momentums/Cache (Bias Correction)
        # Because we started at 0, the first few steps are biased towards 0. 
        # This math fixes that warmup period.
        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))
        
        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        # Update Weights and Biases (The actual learning!)
        # Standard Formula: New = Old - (Learning_Rate * Momentum / sqrt(Cache))
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
                         
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after updates
    def post_update_params(self):
        self.iterations += 1