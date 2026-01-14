import numpy as np

class Activation_ReLU:
 def forward(self,inputs):
     #calculate output from input
    self.inputs=inputs
    self.output=np.maximum(0,inputs)

 def backward(self,dvalues):
   #make a copy so we dont modify the original gradients
   self.dinputs=dvalues.copy()
   #zero out gradient where input values were negative
   #logic:if the neuron was inactive (<=0), the derivative is 0.
   self.dinputs[self.inputs<=0]=0
   

class Activation_Softmax:
  def forward(self,inputs):
    #1. get unnormalized probabilities
    # we subtract the max value from each row to prevent overflow(e^100=inf)
    # keepdims=True is essential so we can subtract a column vector from the input matrix    
    exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))

    #2.normalize them for each sample
    probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)

    self.output=probabilities