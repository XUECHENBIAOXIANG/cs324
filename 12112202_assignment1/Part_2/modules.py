import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features,learningrate):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self.params = {'weight': np.random.randn(in_features, out_features), 'bias': np.zeros(out_features)}
        self.grads = {'weight': None, 'bias': None}
        self.x = None
        self.learningrate = learningrate
    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """
        self.x = x
        return  np.dot(x, self.params['weight']) + self.params['bias']
    def testforward(self, x):

        return  np.dot(x, self.params['weight']) + self.params['bias']

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        self.grads['weight'] = np.dot(self.x.T, dout)/len(self.x)
        self.grads['bias'] = np.mean(dout, axis=0)
        res = np.dot(dout, self.params['weight'].T)
        self.params['weight'] -= self.learningrate*self.grads['weight']
        self.params['bias'] -= self.learningrate*self.grads['bias']
        return res

class ReLU(object):
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """
        self.x = x
        return np.maximum(0, x)

    def testforward(self, x):

        return np.maximum(0, x)
    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        return np.where(self.x > 0, 1, 0)*dout

class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """

        x_max = np.max(x, axis=1, keepdims=True)
        exps = np.exp(x - x_max)
        ans = exps / np.sum(exps, axis=1, keepdims=True)
        return ans

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        return dout

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """
        return -np.sum(y * np.log(x+1e-10))

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        return x - y
