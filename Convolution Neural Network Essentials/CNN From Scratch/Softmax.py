import numpy as np

# A Fully connected layer with softmax activation
class Softmax:
    def __init__(self, input_len, nodes):
        # Divide the input_len to reduce variance of the initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, x):
        """
        Perform Forward Pass for this layer using x.
        Input:
            Any Array with Any dimension
        Return:
            1D NumPy array containing respective probability values
        """ 
        self.last_input_shape = x.shape

        x = x.flatten()
        self.last_input = x

        input_len, nodes = self.weights.shape
        totals = np.dot(x, self.weights) + self.biases # f(x) = W.x + b
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)


    def backward(self, dL_d_out, lr):
        """
        Perform Backpropgation for Softmax Layer.
        Input:
            - dL_d_out -> Loss gradient for the layer's outputs.
            - lr -> Learning Rate

        Return:
            Loss gradient for this layer's input
        """
        # As we know only 1 element of dL_d_out will be nonzero 
        for i, gradient in enumerate(dL_d_out):
            if gradient == 0:
                continue

        # e ^ totals
        t_exp = np.exp(self.last_totals)

        # Sum of all e ^ totals
        S = np.sum(t_exp)

        # Gradient of out[i] against totals
        d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
        d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

        # Gradients of totals against weights / biases / input
        d_t_d_w = self.last_input
        d_t_d_b = 1
        d_t_d_inputs = self.weights

        # Gradients of loss against totals
        d_L_d_t = gradient * d_out_d_t

        # Gradients of loss against weights / biases / input
        d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
        d_L_d_b = d_L_d_t * d_t_d_b
        dL_d_inputs = d_t_d_inputs @ d_L_d_t

        # Update weights / biases
        self.weights -= lr * d_L_d_w
        self.biases -= lr * d_L_d_b

        return dL_d_inputs.reshape(self.last_input_shape)