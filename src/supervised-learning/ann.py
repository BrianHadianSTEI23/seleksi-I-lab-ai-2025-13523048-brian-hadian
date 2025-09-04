
import numpy as np

class ActivationFunction:

    @staticmethod
    def sigmoid(x : np.ndarray):
        sigmoid_val = 1 / (1 + np.e ** (-x))
        return sigmoid_val

    @staticmethod
    def sigmoid_derivative(x : np.ndarray):
        sigmoid_derivative_val = (np.e ** (-x)) / ((1 + np.e ** (-x)) ** 2)
        return sigmoid_derivative_val

    @staticmethod
    def relu(x : np.ndarray):
        relu_val = np.maximum(0, x)
        return relu_val

    @staticmethod
    def relu_derivative(x : np.ndarray):
        relu_derivative_val = np.where(x > 0, 1, 0)
        return relu_derivative_val

    @staticmethod
    def linear(x : np.ndarray):
        return x

    @staticmethod
    def linear_derivative(x : np.ndarray):
        return 1

    @staticmethod
    def softmax(x : np.ndarray):
        exps = np.exp(x)
        return exps / sum(exps)

    @staticmethod
    def softmax_derivative(x : np.ndarray):
        return np.diag(x) - np.outer(x, x)
    
class Layer:
    def __init__(self, input_dim, output_dim, activationFunction, learningRate):
        self.w = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros((output_dim, 1))
        self.activationFunction = activationFunction
        self.learningRate = learningRate

    def forward(self, x: np.ndarray):
        self.x = x  # store input for backward
        self.z = self.w @ x + self.b
        if self.activationFunction == "relu":
            self.a = ActivationFunction.relu(self.z)
        elif self.activationFunction == "sigmoid":
            self.a = ActivationFunction.sigmoid(self.z)
        elif self.activationFunction == "softmax":
            self.a = ActivationFunction.softmax(self.z)
        else:
            self.a = ActivationFunction.linear(self.z)
        return self.a

    def backward(self, da: np.ndarray):
        # da: gradient from next layer
        if self.activationFunction == "relu":
            dz = da * ActivationFunction.relu_derivative(self.z)
        elif self.activationFunction == "sigmoid":
            dz = da * ActivationFunction.sigmoid_derivative(self.z)
        elif self.activationFunction == "softmax":
            dz = da
        else:
            dz = da  # linear

        m = self.x.shape[1]  # batch size
        dw = (dz @ self.x.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        # update params
        self.w -= self.learningRate * dw
        self.b -= self.learningRate * db

        # return gradient to previous layer
        da_prev = self.w.T @ dz
        return da_prev

class ArtificialNeuralNetwork:
    def __init__(self, input_dim, hidden_layers, hidden_size, output_dim, activation="relu", learning_rate=0.1):
        self.layers : list[Layer] = []
        prev_dim = input_dim
        # hidden layers
        for _ in range(hidden_layers):
            self.layers.append(Layer(prev_dim, hidden_size, activation, learning_rate))
            prev_dim = hidden_size
        # output layer
        self.layers.append(Layer(prev_dim, output_dim, "softmax" if output_dim > 1 else "sigmoid", learning_rate))

    def forward(self, x : np.ndarray) :
        # input x into the neuron one by one
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a
        
    def backward(self, y_true, y_pred):
        # Gradient descent
        if y_true.shape[0] == 1:  # binary cross-entropy
            da = -(np.divide(y_true, y_pred+1e-9) - np.divide(1-y_true, 1-y_pred+1e-9))
        else:  # softmax + cross-entropy
            da = y_pred - y_true
        for layer in reversed(self.layers):
            da = layer.backward(da)

    def train(self, x, y, epochs=1000):
        for _ in range(epochs):
            y_pred = self.forward(x)
            self.backward(y, y_pred)
    
    def predict(self, x : np.ndarray):
        # this function will return a calculated value based on each layer that has been initiated
        pred = x
        for layer in self.layers:
            pred = layer.forward(pred)
        return pred
    
