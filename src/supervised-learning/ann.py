
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
        s = ActivationFunction.softmax(x)
        return s * (1 - s)
    
class Layer:
    def __init__(self, inputDim, outputDim, activationFunction, learningRate, lossFunction):
        self.w = np.random.randn(outputDim, inputDim) * 0.01
        self.b = np.zeros((outputDim, 1))
        self.activationFunction = activationFunction
        self.learningRate = learningRate
        self.lossFunction = lossFunction
        

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

    def backward(self, yTrue=None, difference=None):
        # activation derivative
        if self.activationFunction == "sigmoid":
            activationDerivativeVal = ActivationFunction.sigmoid_derivative(self.z)
        elif self.activationFunction == "relu":
            activationDerivativeVal = ActivationFunction.relu_derivative(self.z)
        elif self.activationFunction == "softmax":
            activationDerivativeVal = ActivationFunction.softmax_derivative(self.z)
        else:
            activationDerivativeVal = ActivationFunction.linear_derivative(self.z)

        # calculate dz
        dz = np.zeros_like(self.a)
        if yTrue is not None:
            # Output layer
            if self.lossFunction == "crossEntropy":
                if self.activationFunction in ["sigmoid", "softmax"]:
                    dz = self.a - yTrue  # simplified derivative for CE with sigmoid/softmax
                else:
                    dz = (self.a - yTrue) * activationDerivativeVal
            elif self.lossFunction == "mse":
                dz = 2 * (self.a - yTrue) * activationDerivativeVal
        else:
            # Hidden layer: use incoming gradient
            if difference is None:
                raise ValueError("Hidden layer backward must receive difference from next layer")
            dz = difference * activationDerivativeVal

        # Gradients for weights and bias
        dw = dz @ self.x.T
        db = np.sum(dz, axis=1, keepdims=True)

        # Update parameters
        self.w -= self.learningRate * dw
        self.b -= self.learningRate * db

        # Return gradient for previous layer
        differencePrev = self.w.T @ dz
        return differencePrev


class ArtificialNeuralNetwork:
    def __init__(self, inputDim, hiddenLayerNum, hiddenLayerSize, outputDim, lossFunction : str, activation="relu", alpha=0.1):
        self.layers : list[Layer] = []
        prevDim = inputDim
        self.lossFunction = lossFunction
        # hidden layer
        for _ in range(hiddenLayerNum):
            self.layers.append(Layer(prevDim, hiddenLayerSize, activation, alpha, self.lossFunction))
            prevDim = hiddenLayerSize
        # output layer
        self.layers.append(Layer(prevDim, 1, activation, alpha, self.lossFunction)) # this will be used for regression, therefore the last layer must be one neuron

    def forward(self, x : np.ndarray) :
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a
        
    def backward(self, yTrue):
        # Gradient descent based on lossFUnction
        dz = None
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                # Output layer
                dz = layer.backward(yTrue=yTrue)
            else:
                dz = layer.backward(difference=dz)
        

    def train(self, x, y, epochs=1000):
        for _ in range(epochs):
            yPred = self.forward(x)
            self.backward(y)
    
    def predict(self, x : np.ndarray):
        # this function will return a calculated value based on each layer that has been initiated
        pred = x
        for layer in self.layers:
            pred = layer.forward(pred)
        return pred
    
