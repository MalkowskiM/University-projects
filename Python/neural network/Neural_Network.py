import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import expit
import contextlib
import io

def load_data(filename_X_train:str, filename_X_test:str, filename_Y_train:str, filename_Y_test:str):
    """
    Load data from CSV files.

    Args:
        filename_X_train (str): File path for training data features.
        filename_X_test (str): File path for test data features.
        filename_Y_train (str): File path for training data labels.
        filename_Y_test (str): File path for test data labels.

    Returns:
        tuple: Tuple containing X_train, X_test, Y_train, Y_test.
    """
    X_train = pd.read_csv(filename_X_train)
    X_test = pd.read_csv(filename_X_test)
    Y_train = pd.read_csv(filename_Y_train)
    Y_test = pd.read_csv(filename_Y_test)
    X_train = np.array(X_train).T
    X_test = np.array(X_test).T
    Y_train = np.array(Y_train).T
    Y_test = np.array(Y_test).T
    return X_train, X_test, Y_train, Y_test

def relu(Z: np.array):
    """ReLU activation function."""
    return np.maximum(Z, 0)

def sigmoid(Z: np.array):
    """Sigmoid activation function."""
    return expit(Z)

def relu_deriv(Z: np.array):
    """Derivative of ReLU activation function."""
    return Z > 0

class Hidden_Layer:
    """Hidden layer of the neural network."""
    def __init__(self, n_inputs: int, n_neurons: int, learning_rate = 0.1, bias_offset = 0):
        """
        Initialize hidden layer parameters.

        Args:
            n_inputs (int): Number of input features.
            n_neurons (int): Number of neurons in the layer.
            learning_rate (float): Learning rate for training.
            bias_offset (float, optional): Offset for bias initialization. Defaults to 0.5.
        """
        self.weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(2 / n_inputs)
        self.bias = np.zeros((n_neurons, 1)) - bias_offset
        self.learning_rate = learning_rate
        self.activation_function = relu
        self.activation_function_derivative = relu_deriv

    def forward_propagation(self, inputs: np.array):
        """
        Perform forward propagation through the layer.

        Args:
            inputs (np.array): Input data.

        """
        self.Z = np.dot(self.weights, inputs) + self.bias
        self.output = self.activation_function(self.Z)

    def backward_propagation(self, X: np.ndarray, W_next: np.ndarray, dZ_next: np.ndarray):
        """
        Perform backward propagation through the layer.

        Args:
            X (np.ndarray): Input data.
            W_next (np.ndarray): Weights of the next layer.
            dZ_next (np.ndarray): Gradient from the next layer.

        Returns:
            tuple: Updated weights and gradient for the previous layer.
        """

        m = X.shape[1]
        W = self.weights

        dZ = np.dot(W_next.T, dZ_next) * self.activation_function_derivative(self.Z)
        dW = np.dot(dZ, X.T)/m
        db = np.sum(dZ, axis=1, keepdims=True)/m

        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db
        return W, dZ


class Output_layer(Hidden_Layer):
    """Output layer of the neural network."""
    def __init__(self, n_inputs: int, n_neurons: int, learning_rate=0.1, bias_offset=0):
        """Initialize output layer parameters.

        Args:
            n_inputs (int): Number of input features.
            n_neurons (int): Number of neurons in the layer.
            learning_rate (float): Learning rate for training.
            bias_offset (float, optional): Offset for bias initialization. Defaults to 0.
        """
        super().__init__(n_inputs, n_neurons, learning_rate=0.1, bias_offset=0)
        self.activation_function = sigmoid
        limit = limit = np.sqrt(2 / (n_inputs + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_inputs))

    def backward_propagation(self, Y: np.ndarray, X: np.ndarray):
        """Perform backward propagation through the output layer.

        Args:
            Y (np.ndarray): True labels.
            X (np.ndarray): Input data.

        Returns:
            tuple: Tuple containing dW and dZ.
        """
        m = Y.shape[1]
        W = self.weights

        dZ = self.output - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db

        return W, dZ


class Neural_Network:
    """Neural Network class."""
    def __init__(self, hidden_layers: list, output_layer: Output_layer, learning_rate: float, threshold=0.7):
        """
        Initialize the neural network.

        Args:
            hidden_layers (list): List of hidden layers.
            output_layer (Output_layer): Output layer of the network.
            learning_rate (float): Learning rate for training.
            threshold (float, optional): Threshold for binary classification. Defaults to 0.7. updated in predict method.
        """
        self.hidden_layers = hidden_layers
        for hidden_layer in self.hidden_layers:
            hidden_layer.learning_rate = learning_rate
        self.output_layer = output_layer
        self.output_layer.learning_rate = learning_rate
        self.threshold = threshold

    def cross_entropy(self, Y_true: np.array, Y_prediction: np.array):
        """
        Calculate the binary cross-entropy loss.

        Args:
            Y_true (np.array): True labels.
            Y_prediction (np.array): Predicted probabilities.

        Returns:
            float: Cross-entropy loss.
        """
        Y_prediction = np.clip(Y_prediction, 1e-15, 1 - 1e-15)
        return -np.mean(Y_true * np.log(Y_prediction) + (1 - Y_true) * np.log(1 - Y_prediction))
    def set_learning_rate(self, learning_rate: float):
        for hidden_layer in self.hidden_layers:
            hidden_layer.learning_rate = learning_rate
        self.output_layer.learning_rate = learning_rate

    def calculate_sensitivity_specificity(self, Y_true: np.array, y_prediction: np.array):
        """
        Calculate sensitivity and specificity.

        Args:
            Y_true (np.array): True labels.
            y_prediction (np.array): Predicted labels.

        Returns:
            tuple: Tuple containing sensitivity and specificity.
        """
        TP = np.sum((Y_true == 1) & (y_prediction == 1))
        TN = np.sum((Y_true == 0) & (y_prediction == 0))
        FP = np.sum((Y_true == 0) & (y_prediction == 1))
        FN = np.sum((Y_true == 1) & (y_prediction == 0))

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        return sensitivity, specificity
    def train(self, X: np.array, Y: np.array, epochs: int):
        """
        Train the neural network.

        Args:
            X (np.array): Input data.
            Y (np.array): True labels.
            epochs (int): Number of training epochs.
        """
        input = list()
        input.append(X)
        self.loss_history = list()

        for epoch in range(epochs):
            for i in range(len(self.hidden_layers)):
                self.hidden_layers[i].forward_propagation(input[i])
                input.append(self.hidden_layers[i].output)
            self.output_layer.forward_propagation(input[-1])

            loss = Neural_Network.cross_entropy(self, Y, self.output_layer.output)
            self.loss_history.append(loss)

            W, dZ = self.output_layer.backward_propagation(Y, input[-1])
            for i in reversed(range(len(self.hidden_layers))):
                W, dZ = self.hidden_layers[i].backward_propagation(input[i], W, dZ)

            if epoch % 100 == 0:
                print('Epoch ', epoch, ', Loss: ', loss)

    def predict(self, X: np.array):
        """
        Make predictions using the trained model.

        Args:
            X (np.array): Input data.

        Returns:
            np.array: Predicted labels.
        """
        input = list()
        input.append(X)

        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].forward_propagation(input[i])
            input.append(self.hidden_layers[i].output)
        self.output_layer.forward_propagation(input[-1])
        self.threshold = np.mean(self.output_layer.output)

        return self.output_layer.output

    def plot_loss(self):
        """Plot the loss function over epochs."""
        plt.plot(self.loss_history, linestyle='-')
        plt.title('Loss function graph')
        plt.xlabel('Epoch')
        plt.ylabel('Loss function value')
        plt.grid(True)







if __name__ == '__main__':
    pass





