import Neural_Network
import numpy as np
from matplotlib import pyplot as plt
import contextlib
import io
from mpl_toolkits.mplot3d import Axes3D



def test_network(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array, learning_rate: float, epochs: int, network: Neural_Network.Neural_Network):
    """
    Test the neural network on training and test data.

    Args:
        X_train (np.array): Training data features.
        X_test (np.array): Test data features.
        Y_train (np.array): Training data labels.
        Y_test (np.array): Test data labels.
    """

    network.set_learning_rate(learning_rate)

    network.train(X_train, Y_train, epochs)

    predictions_train = network.predict(X_train)
    loss_train = network.cross_entropy(Y_train, predictions_train)
    predictions_train = (predictions_train[0, :] > network.threshold).astype(int)

    predictions_test = network.predict(X_test)
    loss_test = network.cross_entropy(Y_test, predictions_test)
    predictions_test = (predictions_test[0, :] > network.threshold).astype(int)



    sensitivity_train, specificity_train = network.calculate_sensitivity_specificity(Y_train, predictions_train)
    sensitivity_test, specificity_test = network.calculate_sensitivity_specificity(Y_test, predictions_test)

    print('\nTrain data:')
    print('Predictions: ', predictions_train)
    print('True values: ', Y_train.astype(int)[0, :])
    print("Sensitivity: ", sensitivity_train)
    print("Specificity: ", specificity_train)
    print("loss function value: ", loss_train)
    print('\nTest data:')
    print('Predictions: ', predictions_test)
    print('True values: ', Y_test.astype(int)[0, :])
    print("Sensitivity: ", sensitivity_test)
    print("Specificity: ", specificity_test)
    print("loss function value: ", loss_test)
    # network.plot_loss()
    plot_epoch(X_train, X_test, Y_train, Y_test, network)
    # plot_threshold(X_train, X_test, Y_train, Y_test, network)


def plot_epoch(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array, network: Neural_Network.Neural_Network):
    """
    Plot sensitivity and specificity by epoch and threshold.

    Args:
        X_train (np.array): Training data features.
        X_test (np.array): Test data features.
        Y_train (np.array): Training data labels.
        Y_test (np.array): Test data labels.
        network (Neural_Network): Trained neural network.
    """
    specificity_history_train = list()
    sensitivity_history_train = list()
    specificity_history_test = list()
    sensitivity_history_test = list()

    for epoch in range(100, 5000, 100):
        f = io.StringIO()
        # Używamy contextlib.redirect_stdout do przekierowania stdout
        with contextlib.redirect_stdout(f):
            network.train(X_train, Y_train, epoch)

            predictions_train = network.predict(X_train)
            predictions_test = network.predict(X_test)
            predictions_train = (predictions_train[0, :] > network.threshold).astype(int)
            predictions_test = (predictions_test[0, :] > network.threshold).astype(int)

            sensitivity_train, specificity_train = network.calculate_sensitivity_specificity(Y_train, predictions_train)
            sensitivity_history_train.append(sensitivity_train)
            specificity_history_train.append(specificity_train)

            sensitivity_test, specificity_test = network.calculate_sensitivity_specificity(Y_test, predictions_test)
            specificity_history_test.append(specificity_test)
            sensitivity_history_test.append(sensitivity_test)

    epochs = range(100, 5000, 100)
    plt.figure()
    plt.plot(epochs, specificity_history_train, label='Specificity_train')
    plt.plot(epochs, sensitivity_history_train, label='Sensitivity_train')
    plt.plot(epochs, specificity_history_test, label='Specificity_test')
    plt.plot(epochs, sensitivity_history_test, label='Sensitivity_test')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.grid(True)
    plt.title('Sensitivity and specificity by epoch')
    plt.legend()

def plot_threshold(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array, network: Neural_Network.Neural_Network):
    """
        Plot sensitivity and specificity by threshold.

        Args:
            X_train (np.array): Training data features.
            X_test (np.array): Test data features.
            Y_train (np.array): Training data labels.
            Y_test (np.array): Test data labels.
            network (Neural_Network): Trained neural network.
        """
    specificity_history_train_th = list()
    sensitivity_history_train_th = list()
    specificity_history_test_th = list()
    sensitivity_history_test_th = list()

    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        f = io.StringIO()
        # Używamy contextlib.redirect_stdout do przekierowania stdout
        with contextlib.redirect_stdout(f):
            network.train(X_train, Y_train, 1000)
            predictions_train = network.predict(X_train)
            predictions_test = network.predict(X_test)
            network.threshold = threshold
            predictions_train = (predictions_train[0, :] > network.threshold).astype(int)
            predictions_test = (predictions_test[0, :] > network.threshold).astype(int)

            sensitivity_train, specificity_train = network.calculate_sensitivity_specificity(Y_train, predictions_train)
            sensitivity_history_train_th.append(sensitivity_train)
            specificity_history_train_th.append(specificity_train)

            sensitivity_test, specificity_test = network.calculate_sensitivity_specificity(Y_test, predictions_test)
            specificity_history_test_th.append(specificity_test)
            sensitivity_history_test_th.append(sensitivity_test)

    plt.figure()
    plt.plot(thresholds, specificity_history_train_th, label='Specificity_train')
    plt.plot(thresholds, sensitivity_history_train_th, label='Sensitivity_train')
    plt.plot(thresholds, specificity_history_test_th, label='Specificity_test')
    plt.plot(thresholds, sensitivity_history_test_th, label='Sensitivity_test')
    plt.xlabel('Thresholds')
    plt.ylabel('Values')
    plt.grid(True)
    plt.title('Sensitivity and specificity by threshold')
    plt.legend()
    plt.show()

def evaluate_model(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array, learning_rate: float, epochs: int, network: Neural_Network.Neural_Network):

    network.set_learning_rate(learning_rate)
    network.train(X_train, Y_train, epochs)

    predictions_train = network.predict(X_train)
    predictions_train = (predictions_train[0, :] > network.threshold).astype(int)

    predictions_test = network.predict(X_test)
    predictions_test = (predictions_test[0, :] > network.threshold).astype(int)

    sensitivity_train, specificity_train = network.calculate_sensitivity_specificity(Y_train, predictions_train)
    sensitivity_test, specificity_test = network.calculate_sensitivity_specificity(Y_test, predictions_test)

    return sensitivity_train, specificity_train, sensitivity_test, specificity_test

def plot_learning_rate(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array, network: Neural_Network.Neural_Network):

    results = {}
    learning_rates = np.linspace(0.0001, 0.0015, 100)
    epochs = range(100, 5000, 50)
    for learning_rate in learning_rates:
        network.set_learning_rate(learning_rate)
        print(learning_rate)
        for epoch in epochs:
            f = io.StringIO()
            # Używamy contextlib.redirect_stdout do przekierowania stdout
            with contextlib.redirect_stdout(f):
                sensitivity_train, specificity_train, sensitivity_test, specificity_test = evaluate_model(X_train, X_test, Y_train, Y_test, learning_rate, epoch, network)

                results[(learning_rate, epoch)] = {
                    "sensitivity_train": sensitivity_train,
                    "specificity_train": specificity_train,
                    "sensitivity_test": sensitivity_test,
                    "specificity_test": specificity_test
                }


    sensitivity_train = np.zeros((len(learning_rates), len(epochs)))
    specificity_train = np.zeros((len(learning_rates), len(epochs)))
    sensitivity_test = np.zeros((len(learning_rates), len(epochs)))
    specificity_test = np.zeros((len(learning_rates), len(epochs)))

    for i, lr in enumerate(learning_rates):
        for j, epoch in enumerate(epochs):
            result = results[(lr, epoch)]
            sensitivity_train[i, j] = result['sensitivity_train']
            specificity_train[i, j] = result['specificity_train']
            sensitivity_test[i, j] = result['sensitivity_test']
            specificity_test[i, j] = result['specificity_test']

    fig = plt.figure(figsize=(14, 10))
    print(sensitivity_test.shape)

    # Train Sensitivity
    ax1 = fig.add_subplot(221, projection='3d')
    X, Y = np.meshgrid(epochs, learning_rates)
    ax1.plot_surface(X, Y, sensitivity_train, cmap='viridis')
    ax1.set_title('Train Sensitivity')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Learning Rate')
    ax1.set_zlabel('Sensitivity')

    # Train Specificity
    ax2 = fig.add_subplot(222, projection='3d')
    X, Y = np.meshgrid(epochs, learning_rates)
    ax2.plot_surface(X, Y, specificity_train, cmap='viridis')
    ax2.set_title('Train Specificity')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.set_zlabel('Specificity')

    # Test Sensitivity
    ax3 = fig.add_subplot(223, projection='3d')
    X, Y = np.meshgrid(epochs, learning_rates)
    ax3.plot_surface(X, Y, sensitivity_test, cmap='viridis')
    ax3.set_title('Test Sensitivity')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate')
    ax3.set_zlabel('Sensitivity')

    # Test Specificity
    ax4 = fig.add_subplot(224, projection='3d')
    X, Y = np.meshgrid(epochs, learning_rates)
    ax4.plot_surface(X, Y, specificity_test, cmap='viridis')
    ax4.set_title('Test Specificity')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Learning Rate')
    ax4.set_zlabel('Specificity')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = Neural_Network.load_data('X_train.csv', 'X_test.csv', 'Y_train.csv', 'Y_test.csv')

    learning_rate = 0.1
    epochs = 1000

    layer1 = Neural_Network.Hidden_Layer(7, 8, bias_offset=0)
    output_layer = Neural_Network.Output_layer(8, 1)

    network = Neural_Network.Neural_Network([layer1], output_layer, learning_rate, threshold=0.7)

    # test_network(X_train, X_test, Y_train, Y_test, learning_rate, epochs, network)

    # plot_learning_rate(X_train, X_test, Y_train, Y_test, network)

    layer2 = Neural_Network.Hidden_Layer(8, 12)
    output_layer2 = Neural_Network.Output_layer(12, 1)
    network2 = Neural_Network.Neural_Network([layer1, layer2], output_layer2, learning_rate)
    # test_network(X_train, X_test, Y_train, Y_test, learning_rate, epochs, network2)
    # plot_learning_rate(X_train, X_test, Y_train, Y_test, network2)

    layer3 = Neural_Network.Hidden_Layer(7, 32)
    layer4 = Neural_Network.Hidden_Layer(32, 32)
    layer5 = Neural_Network.Hidden_Layer(32, 16)
    output_layer3 = Neural_Network.Output_layer(16, 1)
    network3 = Neural_Network.Neural_Network([layer3, layer4, layer5], output_layer3,learning_rate)
    # test_network(X_train, X_test, Y_train, Y_test, learning_rate, epochs, network3)
    plot_learning_rate(X_train, X_test, Y_train, Y_test, network3)