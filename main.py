import numpy as np
import matplotlib.pyplot as plt

"""
There are 3 layers, Input, Hidden and Output

Input takes the pixel data  - 784 neurons
Hidden processes it - 20 neurons
Output returns the value - 10 neurons

The goal is to get a single output neuron to 1 and the rest to 0
https://www.youtube.com/watch?v=9RN2Wr8xvro was used as the main learning resource and reference
"""


# Initialises a neural network with a defined shape
class NeuralNetwork:
    def __init__(self, network_shape):
        self.layers_number = len(network_shape)
        # Initialise the weights and biases
        self.weights = []
        self.biases = []

        for layer_index in range(len(network_shape) - 1):
            first_layer = network_shape[layer_index]
            second_layer = network_shape[layer_index + 1]

            # Weights from the first to the second layer
            layer_weights = np.random.uniform(-0.5, 0.5, (second_layer, first_layer))
            # Biases added to the second layer
            layer_biases = np.zeros((second_layer, 1))

            self.weights.append(layer_weights)
            self.biases.append(layer_biases)


def get_mnist(file_path):
    # Load MNIST dataset from file
    # MNIST is a 60,000 long 28*28 grayscale dataset of handwritten digits from 0 to 9
    with np.load(file_path) as raw_data:
        # images is the raw image data
        # labels is what digit each image represents
        mnist_images, mnist_labels = raw_data["x_train"], raw_data["y_train"]

    # Normalise pixel values to be between 0 and 1
    mnist_images = mnist_images.astype("float32") / 255

    # Reshape from a 28*28 array to a 784*1 array so the network can take it as an input
    mnist_images = np.reshape(mnist_images, (mnist_images.shape[0], mnist_images.shape[1] * mnist_images.shape[2]))

    # Convert labels to one-hot encoding, so they can be checked against the network output
    # [5] -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # [1] -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    mnist_labels = np.eye(10)[mnist_labels]

    return mnist_images, mnist_labels


def forward_propagation(layer_input_values, weights, biases):
    # Takes an array of input values, multiplies by an array of weights, adds the biases, normalises it and returns it
    layer_output_values = biases + (weights @ layer_input_values)  # "@" is __matmul__, designed for matrix multiplication
    layer_output_values_normalised = sigmoid(layer_output_values)  # Normalise the hidden_pre values using Sigmoid function.
    return layer_output_values_normalised


def layer_delta_error(layer_index, last_delta_error):
    # How much each neuron participated towards the error in the output
    current_layer = layer_values[layer_index]
    sigmoid_derivative = current_layer * (1 - current_layer)
    error = np.transpose(network.weights[layer_index + 1]) @ last_delta_error * sigmoid_derivative
    return error


def sigmoid(array):
    # There are different / better functions: investigate Leaky RELU
    # Performs Sigmoid function on all elements of a numpy array
    return 1 / (1 + np.exp(-array))


# images is a (60000, 784) array of values ranging from 0 to 1
# labels is a (60000, 10) array with a single 1 and nine 0's
images, labels = get_mnist("mnist.npz")

# Set up the layer sizes
network_structure = [784, 25, 10]
learn_rate = 0.015  # learn_rate is the effect that errors have on the weights and biases.
training_generations = 5

# Initialise a network
network = NeuralNetwork(network_structure)

# Train the network
for generation in range(training_generations):  # Iterate for several generations
    number_correct = 0  # number of images correctly identified
    for i in range(0, 49999):  # Iterate for each image in the dataset. Reserve 10000 for testing
        image = images[i]
        label = labels[i]
        # Convert the vectors to matrices so matrix multiplication can be done
        # From 784 and 10 vectors to (784, 1) and (10, 1) matrices
        # This doesn't alter them in any real way
        image.shape += (1,)
        label.shape += (1,)

        # Forward propagation
        layer_values = [image]
        for layer in range(network.layers_number - 1):  # -1 is because the output values don't need processed
            # Perform the matrix multiplication on the last layer's data
            output_values = forward_propagation(layer_values[-1], network.weights[layer], network.biases[layer])
            layer_values.append(output_values)

        # Back propagation
        for j in range(network.layers_number - 1):
            layer = -(j + 1)
            # Error calculation - how much did each neuron contribute to the error in the output
            if layer == -1:
                delta_error = (layer_values[layer] - label)
            else:
                # noinspection PyUnboundLocalVariable
                delta_error = layer_delta_error(layer, delta_error)

            # Gradient descent - move towards the local minimum
            network.weights[layer] += -learn_rate * delta_error @ np.transpose(layer_values[layer - 1])
            network.biases[layer] += -learn_rate * delta_error

        # Cost / Error calculation - not used in program
        # This is the mean squared error formula. It returns the average error of the entire network's output
        # Calculates the difference between output and label, squares it, sums up all the values and divides by the total number of output neurons
        average_error = 1 / network_structure[-1] * np.sum((layer_values[-1] - label) ** 2, axis=0)
        # Checks if the input was classified correctly
        is_correct = np.argmax(layer_values[-1]) == np.argmax(label)
        number_correct += int(is_correct)

    accuracy = (number_correct / 50000) * 100
    print(f"Out of 50000 training images, {number_correct} were correctly classified, at an accuracy of {round(accuracy, 2)}%")

# Automatically test the network
number_correct = 0
for i in range(50000, 60000):
    image = images[i]
    image.shape += (1,)

    # Forward propagation
    layer_values = [image]
    for layer in range(network.layers_number - 1):  # -1 is because the output values don't need processed
        # Perform the matrix multiplication on the last layer's data
        output_values = forward_propagation(layer_values[-1], network.weights[layer], network.biases[layer])
        layer_values.append(output_values)
    output = np.argmax(layer_values[-1])

    is_correct = output == np.argmax(labels[i])
    number_correct += int(is_correct)

accuracy = (number_correct / 10000) * 100
print(f"Out of 10000 new test images, {number_correct} were correctly identified, at an accuracy of {round(accuracy, 2)}%")

# User test
while True:
    index = int(input("Enter a number (0 - 59999): "))
    image = images[index]
    image.shape += (1,)

    # Forward propagation
    layer_values = [image]
    for layer in range(network.layers_number - 1):  # -1 is because the output values don't need processed
        # Perform the matrix multiplication on the last layer's data
        output_values = forward_propagation(layer_values[-1], network.weights[layer], network.biases[layer])
        layer_values.append(output_values)

    output = np.argmax(layer_values[-1])
    plt.imshow(image.reshape(28, 28), cmap="Greys")
    plt.title(f"Label = {np.argmax(labels[index])}. Output = {output}")
    plt.show()
