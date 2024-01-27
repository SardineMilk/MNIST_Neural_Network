import numpy
import numpy as np
import matplotlib.pyplot as plt

"""
The network consists of an Input layer, an arbitrary number of Hidden layers and an Output layer.
The input of the network is a 1d array consisting of values between 0 and 1, with 0 being black and 1 being white.
The output of the network is a 1d array of values between 0 and 1, with 1 being fully confident and 0 being not at all

The goal is to get a single output neuron to 1 and the rest to 0
https://www.youtube.com/watch?v=9RN2Wr8xvro was used as the main learning resource and reference
"""


class NeuralNetwork:
    def __init__(self, network_shape):  # Initialises a neural network with a defined shape and random weights/biases
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
            layer_biases = np.random.uniform(-0.5, 0.5, (second_layer, 1))

            self.weights.append(layer_weights)
            self.biases.append(layer_biases)

    def train_network(self, training_array, label_array, training_epochs, learn_rate):
        # Train the network
        for epoch in range(training_epochs):  # Iterate over the training data several times. Too many results in over-fitting
            images_correctly_identified = 0  # number of images correctly identified this epoch
            for k in range(len(training_array)):  # Iterate for each image in the dataset. Reserve 10000 for testing
                image = training_array[k]
                label = label_array[k]
                # Convert the vectors to matrices so matrix multiplication can be done
                # From 784 and 10 dimensional vectors to (784, 1) and (10, 1) matrices
                # This doesn't alter them in any real way
                image.shape += (1,)
                label.shape += (1,)

                # Forward propagation
                layer_values = [image]
                for layer in range(self.layers_number - 1):  # -1 is because the output values don't need processed
                    # Perform the matrix multiplication on the last layer's data
                    output_values = forward_propagation(layer_values[-1], self.weights[layer], self.biases[layer])
                    layer_values.append(output_values)

                # Back propagation
                for j in range(self.layers_number - 1):
                    layer = -(j + 1)
                    # Error calculation - how much did each neuron contribute to the error in the output
                    if layer == -1:
                        # The output layer is a special case, only requiring the difference between the output and the label
                        delta_error = (layer_values[layer] - label)
                    else:
                        # How much each neuron participated towards the error in the output
                        current_layer = layer_values[layer]
                        sigmoid_derivative = current_layer * (1 - current_layer)
                        delta_error = np.transpose(network.weights[layer + 1]) @ delta_error * sigmoid_derivative

                    # Gradient descent - move towards the local minimum
                    network.weights[layer] += -learn_rate * delta_error @ np.transpose(layer_values[layer - 1])
                    network.biases[layer] += -learn_rate * delta_error

                # Checks if the input was classified correctly and if it was, add it to number_correct
                images_correctly_identified += int(np.argmax(layer_values[-1]) == np.argmax(label))

            network_accuracy = (images_correctly_identified / len(training_array)) * 100
            print(f"Epoch: {epoch+1}/{training_epochs}   Correctly Identified: {images_correctly_identified}/{len(training_array)}   Accuracy: {round(network_accuracy, 3)}%")

    def query_network(self, input_values):
        layer_values = [input_values]
        for layer in range(self.layers_number - 1):  # -1 is because the output values don't need processed
            # Use the last layer's data, the weights and the biases to get this layer's data
            output_values = forward_propagation(layer_values[-1], self.weights[layer], self.biases[layer])
            layer_values.append(output_values)

        # returns the output the network is most confident in
        return np.argmax(layer_values[-1])


def get_npz_data(file_path):
    # Load MNIST dataset from file
    # MNIST is a 60,000 long 28*28 grayscale dataset of handwritten digits from 0 to 9, labeled by humans
    with np.load(file_path) as raw_data:
        # images is the raw image data
        # labels is what digit each image represents
        mnist_images, mnist_labels = raw_data["x_train"], raw_data["y_train"]

    # Normalise image pixel values to be between 0 and 1 and convert to float
    mnist_images = mnist_images.astype("float32") / 255
    # Reshape from a 60000*28*28 array to a 60000*784 array so the network can take it as an input
    mnist_images = np.reshape(mnist_images, (mnist_images.shape[0], mnist_images.shape[1] * mnist_images.shape[2]))

    # Convert labels to one-hot encoding, so they can be checked against the network output
    # [5] -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # [1] -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    mnist_labels = np.eye(10)[mnist_labels]

    return mnist_images, mnist_labels


def get_csv_data(file_path):
    print("Loading training data...")
    raw_data = numpy.genfromtxt(file_path, delimiter=",")
    print("Splitting data into arrays...")
    data_labels = np.array([int(row[0]) for row in raw_data])  # First column is the label
    data_images = np.array([row[1:] for row in raw_data])  # The remaining 784 columns are pixel values

    # Convert pixel values from 0-255 range to 0-1 range
    print("Converting image format...")
    data_images = data_images.astype("float32") / 255
    # Convert labels to one-hot encoding, so they can be checked against the network output
    print("Converting label format...")
    data_labels = np.eye(27)[data_labels]

    return data_images, data_labels


def forward_propagation(layer_input_values, weights, biases):
    # Takes an array of input values, multiplies by an array of weights, adds the biases, normalises it and returns it
    layer_output_values = biases + (weights @ layer_input_values)  # "@" is __matmul__, designed for matrix multiplication
    layer_output_values_normalised = sigmoid(layer_output_values)  # Normalise the hidden_pre values using Sigmoid function.
    return layer_output_values_normalised


def sigmoid(array):
    # There are different / better functions: investigate Leaky RELU
    # Performs Sigmoid function on all elements of a numpy array
    return 1 / (1 + np.exp(-array))


# images is a (60000, 784) array of values ranging from 0 to 1
# labels is a (60000, 10) array with a single 1 and nine 0's
# images, labels = get_nps_data("mnist.npz")
images, labels = get_csv_data("emnist-letters-train.csv")

# Initialise a network
# The first layer must be the same size as the input array
# The last layer must be the same size as the number of outputs
print("Constructing network...")
# [784, 128, 64, 27]
network = NeuralNetwork([len(images[0]), 128, 64, len(labels[0])])

# Train the network
network_learn_rate = 0.01  # learn_rate is the effect that errors have on the weights and biases. increasing it doesn't make learning faster, just more granular
training_generations = 8
print("Training network...")
network.train_network(images[:50000], labels[:50000], training_generations, network_learn_rate)

# Automatically test the network
print("Testing network...")
number_correct = 0
for i in range(50000, 60000):
    image = images[i]
    image.shape += (1,)

    output = network.query_network(image)

    is_correct = output == np.argmax(labels[i])
    number_correct += int(is_correct)

accuracy = (number_correct / 10000) * 100
print(f"Out of 10000 new test images, {number_correct} were correctly identified, at an accuracy of {round(accuracy, 2)}%")

# User test
# alphabet is so output is human-readable, not just an index
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
while True:
    index = int(input(f"Enter a number (0 - {len(images)-1}): "))
    try:
        image = images[index]
        image.shape += (1,)

        output = network.query_network(image)

        # Display the image using matplotlib
        plt.imshow(image.reshape(28, 28), cmap="Greys")
        plt.title(f"Label = {alphabet[np.argmax(labels[index])-1]} Output = {alphabet[output-1]}")
        plt.show()
    except:
        print("Please input valid index")
