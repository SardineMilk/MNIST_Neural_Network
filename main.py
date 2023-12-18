import numpy as np

"""
There are 3 layers, Input, Hidden and Output

Input takes the pixel data  - 784 neurons
Hidden processes it - 20 neurons
Output returns the value - 10 neurons

The goal is to get a single output neuron to 1 and the rest to 0
https://www.youtube.com/watch?v=9RN2Wr8xvro was used as the main learning resource and reference
"""


def forward_propagation(layer_input_values, weights, biases):
    # Takes an array of input values, multiplies by an array of weights, adds the biases, normalises it and returns it
    layer_output_values = biases + (weights @ layer_input_values)  # "@" is __matmul__, designed for matrix multiplication
    layer_output_values_normalised = sigmoid(layer_output_values)  # Normalise the hidden_pre values using Sigmoid function
    return layer_output_values_normalised


def layer_delta_error(layer_index, last_delta_error):
    # How much each neuron participated towards the error in the output
    current_layer = layer_outputs[layer_index]
    sigmoid_derivative = current_layer * (1 - current_layer)
    delta_error = np.transpose(network_weights[layer_index + 1]) @ last_delta_error * sigmoid_derivative
    return delta_error


def sigmoid(array):
    # Performs Sigmoid function on all elements of a numpy array
    return 1 / (1 + np.exp(-array))


def get_mnist(file_path):
    # Load MNIST dataset from file
    # MNIST is a 60,000 long 28*28 grayscale dataset of handwritten digits from 0 to 9
    with np.load(file_path) as raw_data:
        # images is the raw image data
        # labels is what digit each image represents
        mnist_images, mnist_labels = raw_data["x_train"], raw_data["y_train"]

    # Normalise pixel values to be between 0 and 1
    mnist_images = mnist_images.astype("float32") / 255

    # Reshape from a 2d 28*28 array to a 1d 784 array
    # So the network can take it as input
    mnist_images = np.reshape(mnist_images, (mnist_images.shape[0], mnist_images.shape[1] * mnist_images.shape[2]))

    # Convert labels to one-hot encoding, so they can be checked against the network output
    # [5] -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # [1] -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    mnist_labels = np.eye(10)[mnist_labels]

    return mnist_images, mnist_labels


# images is a (60000, 784) array of values ranging from 0 to 1
# labels is a (60000, 10) array with a single 1 and nine 0's
images, labels = get_mnist("mnist.npz")

# Set up the layer sizes
network_structure = [784, 40, 10]
# learn_rate is the effect that errors have on the weights and biases.
learn_rate = 0.01
training_generations = 5

layers_number = len(network_structure)
# Initialise the weights and biases
network_weights = []  # 0 = input -> hidden, 1 = hidden -> output
network_biases = []  # 0 = hidden, 1 = output
for i in range(len(network_structure) - 1):
    first_layer = network_structure[i]
    second_layer = network_structure[i+1]

    # Weights from the first to the second layer
    layer_weights = np.random.uniform(-0.5, 0.5, (second_layer, first_layer))
    # Biases added to the second layer
    # layer_biases = np.zeros((second_layer, 1))
    layer_biases = np.zeros((second_layer, 1))

    network_weights.append(layer_weights)
    network_biases.append(layer_biases)

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

        # hidden_values = process_layer(image, weight_input_hidden, bias_input_hidden)
        # output_values = process_layer(hidden_values, weight_hidden_output, bias_hidden_output)

        # Forward propagation
        layer_outputs = []
        input_values = image

        for layer in range(layers_number - 1):  # -1 is because the output values don't need processed
            # Perform the matrix multiplication on the last layer's data
            output_values = forward_propagation(input_values, network_weights[layer], network_biases[layer])
            input_values = output_values
            layer_outputs.append(output_values)

        hidden_values = layer_outputs[0]

        # Back propagation output -> hidden
        delta_output = layer_outputs[-1] - label  # How wrong was the output?
        # Adjust the weights and biases based on delta_output and learn_rate - gradient descent
        network_weights[1] += -learn_rate * delta_output @ np.transpose(hidden_values)  # Update hidden -> output weights using delta output and learn rate
        network_biases[1] += -learn_rate * delta_output  # Update all biases for hidden -> output layer
        # Back propagation hidden -> input
        delta_hidden = layer_delta_error(0, delta_output)
        network_weights[0] += -learn_rate * delta_hidden @ np.transpose(image)
        network_biases[0] += -learn_rate * delta_hidden

        # Cost / Error calculation
        # It is just used by the user, not the program
        # This is the mean squared error formula. It returns the average error of the entire network
        # Calculates the difference between each output and the corresponding label and squares it then sums up all the values and divides by the total number of output neurons
        average_error = 1 / network_structure[-1] * np.sum((layer_outputs[-1] - label) ** 2, axis=0)
        # Checks if the input was classified correctly
        # Is the highest output neuron the same as the label and if it is, add 1 to number_correct
        number_correct += int(np.argmax(layer_outputs[-1]) == np.argmax(label))

    accuracy = (number_correct / 50000) * 100
    print(f"Out of 50000 training images, {number_correct} were correctly classified, at an accuracy of {round(accuracy, 2)}%")

"""
with open("network_weights", "w") as file:
    for layer in network_weights:
        np.savetxt(file, layer)
with open("network_biases", "w") as file:
    for layer in network_biases:
        np.savetxt(file, layer)
"""

# Test the network
number_correct = 0
for i in range(50000, 60000):
    image = images[i]
    image.shape += (1,)

    # Forward propagation
    layer_outputs = []
    input_values = image

    for layer in range(layers_number - 1):  # -1 is because the output values don't need processed
        # Perform the matrix multiplication on the output from the last layer
        output_values = forward_propagation(input_values, network_weights[layer], network_biases[layer])
        input_values = output_values
        layer_outputs.append(output_values)

    output = np.argmax(layer_outputs[-1])
    if output == np.argmax(labels[i]):
        number_correct += 1
accuracy = (number_correct / 10000) * 100
print(f"Out of 10000 new test images, {number_correct} were correctly identified, at an accuracy of {round(accuracy, 2)}%")
