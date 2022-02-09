import sys
import time
import numpy as np
from random import shuffle


def sigmoid(x):
    # Prevent overflow.
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    """Compute tanh values for each value in the array"""
    return np.tanh(x)


def derivative_tanh(x):
    return 1 - np.power(np.tanh(x), 2)


def reLu(x):
    """Compute ReLu values for each value in the array"""
    return np.maximum(x, 0)


def derivative_relu(x):
    return np.array(x > 0, dtype=np.float32)


def leakyReLu(x):
    """Compute Leaky ReLu values for each value in the array"""
    return np.maximum(x, 0.1 * x)


def leaky_rule(x):
    if x > 0:
        return 1
    return 0.1


def derivative_leakyReLu(x):
    """Compute Leaky ReLu values for each value in the array"""
    vfunc = np.vectorize(leaky_rule)
    return vfunc(x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if np.isnan(x).any():
        print()
    x -= np.max(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()


def derivative_softmax(x, y):
    mat = np.copy(x)
    mat[y] -= 1
    return mat


def log_loss(prediction, y):
    return -(prediction * np.log(y) + (1 - prediction) * np.log(1 - y))


def derive(function_name, x):
    if function_name == "ReLu":
        return derivative_relu(x)
    elif function_name == "tanh":
        return derivative_tanh(x)
    elif function_name == "sigmoid":
        return derivative_sigmoid(x)
    elif function_name == 'leakyReLu':
        return derivative_leakyReLu(x)
    else:
        print("error")
        return None


def activate(product, activation_name):
    if activation_name == "ReLu":
        return reLu(product)
    elif activation_name == "tanh":
        return tanh(product)
    elif activation_name == "sigmoid":
        return sigmoid(product)
    elif activation_name == "softmax":
        return softmax(product)
    elif activation_name == 'leakyReLu':
        return leakyReLu(product)
    else:
        print("error")
        return None


def forward_propagation(sample, all_weights, activations, network_shape, biases):
    layers = [sample,
              np.zeros(shape=(network_shape[1], 1), dtype=float),
              np.zeros(shape=(network_shape[2], 1), dtype=float)]

    for k in range(1, len(layers)):
        layers[k] = np.dot(all_weights[k - 1], layers[k - 1]) + biases[k - 1]
        layers[k] = activate(layers[k], activations[k - 1])

    predict = np.argmax(layers[-1])
    return predict, layers


def back_propagation(sample, y, all_weights, layers, activations):
    w1, w2 = all_weights

    a1, a2 = layers[1:]
    dz2 = derivative_softmax(a2, y)
    dw2 = np.dot(dz2, a1.T)
    db2 = dz2

    dz1 = np.dot(w2.T, dz2) * derive(activations[0], a1)
    dw1 = np.dot(dz1, sample.T)
    db1 = dz1

    gradients = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }

    return gradients


def update_weights(all_weights, biases, gradients, learning_rate):
    dw = [gradients['dw1'], gradients['dw2']]
    db = [gradients['db1'], gradients['db2']]

    for i in range(len(all_weights)):
        all_weights[i] -= learning_rate * dw[i]
        biases[i] -= learning_rate * db[i]

    return all_weights, biases


def make_predictions(test_x, all_weights, activations, network_shape, biases):
    predictions = []
    for sample in test_x:
        sample = sample.reshape(network_shape[0], 1)
        predictions.append(forward_propagation(sample, all_weights, activations, network_shape, biases)[0])
    return predictions


def shuffle_training(train_x, train_y):
    ind_list = list(range(len(train_x)))
    shuffle(ind_list)
    train_new = train_x[ind_list, :]
    target_new = train_y[ind_list, ]
    return train_new, target_new


def neural_network(train_x, train_y, test_x):
    # here we declare values for the learning rate and number of epochs
    network_shape = (784, 150, 10)
    all_weights = [0.001 * np.random.rand(150, 784),
                   0.001 * np.random.rand(10, 150)]
    activations = ["leakyReLu", "softmax"]
    biases = [np.zeros(shape=(network_shape[1], 1)),
              np.zeros(shape=(network_shape[2], 1))]
    learning_rate = 0.005
    epochs = 8

    for epoch in range(epochs):
        train_x, train_y = shuffle_training(train_x, train_y)
        cost, index = 0, 0
        for sample, y in zip(train_x, train_y):
            sample = sample.reshape(network_shape[0], 1)
            predict, layers = forward_propagation(sample, all_weights, activations, network_shape, biases)
            loss = log_loss(predict, y)
            gradients = back_propagation(sample, y, all_weights, layers, activations)
            all_weights, biases = update_weights(all_weights, biases, gradients, learning_rate)
            index += 1

    predictions = make_predictions(test_x, all_weights, activations, network_shape, biases)
    return predictions


def log_results(y_hat):
    with open("test_y", "w") as file_name:
        for i in range(len(y_hat) - 1):
            file_name.write(f"{int(y_hat[i])}\n")
        file_name.write(f"{int(y_hat[-1])}")


def read_info():
    train_x_fname, train_y_fname = sys.argv[1], sys.argv[2]
    test_x_fname = sys.argv[3]

    train_x = np.loadtxt(train_x_fname, dtype=np.longdouble)
    train_y = np.loadtxt(train_y_fname, dtype='uint8')
    test_x = np.loadtxt(test_x_fname, dtype=np.longdouble)

    train_x = train_x / 255
    test_x = test_x / 255
    return train_x, train_y, test_x


def main():
    train_x, train_y, test_x = read_info()
    predictions = neural_network(train_x, train_y, test_x)
    log_results(predictions)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'time took is: {end - start}')
