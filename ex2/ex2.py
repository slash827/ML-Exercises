import sys
import numpy as np


def z_score_array(array, mean=None, std=None):
    if mean is None:
        mean = np.mean(array, axis=0)
    if std is None:
        std = np.std(array, axis=0)
    for i in range(len(mean)):
        array[:, i] -= mean[i]
        array[:, i] /= std[i]
    return array, mean, std


def read_info():
    train_x_fname, train_y_fname = sys.argv[1], sys.argv[2]
    test_x_fname, out_fname = sys.argv[3], sys.argv[4]

    train_x = np.loadtxt(train_x_fname, delimiter=",")
    train_y = np.loadtxt(train_y_fname, delimiter=",")
    test_x = np.loadtxt(test_x_fname, delimiter=",")

    train_x, mean, std = z_score_array(train_x)
    test_x, mean, std = z_score_array(test_x, mean, std)
    return train_x, train_y, test_x, out_fname


def knn(train_x, train_y, test_x, k=4):
    y_hat = np.zeros(shape=test_x.shape[0], dtype=int)
    count = 0
    for i, test_sample in enumerate(test_x):
        count += 1
        distance_matrix = np.c_[train_y, np.zeros(train_y.shape[0])]
        for j, train_sample in enumerate(train_x):
            subtract = np.subtract(test_sample, train_sample)
            distance_matrix[j][1] = np.linalg.norm(subtract)

        distance_matrix = distance_matrix[distance_matrix[:, 1].argsort()][:k]
        distance_matrix = np.delete(distance_matrix, obj=1, axis=1).reshape(k).tolist()
        y_hat[i] = int(max(set(distance_matrix), key=distance_matrix.count))
    return y_hat


def pa(train_x, train_y, test_x, weights):
    # here we declare values for the learning rate and number of epochs
    epochs = 300

    for epoch in range(epochs):
        for i in range(len(train_x)):
            x_i, y_i = train_x[i], train_y[i]
            predicted = np.argmax(np.dot(weights, x_i))
            r = max(0, 1 + np.dot(weights[y_i], x_i) - np.dot(weights[predicted], x_i))
            r /= 2 * np.dot(x_i, x_i)
            if predicted != y_i:
                weights[y_i] += r * x_i
                weights[predicted] -= r * x_i

    predictions = [np.argmax(np.dot(weights, sample)) for sample in test_x]
    return predictions


def svm(train_x, train_y, test_x, weights):
    # here we declare values for the learning rate and number of epochs
    learning_rate, lamda, epochs = 0.01, 0.35, 300

    for epoch in range(epochs):
        for i in range(len(train_x)):
            x_i, y_i = train_x[i], int(train_y[i])
            predicted = int(np.argmax(np.dot(weights, x_i)))
            if predicted != y_i:
                weights[y_i] = (1 - lamda * learning_rate) * weights[y_i] + lamda * x_i
                weights[predicted] = (1 - lamda * learning_rate) * weights[predicted] + lamda * x_i
                other = 3 - y_i - predicted
                weights[other] = (1 - lamda * learning_rate) * weights[other]

    predictions = [np.argmax(np.dot(weights, sample)) for sample in test_x]
    return predictions


def perceptron(train_x, train_y, test_x, weights):
    # here we declare values for the learning rate and number of epochs
    learning_rate = 0.1
    epochs = 500

    for epoch in range(epochs):
        for i in range(len(train_x)):
            x_i, y_i = train_x[i], train_y[i]
            predicted = np.argmax(np.dot(weights, x_i))
            if predicted != y_i:
                weights[y_i] += learning_rate * x_i
                weights[predicted] -= learning_rate * x_i

    print(f"biases of perceptron are:\n {weights}")
    predictions = [np.argmax(np.dot(weights, sample)) for sample in test_x]
    return predictions


def main():
    train_x, train_y, test_x, out_fname = read_info()
    # initializing the weights (and bias included)
    weights = np.zeros(shape=(3, 6), dtype=float)
    train_x = np.c_[train_x, np.ones(train_x.shape[0])]
    train_y = train_y.astype(int)
    test_x = np.c_[test_x, np.ones(test_x.shape[0])]

    knn_y_hat = knn(train_x, train_y, test_x)
    perceptron_y_hat = perceptron(train_x, train_y, test_x, weights)
    svm_y_hat = svm(train_x, train_y, test_x, weights)
    pa_y_hat = pa(train_x, train_y, test_x, weights)

    log_results(out_fname, knn_y_hat, perceptron_y_hat, svm_y_hat, pa_y_hat)


def log_results(out_fname, knn_y_hat, perceptron_y_hat, svm_y_hat, pa_y_hat):
    with open(out_fname, "w") as file_name:
        for i in range(len(knn_y_hat)):
            file_name.write(f"knn: {knn_y_hat[i]}, perceptron: {perceptron_y_hat[i]}, ")
            file_name.write(f"svm: {svm_y_hat[i]}, pa: {pa_y_hat[i]}\n")


if __name__ == '__main__':
    main()
