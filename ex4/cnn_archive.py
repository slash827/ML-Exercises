import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import seaborn as sns
from ex4 import log_results


def read_info(max_rows=55000):
    train_x = np.loadtxt("train_x", dtype=np.float, max_rows=max_rows)
    train_y = np.loadtxt("train_y", dtype='uint8', max_rows=max_rows)
    test_x = np.loadtxt("test_x", dtype=np.float, max_rows=max_rows // 11)
    test_y = np.loadtxt("test_labels", dtype='uint8', max_rows=max_rows // 11)

    train_x = train_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)

    train_x = train_x / 255
    test_x = test_x / 255

    return train_x, train_y, test_x, test_y


def one_hot_encoding(labels, num_classes=10):
    # Convert labels into one-hot format
    temp = []
    for i in range(len(labels)):
        temp.append(to_categorical(labels[i], num_classes=num_classes))
    return np.array(temp)


def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_normal',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def plot_bar_graph(labels, y_samples, title):
    data = []
    for y_value in set(y_samples):
        data.append([y for y in y_samples if y == y_value])

    data_amount = [len(item) for item in data]
    print(f'for title {title} the data is: {data_amount}')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.countplot(data=data)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    plt.savefig(f'{title}.png')
    plt.show()


def main():
    train_x, train_y, test_x, test_y = read_info()
    labels = range(10)
    plot_bar_graph(labels, train_y, "Train images amounts")
    plot_bar_graph(labels, test_y, "Test images amounts")
    plot_bar_graph(labels, train_y.tolist() + test_y.tolist(), "All images amounts")

    return
    # Convert y_train into one-hot format
    train_y = one_hot_encoding(train_y)
    test_y = one_hot_encoding(test_y)

    model = cnn_model()

    model.fit(train_x, train_y, epochs=10,
              validation_data=(test_x, test_y))

    predictions = model.predict(test_x)
    predictions = np.argmax(predictions, axis=1)

    test_y = np.argmax(test_y, axis=1)
    print(classification_report(test_y, predictions))

    print(accuracy_score(test_y, predictions))
    log_results(predictions, "test_y_CNN")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'time took is: {end - start}')
