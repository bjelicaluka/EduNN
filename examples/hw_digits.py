import numpy as np
from lib.Model import Model
import mnist


def prepare_data():
    # load data
    num_classes = 10
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    # data processing
    # flatten 28x28 to 784x1 vectors, [60000, 784]
    x_train_ = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]).astype('float32')
    x_train_ = x_train_ / 255
    y_train_ = np.eye(num_classes)[train_labels]

    # flatten 28x28 to 784x1 vectors, [60000, 784]
    x_test_ = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32')
    x_test_ = x_test_ / 255
    y_test_ = np.eye(num_classes)[test_labels]
    return x_train_, y_train_, x_test_, y_test_


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prepare_data()
    model = Model([[784], [30, "relu"], [10, "softmax"]], loss_function="cross_entropy", learning_rate=0.001)

    model.train("gradient_descent", x_train, y_train, epochs=5, shuffle_data=True, batch_size=10)

    print(model.calculate_accuracy(inputs=x_test, labels=y_test))

    # model.save("../../saved/hw_digits/", "saved.pck")
