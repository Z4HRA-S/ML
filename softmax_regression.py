import numpy as np


def encode_label(label):
    label = np.array(label)
    classes = len(set(label))
    encoded = np.zeros((label.shape[0], classes))
    for i in range(label.shape[0]):
        encoded[i][label[i] - 1] = 1
    return encoded


def softmax(z):
    """
    :param z: input vector
    :return: softmax(z)
    """
    z = z - np.max(z)  # normalize the vector to avoid overflow in exp
    return np.exp(z) / np.sum(np.exp(z))


def gradient_log_likelihood(train_data, train_label, coef, lamda=0.0001):
    """
    Calculate hypothesis
    :param train_data:
    :param train_label:
    :param coef: The matrix parameters of our hypothesis
    :param lamda: Regularization
    :return: gradient of log-likelihood with respect to coef
    """
    train_label = encode_label(train_label)
    n = train_data.shape[0]
    gradient = np.zeros((coef.shape[0], coef.shape[1]))
    for i in range(n):
        data = train_data[i].reshape(train_data[i].shape[0], 1)
        error = (softmax(np.matmul(coef, train_data[i])) - train_label[i]).reshape(train_label.shape[1], 1)
        gradient = gradient + np.transpose(np.matmul(data, error.T))
    gradient = (gradient / n) + lamda * coef
    return gradient


def predict(test_data, coef):
    predicted_label = [np.argmax(softmax(np.matmul(coef, test_data[i]))) + 1  # Our label are started from 1
                       for i in range(test_data.shape[0])]
    return predicted_label
