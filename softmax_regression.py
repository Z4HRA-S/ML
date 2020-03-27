import numpy as np
import time


def softmax(z):
    """
    :param index: index of the element that match the label
    :param z: input vector
    :return: softmax(z)
    """
    z = z - np.max(z)  # normalize the vector to avoid overflow in exp
    return np.exp(z) / np.sum(np.exp(z))


def log_likelihood(data, coef, label):
    """
    Calculate hypothesis
    :param data: Training data - numpy matrix
    :param coef: The matrix parameters of our hypothesis
    :param label: Training label - numpy vector
    :return: Output of log-likelihood
    """
    loss = [np.log10(softmax(np.matmul(np.transpose(coef), data[:, i]))[label[i]])
            for i in range(data.shape[1])]
    return -sum(loss)


def gradient_descent(data, label, step):
    eps = 0.000001
    number_of_classes = len(set(label))
    number_of_features = data.shape[0]
    coef = np.ones((number_of_features, number_of_classes))
    gradient = 1
    iteration = 0
    tik = time.time()
    while abs(gradient) > 0.0001:
        # for i in range(100):
        iteration += 1
        prev_gradient = gradient
        gradient = (log_likelihood(data, coef + eps, label) - log_likelihood(data, coef, label)) / eps
        if prev_gradient * gradient < 0:
            step = step / 2
        if step < eps:
            eps = eps / 2
        coef = coef - step * gradient
    tok = time.time()
    print("We reached gradient:", str(gradient), "in ", str(iteration), "iteration", "and in",
          tok - tik, "seconds")
    return coef

