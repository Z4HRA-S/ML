import numpy as np
import gradient_descent as gr


class SoftMaxRegression:

    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label
        self.lamda = 0.0001

    def set_param(self, lamda=0.0001):
        self.lamda = lamda

    def encode_label(self, label):
        label = np.array(label)
        classes = len(set(label))
        encoded = np.zeros((label.shape[0], classes))
        for i in range(label.shape[0]):
            encoded[i][int(label[i] - 1)] = 1
        return encoded

    def softmax(self, z):
        """
        :param z: input vector
        :return: softmax(z)
        """
        z = -1 * z
        z = z - np.max(z)  # normalize the vector to avoid overflow in exp
        return np.exp(z) / np.sum(np.exp(z))

    def gradient_log_likelihood(self, coef):
        """
        Calculate hypothesis
        :param coef: The matrix parameters of our hypothesis
        :return: gradient of log-likelihood with respect to coef
        """
        train_label = self.encode_label(self.train_label)
        n = self.train_data.shape[0]
        gradient = np.zeros(coef.shape)
        for i in range(n):
            data = self.train_data[i].reshape(self.train_data[i].shape[0], 1)
            error = (train_label[i] - self.softmax(np.matmul(coef, self.train_data[i]))).reshape(
                self.train_label.shape[1], 1)
            gradient = gradient + np.transpose(np.matmul(data, error.T))
        gradient = gradient + self.lamda * coef
        return gradient

    def predict(self, test_data):
        features = self.train_data.shape[1] - 1
        classes = len(set(self.train_label))
        coef = gr.gradient_descent(self.gradient_log_likelihood, coef_shape=(classes, features))
        predicted_label = [np.argmax(self.softmax(np.matmul(coef, test_data[i]))) + 1  # Our label are started from 1
                           for i in range(test_data.shape[0])]
        return predicted_label
