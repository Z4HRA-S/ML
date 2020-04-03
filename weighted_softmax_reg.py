import numpy as np
import gradient_descent as gr


class WSMRegression:

    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label
        self.lamda = 0.0001
        self.tau = 0.8
        self.weight = np.ones((train_data.shape[0],))

    def set_param(self, tau=0.8, lamda=0.0001):
        self.lamda = lamda
        self.tau = tau

    def __set_weight(self, data_point):
        magnitude = lambda vector: sum(map(lambda x: x ** 2, vector))
        self.weight = [np.exp(-magnitude(data_point - data)) / (2 * self.tau ** 2) for data in self.train_data]

    def __encode_label(self, label):
        label = np.array(label)
        classes = len(set(label))
        encoded = np.zeros((label.shape[0], classes))
        for i in range(label.shape[0]):
            encoded[i][int(label[i] - 1)] = 1
        return encoded

    def __softmax(self, z):
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

        train_label = self.__encode_label(self.train_label)
        n = self.train_data.shape[0]
        gradient = np.zeros(coef.shape)
        for i in range(n):
            data = self.train_data[i].reshape(self.train_data[i].shape[0], 1)
            error = (train_label[i] - self.__softmax(np.matmul(coef, self.train_data[i]))).reshape(train_label.shape[1],1)
            gradient = gradient + self.weight[i] * np.transpose(np.matmul(data, error.T))
        gradient = gradient + self.lamda * coef
        return gradient

    def predict(self, test_data):
        classes = len(set(self.train_label))
        features = self.train_data.shape[1]
        predicted = []
        for t in test_data:
            self.__set_weight(t)
            coef = gr.gradient_descent(self.gradient_log_likelihood, (classes, features))
            label = np.argmax(self.__softmax(np.matmul(coef, t))) + 1
            predicted.append(label)
        return predicted
