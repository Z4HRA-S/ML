import numpy as np
import gradient_descent as gr

class BLRegression:

    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label
        self.mean = np.zeros((train_data.shape[1], 1))
        self.cov_matrix = 0.0001 * np.eye(train_data.shape[1])

    def set_param(self, mean, cov_matrix):
        self.mean = mean
        self.cov_matrix = cov_matrix

    def sigmoid(self, z):
        """
        :param z: input scalar
        :return: sigmoid(z)
        """
        z = -1 * z
        return 1 / (1 + np.exp(z))

    def train_newton_method(self):
        n = self.train_data.shape[1]
        label = self.train_label
        coef = np.ones((n, 1))*-10
        prev = np.zeros((n, 1))
        s_n = np.ones((n, n))
        gradient = np.ones((n, 1))
        # train phase: Newton-Raphson method for finding coef
        while np.max(np.abs(prev - coef)) > 0.1:
        #while np.max(np.abs(gradient)) > 0.1:
            prev = coef
            model = np.matmul(coef.T, self.train_data.T)
            logistic_model = np.array([self.sigmoid(m) for m in model], dtype='float64')
            error = (label - logistic_model).reshape(len(label), 1)
            gradient = np.matmul(self.cov_matrix, (coef - self.mean)) - np.matmul(self.train_data.T, error)
            s_diag = np.diag([s * (1 - s) for s in logistic_model[0]])
            hessian = self.cov_matrix + np.matmul(np.matmul(self.train_data.T, s_diag), self.train_data)
            coef = coef - np.matmul(np.linalg.inv(hessian), gradient)
            s_n = hessian
            #print(coef[:5].T)

        return coef, s_n

    def gradient(self, theta):
        label = self.train_label
        model = np.matmul(theta.T, self.train_data.T)
        logistic_model = np.array([self.sigmoid(m) for m in model], dtype='float64')
        error = (label - logistic_model).reshape(len(label), 1)
        grad = np.matmul(self.cov_matrix, (theta - self.mean)) - np.matmul(self.train_data.T, error)
        return grad

    def train_gradient_descent(self):
        coef = gr.gradient_descent(self.gradient, (self.train_data.shape[1], 1), iteration=1000, step=1)
        model = np.matmul(coef.T, self.train_data.T)
        logistic_model = np.array([self.sigmoid(m) for m in model], dtype='float64')
        s_diag = np.diag([s * (1 - s) for s in logistic_model[0]])
        s_n = self.cov_matrix + np.matmul(np.matmul(self.train_data.T, s_diag), self.train_data)
        return coef, s_n

    def predict(self, test_data):
        w_MAP, s_n = self.train_newton_method()
        #w_MAP, s_n = self.train_gradient_descent()
        predicted = []
        for test in test_data:
            test = test.reshape(test.shape[0], 1)
            mu = np.matmul(w_MAP.T, test)
            sigma = np.matmul(np.matmul(test.T, s_n), test)
            c1_posterior = self.sigmoid(1 / np.sqrt(1 + (np.pi * sigma / 8)) * mu)
            label = 1 if c1_posterior >= 0.5 else 0
            predicted.append(label)
        return predicted

    def one_vs_all(self):
        pass
