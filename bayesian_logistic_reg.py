import numpy as np
import gradient_descent as gr


class BLRegression:

    def __init__(self):
        self.name = "Bayesian Logistic Regression"

    def fit(self, train_data, label):
        self.train_data = train_data
        self.label = label
        self.train_label = label
        self.mean = np.mean(train_data, axis=0)
        self.cov_matrix = np.linalg.inv(np.cov(train_data.T))

    def set_param(self, mean, cov_matrix):
        self.mean = mean
        self.cov_matrix = cov_matrix

    def __sigmoid(self, z):
        """
        :param z: input scalar
        :return: sigmoid(z)
        """
        if z >= 0:
            return 1 / (1 + np.exp(-z, dtype=np.float128))
        else:
            return np.exp(z, dtype=np.float128) / (1 + np.exp(z, dtype=np.float128))

    def __train_newton_method(self):
        n = self.train_data.shape[1]
        label = self.train_label
        coef = np.ones((n, 1))
        prev = np.zeros((n, 1))
        s_n = np.ones((n, n))
        gradient = np.ones((n, 1))
        # train phase: Newton-Raphson method for finding coef
        while np.max(np.abs(prev - coef)) > 0.1:
            # while np.max(np.abs(gradient)) > 0.1:
            prev = coef
            model = np.matmul(coef.T, self.train_data.T)
            logistic_model = np.array([self.__sigmoid(m) for m in model], dtype='float64')
            error = (label - logistic_model).reshape(len(label), 1)
            gradient = np.matmul(self.cov_matrix, (coef - self.mean)) - np.matmul(self.train_data.T, error)
            s_diag = np.diag([s * (1 - s) for s in logistic_model[0]])
            hessian = self.cov_matrix + np.matmul(np.matmul(self.train_data.T, s_diag), self.train_data)
            coef = coef - np.matmul(np.linalg.inv(hessian), gradient)
            s_n = hessian
        # print(coef[:5].T)

        return coef, s_n

    def __gradient(self, theta):
        model = np.matmul(theta.T, self.train_data.T, dtype=np.float128)
        logistic_model = np.array([self.__sigmoid(m) for m in model], dtype=np.float128)
        error = (logistic_model - self.train_label)
        grad = np.matmul(self.cov_matrix, (theta - self.mean),
                         dtype=np.float128) + np.matmul(self.train_data.T, error.T, dtype=np.float128)
        return grad

    def __train_gradient_descent(self):
        coef = gr.gradient_descent(self.__gradient, (self.train_data.shape[1],), iteration=1000, step=1)
        model = np.matmul(coef.T, self.train_data.T)
        logistic_model = np.array([self.__sigmoid(m) for m in model], dtype='float64')
        s_diag = np.diag([s * (1 - s) for s in logistic_model])
        s_n = self.cov_matrix + np.matmul(np.matmul(self.train_data.T, s_diag), self.train_data)
        return coef, s_n

    def __predict(self, test_data):
        # w_MAP, s_n = self.train_newton_method()
        w_MAP, s_n = self.__train_gradient_descent()
        predicted = []
        for test in test_data:
            test = test.reshape(test.shape[0], 1)
            mu = np.matmul(w_MAP.T, test)
            sigma = np.matmul(np.matmul(test.T, s_n), test)
            c1_posterior = self.__sigmoid(1 / np.sqrt(1 + (np.pi * sigma / 8)) * mu)
            predicted.append(c1_posterior)
            # label = 1 if c1_posterior >= 0.5 else 0
            # predicted.append(label)
        return predicted

    def predict(self, test_data):  # use on vs all method
        classes = set(self.label)
        predicted = np.zeros((len(classes), len(test_data)))
        for c in classes:
            self.train_label = [1 if t == c else 0 for t in self.label]
            index = int(c - 1)
            predicted[index] = self.__predict(test_data)

        predicted = [np.argmax(predicted[:, i]) + 1 for i in range(len(test_data))]
        return predicted
