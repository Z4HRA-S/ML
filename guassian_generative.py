import numpy as np


class Generative:

    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def set_param(self):
        pass

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

    def train(self):
        classes = len(set(self.train_label))
        features = self.train_data.shape[1]
        n_k = np.zeros((classes,))
        for lb in self.train_label:
            i = int(lb - 1)
            n_k[i] += 1

        prior = n_k / len(self.train_label)
        label = self.encode_label(self.train_label)

        mean_k = np.matmul(self.train_data.T, label) / n_k
        cov_matrix = np.zeros((features, features), dtype="float64")
        for k in range(classes):
            for n in range(self.train_data.shape[0]):
                error = (self.train_data[n] - mean_k[:, k]).reshape(features, 1)
                cov_matrix += label[n][k] * np.matmul(error, error.T)
        cov_matrix = cov_matrix / len(self.train_label)
        inv_cov = np.linalg.inv(cov_matrix)
        bias_term = np.array(
            [np.matmul(np.matmul(mean_k[:, k].T, inv_cov), mean_k[:, k]) * (-0.5) + np.log(prior[k]) for k in
             range(classes)])
        coef = np.matmul(inv_cov, mean_k)
        return coef, bias_term

    def predict(self, test_data):
        coef, bias = self.train()
        predicted = []
        for t in test_data:
            posterior = self.softmax(np.matmul(coef.T, t) + bias)
            label = np.argmax(posterior) + 1
            predicted.append(label)
        return predicted
