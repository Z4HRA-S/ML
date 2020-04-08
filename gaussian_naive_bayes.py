import numpy as np


class GNB:

    def __init__(self):
        self.name = "Gaussian Naive Bayes"

    def fit(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label
        classes = len(set(train_label))
        self.__mean, self.__var = self.__train()

        prior = np.zeros((classes,))
        for label in train_label:
            i = int(label - 1)
            prior[i] += 1
        self.__prior = prior / (len(train_label))

    def __encode_label(self, label):
        label = np.array(label)
        classes = len(set(label))
        encoded = np.zeros((label.shape[0], classes))
        for i in range(label.shape[0]):
            encoded[i][int(label[i] - 1)] = 1
        return encoded

    def __train(self):
        classes = len(set(self.train_label))
        features = self.train_data.shape[1]
        label = self.__encode_label(self.train_label)

        n_k = np.zeros((classes,))
        for lb in self.train_label:
            i = int(lb - 1)
            n_k[i] += 1

        mean = np.matmul(self.train_data.T, label, dtype=np.float128) / n_k
        variance = np.zeros((features, classes), dtype=np.float128)
        for i in range(features):
            for k in range(classes):
                variance[i][k] = sum([(self.train_data[n][i] - mean[i][k]) ** 2 * label[n][k]
                                      for n in range(len(self.train_label))])
        variance = variance / n_k
        return mean, variance

    def predict(self, test_data):
        classes = len(set(self.train_label))
        features = self.train_data.shape[1]
        predicted = []
        guassian = lambda x, mu, sigma: np.exp((x - mu / sigma) ** 2 / -2, dtype=np.float128) / (
                sigma * np.sqrt(2 * np.pi))

        for t in test_data:
            label = np.argmax([self.__prior[k] * np.prod([guassian(t, self.__mean[i][k], self.__var[i][k])
                                                   for i in range(features)], dtype="float64") for k in range(classes)])
            predicted.append(label)
        return predicted