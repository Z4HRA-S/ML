import numpy as np
import scipy.stats


class GNB:

    def __init__(self):
        self.name = "Gaussian Naive Bayes"

    def fit(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def separate_class(self):
        classes = set(self.train_label)
        separated = [[] for i in range(len(classes))]

        for i in range(self.train_data.shape[0]):
            separated[int(self.train_label[i]) - 1].append(self.train_data[i])
        return separated

    def _train(self):
        separated = self.separate_class()
        mean = np.array([np.mean(s, axis=0) for s in separated])
        std = np.array([np.std(s, axis=0) for s in separated])
        prior = [len(data) / self.train_data.shape[0] for data in separated]
        return mean, std, prior

    def predict(self, test_data):
        mean, std, prior = self._train()
        gaussian = lambda x, mu, sigma: scipy.stats.norm(mu, sigma).pdf(x)
        predicted = []
        for t in test_data:
            prob = np.log(prior) + np.sum(np.log(gaussian(t, mean, std)), axis=1)
            #prob = prior * np.prod(gaussian(t, mean, std), axis=1)
            label = np.argmax(prob) + 1
            predicted.append(label)
        return predicted
