import softmax_regression as sreg
import numpy as np


class TestClass:
    def test_softmax(self):
        random_vec = np.random.randint(1, 100) * np.random.rand(1, 10)
        normalized = sreg.softmax(random_vec)
        assert abs(np.sum(normalized) - 1) < 0.000001
"""
    def test_log_likelihood(self):
        labels_count = 4
        data_points_count = 5
        features_count = 3
        label = np.array([1, 2, 3, 0, 3])
        data = np.random.rand(features_count, data_points_count)
        coef = np.random.rand(features_count, labels_count)
        h = sreg.log_likelihood(data, coef, label)
        l_0 = np.log10(sreg.softmax(np.matmul(np.transpose(coef), data[:, 0]))[1])
        l_1 = np.log10(sreg.softmax(np.matmul(np.transpose(coef), data[:, 1]))[2])
        l_2 = np.log10(sreg.softmax(np.matmul(np.transpose(coef), data[:, 2]))[3])
        l_3 = np.log10(sreg.softmax(np.matmul(np.transpose(coef), data[:, 3]))[0])
        l_4 = np.log10(sreg.softmax(np.matmul(np.transpose(coef), data[:, 4]))[3])
        true_h = [l_0, l_1, l_2, l_3, l_4]
        assert -sum(true_h) == h
"""