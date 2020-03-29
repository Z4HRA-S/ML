import numpy as np
import time


def gradient_descent(gradient_func, number_of_classes, number_of_features,
                     step=0.001, gradient_precision=0.0001):
    coef = np.ones((number_of_classes, number_of_features))
    gradient = 1
    iteration = 0
    tik = time.time()
    #while np.linalg.norm(gradient) > gradient_precision:
    for i in range(10000):
        iteration += 1
        gradient = gradient_func(coef)
        coef = coef - step * gradient
    tok = time.time()
    print("We reached gradient:", np.linalg.norm(gradient), "in ", str(iteration), "iteration", "and in",
          tok - tik, "seconds")
    return coef
