import numpy as np
import time


def gradient_descent(gradient_func, coef_shape, iteration=1000, step=0.001):
    coef = np.ones(coef_shape)
    gradient = np.ones(coef_shape)
    gr_diff = 0.01
    iter = 0
    tik = time.time()
    for i in range(iteration):
        iter += 1
        prev = gradient
        gradient = gradient_func(coef)
        coef = coef - step * gradient
        if np.max(np.abs(gradient - prev)) < gr_diff:
            break

    tok = time.time()
    print("We reached gradient:", gradient, "in ", str(iter), "iteration", "and in",
          tok - tik, "seconds")
    return coef
