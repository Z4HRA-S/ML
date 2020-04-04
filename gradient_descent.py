import numpy as np
import time


def gradient_descent(gradient_func, coef_shape, iteration=1000, step=0.001):
    coef = np.ones(coef_shape)
    gradient = np.ones(coef_shape)
    gr_diff = 0.1
    iter = 0
    tik = time.time()
    for i in range(iteration):
        iter += 1
        prev = gradient
        gradient = gradient_func(coef)
        coef = coef - step * gradient
        if np.max(np.abs(gradient)) < gr_diff:
            break
    tok = time.time()
    print("We reached gradient:", np.linalg.norm(gradient), "with step:", step, "in ", str(iter), "iteration", "and in",
          tok - tik, "seconds")
    return coef
