import numpy as np
import numpy.linalg as la

# gaussian kernel function
def rbf_kernel(sigma, **kwargs):

    def f(x1, x2):
        return np.exp(-np.sqrt(la.norm(x1-x2) ** 2 / (2 * sigma ** 2)))
    return f

# polynomial kernel function
def polynomial_kernel(power, coef, **kwargs):
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f

