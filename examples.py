import numpy as np

# dy/dt = k y
def exponential_growth(t, y, k=1.0):
    return k * y

def exact_exponential(t, y0, k=1.0):
    return y0 * np.exp(k * t)


# Logistic equation
def logistic(t, y, r=1.0, K=10.0):
    return r * y * (1 - y / K)
