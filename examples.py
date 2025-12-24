import numpy as np

# 1️⃣ 指數成長 dy/dt = k y
def exponential_growth(t, y, k=1.0):
    return k * y

def exact_exponential(t, y0, k=1.0):
    return y0 * np.exp(k * t)


# 2️⃣ Logistic equation
def logistic(t, y, r=1.0, K=10.0):
    return r * y * (1 - y / K)

def exact_logistic(t, y0, r=1.0, K=10.0):
    return K / (1 + (K/y0 - 1) * np.exp(-r * t))


# 3️⃣ 受迫衰減 dy/dt = -y + sin(t)
def forced_decay(t, y):
    return -y + np.sin(t)
