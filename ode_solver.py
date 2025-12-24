import numpy as np

def euler(f, t0, y0, h, n):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)

    t[0] = t0
    y[0] = y0

    for i in range(n):
        y[i+1] = y[i] + h * f(t[i], y[i])
        t[i+1] = t[i] + h

    return t, y


def improved_euler(f, t0, y0, h, n):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)

    t[0] = t0
    y[0] = y0

    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h * k1)
        y[i+1] = y[i] + h * (k1 + k2) / 2
        t[i+1] = t[i] + h

    return t, y


def rk4(f, t0, y0, h, n):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)

    t[0] = t0
    y[0] = y0

    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h*k1/2)
        k3 = f(t[i] + h/2, y[i] + h*k2/2)
        k4 = f(t[i] + h, y[i] + h*k3)

        y[i+1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t[i+1] = t[i] + h

    return t, y
