import numpy as np
import matplotlib.pyplot as plt
from ode_solver import euler, improved_euler, rk4
from examples import exponential_growth, exact_exponential

# Initial conditions
t0 = 0
y0 = 1
h = 0.1
t_end = 5
n = int((t_end - t0) / h)

# Numerical solutions
t_e, y_e = euler(lambda t, y: exponential_growth(t, y), t0, y0, h, n)
t_ie, y_ie = improved_euler(lambda t, y: exponential_growth(t, y), t0, y0, h, n)
t_rk, y_rk = rk4(lambda t, y: exponential_growth(t, y), t0, y0, h, n)

# Exact solution
t_exact = np.linspace(t0, t_end, 200)
y_exact = exact_exponential(t_exact, y0)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t_exact, y_exact, 'k-', label='Exact Solution')
plt.plot(t_e, y_e, 'o--', label='Euler Method')
plt.plot(t_ie, y_ie, 's--', label='Improved Euler Method')
plt.plot(t_rk, y_rk, 'd--', label='RK4 Method')

plt.xlabel("t")
plt.ylabel("y")
plt.title("Numerical Solution of ODE")
plt.legend()
plt.grid()
plt.show()
