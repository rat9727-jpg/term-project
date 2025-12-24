import numpy as np
import matplotlib.pyplot as plt
from ode_solver import euler, improved_euler, rk4
from examples import (
    exponential_growth, exact_exponential,
    logistic, exact_logistic,
    forced_decay
)

# =========================
# Menu
# =========================
print("Choose an ODE to solve:")
print("1. Exponential Growth  dy/dt = k y")
print("2. Logistic Equation   dy/dt = r y (1 - y/K)")
print("3. Forced Decay        dy/dt = -y + sin(t)")

choice = input("Enter your choice (1/2/3): ")

# =========================
# Initial conditions
# =========================
t0 = 0
t_end = 5
h = 0.1
n = int((t_end - t0) / h)

# =========================
# Select equation
# =========================
if choice == "1":
    y0 = 1
    f = lambda t, y: exponential_growth(t, y)
    exact = lambda t: exact_exponential(t, y0)
    title = "Exponential Growth"

elif choice == "2":
    y0 = 1
    f = lambda t, y: logistic(t, y)
    exact = lambda t: exact_logistic(t, y0)
    title = "Logistic Equation"

elif choice == "3":
    y0 = 0
    f = lambda t, y: forced_decay(t, y)
    exact = None
    title = "Forced Decay"

else:
    print("Invalid choice")
    exit()

# =========================
# Numerical solutions
# =========================
t_e, y_e = euler(f, t0, y0, h, n)
t_ie, y_ie = improved_euler(f, t0, y0, h, n)
t_rk, y_rk = rk4(f, t0, y0, h, n)

# =========================
# Plot solutions
# =========================
plt.figure(figsize=(8, 5))

if exact is not None:
    t_plot = np.linspace(t0, t_end, 2000)
    y_exact = exact(t_plot)
    plt.plot(t_plot, y_exact, 'k-', label="Exact Solution")

plt.plot(t_e, y_e, 'o--', label="Euler")
plt.plot(t_ie, y_ie, 's--', label="Improved Euler")
plt.plot(t_rk, y_rk, 'd--', label="RK4")

plt.xlabel("t")
plt.ylabel("y")
plt.title(title)
plt.legend()
plt.grid()
plt.show()

# =========================
# Error analysis (if exact exists)
# =========================
if exact is not None:
    y_exact_e = exact(t_e)
    y_exact_ie = exact(t_ie)
    y_exact_rk = exact(t_rk)

    print("Maximum Error:")
    print("Euler:", np.max(np.abs(y_exact_e - y_e)))
    print("Improved Euler:", np.max(np.abs(y_exact_ie - y_ie)))
    print("RK4:", np.max(np.abs(y_exact_rk - y_rk)))
else:
    print("No exact solution available. Error analysis skipped.")

print("Program finished successfully.")
