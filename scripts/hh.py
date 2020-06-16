"""
Basic script for interacting with the Hodgkin-Huxley model.

About Units:
All voltages are given in mV.
n, m, and h are non-dimensional probabilities that any given gate is open.
"""


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from input_functions import pulse, constant

C_m = 1
Gmax_K = 120
Gmax_Na = 36
Gmax_Leak  =   0.3
V_Na =  50.0
V_K  = -77.0
V_Leak  = -54.387

# All these are ripped from the Hodgkin-Huxley paper.
def alpha_n(V):
    return 0.01 * (V + 10) / (np.exp((V + 10) / 10) - 1)

def alpha_h(V):
    return 0.07 * np.exp(V / 20)

def alpha_m(V):
    return 0.1 * (V + 25) / (np.exp((V + 25) / 10) - 1)

def beta_n(V):
    return 0.125 * np.exp(V / 80)

def beta_h(V):
    return 1 / (np.exp((V + 30) / 10) + 1)

def beta_m(V):
    return 4 * np.exp(V / 18)

def I_K(V, n):
    return np.power(n, 4) * Gmax_K * (V - V_K)

def I_Na(V, m, h):
    return np.power(m, 3) * h * Gmax_Na * (V - V_Na)

def I_Leak(V):
    return Gmax_Leak * (V - V_Leak)

def hh(X, t, I):
    V, n, m, h = X

    dVdt = (I(t) - I_K(V, n) - I_Na(V, m, h) - I_Leak(V)) / C_m
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h

    return [dVdt, dndt, dmdt, dhdt]

def main():
    # Values taken from HH tutorial.
    X_0 = [-65, 0.05, 0.6, 0.32]
    t = np.linspace(0, 10, 1000)
    X = odeint(hh, X_0, t, args=((pulse(45, 1),)))
    V = X[:, 0]

    plt.plot(t, V)
    plt.show()

def check_pulse():
    t = np.linspace(0, 10, 1000)
    y = pulse(45, 0.1, False)

    plt.plot(t, y(t))
    plt.show()

if __name__ == '__main__':
    # check_pulse()
    main()
