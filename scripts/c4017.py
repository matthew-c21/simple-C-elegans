import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Values from PLM type neurons.
C_m = 9.1 * 1e-12    # 9.1 pF
R_m = 16 * 1e-12     # 16 GOhms
V_leak = -35 * 1e-3  # 35 mV
V_eq = 0

def Vprime(V, t, I_ext = None):
    i = 0 if I_ext is None else I_ext(t)
    return ((V_leak - V) / R_m + i) / C_m


def main(args = None):
    t = np.linspace(0, 1500) / 1000
    V = odeint(Vprime, V_eq, t, args = args)

    plt.plot(t, V)

    plt.show()

