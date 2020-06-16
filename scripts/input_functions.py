import numpy as np

def constant(k):
    return lambda _: k


def pulse(k, T, repeat = True):
    """Acts as a rectangular pulse. Returns a function of t that returns either k or 0 depending where in the pulse you are.

    Params:
    k: value of function when active
    T: total time length of pulse. k will be applied for T / 2.
    repeat: whether multiple pulses happen. """
    def f(t):
        t = t % T if repeat else t
        active = t < T / 2
        return np.where(active, k, 0)

    return f
