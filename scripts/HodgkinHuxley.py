import scipy as sp
import numpy as np
import pylab as plt
from scipy.integrate import odeint

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    C_m  =   1.0
    """membrane capacitance, in uF/cm^2"""

    g_L  =   0.3
    """Leak maximum conductances, in mS/cm^2"""

    E_L  = -54.387
    """Leak Nernst reversal potentials, in mV"""

    t = sp.arange(0.0, 450.0, 0.01)
    """ The time to integrate over """

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        magnitude1 = 1e6
        magnitude2 = 3.5e1
        return magnitude1*(t>100) - magnitude1*(t>200) + magnitude2*(t>300) - magnitude2*(t>400)

    @staticmethod
    def dALLdt(V, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        dVdt = (self.I_inj(t) - self.I_L(V)) / self.C_m
        return dVdt

    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """

        V = odeint(self.dALLdt, -65, self.t, args=(self,))
        il = self.I_L(V)

        plt.figure()

        ax1 = plt.subplot(2,1,1)
        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(self.t, V, 'k')
        plt.ylabel('V (mV)')

        plt.subplot(2,1,2, sharex = ax1)
        i_inj_values = [self.I_inj(t) for t in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        # plt.ylim(-1, 0.)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()
