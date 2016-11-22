import numpy as np
import lmfit

class SinglePoleStep:
    def __init__(self, amp, tau):
        self.amp_s = amp
        self.tau_s = tau
        self.amp_k = self.amp_s/(self.amp_s-1.)
        self.tau_k = (1.-self.amp_s)*self.tau_s

        self.step_func = lambda t: 1.-self.amp_s*np.exp(-t/self.tau_s)

        self.kernel_step_func = lambda t: 1.+self.amp_k*np.exp(-t/self.tau_k)
        self.step_model = lmfit.Model(self.step_func)
        self.kernel_step_model = lmfit.Model(self.kernel_step_func)