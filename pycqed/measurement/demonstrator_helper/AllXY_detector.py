import numpy as np
import time
from pycqed.measurement.detector_functions import Detector_Function


class AllXYDetector(Detector_Function):

    def __init__(self, noise=0, delay=0, **kw):
        self.name = "AllXYDetector"
        self.value_names = ['q0', 'q1']
        self.value_units = ['\%', '\%']

        self.noise = noise
        self.delay = delay
        self.detector_control = 'hard'

    def get_values(self):
        dat = np.zeros(21, len(self.value_names))
        for i in range(len(self.value_names)):
            start = (np.random.random(5) - 0.5) * self.noise
            middel = (np.random.random(12) - 0.5) * self.noise + 0.5
            end = (np.random.random(4) - 0.5) * self.noise + 1
            dat[:, i] = np.concatenate((start, middel, end))

        time.sleep(self.delay)

        return dat 
