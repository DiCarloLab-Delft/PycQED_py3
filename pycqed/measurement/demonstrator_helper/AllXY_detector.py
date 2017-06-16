import numpy as np
import time
from pycqed.measurement.detector_functions import Detector_Function


class AllXYDetector(Detector_Function):

    def __init__(self, noise=0, delay=0, **kw):
        super(AllXYDetector, self).__init__()
        self.name = "AllXYDetector"
        self.noise = noise
        self.delay = delay
        self.detector_control = 'hard'

    def get_values(self):
        start = (np.random.random(5) - 0.5) * self.noise
        middel = (np.random.random(12) - 0.5) * self.noise + 0.5
        end = (np.random.random(4) - 0.5) * self.noise + 1
        time.sleep(self.delay)
        return np.concatenate((start, middel, end))
