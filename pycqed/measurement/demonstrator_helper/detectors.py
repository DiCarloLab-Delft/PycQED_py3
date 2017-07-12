import numpy as np
import time
from pycqed.measurement.detector_functions import Detector_Function, Hard_Detector
import quantumsim.qasm
from quantumsim import sparsedm as sdm

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


class Quantumsim_Two_QB_Hard_Detector(Hard_Detector):

    def __init__(self, qasm_file, **kwargs):
        with open(qasm_file) as f:
            file_content = f.read()

        super().__init__()

        self.parser = quantumsim.qasm.QASMParser(**kwargs)
        self.parser.parse(file_content)
        self.name = 'Quantumsim_Two_QB_Detector'
        self.value_names = ['Q0 ', 'Q1 ', 'corr. (Q0, Q1) ']
        self.value_units = ['prob.']*3

    def prepare(self, sweep_points):
        if len(sweep_points) > len(self.parser.circuits):
            raise ValueError("More sweep points than circuits")

    def get_values(self):
        results = []

        for c in self.parser.circuits:
            d = sdm.SparseDM(['q1', 'q0'])
            c.apply_to(d)
            diag = d.full_dm.get_diag()
            parity = diag[[0, 3]].sum()
            p0 = diag[[1, 3]].sum()
            p1 = diag[[2, 3]].sum()
            results.append((p0, p1, parity))

        return np.array(results).T
