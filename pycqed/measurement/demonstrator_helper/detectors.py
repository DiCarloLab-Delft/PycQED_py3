import numpy as np
from pycqed.measurement.detector_functions import Hard_Detector
import quantumsim.qasm
from quantumsim import sparsedm as sdm


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
            d = sdm.SparseDM(['q0', 'q1'])
            # this ensures that the qubits are in the density matrix
            # in the right order. (fix this, should use d.idx_in_full!)
            d.ensure_dense("q0")
            d.ensure_dense("q1")
            c.apply_to(d)
            diag = d.full_dm.get_diag()
            parity = diag[[0, 3]].sum()
            p0 = diag[[1, 3]].sum()
            p1 = diag[[2, 3]].sum()
            results.append((p0, p1, parity))

        return np.array(results).T
