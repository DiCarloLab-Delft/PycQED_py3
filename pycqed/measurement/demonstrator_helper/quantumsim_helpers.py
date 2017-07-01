import numpy as np
import os
import re
import urllib.request
import pycqed as pq
from pycqed.measurement import measurement_control
from pycqed.measurement import detector_functions as det
from pycqed.measurement import sweep_functions as swf
from quantumsim import sparsedm as sdm
from tess.TessConnect import TessConnection
from qcodes import station

import quantumsim.qasm

tc = TessConnection()
tc.connect("simulate")
default_simulate_options = {
    "num_avg": 10000,
    "iterations": 1
}

st = station.Station()
# Connect to the qx simulator
MC = measurement_control.MeasurementControl(
    'MC', live_plot_enabled=False, verbose=True)
MC.station = st

st.add_component(MC)


class Quantumsim_Two_QB_Hard_Detector(det.Hard_Detector):

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
            d = sdm.SparseDM(c.get_qubit_names())
            c.apply_to(d)
            diag = d.full_dm.get_diag()
            parity = diag[[0, 3]].sum()
            p0 = diag[[1, 3]].sum()
            p1 = diag[[2, 3]].sum()
            results.append((p0, p1, parity))

        return np.array(results).T


def simulate_qasm_file(file_url, options={}):
    # file_url="http://localhost:3000/uploads/asset/file/75/ac5bc9e8-3929-4205-babf-2cf9c4490225.qasm"
    file_path = _retrieve_file_from_url(file_url)
    print('simulation_called')

    quantumsim_sweep = swf.None_Sweep()
    quantumsim_sweep.parameter_name = 'Circuit number '
    quantumsim_sweep.unit = '#'

    qubit_parameters = {
        'Q0': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.0189, 'frac1_1': 0.918},
        'Q1': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.068, 'frac1_1': 0.949},
        'q0': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.0189, 'frac1_1': 0.918},
        'q1': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.068, 'frac1_1': 0.949}}

    quantumsim_det = Quantumsim_Two_QB_Hard_Detector(
        file_path, dt=(40, 280), qubit_parameters=qubit_parameters)
    sweep_points = range(len(quantumsim_det.parser.circuits))

    MC.set_detector_function(quantumsim_det)
    MC.set_sweep_function(quantumsim_sweep)
    MC.set_sweep_points(sweep_points)
    dat = MC.run("run QASM")
    print('simulation finished')
    return _MC_result_to_chart_dict(dat)


# Private
# -------


def _get_qasm_sweep_points(file_path):
    counter = 0
    with open(file_path) as f:
        line = f.readline()
        while(line):
            if re.match(r'(^|\s+)(measure|RO)(\s+|$)', line):
                counter += 1
            line = f.readline()

    return range(counter)


def _retrieve_file_from_url(file_url):

    file_name = file_url.split("/")[-1]
    base_path = os.path.join(
        pq.__path__[0], 'measurement', 'demonstrator_helper',
        'qasm_files', file_name)
    file_path = base_path
    # download file from server
    urllib.request.urlretrieve(file_url, file_path)
    return file_path


def _MC_result_to_chart_dict(result):
    for i in result:
        if(isinstance(result[i], np.ndarray)):
            result[i] = result[i].tolist()
    return [{
        "data-type": "chart",
        "data": result
    }]
