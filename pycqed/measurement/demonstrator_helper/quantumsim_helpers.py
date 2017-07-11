import logging
logging.warnings.warn('This file is now included in simulation helpers', DeprecationWarning)

import numpy as np
import os
import re
import urllib.request
import pycqed as pq
from pycqed.measurement import measurement_control
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.demonstrator_helper.detectors import Quantumsim_Two_QB_Hard_Detector


from tess.TessConnect import TessConnection
from qcodes import station


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


def simulate_qasm_file(file_url, options={}):
    # file_url="http://localhost:3000/uploads/asset/file/75/ac5bc9e8-3929-4205-babf-2cf9c4490225.qasm"
    file_path = _retrieve_file_from_url(file_url)
    print('simulation_called')



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
