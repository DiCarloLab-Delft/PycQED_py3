import logging
import numpy as np
import os
import re
import urllib.request
import pycqed as pq
import json

from pycqed.measurement import measurement_control
from pycqed.instrument_drivers.virtual_instruments.pyqx.qx_client import \
    qx_client
from pycqed.measurement.detector_functions import QX_Hard_Detector
from pycqed.measurement.demonstrator_helper.detectors import \
    Quantumsim_Two_QB_Hard_Detector
from pycqed.measurement import sweep_functions as swf

from tess.TessConnect import TessConnection
from qcodes import station

tc = TessConnection()
tc.connect("simulate")


st = station.Station()
# Connect to the qx simulator
MC = measurement_control.MeasurementControl(
    'MC', live_plot_enabled=False, verbose=True)

datadir = os.path.abspath(os.path.join(
    os.path.dirname(pq.__file__), os.pardir, 'simulator_data'))
MC.datadir(datadir)
MC.station = st

st.add_component(MC)


def simulate_qasm_file(file_url: str,  config_json: str):
    file_path = _retrieve_file_from_url(file_url)
    options = json.loads(config_json)
    if("simulator" in options):
        if options['simulator'].lower() == "qx":
            return _simulate_QX(file_path, options)
        elif options['simulator'].lower() == "quantumsim":
            return _simulate_quantumsim(file_path, options)
        else:
            raise Exception("ERROR: simulator "+options['quantumsim'] +
                            " not recognized")
    else:
        raise Exception("ERROR: No simulator selected")

# Private
# -------


def _simulate_QX(file_path, options):
    qxc = qx_client()
    try:
        qxc.connect()
        qxc.create_qubits(2)
        qx_sweep = swf.QX_Hard_Sweep(qxc, file_path)
        num_avg = options.get('num_avg', 10000)  # 10000 is the default

        qx_detector = QX_Hard_Detector(qxc, [file_path], num_avg=num_avg)
        sweep_points = range(len(qx_detector.randomizations[0]))
        # qx_detector.prepare(sweep_points)
        # Start measurment
        MC.set_detector_function(qx_detector)
        MC.set_sweep_function(qx_sweep)
        MC.set_sweep_points(sweep_points)
        dat = MC.run("run QASM")
        return _MC_result_to_chart_dict(dat)

    finally:
        qxc.disconnect()


def _simulate_quantumsim(file_path, options):
    quantumsim_sweep = swf.None_Sweep()
    quantumsim_sweep.parameter_name = 'Circuit number '
    quantumsim_sweep.unit = '#'

    qubit_parameters = {
        'Q0': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.0189, 'frac1_1': 0.918},
        'Q1': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.068, 'frac1_1': 0.949},
        'q0': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.0189, 'frac1_1': 0.918},
        'q1': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.068, 'frac1_1': 0.949},
        'default': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.068, 'frac1_1': 0.949}}

    quantumsim_det = Quantumsim_Two_QB_Hard_Detector(
        file_path, timegrid=20, gate_1_step=1, gate_2_step=5,
            qubit_parameters=qubit_parameters)
    sweep_points = range(len(quantumsim_det.parser.circuits))

    MC.set_detector_function(quantumsim_det)
    MC.set_sweep_function(quantumsim_sweep)
    MC.set_sweep_points(sweep_points)
    dat = MC.run("run QASM")
    print('simulation finished')
    return _MC_result_to_chart_dict(dat)


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
