import numpy as np
import os
import json
import re
import urllib.request
import pycqed as pq
import qcodes as qc

from pycqed.measurement import measurement_control
from qcodes.instrument.base import Instrument
from pycqed.measurement.demonstrator_helper.detectors import \
    Quantumsim_Two_QB_Hard_Detector
from pycqed.measurement import sweep_functions as swf

"""
MeasurementControl and other legacy imports
"""

from tess.TessConnect import TessConnection
import logging
"""
TessConnection is used to tell Tess that we are a kernel open for use
"""

default_execute_options = {}

tc = TessConnection()
tc.connect("execute_CCL")
default_simulate_options = {
    "num_avg": 10000,
    "iterations": 1
}
"""
We connect to tess by giving the kernel type as "execute_CCL"
"""

if hasattr(qc.station,'component'):
    st = qc.station
    new_station = False
else:
    st = qc.station.Station()
    new_station = True

"""
Create the station for which the Instruments can connect to. A virtual representation
of the physical setup. In our case, the CCL. Since we're calling the station, 
"""

MC_demo = measurement_control.MeasurementControl(
    'Demonstrator_MC', live_plot_enabled=False, verbose=True)

datadir = os.path.abspath(os.path.join(
    os.path.dirname(pq.__file__), os.pardir, 'demonstrator_execute_data'))
MC_demo.datadir(datadir)
MC_demo.station = st

st.add_component(MC_demo)

def execute_qisa_file(qisa_file_url: str,  config_json: str,
                      verbosity_level: int=0):
    options = json.loads(config_json)

    MC = Instrument.find_instrument('MC')
    CCL = Instrument.find_instrument('CCL')
    device = Instrument.find_instrument('CCL_transmon')

    num_avg = int(options.get('num_avg', 512))
    
    nr_soft_averages = int(np.round(num_avg/512))
    MC.soft_avg(nr_soft_averages)

    device.RO_acq_averages(512)

    # Get the qisa file
    qisa_fp = _retrieve_file_from_url(qisa_file_url)

    # Two ways to generate the sweep_points. Either I get from the file_url
    # or I get the appended options file which has the kw "measurement_points"
    #sweep_points_fp = _retrieve_file_from_url(sweep_points_file_url)
    #sweep_points = json.loads(sweep_points_fp)
    #sweep_points = sweep_points["measurement_points"]

    # Ok, I am assured by stanvn that he will provide me a options with kw
    sweep_points = options["measurement_points"]

    s = swf.OpenQL_File_Sweep(file_path_sweep = qisa_fp, CCL=CCL,
                 parameter_name ='Points', unit ='a.u.',
                 upload=True)

    # To be modified depending on what we're gonna name the S-7 qubits? <<<<<
    d = device.get_integrated_average_detector()
    #d.value_names = ['Q0 ', 'Q1 ', 'Corr. (Q0, Q1) ']
    #d.value_units = ['frac.', 'frac.', 'frac.']
    #>>>>>>

    MC.set_sweep_function(s)
    MC.set_sweep_points(sweep_points)

    MC.set_detector_function(d)
    data = MC.run('CCL_execute')  # FIXME <- add the proper name

    return _MC_result_to_chart_dict(data)

def calibrate(config_json: str):
    """
    Perform calibrations based on the options specified in the config_json.
    Calibrations are performed using the dependency graph
    """
    print('*'*80)
    print('\t options')
    print('*'*80)

    print(config_json)

    print('*'*80)
    options = json.loads(config_json)


    # relies on this being added explicitly
    cal_graph = station.calibration_graph
    if 'readout' in options:
        if options['readout']:
            cal_graph.multiplexed_RO.state('needs calibration')

    if 'single_qubit_gates' in options:
        if options['single_qubit_gates']:
            sqg_nodes = ['QL_freq_ramsey', 'QL_motzoi',
                         'QL_amplitude_fine', 'QL_RB',
                         'QR_freq_ramsey', 'QR_motzoi',
                         'QR_amplitude_fine', 'QR_RB', 'TD_char']
            for node in sqg_nodes:
                cal_graph.nodes[node].state('needs calibration')

    if 'time_domain_char' in options:
        if options['time_domain_char']:
            tdc_nodes = ['QL_T1', 'QL_T2s', 'QL_echo',
                         'QR_T1', 'QR_T2s', 'QR_echo', 'TD_char']
            for node in tdc_nodes:
                cal_graph.nodes[node].state('needs calibration')

    if 'cz_single_qubit_phase' in options:
        if options['cz_single_qubit_phase']:
            sqp_nodes = ['CZ_QR_phase', 'CZ_QL_phase', 'CZ']
            for node in sqp_nodes:
                cal_graph.nodes[node].state('needs calibration')

    if 'two_qubit_gate' in options:
        if options['two_qubit_gate']:
            cz_nodes = ['CZ_conditional_phase',
                        'CZ_QR_phase', 'CZ_QL_phase', 'CZ']
            for node in cz_nodes:
                cal_graph.nodes[node].state('needs calibration')
    cal_graph.demonstrator_cal(verbose=True)

    # Send over the results of the calibrations
    send_calibration_data()


def _retrieve_file_from_url(file_url: str):
    """
    Self explanatory: we retrieve the file from the url given
    """

    file_name = file_url.split("/")[-1]
    base_path = os.path.join(
        pq.__path__[0], 'measurement', 'demonstrator_helper',
        'qasm_files', file_name)
    file_path = base_path
    # download file from server
    urllib.request.urlretrieve(file_url, file_path)
    return file_path


def _get_qasm_sweep_points(file_path):
    """
    I am unsure what this does. Will need to grep RO_acq_averages.
    Am guessing this is related to qubit_object, CCL_transmon.py.
    """
    counter = 0
    with open(file_path) as f:
        line = f.readline()
        while(line):
            if re.match(r'(^|\s+)(measure|RO)(\s+|$)', line):
                counter += 1
            line = f.readline()

    return range(counter)


def _MC_result_to_chart_dict(result):
    for i in result:
        if(isinstance(result[i], np.ndarray)):
            result[i] = result[i].tolist()
    return [{
        "data-type": "chart",
        "data": result
    }]

def _simulate_quantumsim(file_path, options):
    """
    We can remove this function as I am using this to dummy execute
    """
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
    return dat


# Send the callibration of the machine every 10 minutes
# This function is blocking!
def send_calibration_data():

    banned_pars = ['IDN', 'RO_optimal_weights_I', 'RO_optimal_weights_Q',
                   'qasm_config']
    # threading.Timer(10, _send_calibration).start()
    # snapshot = MC.station.snapshot()
    snapshot = qc.station.snapshot()
    calibration = {
        "q0": snapshot["instruments"]["QL"],
        "q1": snapshot["instruments"]["QR"],
        'fridge': snapshot["instruments"]["Maserati_fridge_mon"]
    }
    for par in banned_pars:
        try:
            del calibration['q0']['parameters'][par]
            del calibration['q1']['parameters'][par]
        except KeyError as e:
            logging.warning(e)
    tc.client.publish_custom_msg({
        "calibration": calibration
    })
    print('Calibration data send')
