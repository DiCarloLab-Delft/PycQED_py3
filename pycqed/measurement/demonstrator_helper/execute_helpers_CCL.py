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

try:
    MC = Instrument.find_instrument('MC')
    st = MC.station
    new_station = False
except KeyError:
    st = qc.station.Station()
    new_station = True

"""
Create the station for which the Instruments can connect to.
A virtual representation of the physical setup. In our case, the CCL.
Since we're calling the station,
"""
try:
    MC_demo = measurement_control.MeasurementControl(
        'Demonstrator_MC', live_plot_enabled=True, verbose=True)

    datadir = os.path.abspath(os.path.join(
        os.path.dirname(pq.__file__), os.pardir, 'demonstrator_execute_data'))
    MC_demo.datadir(datadir)
    MC_demo.station = st

    st.add_component(MC_demo)
except KeyError:
    MC_demo = Instrument.find_instrument('Demonstrator_MC')


def execute(qisa_file_url: str, tqisa_file_url:str, qasm_file_url:str,  config_json: str,
            verbosity_level: int=0):
    options = json.loads(config_json)

    if (not new_station):
        write_to_log('options:')
        write_to_log(options)
        write_to_log(qisa_file_url)
        write_to_log(tqisa_file_url)
        write_to_log(qasm_file_url)

        CCL = Instrument.find_instrument('CCL')
        device = Instrument.find_instrument('device')

        num_avg = int(options.get('num_avg', 512))

        nr_soft_averages = int(np.round(num_avg/512))
        MC_demo.soft_avg(nr_soft_averages)

        device.ro_acq_averages(512)

        # Get the qisa file
        qisa_fp = _retrieve_file_from_url(qisa_file_url)

        # Two ways to generate the sweep_points. Either I get from the file_url
        # or I get the appended options file which has the kw "measurement_points"
        # sweep_points_fp = _retrieve_file_from_url(sweep_points_file_url)
        # sweep_points = json.loads(sweep_points_fp)
        # sweep_points = sweep_points["measurement_points"]

        # Ok, I am assured by stanvn that he will provide me a options with kw
        sweep_points = options["measurement_points"]

        s = swf.OpenQL_File_Sweep(filename=qisa_fp, CCL=CCL,
                                  parameter_name='Points', unit='a.u.',
                                  upload=True)

        d = device.get_correlation_detector()

        MC_demo.set_sweep_function(s)
        MC_demo.set_sweep_points(sweep_points)
        MC_demo.set_detector_function(d)
        data = MC_demo.run('CCL_execute')  # FIXME <- add the proper name

    else:
        qisa_fp = _retrieve_file_from_url(qisa_file_url)
        data = _simulate_quantumsim(qisa_fp, options)

    return _MC_result_to_chart_dict(data)


def calibrate(config_json: str):
    """
    Perform calibrations based on the options specified in the config_json.
    Calibrations are performed using the dependency graph.

    N.B. on this helper no calibration protocol is defined so it only
    updates the calibration data in the overview.
    """
    options = json.loads(config_json)

    # Get the kernel_type
    try:
        kernel_type = options['kernel_type']
    except:
        print('Could not find kernel_type in the json options file')
        kernel_type = 'execute_CCL'

    # Send over the results of the calibrations
    send_calibration_data(kernel_type)


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
    quantumsim_sweep.parameter_name = 'CCL number '
    quantumsim_sweep.unit = '#'

    qubit_parameters = {
        'Q0': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.0189, 'frac1_1': 0.918},
        'Q1': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.068, 'frac1_1': 0.949},
        'q0': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.0189, 'frac1_1': 0.918},
        'q1': {'T1': 30e3, 'T2': 17e3, 'frac1_0': 0.068, 'frac1_1': 0.949}}

    quantumsim_det = Quantumsim_Two_QB_Hard_Detector(
        file_path, dt=(40, 280), qubit_parameters=qubit_parameters)
    sweep_points = range(len(quantumsim_det.parser.circuits))

    MC_demo.set_detector_function(quantumsim_det)
    MC_demo.set_sweep_function(quantumsim_sweep)
    MC_demo.set_sweep_points(sweep_points)
    dat = MC_demo.run("run QASM")
    print('simulation execute_CCL finished')
    return dat


def send_calibration_data(kernel_type: str):
    """
    Sends a snapshot containing the latest calibration data
    """

    banned_pars = ['IDN', 'ro_acq_weight_func_I', 'ro_acq_weight_func_Q',
                   'qasm_config']
    snapshot = st.snapshot()
    calibration = {
        "q0": snapshot["instruments"]["QL"],
        "q1": snapshot["instruments"]["QR"]
        # 'fridge': snapshot["instruments"]["Maserati_fridge_mon"]
    }
    for par in banned_pars:
        try:
            del calibration['q0']['parameters'][par]
            del calibration['q1']['parameters'][par]
        except KeyError as e:
            logging.warning(e)
    tc.client.publish_custom_msg({
        "calibration": calibration,
        "kernel_type": kernel_type
    })

    # print({
    #     "calibration": calibration,
    #     "kernel_type": kernel_type})
    print('Calibration data send')


def write_to_log(string):
    with open(r'D:\Experiments\1709_M18\demo_log.txt', 'a+') as f:
        f.write(str(string) + '\n')
