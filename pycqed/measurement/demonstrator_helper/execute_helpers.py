import numpy as np
import os
import re
import urllib.request
import pycqed_scripts as pqs  # N.B. should not be defined in pycqed
# import time
import pycqed as pq
import qcodes as qc
from qcodes.instrument.base import Instrument
from pycqed.measurement import detector_functions as det
from pycqed.measurement import sweep_functions as swf
from tess.TessConnect import TessConnection
import logging

import threading

defualt_execute_options = {}

tc = TessConnection()
tc.connect("execute")
defualt_simulate_options = {
    "num_avg": 10000,
    "iterations": 1
}

# Extract some "hardcoded" instruments from the global namespace"


def execute_qasm_file(file_url: str,  # config_json: str=None,
                      verbosity_level: int=0):

    MC = Instrument.find_instrument('MC')
    CBox = Instrument.find_instrument('CBox')
    device = Instrument.find_instrument('Starmon')
    # N.B. hardcoded fixme
    cfg = device.qasm_config()
    qasm_fp = _retrieve_file_from_url(file_url)
    sweep_points = _get_qasm_sweep_points(qasm_fp)

    s = swf.QASM_Sweep_v2(qasm_fn=qasm_fp, config=cfg, CBox=CBox,
                          verbosity_level=verbosity_level)

    d = device.get_correlation_detector()
    MC.set_sweep_function(s)
    MC.set_sweep_points(sweep_points)
    MC.set_detector_function(d)
    data = MC.run('demonstrator')  # FIXME <- add the proper name

    return _MC_result_to_chart_dict(data)


def _retrieve_file_from_url(file_url):

    file_name = file_url.split("/")[-1]
    base_path = os.path.join(
        pq.__path__[0], 'measurement', 'demonstrator_helper',
        'qasm_files', file_name)
    file_path = base_path
    # download file from server
    urllib.request.urlretrieve(file_url, file_path)
    return file_path


def _get_qasm_sweep_points(file_path):
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

# Send the callibration of the machine every 10 minutes


def _send_calibration():

    banned_pars = ['IDN', 'RO_optimal_weights_I', 'RO_optimal_weights_Q',
                   'qasm_config']
    # threading.Timer(10, _send_calibration).start()
    # snapshot = MC.station.snapshot()
    snapshot = qc.station.snapshot()
    calibration = {
        "q0": snapshot["instruments"]["QL"],
        "q1": snapshot["instruments"]["QR"]
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
    threading.Timer(10*60, _send_calibration).start()


_send_calibration()
