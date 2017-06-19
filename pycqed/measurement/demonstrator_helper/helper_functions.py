import numpy as np
import os
import re
import urllib.request
import pycqed_scripts as pqs  # N.B. should not be defined in pycqed
import time
import pycqed as pq

from qcodes import station

from pycqed.measurement import measurement_control
from pycqed.measurement.sweep_functions import None_Sweep
from pycqed.measurement.sweep_functions import QX_Hard_Sweep

from pycqed.measurement.demonstrator_helper.AllXY_detector import AllXYDetector
from pycqed.measurement.detector_functions import QX_Hard_Detector
from pycqed.instrument_drivers.virtual_instruments.pyqx.qx_client import qx_client
from pycqed.measurement.waveform_control_CC import qasm_helpers as qh
from qcodes.instrument.base import Instrument
from pycqed.measurement import detector_functions as det

defualt_options = {
    "shots": 1000,
    "iterations": 1
}
server_host = "http://localhost:3000/"


def execute_qasm_file(file_url: str, config_json: str=None,
                      verbosity_level: int=0):

    # Extract some "hardcoded" instruments from the global namespace"
    MC = Instrument.find_instrument('MC')
    CBox = Instrument.find_instrument('CBox')
    device = Instrument.find_instrument('Starmon')

    # N.B. hardcoded fixme
    q0 = device.QR

    if config_json is None:
        config_dir = os.path.join(
            pqs.__path__[0], 'experiments', 'Demonstrator_1706')
        config_fn = os.path.join(config_dir, 'demo_config.json')
    else:
        raise NotImplementedError('No support for json input yet')

    qasm_fp = _retrieve_file_from_url(file_url)
    sweep_points = _get_qasm_sweep_points(qasm_fp)

    s = qh.QASM_Sweep_v2(qasm_fn=qasm_fp, config_fn=config_fn, CBox=CBox,
                         verbosity_level=verbosity_level)
    d = det.UHFQC_integrated_average_detector(
        device.acquisition_instrument.get_instr(),
        AWG=device.seq_contr.get_instr(),
        nr_averages=q0.RO_acq_averages(),
        integration_length=q0.RO_acq_integration_length(),
        result_logging_mode='lin_trans',
        channels=[q0.RO_acq_weight_function_I()])
    MC.set_sweep_function(s)
    MC.set_sweep_points(sweep_points)
    MC.set_detector_function(d)
    data = MC.run('demonstrator')  # FIXME <- add the proper name

    return _MC_result_to_chart_dict(data)


def execute_AllXY(options={}):
    MC = Instrument.find_instrument('MC')
    defualt_options.update(options)
    options = defualt_options
    MC.set_detector_function(AllXYDetector(noise=0.1, delay=5))
    return_value = []
    for i in range(options["iterations"]):
        MC.set_sweep_function(None_Sweep(sweep_control="hard"))
        MC.set_sweep_points(np.arange(21))
        dat = MC.run('AllXY')
        return_value.append(_MC_result_to_chart_dict(dat))
    MC.close()
    return return_value


def _retrieve_file_from_url(file_url):

    file_name = file_url.split("/")[-1]
    base_path = os.path.join(
        pq.__path__[0], 'measurement', 'demonstrator_helper',
         'qasm_files', file_name)
    file_path = base_path
    # download file from server
    urllib.request.urlretrieve(server_host+file_url, file_path)
    return file_path


def simulate_qasm_file(file_url, options={}):
    # file_url="uploads/asset/file/65/f27d92be-8505-43dc-af7d-4c395c70aaf9.qasm"
    file_path = _retrieve_file_from_url(file_url)

    # Connect to the qx simulator

    MC = Instrument.find_instrument('MC')

    qxc = qx_client()
    qxc.connect()
    time.sleep(1)
    qxc.create_qubits(5)
    try:

        qx_sweep = QX_Hard_Sweep(qxc, file_path)
        qx_detector = QX_Hard_Detector(qxc, [file_path], num_avg=10000)
        sweep_points = range(len(qx_detector.randomizations[0]))
        print("SWEEP")
        print(sweep_points)
        # qx_detector.prepare(sweep_points)
        # Start measurment
        MC.set_detector_function(qx_detector)
        MC.set_sweep_function(qx_sweep)
        MC.set_sweep_points(sweep_points)
        dat = MC.run("run QASM")
        return _MC_result_to_chart_dict(dat)
    except:
        raise

        return []
    finally:
        qxc.disconnect()


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
