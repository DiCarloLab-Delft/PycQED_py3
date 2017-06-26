import numpy as np
import os
import re
import urllib.request
import sys
import time
import linecache

from qcodes import station

from pycqed.measurement import measurement_control
from pycqed.measurement.sweep_functions import None_Sweep
from pycqed.measurement.sweep_functions import QX_Hard_Sweep

from pycqed.measurement.demonstrator_helper.AllXY_detector import AllXYDetector
from pycqed.measurement.detector_functions import QX_Hard_Detector

from pycqed.instrument_drivers.virtual_instruments.pyqx.qx_client import qx_client

defualt_simulate_options = {
    "shots": 1000,
    "iterations": 1
}
defualt_execute_options = {}


def execute_AllXY(options={}):
    MC = measurement_control.MeasurementControl(
        'MC', live_plot_enabled=False, verbose=True)
    defualt_simulate_options.update(options)
    options = defualt_simulate_options
    MC.set_detector_function(AllXYDetector(noise=0.1, delay=5))
    return_value = []
    for i in range(options["iterations"]):
        MC.set_sweep_function(None_Sweep(sweep_control="hard"))
        MC.set_sweep_points(np.arange(21))
        dat = MC.run('AllXY')
        return_value.append(_MC_result_to_chart_dict(dat))
    MC.close()
    return return_value


def execute_qasm_file(file_url, options={}):
    # file_url="uploads/asset/file/65/f27d92be-8505-43dc-af7d-4c395c70aaf9.qasm"
    file_path = _get_file_from_url(file_url)


def simulate_qasm_file(file_url, options={}):
    # file_url="uploads/asset/file/65/f27d92be-8505-43dc-af7d-4c395c70aaf9.qasm"
    file_path = _get_file_from_url(file_url)

    # Connect to the qx simulator

    MC = measurement_control.MeasurementControl(
        'MC', live_plot_enabled=False, verbose=True)
    st = station.Station()
    MC.station = st
    st.add_component(MC)
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
        MC.close()
        qxc.disconnect()


def _get_file_from_url(file_url):
    # file_url="uploads/asset/file/65/f27d92be-8505-43dc-af7d-4c395c70aaf9.qasm"
    file_name = file_url.split("/")[-1]
    base_path = os.path.dirname(os.path.abspath(__file__))+"\\QASM_files\\"
    file_path = base_path + file_name
    print(file_url)
    # download file from server
    urllib.request.urlretrieve(file_url, file_path)
    return file_path


def _get_qasm_sweep_points(file_path):
    counter = 0
    with open(file_path) as f:
        line = f.readline()
        while(line):
            if re.match(r'(^|\s+)(measure)(\s+|$)', line):
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

# operation_dict # Exists in global namespace for you
# CBox = qc.station['CBox']

# def measure_allxy(self, MC=None, label='',
#                   analyze=True, close_fig=True, verbose=True):

#     self.prepare_for_timedomain()
#     if MC is None:
#         MC = self.MC.get_instr()

# This line code generates a QASM file for the AllXY experiment
# AllXY = sqqs.AllXY(self.name, double_points=True)
# This retuns
