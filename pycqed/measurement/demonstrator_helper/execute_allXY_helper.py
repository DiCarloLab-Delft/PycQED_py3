import numpy as np
import os
import re
import urllib.request

import pycqed as pq
from pycqed.measurement import measurement_control as mc
from pycqed.measurement.demonstrator_helper.AllXY_detector import AllXYDetector
from pycqed.measurement import sweep_functions as swf

from tess.TessConnect import TessConnection
from qcodes import station

tc = TessConnection()
tc.connect("execute")

defualt_simulate_options = {
    "shots": 1000,
    "iterations": 1
}
defualt_execute_options = {}


def execute_AllXY(options={}):
    st = station.Station()

    MC = mc.MeasurementControl(
        'MC', live_plot_enabled=True, verbose=True)
    MC.soft_avg(40)
    MC.station = st
    st.add_component(MC)

    defualt_simulate_options.update(options)
    options = defualt_simulate_options

    try:
        MC.set_detector_function(AllXYDetector(noise=0.1, delay=0.2))
        return_value = []
        for i in range(options["iterations"]):
            MC.set_sweep_function(swf.None_Sweep(sweep_control="hard"))
            MC.set_sweep_points(np.arange(21))
            dat = MC.run('AllXY')
            return_value.append(_MC_result_to_chart_dict(dat))
        MC.close()
        return return_value
    except:
        raise
        return []
    finally:
        MC.close()


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
    return {
        "data-type": "chart",
        "data": result
    }
