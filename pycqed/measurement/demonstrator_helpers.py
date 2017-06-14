import numpy as np
import time
from pycqed.measurement import measurement_control
from pycqed.measurement.sweep_functions import None_Sweep
from pycqed.measurement.detector_functions import Detector_Function

defualt_options = {
    "shots": 1000,
    "iterations": 1
}


class AllXYDetector(Detector_Function):

    def __init__(self, noise=0, delay=0, **kw):
        super(AllXYDetector, self).__init__()
        self.name = "AllXYDetector"
        self.noise = noise
        self.delay = delay
        self.detector_control = 'hard'

    def get_values(self):
        start = (np.random.random(5) - 0.5) * self.noise
        middel = (np.random.random(12) - 0.5) * self.noise + 0.5
        end = (np.random.random(4) - 0.5) * self.noise + 1
        time.sleep(self.delay)
        return np.concatenate((start, middel, end))


def execute_AllXY(options={}):
    print(options)
    MC = measurement_control.MeasurementControl(
        'MC', live_plot_enabled=False, verbose=True)
    defualt_options.update(options)
    options = defualt_options
    print("PRINT")
    print(options)
    MC.set_detector_function(AllXYDetector(noise=0.1, delay=5))
    return_value = []
    for i in range(options["iterations"]):
        MC.set_sweep_function(None_Sweep(sweep_control="hard"))
        MC.set_sweep_points(np.arange(21))
        dat = MC.run('AllXY')
        return_value.append(MC_result_to_chart_dict(dat))
    MC.close()
    return return_value


def MC_result_to_chart_dict(result):
    for i in result:
        if(isinstance(result[i], np.ndarray)):
            result[i] = result[i].tolist()
    return {
        "data-type": "chart",
        "data": result
    }

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
