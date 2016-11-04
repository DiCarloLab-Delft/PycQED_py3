import unittest
import numpy as np
from pycqed.measurement import measurement_control
from pycqed.measurement.sweep_functions import None_Sweep
from pycqed.measurement.detector_functions import Dummy_Detector_Soft

from qcodes import station
station = station.Station()


class Test_SingleQubitTek(unittest.TestCase):

    def setUp(self):
        # set up a pulsar with some mock settings for the element
        self.MC = measurement_control.MeasurementControl('MC')
        self.MC.station = station

    def test_soft_sweep_1D(self):
        sweep_pts = np.linspace(0, 10, 30)
        self.MC.set_sweep_function(None_Sweep())
        self.MC.set_sweep_points(sweep_pts)
        self.MC.set_detector_function(Dummy_Detector_Soft())
        dat = self.MC.run('dummy')
        x = dat[:, 0]
        xr = np.arange(len(x))/15
        y = np.array([np.sin(xr/np.pi), np.cos(xr/np.pi)])
        y0 = dat[:, 1]
        y1 = dat[:, 2]
        np.testing.assert_array_almost_equal(y0, y[0, :])
        np.testing.assert_array_almost_equal(y1, y[1, :])


class Detector_Function(object):

    '''
    Detector_Function class for MeasurementControl
    '''

    def __init__(self, **kw):
        self.set_kw()
        self.name = 'Experiment_name'
        self.value_names = ['val A', 'val B']
        self.value_units = ['arb. units', 'arb. units']

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s' % (key, kw[key]))

    def get_values(self):
        pass

    def prepare(self, **kw):
        pass

    def initialize_data_arrays(self, sweep_points):
        pass

    def finish(self, **kw):
        pass


