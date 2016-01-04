import qt
import numpy as np
from instrument import Instrument
import instruments
import sys
import logging
import types


class ATS_TD(Instrument):
    '''
    meta-instrument used to control the ATS in TD mode.
    TD measurements talkt to the ATS_TD instead of the ATS itself.
    '''
    def __init__(self, name, **kw):
        logging.info(__name__ + ':Initializing measurement control instrument')
        Instrument.__init__(self, name, tags=['Meta-Instrument'])
        self.add_parameter('NoSweeps', flags=Instrument.FLAG_GETSET,
                           type=float)
        self.add_parameter('NoSegments', flags=Instrument.FLAG_GETSET,
                           type=float)
        self.add_parameter('points_per_trace', flags=Instrument.FLAG_GETSET,
                           type=float)
        self.add_parameter('finite_IF', flags=Instrument.FLAG_GETSET,
                           type=bool)
        self.add_parameter('range', flags=Instrument.FLAG_GETSET,
                           type=float, units='Volt (Vp)')
        self.add_parameter('sample_rate', flags=Instrument.FLAG_GETSET,
                           type=int, units='MSPS')

        ATS_name = kw.pop('ATS_name', 'ATS')
        self.ATS = qt.instruments[ATS_name]
        self.set_sample_rate(1.e3)
        self.set_range(self.ATS.get_ch1_range())

    def set_TD_mode(self, TD_mode):
        self.ATS.set_sample_rate(self._MSPS*1e3)  # in KSPS
        self.ATS.set_ch1_range(self._range)
        self.ATS.set_ch2_range(self._range)
        self.ATS.set_eng1_trigger_level(150)  # hardcoded val, bad!
        self.ATS.set_ext_trigger_delay(0)
        if self.finite_IF is True:
            self.ATS.set_ch1_coupling('AC')
            self.ATS.set_ch2_coupling('AC')
        else:
            self.ATS.set_ch1_coupling('DC')
            self.ATS.set_ch2_coupling('DC')

        self.ATS.abort()
        return True

    def init(self):

        self.get_points_per_trace()
        self.ATS.get_trace_length()

    def get_TD_mode(self):
        ATS_mode = self.ATS.get_acquisition_mode()
        if ATS_mode == 'TD':
            return True
        else:
            return False

    #  Functions that talk to ATS directly
    def _do_set_NoSweeps(self, NoSweeps):
        self.ATS.set_number_of_buffers(NoSweeps)

    def _do_get_NoSweeps(self):
        return self.ATS.get_number_of_buffers()

    def _do_set_NoSegments(self, NoSegments):
        self.ATS.set_records_per_buffer(NoSegments)

    def _do_get_NoSegments(self):
        return self.ATS.get_records_per_buffer()

    def _do_set_points_per_trace(self, points_per_trace):
        self.ATS.set_points_per_trace(points_per_trace)

    def _do_get_points_per_trace(self):
        return self.ATS.get_points_per_trace()

    def _do_set_finite_IF(self, finite_IF):
        self.finite_IF = finite_IF

    def _do_get_finite_IF(self):
        return self.finite_IF

    #  Directly wrapped functions
    def get_t_base(self):
        return (np.arange(self.ATS.get_points_per_trace())
                / (self.ATS.get_sample_rate()*1e3))

    def start_acquisition(self):
        return self.ATS.start_acquisition()

    def average_data(self, channel):
        return self.ATS.average_data(channel)

    def get_data(self, silent=False):
        return self.ATS.get_data(silent=silent)

    def do_get_range(self):
        '''
        Get the range (in volts) of the ATS.
        range is specified as peak voltage.
        '''
        return self._range

    def do_set_range(self, value):
        '''
        Set the range (in volts) of the ATS.
        range is specified as peak voltage.
        Allowed values: 0.04, 0.1, 0.2, 0.4, 1, 2, 4
        '''
        self._range = value
        self.ATS.set_ch1_range(self._range)
        self.ATS.set_ch2_range(self._range)

    def do_set_sample_rate(self, MSPS):
        self._MSPS = MSPS
        self.ATS.set_sample_rate(self._MSPS*1e3)

    def do_get_sample_rate(self):
        return self._MSPS

    def get_range(self, channel):
        return eval('self.ATS.get_ch%s_range()' % channel)
