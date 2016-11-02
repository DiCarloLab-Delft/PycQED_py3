# TODO make this file run :)

import numpy as np
import unittest

from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control.pulse import SquarePulse
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sqs


class Test_SingleQubitTek(unittest.TestCase):

    def setUp(self):
        # set up a pulsar with some mock settings for the element
        self.pulsar = Pulsar()
        for i in range(4):
            self.pulsar.define_channel(id='ch{}'.format(i+1),
                                          name='ch{}'.format(i+1),
                                          type='analog',
                                          # max safe IQ voltage
                                          high=.7, low=-.7,
                                          offset=0.0, delay=0, active=True)
            self.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                          name='ch{}_marker1'.format(i+1),
                                          type='marker',
                                          high=2.0, low=0, offset=0.,
                                          delay=0, active=True)
            self.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                          name='ch{}_marker2'.format(i+1),
                                          type='marker',
                                          high=2.0, low=0, offset=0.,
                                          delay=0, active=True)

        self.pulse_pars = {
            'I_channel': 'ch1',
            'Q_channel': 'ch2',
            'amplitude': .5,
            'amp90_scale': .5,
            'sigma': 10e-9,
            'nr_sigma': 4,
            'motzoi': .8,
            'mod_frequency': 100e-6,
            'pulse_delay': 5e-9,
            'phi_skew': 0,
            'alpha': 1,
            'phase': 0,
            'pulse_type': 'SSB_DRAG_pulse'}


        self.RO_pars = {
            'I_channel': 'ch3',
            'Q_channel': 'ch4',
            'RO_pulse_marker_channel': 'ch3_marker1',
            'amplitude': '.5',
            'length': 300e-9,
            'pulse_delay': 0,
            'mod_frequency': 50e6,
            'fixed_point_frequency': 50e6,
            'acq_marker_delay': 0,
            'acq_marker_channel': 'ch1_marker1',
            'phase': 0,
            'pulse_type': 'MW_IQmod_pulse_tek'}

        station = Bunch(pulsar=self.pulsar)
        sqs.station = station



    def test_ramsey(self):
        times = np.linspace(0, 100e-6, 21)
        seq, el_list = sqs.Ramsey_seq(times, self.pulse_pars, self.RO_pars,
                       artificial_detuning=None,
                       cal_points=True,
                       verbose=False,
                       upload=False,
                       return_seq=True)


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
