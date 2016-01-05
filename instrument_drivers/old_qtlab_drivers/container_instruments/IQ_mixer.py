from instrument import Instrument
import logging
import numpy as np
import qt


class IQ_mixer(Instrument):
    '''
    Passive object designed to hold paramters for the mixer.
    '''

    def __init__(self, name, reset=False, **kw):

        logging.info(__name__ + ': Initializing instrument')
        Instrument.__init__(self, name, tags=['Container', 'Qubit-Object'])

        # Qubit parameters
        self.add_parameter('QI_amp_ratio', units='deg',
                           type=float, flags=Instrument.FLAG_GETSET)

        self.add_parameter('IQ_phase_skewness', units='deg',
                           type=float, flags=Instrument.FLAG_GETSET)

    def do_get_QI_amp_ratio(self):
        return self.QI_amp_ratio

    def do_set_QI_amp_ratio(self, QI_amp_ratio):
        self.QI_amp_ratio = QI_amp_ratio

    def do_get_IQ_phase_skewness(self):
        return self.IQ_phase_skewness

    def do_set_IQ_phase_skewness(self, IQ_phase_skewness):
        self.IQ_phase_skewness = IQ_phase_skewness


