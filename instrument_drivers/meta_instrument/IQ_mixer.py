import logging

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals


class IQ_mixer(Instrument):
    '''
    Passive object designed to hold paramters for the mixer.
    '''
    def __init__(self, name):
        logging.info(__name__ + ': Initializing instrument')
        super().__init__(name)

        # Qubit parameters
        self.add_parameter('QI_amp_ratio',
                           set_cmd=self._set_QI_amp_ratio,
                           get_cmd=self._get_QI_amp_ratio,
                           vals=vals.Numbers(0, 2))

        self.add_parameter('IQ_phase_skewness', units='deg',
                           get_cmd=self._get_IQ_phase_skewness,
                           set_cmd=self._set_IQ_phase_skewness,
                           vals=vals.Numbers(0, 360))
        self.IQ_phase_skewness.set(0)
        self.QI_amp_ratio.set(1)

    def _get_QI_amp_ratio(self):
        return self._QI_amp_ratio

    def _set_QI_amp_ratio(self, QI_amp_ratio):
        self._QI_amp_ratio = QI_amp_ratio

    def _get_IQ_phase_skewness(self):
        return self._IQ_phase_skewness

    def _set_IQ_phase_skewness(self, IQ_phase_skewness):
        self._IQ_phase_skewness = IQ_phase_skewness
