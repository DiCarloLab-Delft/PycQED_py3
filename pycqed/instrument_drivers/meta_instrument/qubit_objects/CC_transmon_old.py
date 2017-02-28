import logging
import numpy as np

from .qubit_object import Transmon
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


class CC_transmon(Transmon):
    '''
    Transmon that is controlled using the central controller.
    '''
    def __init__(self, name,
                 LO, cw_source, td_source,
                 IVVI, AWG, CBox,
                 heterodyne_instr, MC, rf_RO_source=None, **kw):
        super().__init__(name, **kw)
        # Change this when inheriting directly from Transmon instead of
        self.LO = LO
        self.cw_source = cw_source
        self.td_source = td_source
        self.rf_RO_source = rf_RO_source
        self.IVVI = IVVI
        self.heterodyne_instr = heterodyne_instr
        self.AWG = AWG
        self.MC = MC
        self.add_parameters()
        self.set_acquisition_detectors()

    def add_parameters(self):
        self.add_parameter('mod_amp_td', label='RO modulation ampl td',
                           unit='V', initial_value=0.5,
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter)

        self.add_parameter('f_RO_mod',
                           label='Readout-modulation frequency', unit='Hz',
                           initial_value=-2e7,
                           parameter_class=ManualParameter)
        # Time-domain parameters
        self.add_parameter('f_pulse_mod',
                           initial_value=-50e6,
                           label='pulse-modulation frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('awg_nr', label='CBox awg nr', unit='#',
                           parameter_class=ManualParameter)

        self.add_parameter('amp180',
                           label='Pi-pulse amplitude', unit='mV',
                           initial_value=300,
                           parameter_class=ManualParameter)
        # Amp 90 is hardcoded to be half amp180
        self.add_parameter('amp90',
                           label='Pi/2-pulse amplitude', unit='mV',
                           get_cmd=self._get_amp90)
        self.add_parameter('gauss_width', unit='s',
                           initial_value=40e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('motzoi', label='Motzoi parameter', unit='',
                           initial_value=0,
                           parameter_class=ManualParameter)

        # Single shot readout specific parameters
        self.add_parameter('RO_threshold', unit='dac-value',
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('signal_line', parameter_class=ManualParameter,
                           vals=vals.Enum(0, 1), initial_value=0)

        # Mixer skewness correction
        self.add_parameter('phi', unit='deg',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('alpha', unit='',
                           parameter_class=ManualParameter, initial_value=1)
        # Mixer offsets correction, qubit drive
        self.add_parameter('mixer_offs_drive_I',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_offs_drive_Q',
                           parameter_class=ManualParameter, initial_value=0)

        self.add_parameter('mixer_offs_RO_I',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_offs_RO_Q',
                           parameter_class=ManualParameter, initial_value=0)


        self.add_parameter('mod_amp_cw', label='RO modulation ampl cw',
                           unit='V', initial_value=0.5,
                           parameter_class=ManualParameter)

        self.add_parameter('RO_power_cw', label='RO power cw',
                           unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pow', label='spectroscopy power',
                           unit='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pow_pulsed',
                           label='pulsed spectroscopy power',
                           unit='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('td_source_pow',
                           label='Time-domain power',
                           unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pulse_type', label='Pulsed spec pulse type',
                           parameter_class=ManualParameter,
                           initial_value='SquarePulse',
                           vals=vals.Enum('SquarePulse'))  # , SSB_DRAG_pulse))
        # we should also implement SSB_DRAG_pulse for pulsed spec
        self.add_parameter('spec_pulse_length',
                           label='Pulsed spec pulse duration',
                           unit='s',
                           vals=vals.Numbers(1e-9, 20e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pulse_marker_channel',
                           unit='s',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pulse_depletion_time',
                           unit='s',
                           vals=vals.Numbers(1e-9, 50e-6),
                           parameter_class=ManualParameter)

        # Rename f_RO_mod
        # Time-domain parameters
        self.add_parameter('pulse_I_channel', initial_value='ch1',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_Q_channel', initial_value='ch2',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)

        self.add_parameter('pulse_I_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_Q_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)

        # These parameters are only relevant if using MW_IQmod_pulse type
        # RO
        self.add_parameter('RO_I_channel', initial_value='ch3',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_Q_channel', initial_value='ch4',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_I_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_Q_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)

        self.add_parameter('RO_pulse_type', initial_value='MW_IQmod_pulse',
                           vals=vals.Enum('MW_IQmod_pulse',
                                          'Gated_MW_RO_pulse'),
                           parameter_class=ManualParameter)
        # Relevant when using a marker channel to gate a MW-RO tone.
        self.add_parameter('RO_pulse_marker_channel',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_pulse_power', unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('f_pulse_mod',
                           initial_value=-100e6,
                           label='pulse-modulation frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_RO_mod',
                           label='Readout-modulation frequency', unit='Hz',
                           initial_value=-2e7,
                           parameter_class=ManualParameter)

        # Single shot readout specific parameters
        self.add_parameter('RO_threshold', unit='dac-value',
                           initial_value=0,
                           parameter_class=ManualParameter)
        # CBox specific parameter
        self.add_parameter('signal_line', parameter_class=ManualParameter,
                           vals=vals.Enum(0, 1), initial_value=0)

    def get_pulse_pars(self):
        """
        returns the pulse parameter as a dict
        """
        pulse_pars = {
            'control_pulse {}'.format(self.name): {
                'prepare_function': 'QWG_pulse_prepare',
                'pulse_type': 'SSB_DRAG_pulse',
                'I_channel': self.pulse_I_channel.get(),
                'Q_channel': self.pulse_Q_channel.get(),
                'amplitude': self.amp180.get(),
                'amp90_scale': self.amp90_scale.get(),
                'sigma': self.gauss_sigma.get(),
                'nr_sigma': 4,
                'motzoi': self.motzoi.get(),
                'pulse_delay': self.pulse_delay.get(),
                'phi_skew': self.phi_skew.get(),
                'alpha': self.alpha.get(),
                },

            'RO {}'.format(self.name): {
                'prepare_function': 'QWG_pulse_prepare',
                'pulse_type': self.RO_pulse_type.get(),
                'I_channel': self.RO_I_channel.get(),
                'Q_channel': self.RO_Q_channel.get(),
                'RO_pulse_marker_channel': self.RO_pulse_marker_channel.get(),
                'amplitude': self.RO_amp.get(),
                'length': self.RO_pulse_length.get(),
                'pulse_delay': self.RO_pulse_delay.get(),
                'mod_frequency': self.f_RO_mod.get(),
                'acq_marker_delay': self.RO_acq_marker_delay.get(),
                'acq_marker_channel': self.RO_acq_marker_channel.get(),
                'phase': 0}}

        return pulse_pars

