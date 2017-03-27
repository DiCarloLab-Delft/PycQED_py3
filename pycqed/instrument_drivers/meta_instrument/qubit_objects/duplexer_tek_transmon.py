from .Tektronix_driven_transmon import Tektronix_driven_transmon
from math import gcd
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter


class Duplexer_tek_transmon(Tektronix_driven_transmon):

    def __init__(self, name, **kw):
        """
        Basic Duplexer controlled transmon.
        This splits up the I and Q qaudratures over two channels on the
        duplexer. It does not support gated pulses yet but this should be
        implemntable
        """
        super().__init__(name, **kw)

        self.add_parameter('VSM', parameter_class=InstrumentParameter)
        # Delete any unused parameters from the parent class
        del self.parameters['phi_skew']
        del self.parameters['alpha']
        del self.parameters['pulse_I_channel']
        del self.parameters['pulse_Q_channel']
        del self.parameters['pulse_I_offset']
        del self.parameters['pulse_Q_offset']

        self.add_parameter('G_phi_skew', label='IQ phase skewness',
                           unit='deg', vals=vals.Numbers(-180, 180),
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('G_alpha', label='QI amplitude skewness',
                           unit='', vals=vals.Numbers(.1, 2),
                           initial_value=1,
                           parameter_class=ManualParameter)
        self.add_parameter('D_phi_skew', label='IQ phase skewness',
                           unit='deg', vals=vals.Numbers(-180, 180),
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('D_alpha', label='QI amplitude skewness',
                           unit='', vals=vals.Numbers(.1, 2),
                           initial_value=1,
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_GI_channel', initial_value='ch1',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_GQ_channel', initial_value='ch2',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_DI_channel', initial_value='ch3',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_DQ_channel', initial_value='ch4',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)

        self.add_parameter('pulse_GI_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_GQ_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_DI_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_DQ_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)

        self.add_parameter('Mux_G_ch', initial_value='in1_out1',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('Mux_D_ch', initial_value='in2_out1',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('Mux_G_att', initial_value=0,
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter)
        self.add_parameter('Mux_D_att', initial_value=0,
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter)
        self.add_parameter('Mux_G_phase', initial_value=30000,
                           unit='dac',
                           vals=vals.Ints(0, 65536),
                           parameter_class=ManualParameter)
        self.add_parameter('Mux_D_phase', initial_value=30000,
                           unit='dac',
                           vals=vals.Ints(0, 65536),
                           parameter_class=ManualParameter)

    def prepare_for_timedomain(self):
        self.LO.on()
        self.cw_source.off()
        self.td_source.on()

        # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
        self.td_source.frequency.set(self.f_qubit.get()
                                     - self.f_pulse_mod.get())
        # setting Mux parameters
        self.VSM.get_instr().set(self.Mux_G_ch()+'_switch', 'ON')
        self.VSM.get_instr().set(self.Mux_D_ch()+'_switch', 'ON')
        self.VSM.get_instr().set(
            self.Mux_G_ch()+'_attenuation', self.Mux_G_att())
        self.VSM.get_instr().set(
            self.Mux_D_ch()+'_attenuation', self.Mux_D_att())
        self.VSM.get_instr().set(self.Mux_G_ch()+'_phase', self.Mux_G_phase())
        self.VSM.get_instr().set(self.Mux_D_ch()+'_phase', self.Mux_D_phase())

        # Use resonator freq unless explicitly specified
        if self.f_RO.get() is None:
            f_RO = self.f_res.get()
        else:
            f_RO = self.f_RO.get()
        self.LO.get_instr().frequency.set(f_RO - self.f_RO_mod.get())
        self.td_source.get_instr().power.set(self.td_source_pow.get())
        self.get_pulse_pars()

        self.AWG.get_instr().set(self.pulse_GI_channel.get()+'_offset',
                                 self.pulse_GI_offset.get())
        self.AWG.get_instr().set(self.pulse_GQ_channel.get()+'_offset',
                                 self.pulse_GQ_offset.get())

        self.AWG.get_instr().set(self.pulse_DI_channel.get()+'_offset',
                                 self.pulse_DI_offset.get())
        self.AWG.get_instr().set(self.pulse_DQ_channel.get()+'_offset',
                                 self.pulse_DQ_offset.get())

        if self.RO_pulse_type.get() is 'MW_IQmod_pulse':
            self.AWG.get_instr().set(self.RO_I_channel.get()+'_offset',
                                     self.RO_I_offset.get())
            self.AWG.get_instr().set(self.RO_Q_channel.get()+'_offset',
                                     self.RO_Q_offset.get())
        elif self.RO_pulse_type.get() is 'Gated_MW_RO_pulse':
            self.RF_RO_source.get_instr().on()
            self.RF_RO_source.get_instr().pulsemod_state.set('on')
            self.RF_RO_source.get_instr().frequency.set(self.f_RO.get())
            self.RF_RO_source.get_instr().power.set(self.RO_pulse_power.get())

    def get_pulse_pars(self):
        self.pulse_pars = {
            'GI_channel': self.pulse_GI_channel(),
            'GQ_channel': self.pulse_GQ_channel(),
            'DI_channel': self.pulse_DI_channel(),
            'DQ_channel': self.pulse_DQ_channel(),
            'amplitude': self.amp180.get(),
            'amp90_scale': self.amp90_scale(),
            'sigma': self.gauss_sigma.get(),
            'nr_sigma': 4,
            'motzoi': self.motzoi.get(),
            'mod_frequency': self.f_pulse_mod.get(),
            'pulse_delay': self.pulse_delay.get(),
            'G_phi_skew': self.G_phi_skew(),
            'D_phi_skew': self.D_phi_skew.get(),
            'G_alpha': self.G_alpha(),
            'D_alpha': self.D_alpha.get(),
            'phase': 0,
            'pulse_type': 'Mux_DRAG_pulse'}

        self.RO_pars = {
            'I_channel': self.RO_I_channel.get(),
            'Q_channel': self.RO_Q_channel.get(),
            'RO_pulse_marker_channel': self.RO_pulse_marker_channel.get(),
            'amplitude': self.RO_amp.get(),
            'length': self.RO_pulse_length.get(),
            'pulse_delay': self.RO_pulse_delay.get(),
            'mod_frequency': self.f_RO_mod.get(),
            'acq_marker_delay': self.RO_acq_marker_delay.get(),
            'acq_marker_channel': self.RO_acq_marker_channel.get(),
            'phase': 0,
            'pulse_type': self.RO_pulse_type.get()}
        return self.pulse_pars, self.RO_pars
