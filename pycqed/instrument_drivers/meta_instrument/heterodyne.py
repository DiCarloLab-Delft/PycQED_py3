# This is a virtual instrument abstracting a homodyne
# source which controls RF and LO sources

import logging
import numpy as np
from time import time
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
# Used for uploading the right AWG sequences
from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
import time


class HeterodyneInstrument(Instrument):
    """
    This is a virtual instrument for a homodyne source

    Instrument is CBox, UHFQC, ATS and DDM compatible

    """
    shared_kwargs = ['RF', 'LO', 'AWG']

    def __init__(self,
                 name,
                 RF,
                 LO,
                 AWG=None,
                 acquisition_instr=None,
                 acquisition_instr_controller=None,
                 single_sideband_demod=False,
                 **kw):

        self.RF = RF

        self.common_init(name, LO, AWG, acquisition_instr,
                         single_sideband_demod, **kw)

        self.add_parameter(
            'RF_power',
            label='RF power',
            unit='dBm',
            vals=vals.Numbers(),
            set_cmd=self._set_RF_power,
            get_cmd=self._get_RF_power)
        self.add_parameter(
            'acquisition_instr_controller',
            set_cmd=self._set_acquisition_instr_controller,
            get_cmd=self._get_acquisition_instr_controller,
            vals=vals.Anything())
        self.acquisition_instr_controller(acquisition_instr_controller)
        self._RF_power = None

    def common_init(self,
                    name,
                    LO,
                    AWG,
                    acquisition_instr='CBox',
                    single_sideband_demod=False,
                    **kw):
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, **kw)

        self.add_parameter(
            'set_delay',
            vals=vals.Numbers(),
            label='Delay after setting the frequency',
            parameter_class=ManualParameter,
            initial_value=0.0)

        self.LO = LO
        self.AWG = AWG
        self.add_parameter(
            'frequency',
            label='Heterodyne frequency',
            unit='Hz',
            vals=vals.Numbers(9e3, 40e9),
            get_cmd=self._get_frequency,
            set_cmd=self._set_frequency)
        self.add_parameter(
            'f_RO_mod',
            label='Intermodulation frequency',
            unit='Hz',
            vals=vals.Numbers(-600e6, 600e6),
            set_cmd=self._set_f_RO_mod,
            get_cmd=self._get_f_RO_mod)
        self.add_parameter(
            'single_sideband_demod',
            vals=vals.Bool(),
            label='Single sideband demodulation',
            parameter_class=ManualParameter,
            initial_value=single_sideband_demod)
        self.add_parameter(
            'acquisition_instr',
            vals=vals.Strings(),
            label='Acquisition instrument',
            set_cmd=self._set_acquisition_instr,
            get_cmd=self._get_acquisition_instr)
        self.add_parameter(
            'nr_averages',
            label='Number of averages',
            vals=vals.Numbers(min_value=0, max_value=1e6),
            parameter_class=ManualParameter,
            initial_value=1024)
        self.add_parameter(
            'status',
            vals=vals.Enum('On', 'Off'),
            set_cmd=self._set_status,
            get_cmd=self._get_status)
        self.add_parameter(
            'trigger_separation',
            label='Trigger separation',
            unit='s',
            vals=vals.Numbers(0),
            set_cmd=self._set_trigger_separation,
            get_cmd=self._get_trigger_separation)
        self.add_parameter(
            'RO_length',
            label='Readout length',
            unit='s',
            vals=vals.Numbers(0),
            set_cmd=self._set_RO_length,
            get_cmd=self._get_RO_length)
        self.add_parameter(
            'auto_seq_loading',
            vals=vals.Bool(),
            label='Automatic AWG sequence loading',
            parameter_class=ManualParameter,
            initial_value=True)
        self.add_parameter(
            'acq_marker_channels',
            vals=vals.Strings(),
            label='Acquisition trigger channels',
            docstring='comma (,) separated string of marker channels',
            set_cmd=self._set_acq_marker_channels,
            get_cmd=self._get_acq_marker_channels)

        self._trigger_separation = 10e-6
        self._RO_length = 2274e-9
        self._awg_seq_filename = ''
        self._awg_seq_parameters_changed = True
        self._UHFQC_awg_parameters_changed = True
        self.acquisition_instr(acquisition_instr)
        self.status('Off')
        self._f_RO_mod = 10e6
        self._frequency = 5e9
        self.frequency(5e9)
        self.f_RO_mod(10e6)
        self._eps = 0.01  # Hz slack for comparing frequencies
        self._acq_marker_channels = (
            'ch4_marker1,ch4_marker2,' + 'ch3_marker1,ch3_marker2')

    def prepare(self, get_t_base=True):
        # Uploading the AWG sequence
        if self.AWG != None:
            if (self._awg_seq_filename not in self.AWG.setup_filename()
                    or self._awg_seq_parameters_changed
                ) and self.auto_seq_loading():
                self._awg_seq_filename = \
                    st_seqs.generate_and_upload_marker_sequence(
                        5e-9, self.trigger_separation(), RF_mod=False,
                        acq_marker_channels=self.acq_marker_channels())
                self._awg_seq_parameters_changed = False

        # Preparing the acquisition instruments
        if 'CBox' in self.acquisition_instr():
            self.prepare_CBox(get_t_base)
        elif 'UHFQC' in self.acquisition_instr():
            self.prepare_UHFQC()
        elif 'ATS' in self.acquisition_instr():
            self.prepare_ATS(get_t_base)
        elif 'DDM' in self.acquisition_instr():
            self.prepare_DDM()

        else:
            raise ValueError("Invalid acquisition instrument {} in {}".format(
                self.acquisition_instr(), self.__class__.__name__))

        # turn on the AWG and the MWGs
        if self.AWG != None:
            self.AWG.run()
        self.on()

    def prepare_CBox(self, get_t_base=True):
        if get_t_base:
            trace_length = 512
            tbase = np.arange(0, 5 * trace_length, 5) * 1e-9
            self.cosI = np.floor(
                127. * np.cos(2 * np.pi * self.f_RO_mod() * tbase))
            self.sinI = np.floor(
                127. * np.sin(2 * np.pi * self.f_RO_mod() * tbase))
            self._acquisition_instr.sig0_integration_weights(self.cosI)
            self._acquisition_instr.sig1_integration_weights(self.sinI)
            # because using integrated avg
            self._acquisition_instr.set('nr_samples', 1)
            self._acquisition_instr.nr_averages(int(self.nr_averages()))

    def prepare_UHFQC(self):
        # Upload the correct integration weigths
        if self.single_sideband_demod():
            self._acquisition_instr.prepare_SSB_weight_and_rotation(
                IF=self.f_RO_mod(), weight_function_I=0, weight_function_Q=1)
        else:
            self._acquisition_instr.prepare_DSB_weight_and_rotation(
                IF=self.f_RO_mod(), weight_function_I=0, weight_function_Q=1)

        if self._UHFQC_awg_parameters_changed and self.auto_seq_loading():
            self._acquisition_instr.awg_sequence_acquisition()
            self._UHFQC_awg_parameters_changed = False

        # this sets the result to integration and rotation outcome
        self._acquisition_instr.qas_0_result_source(2)
        self._acquisition_instr.qas_0_integration_mode(0)
        # only one sample to average over
        self._acquisition_instr.qas_0_result_length(1)
        self._acquisition_instr.qas_0_result_averages(self.nr_averages())
        self._acquisition_instr.qas_0_integration_length(
            int(self.RO_length() * 1.8e9))
        # Configure the result logger to not do any averaging
        # The AWG program uses userregs/0 to define the number o
        # iterations in the loop
        self._acquisition_instr.awgs_0_userregs_0(int(self.nr_averages()))
        self._acquisition_instr.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg
        self._acquisition_instr.acquisition_initialize([0, 1], 'rl')
        self.scale_factor_UHFQC = 1 / (
            1.8e9 * self.RO_length() * int(self.nr_averages()))

    def prepare_ATS(self, get_t_base=True):
        if self.AWG != None:
            if (self._awg_seq_filename not in self.AWG.setup_filename() or
                    self._awg_seq_parameters_changed) and \
                    self.auto_seq_loading():
                self._awg_seq_filename = \
                    st_seqs.generate_and_upload_marker_sequence(
                        self.RO_length(), self.trigger_separation(),
                        RF_mod=False,
                        acq_marker_channels=self.acq_marker_channels())
                self._awg_seq_parameters_changed = False

        if get_t_base:
            self._acquisition_instr_controller.demodulation_frequency = \
                self.f_RO_mod()
            buffers_per_acquisition = 8
            self._acquisition_instr_controller.update_acquisitionkwargs(
                # mode='NPT',
                samples_per_record=64 * 1000,  # 4992,
                records_per_buffer=int(self.nr_averages() /
                                       buffers_per_acquisition),  # 70 segments
                buffers_per_acquisition=buffers_per_acquisition,
                channel_selection='AB',
                transfer_offset=0,
                external_startcapture='ENABLED',
                enable_record_headers='DISABLED',
                alloc_buffers='DISABLED',
                fifo_only_streaming='DISABLED',
                interleave_samples='DISABLED',
                get_processed_data='DISABLED',
                allocated_buffers=buffers_per_acquisition,
                buffer_timeout=1000)

    def probe(self):
        if 'CBox' in self.acquisition_instr():
            return self.probe_CBox()
        elif 'UHFQC' in self.acquisition_instr():
            return self.probe_UHFQC()
        elif 'ATS' in self.acquisition_instr():
            return self.probe_ATS()
        elif 'DDM' in self.acquisition_instr():
            return self.probe_DDM()
        else:
            raise ValueError("Invalid acquisition instrument {} in {}".format(
                self.acquisition_instr(), self.__class__.__name__))

    def probe_CBox(self):
        if self.single_sideband_demod():
            demodulation_mode = 'single'
        else:
            demodulation_mode = 'double'
        self._acquisition_instr.acquisition_mode('idle')
        self._acquisition_instr.acquisition_mode('integration averaging')
        self._acquisition_instr.demodulation_mode(demodulation_mode)
        # d = self.CBox.get_integrated_avg_results()
        # quick fix for spec units. Need to properrly implement it later
        # after this, output is in V
        scale_factor_dacV = 1. * 0.75 / 128.
        # scale_factor_integration = 1./float(self.f_RO_mod() *
        #     self.CBox.nr_samples()*5e-9)
        scale_factor_integration = \
            1 / (64.*self._acquisition_instr.integration_length())
        factor = scale_factor_dacV * scale_factor_integration
        d = np.double(self._acquisition_instr.get_integrated_avg_results()) \
            * np.double(factor)
        # print(np.size(d))
        return d[0][0] + 1j * d[1][0]

    def probe_UHFQC(self):
        if (self._awg_seq_parameters_changed
                or self._UHFQC_awg_parameters_changed
            ) and self.auto_seq_loading():
            self.prepare()

        dataset = self._acquisition_instr.acquisition_poll(
            samples=1, acquisition_time=0.001)
        dat = (self.scale_factor_UHFQC * dataset[0][0] +
               self.scale_factor_UHFQC * 1j * dataset[1][0])
        return dat

    def probe_ATS(self):
        dat = self._acquisition_instr_controller.acquisition()

        return dat

    def finish(self):
        self.off()
        if self.AWG is not None:
            self.AWG.stop()
        if 'UHFQC' in self.acquisition_instr():
            self._acquisition_instr.acquisition_finalize()

    def _set_frequency(self, val):
        self._frequency = val
        # this is the definition agreed upon in issue 131
        self.RF.frequency(val)
        self.LO.frequency(val - self._f_RO_mod)
        time.sleep(self.set_delay())

    def _get_frequency(self):
        freq = self.RF.frequency()
        LO_freq = self.LO.frequency()
        if abs(LO_freq - freq + self._f_RO_mod) > self._eps:
            logging.warning('f_RO_mod between RF and LO is not set correctly')
            logging.warning(
                '\tf_RO_mod = {}, LO_freq = {}, RF_freq = {}'.format(
                    self._f_RO_mod, LO_freq, freq))
        if abs(self._frequency - freq) > self._eps:
            logging.warning('Heterodyne frequency does not match RF frequency')
        return self._frequency

    def _set_f_RO_mod(self, val):
        if val != self._f_RO_mod and 'UHFQC' in self.acquisition_instr():
            self._UHFQC_awg_parameters_changed = True
        self._f_RO_mod = val
        self.LO.frequency(self._frequency - val)

    def _get_f_RO_mod(self):
        freq = self.RF.frequency()
        LO_freq = self.LO.frequency()
        if abs(LO_freq - freq + self._f_RO_mod) > self._eps:
            logging.warning('f_RO_mod between RF and LO is not set correctly')
            logging.warning(
                '\tf_RO_mod = {}, LO_freq = {}, RF_freq = {}'.format(
                    self._f_RO_mod, LO_freq, freq))
        return self._f_RO_mod

    def _set_RF_power(self, val):
        self.RF.power(val)
        self._RF_power = val
        # internally stored to allow setting RF from stored setting

    def _get_RF_power(self):
        return self._RF_power

    def _set_status(self, val):
        if val == 'On':
            self.on()
        else:
            self.off()

    def _get_status(self):
        if (self.LO.status().startswith('On')
                and self.RF.status().startswith('On')):
            return 'On'
        elif (self.LO.status().startswith('Off')
              and self.RF.status().startswith('Off')):
            return 'Off'
        else:
            return 'LO: {}, RF: {}'.format(self.LO.status(), self.RF.status())

    def on(self):
        if self.LO.status().startswith('Off') or \
           self.RF.status().startswith('Off'):
            wait = True
        else:
            wait = False
        self.LO.on()
        self.RF.on()
        if wait:
            # The R&S MWG-s take some time to stabilize their outputs
            time.sleep(1.0)

    def off(self):
        self.LO.off()
        self.RF.off()

    def _get_acquisition_instr(self):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily
        if self._acquisition_instr is None:
            return None
        else:
            return self._acquisition_instr.name

    def _set_acquisition_instr(self, acquisition_instr):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily
        if acquisition_instr is None:
            self._acquisition_instr = None
        else:
            self._acquisition_instr = self.find_instrument(acquisition_instr)
        self._awg_seq_parameters_changed = True
        self._UHFQC_awg_parameters_changed = True

    def _get_acquisition_instr_controller(self):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily
        if self._acquisition_instr_controller == None:
            return None
        else:
            return self._acquisition_instr_controller.name

    def _set_acquisition_instr_controller(self, acquisition_instr_controller):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily
        if acquisition_instr_controller == None:
            self._acquisition_instr_controller = None
        else:
            self._acquisition_instr_controller = \
                self.find_instrument(acquisition_instr_controller)
            print("controller initialized")

    def _set_trigger_separation(self, val):
        if val != self._trigger_separation:
            self._awg_seq_parameters_changed = True
        self._trigger_separation = val

    def _get_trigger_separation(self):
        return self._trigger_separation

    def _set_RO_length(self, val):
        if val != self._RO_length and 'UHFQC' not in self.acquisition_instr():
            self._awg_seq_parameters_changed = True
        self._RO_length = val

    def _get_RO_length(self):
        return self._RO_length

    def _set_acq_marker_channels(self, channels):
        if channels != self._acq_marker_channels:
            self._awg_seq_parameters_changed = True
        self._acq_marker_channels = channels

    def _get_acq_marker_channels(self):
        return self._acq_marker_channels

    def finish(self):
        if 'UHFQC' in self.acquisition_instr():
            self._acquisition_instr.acquisition_finalize()

    def get_demod_array(self):
        return self.cosI, self.sinI

    def demodulate_data(self, dat):
        """
        Returns a complex point in the IQ plane by integrating and demodulating
        the data. Demodulation is done based on the 'f_RO_mod' and
        'single_sideband_demod' parameters of the Homodyne instrument.
        """
        if self._f_RO_mod != 0:
            # self.cosI is based on the time_base and created in self.init()
            if self._single_sideband_demod is True:
                # this definition for demodulation is consistent with
                # issue #131
                I = np.average(self.cosI * dat[0] + self.sinI * dat[1])
                Q = np.average(-self.sinI * dat[0] + self.cosI * dat[1])
            else:  # Single channel demodulation, defaults to using channel 1
                I = 2 * np.average(dat[0] * self.cosI)
                Q = 2 * np.average(dat[0] * self.sinI)
        else:
            I = np.average(dat[0])
            Q = np.average(dat[1])
        return I + 1.j * Q


class LO_modulated_Heterodyne(HeterodyneInstrument):
    """
    Homodyne instrument for pulse modulated LO.
    Inherits functionality from the HeterodyneInstrument

    AWG is used for modulating signal and triggering the CBox for acquisition
    or AWG is used for triggering and UHFQC for modulation and acquisition
    """
    shared_kwargs = ['RF', 'LO', 'AWG']

    def __init__(self,
                 name,
                 LO,
                 AWG,
                 acquisition_instr='CBox',
                 single_sideband_demod=False,
                 **kw):

        self.common_init(name, LO, AWG, acquisition_instr,
                         single_sideband_demod, **kw)

        self.add_parameter(
            'mod_amp',
            label='Modulation amplitude',
            unit='V',
            vals=vals.Numbers(0, 1),
            set_cmd=self._set_mod_amp,
            get_cmd=self._get_mod_amp)
        self.add_parameter(
            'I_channel',
            vals=vals.Strings(),
            label='I channel',
            set_cmd=self._set_I_channel,
            get_cmd=self._get_I_channel)
        self.add_parameter(
            'Q_channel',
            vals=vals.Strings(),
            label='Q channel',
            set_cmd=self._set_Q_channel,
            get_cmd=self._get_Q_channel)
        self.add_parameter(
            'alpha',
            vals=vals.Numbers(),
            initial_value=1,
            parameter_class=ManualParameter)
        self.add_parameter(
            'phi_skew',
            vals=vals.Numbers(),
            initial_value=0,
            parameter_class=ManualParameter,
            unit='deg')

        self._f_RO_mod = 10e6
        self._frequency = 5e9
        self.f_RO_mod(10e6)
        self.frequency(5e9)
        self._mod_amp = 0
        self.mod_amp(.5)
        self._I_channel = 'ch3'
        self._Q_channel = 'ch4'

    def prepare(self, get_t_base=True):
        # Preparing the acquisition instruments
        if 'CBox' in self.acquisition_instr():
            self.prepare_CBox(get_t_base)
        elif 'UHFQC' in self.acquisition_instr():
            self.prepare_UHFQC()
        elif 'ATS' in self.acquisition_instr():
            self.prepare_ATS(get_t_base)
        elif 'DDM' in self.acquisition_instr():
            self.prepare_DDM()

        else:
            raise ValueError("Invalid acquisition instrument {} in {}".format(
                self.acquisition_instr(), self.__class__.__name__))

        # turn on the AWG and the MWGs
        self.AWG.run()
        self.on()

    def prepare_DDM(self):
        for i, channel in enumerate([1, 2]):
            eval(
                "self._acquisition_instr.ch_pair1_weight{}_wint_intlength({})".
                format(channel, self.RO_length * 500e6))
        self._acquisition_instr.ch_pair1_tvmode_naverages(self.nr_averages())
        self._acquisition_instr.ch_pair1_tvmode_nsegments(1)
        self.scale_factor = 1 / (500e6 * self.RO_length) / 127

    def prepare_CBox(self, get_t_base=True):
        """
        uses the AWG to generate the modulating signal and CBox for readout
        """
        # only uploads a seq to AWG if something changed
        if (self._awg_seq_filename not in self.AWG.setup_filename() or
                self._awg_seq_parameters_changed) and self.auto_seq_loading():
            self._awg_seq_filename = \
                st_seqs.generate_and_upload_marker_sequence(
                    self.RO_length(), self.trigger_separation(), RF_mod=True,
                    IF=self.f_RO_mod(), mod_amp=0.5,
                    acq_marker_channels=self.acq_marker_channels(),
                    I_channel=self.I_channel(), Q_channel=self.Q_channel())
            self.AWG.ch3_amp(self.mod_amp())
            self.AWG.ch4_amp(self.mod_amp())
            self._awg_seq_parameters_changed = False

        self.AWG.ch3_amp(self.mod_amp())
        self.AWG.ch4_amp(self.mod_amp())

        if get_t_base is True:
            trace_length = self.CBox.nr_samples()
            tbase = np.arange(0, 5 * trace_length, 5) * 1e-9
            self.cosI = np.cos(2 * np.pi * self.f_RO_mod() * tbase)
            self.sinI = np.sin(2 * np.pi * self.f_RO_mod() * tbase)

        self.CBox.nr_samples(1)  # because using integrated avg

    def prepare_UHFQC(self):
        """
        uses the UHFQC to generate the modulating signal and readout
        """
        # only uploads a seq to AWG if something changed
        if (self._awg_seq_filename not in self.AWG.setup_filename() or
                self._awg_seq_parameters_changed) and self.auto_seq_loading():
            self._awg_seq_filename = \
                st_seqs.generate_and_upload_marker_sequence(
                    5e-9, self.trigger_separation(), RF_mod=False,
                    acq_marker_channels=self.acq_marker_channels())
            self._awg_seq_parameters_changed = False

        # Upload the correct integration weigths
        if self.single_sideband_demod():
            self._acquisition_instr.prepare_SSB_weight_and_rotation(
                IF=self.f_RO_mod(), weight_function_I=0, weight_function_Q=1)
        else:
            self._acquisition_instr.prepare_DSB_weight_and_rotation(
                IF=self.f_RO_mod(), weight_function_I=0, weight_function_Q=1)

        if self._UHFQC_awg_parameters_changed and self.auto_seq_loading():
            self._acquisition_instr.awg_sequence_acquisition_and_pulse_SSB(
                self.f_RO_mod.get(),
                self.mod_amp(),
                RO_pulse_length=self.RO_length(),
                alpha=self.alpha(),
                phi_skew=self.phi_skew())
            self._UHFQC_awg_parameters_changed = False

        # this sets the result to integration and rotation outcome
        self._acquisition_instr.qas_0_result_source(2)
        self._acquisition_instr.qas_0_result_length(1)
        self._acquisition_instr.qas_0_result_averages(self.nr_averages())
        self._acquisition_instr.qas_0_integration_length(
            int(self.RO_length() * 1.8e9))
        # Configure the result logger to not do any averaging
        # The AWG program uses userregs/0 to define the number o
        # iterations in the loop
        self._acquisition_instr.awgs_0_userregs_0(int(self.nr_averages()))
        self._acquisition_instr.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg
        self._acquisition_instr.acquisition_initialize([0, 1], 'rl')
        self.scale_factor_UHFQC = 1 / (
            1.8e9 * self.RO_length() * int(self.nr_averages()))

    def probe_CBox(self):
        if self._awg_seq_parameters_changed:
            self.prepare()
        self.CBox.acquisition_mode(0)
        self.CBox.acquisition_mode(4)
        d = self.CBox.get_integrated_avg_results()
        return d[0][0] + 1j * d[1][0]

    def probe_DDM(self):
        # t0 = time.time()
        self._acquisition_instr.ch_pair1_tvmode_enable.set(1)
        self._acquisition_instr.ch_pair1_run.set(1)
        dataI = eval(
            "self._acquisition_instr.ch_pair1_weight{}_tvmode_data()".format(
                1))
        dataQ = eval(
            "self._acquisition_instr.ch_pair1_weight{}_tvmode_data()".format(
                2))
        dat = (self.scale_factor * dataI + self.scale_factor * 1j * dataQ)
        # t1 = time.time()
        # print("time for DDM polling", t1-t0)
        return dat

    def _set_frequency(self, val):
        self._frequency = val
        # this is the definition agreed upon in issue 131
        # AWG modulation ensures that signal ends up at RF-frequency
        self.LO.frequency(val - self._f_RO_mod)

    def _get_frequency(self):
        freq = self.LO.frequency() + self._f_RO_mod
        if abs(self._frequency - freq) > self._eps:
            logging.warning('Homodyne frequency does not match LO frequency'
                            ' + RO_mod frequency')
        return self._frequency

    def _set_f_RO_mod(self, val):
        if val != self._f_RO_mod:
            self._f_RO_mod = val
            if 'CBox' in self.acquisition_instr():
                self._awg_seq_parameters_changed = True
            elif 'UHFQC' in self.acquisition_instr():
                self._UHFQC_awg_parameters_changed = True
            self.frequency(self._frequency)

    def _get_f_RO_mod(self):
        return self._f_RO_mod

    def _set_mod_amp(self, val):
        if val != self._mod_amp:
            if 'UHFQC' in self.acquisition_instr():
                self._UHFQC_awg_parameters_changed = True
        self._mod_amp = val

    def _get_mod_amp(self):
        return self._mod_amp

    def on(self):
        if self.LO.status().startswith('Off'):
            self.LO.on()
            time.sleep(1.0)

    def off(self):
        self.LO.off()

    def _get_status(self):
        return self.LO.get('status')

    def _set_I_channel(self, channel):
        if channel != self._I_channel and \
                not 'UHFQC' in self.acquisition_instr():
            self._awg_seq_parameters_changed = True
        self._I_channel = channel

    def _get_I_channel(self):
        return self._I_channel

    def _set_Q_channel(self, channel):
        if channel != self._Q_channel and \
                not 'UHFQC' in self.acquisition_instr():
            self._awg_seq_parameters_changed = True
        self._Q_channel = channel

    def _get_Q_channel(self):
        return self._Q_channel
