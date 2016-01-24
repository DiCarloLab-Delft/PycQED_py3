# This is a virtual instrument abstracting a homodyne
# source which controls RF and LO sources

import logging
import numpy as np
from time import time
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals

# Used for uploading the right AWG sequences
from modules.measurement.pulse_sequences import standard_sequences as st_seqs


class HeterodyneInstrument(Instrument):
    '''
    This is a virtual instrument for a homodyne source
    because I am working with the CBox while developing QCodes I will
    start with a CBox version of this instrument.

    An ATS version can then inherit from this instrument.

    Todo:
        - Add power settings
        - Add integration time settings
        - Build test suite
        - Add parameter Heterodyne voltage (that returns a complex value/ 2 values)
        - Add different demodulation settings.
        - Add fading plots that shows the last measured avg transients
          and points in the IQ-plane in the second window
        - Add option to use CBox integration averaging mode and verify
           identical results
    '''
    def __init__(self, name,  RF, LO, CBox,
                 single_sideband_demod=False):
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name)

        self.LO = LO
        self.RF = RF
        self.CBox = CBox
        self._max_tint = 2000

        self.add_parameter('frequency',
                           label='Heterodyne frequency',
                           units='Hz',
                           get_cmd=self.do_get_frequency,
                           set_cmd=self.do_set_frequency,
                           vals=vals.Numbers(9e3, 40e9))
        self.add_parameter('IF',
                           set_cmd=self.do_set_IF,
                           get_cmd=self.do_get_IF,
                           vals=vals.Numbers(-200e6, 200e6),
                           label='Intermodulation frequency',
                           units='Hz')

        # self.add_parameter('t_int', type=float,
        #                    flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #                    minval=100e-6, maxval=self._max_tint, units='ms')
        # self.add_parameter('trace_length', type=float,
        #                    flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #                    units='ms')
        # self.add_parameter('Navg', type=int,
        #                    flags=Instrument.FLAG_GETSET)
        self.add_parameter('single_sideband_demod',
                           label='Single sideband demodulation',
                           get_cmd=self.do_get_single_sideband_demod,
                           set_cmd=self.do_set_single_sideband_demod)
        self.set('IF', 10e6)
        # self.LO_power = 13  # dBm
        # self.RF_power = -60  # dBm
        # self.set_Navg(1)
        self.set('single_sideband_demod', single_sideband_demod)
        # self.init()
        # self.get_status()

    def set_sources(self, status):
        self.RF.set('status', status)
        self.LOset('status', status)

    def do_set_frequency(self, val):
        self._frequency = val
        # this is the definition agreed upon in issue 131
        self.RF.set('frequency', val)
        self.LO.set('frequency', val-self.IF.get())

    def do_get_frequency(self):
        freq = self.RF.get_frequency()
        LO_freq = self.LO.get_frequency()
        if LO_freq != freq-self._IF:
            logging.warning('IF between RF and LO is not set correctly')
        return freq

    def do_set_IF(self, val):
        self._IF = val

    def do_get_IF(self):
        return self._IF

    def do_set_Navg(self, val):
        self.Navg = val

    def do_get_Navg(self):
        '''
        Navg is only used in the detector function
        '''
        return self.Navg

    def do_set_LO_source(self, val):
        self.LO = val

    def do_get_LO_source(self):
        return self.LO

    # def do_set_LO_power(self, val):
    #     self.LO.set_power(val)
    #     self.LO_power = val
    #     # internally stored to allow setting RF from stored setting

    # def do_get_LO_power(self):
    #     return self.LO_power

    def do_set_RF_source(self, val):
        self.RF = val

    def do_get_RF_source(self):
        return self.RF

    # def do_set_RF_power(self, val):
    #     self.RF.set_power(val)
    #     self.RF_power = val
        # internally stored to allow setting RF from stored setting

    # def do_get_RF_power(self):
    #     return self.RF_power

    def do_set_status(self, val):
        self.state = val
        if val == 'On':
            self.LO.on()
            self.RF.on()
        else:
            self.LO.off()
            self.RF.off()

    def do_get_status(self):
        if (self.LO.get('status').startswith('On')and
                self.RF.get('status').startswith('On')):
            return 'On'
        elif (self.LO.get('status').startswith('Off') and
                self.RF.get('status').startswith('Off')):
            return 'Off'
        else:
            return 'LO: %s, RF: %s' % (self.LO.get('status'),
                                       self.RF.get('status'))

    def on(self):
        self.set('status', 'On')

    def off(self):
        self.set('status', 'Off')
        return

    # def do_set_t_int(self, t_int):
    #     '''
    #     sets the integration time per probe shot
    #     '''
    #     self.ATS_CW.set_t_int(t_int)
    #     self.get_trace_length()
    #     self.get_number_of_buffers()

    # def do_get_t_int(self):
    #     return self.ATS_CW.get_t_int()

    # def do_set_trace_length(self, trace_length):
    #     self.ATS_CW.set_trace_length(trace_length)
    #     self.get_t_int()

    # def do_get_trace_length(self):
    #     return self.ATS_CW.get_trace_length()

    # def do_set_number_of_buffers(self, n_buff):
    #     self.ATS_CW.set_number_of_buffers(n_buff)
    #     self.get_t_int()

    # def do_get_number_of_buffers(self):
    #     return self.ATS_CW.get_number_of_buffers()

    def prepare(self, get_t_base=True):
        '''
        This function needs to be overwritten for the ATS based version of this
        driver

        Sets parameters in the ATS_CW and turns on the sources.
        if optimize == True it will optimze the acquisition time for a fixed
        t_int.
        '''
        # self.RF.set_power(self.do_get_RF_power())
        # self.LO.set_power(self.do_get_LO_power())
        if get_t_base is True:
            trace_length = self.CBox.get('nr_samples')
            tbase = np.arange(0, 5*trace_length, 5)*1e-9
            self.cosI = np.cos(2*np.pi*self.get('IF')*tbase)
            self.sinI = np.sin(2*np.pi*self.get('IF')*tbase)
        self.RF.on()
        self.LO.on()


    # def do_set_channel(self, ch):
    #     self.channel = ch

    # def do_get_channel(self, ch):
    #     return self.channel

    def get_averaged_transient(self):
        self.CBox.set('acquisition_mode', 0)
        self.CBox.set('acquisition_mode', 'input averaging mode')
        dat = self.CBox.get_input_avg_results()
        # requires rescaling to dac voltages
        return dat

    # def get_averaged_transient_ATS(self):
    #     # version that can be used for the ATS
    #     self.ATS_CW.start()
    #     data = self.ATS.get_rescaled_avg_data()

    def probe(self, **kw):
        '''
        Starts acquisition and returns the data
            'COMP' : returns data as a complex point in the I-Q plane in Volts
        '''
        i = 0
        succes = False
        while succes is False and i < 10:
            try:
                data = self.get_averaged_transient()
                succes = True
            except:
                succes = False
            i += 1
        s21 = self.demodulate_data(data)  # returns a complex point in the IQ-plane
        return s21

    def get_demod_array(self):
        return self.cosI, self.sinI

    def demodulate_data(self, dat):
        '''
        Returns a complex point in the IQ plane by integrating and demodulating
        the data. Demodulation is done based on the 'IF' and
        'single_sideband_demod' parameters of the Homodyne instrument.
        '''
        if self._IF != 0:
            # self.cosI is based on the time_base and created in self.init()
            if self._single_sideband_demod is True:
                # this definition for demodulation is consistent with
                # issue #131
                I = np.average(self.cosI * dat[0] + self.sinI * dat[1])
                Q = np.average(-self.sinI * dat[0] + self.cosI * dat[1])
            else:  # Single channel demodulation, defaults to using channel 1
                I = 2*np.average(dat[0]*self.cosI)
                Q = 2*np.average(dat[0]*self.sinI)
        else:
            I = np.average(dat[0])
            Q = np.average(dat[1])
        return I+1.j*Q

    def do_set_single_sideband_demod(self, val):
        self._single_sideband_demod = val

    def do_get_single_sideband_demod(self):
        return self._single_sideband_demod


class LO_modulated_Heterodyne(HeterodyneInstrument):
    '''
    Heterodyne instrument for pulse modulated LO.
    Inherits functionality for the HeterodyneInstrument

    AWG is used for modulating signal and triggering the CBox
    CBox is used for acquisition.
    '''
    def __init__(self, name,  LO, CBox, AWG,
                 single_sideband_demod=False):
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name)
        self.LO = LO
        self.CBox = CBox
        self.AWG = AWG
        self._awg_seq_filename = 'Heterodyne_marker_seq_RF_mod'
        self.add_parameter('frequency',
                           label='Heterodyne frequency',
                           units='Hz',
                           get_cmd=self.do_get_frequency,
                           set_cmd=self.do_set_frequency,
                           vals=vals.Numbers(9e3, 40e9))
        self.add_parameter('IF',
                           set_cmd=self.do_set_IF,
                           get_cmd=self.do_get_IF,
                           vals=vals.Numbers(-200e6, 200e6),
                           label='Intermodulation frequency',
                           units='Hz')

        self.add_parameter('single_sideband_demod',
                           label='Single sideband demodulation',
                           get_cmd=self.do_get_single_sideband_demod,
                           set_cmd=self.do_set_single_sideband_demod)
        self.set('single_sideband_demod', single_sideband_demod)

        self.add_parameter('mod_amp',
                           label='Modulation amplitud',
                           units='V',
                           set_cmd=self._do_set_mod_amp,
                           get_cmd=self._do_get_mod_amp,
                           vals=vals.Numbers(0, 1))
        # Negative vals should be done by setting the IF negative

        self._IF = 0  # ensures that awg_seq_par_changed flag is True
        self._mod_amp = .5
        self._frequency = None
        self.set('IF', -10e6)
        self.set('mod_amp', .5)
        self._disable_auto_seq_loading = False
        self._awg_seq_paramters_changed = True
        # internally used to reload awg sequence implicitly

    def prepare(self, get_t_base=True):
        '''
        This function needs to be overwritten for the ATS based version of this
        driver

        Sets parameters in the ATS_CW and turns on the sources.
        if optimize == True it will optimze the acquisition time for a fixed
        t_int.
        '''
        if ((self._awg_seq_filename not in self.AWG.get('setup_filename') or
                self._awg_seq_paramters_changed) and
                not self._disable_auto_seq_loading):
            self.seq_name = st_seqs.generate_and_upload_marker_sequence(
                500e-9, 20e-6, RF_mod=True,
                IF=self.get('IF'), mod_amp=0.5)
        self.AWG.set('ch3_amp', self.get('mod_amp'))
        self.AWG.set('ch4_amp', self.get('mod_amp'))
        self.AWG.run()
        if get_t_base is True:
            trace_length = self.CBox.get('nr_samples')
            tbase = np.arange(0, 5*trace_length, 5)*1e-9
            self.cosI = np.cos(2*np.pi*self.get('IF')*tbase)
            self.sinI = np.sin(2*np.pi*self.get('IF')*tbase)
        self.LO.on()
        # Changes are now incorporated in the awg seq
        self._awg_seq_paramters_changed = False

        self.CBox.set('nr_samples', 1)  # because using integrated avg

    def do_set_frequency(self, val):
        self._frequency = val
        # this is the definition agreed upon in issue 131
        # AWG modulation ensures that signal ends up at RF-frequency
        self.LO.set('frequency', val-self.IF.get())

    def do_get_frequency(self):
        LO_freq = self.LO.get('frequency')
        freq = LO_freq + self._IF
        return freq

    def probe(self):
        if self._awg_seq_paramters_changed:
            self.prepare()
        self.CBox.set('acquisition_mode', 0)
        self.CBox.set('acquisition_mode', 4)
        d = self.CBox.get_integrated_avg_results()
        dat = d[0][0]+1j*d[1][0]
        return dat

    def _do_set_mod_amp(self, val):
        self._mod_amp = val

    def _do_get_mod_amp(self):
        return self._mod_amp

    def do_set_IF(self, val):
        if val != self._IF:
            self._awg_seq_paramters_changed = True
        self._IF = val
