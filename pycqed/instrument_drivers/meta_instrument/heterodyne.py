# This is a virtual instrument abstracting a homodyne
# source which controls RF and LO sources

import logging
import numpy as np
from time import time
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
# Used for uploading the right AWG sequences
from measurement.pulse_sequences import standard_sequences as st_seqs


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
    shared_kwargs = ['RF', 'LO', 'CBox', 'AWG']

    def __init__(self, name,  RF, LO, CBox, AWG,
                 single_sideband_demod=False, **kw):
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, **kw)

        self.LO = LO
        self.RF = RF
        self.AWG = AWG
        self.CBox = CBox
        self._max_tint = 2000

        self.add_parameter('frequency',
                           label='Heterodyne frequency',
                           units='Hz',
                           get_cmd=self.do_get_frequency,
                           set_cmd=self.do_set_frequency,
                           vals=vals.Numbers(9e3, 40e9))
        self.add_parameter('IF', parameter_class=ManualParameter,
                           vals=vals.Numbers(-200e6, 200e6),
                           label='Intermodulation frequency',
                           units='Hz')
        self.add_parameter('RF_power', units='dBm',
                           set_cmd=self.do_set_RF_power,
                           get_cmd=self.do_get_RF_power)

        self.add_parameter('single_sideband_demod',
                           label='Single sideband demodulation',
                           parameter_class=ManualParameter)
        self.set('IF', 10e6)
        # self.LO_power = 13  # dBm
        # self.RF_power = -60  # dBm
        # self.set_Navg(1)
        self.set('single_sideband_demod', single_sideband_demod)
        # self.init()
        # self.get_status()
        self._awg_seq_filename = 'Heterodyne_marker_seq_RF_mod'
        self._disable_auto_seq_loading = False

    def set_sources(self, status):
        self.RF.set('status', status)
        self.LOset('status', status)

    def do_set_frequency(self, val):
        self._frequency = val
        # this is the definition agreed upon in issue 131
        self.RF.set('frequency', val)
        self.LO.set('frequency', val-self.IF.get())

    def do_get_frequency(self):
        freq = self.RF.frequency()
        LO_freq = self.LO.frequency()
        if LO_freq != freq-self.IF():
            logging.warning('IF between RF and LO is not set correctly')
        return freq

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

    def do_set_RF_source(self, val):
        self.RF = val

    def do_get_RF_source(self):
        return self.RF

    def do_set_RF_power(self, val):
        self.RF.power(val)
        self._RF_power = val
        # internally stored to allow setting RF from stored setting

    def do_get_RF_power(self):
        return self._RF_power

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

    def prepare(self, get_t_base=True, RO_length=500e-9, trigger_separation=20e-6):
        '''
        This function needs to be overwritten for the ATS based version of this
        driver
        '''

        if ((self._awg_seq_filename not in self.AWG.get('setup_filename')) and
                not self._disable_auto_seq_loading):
            self.seq_name = st_seqs.generate_and_upload_marker_sequence(
                RO_length, trigger_separation, RF_mod=False,
                IF=self.get('IF'), mod_amp=0.5)

        self.AWG.run()
        if get_t_base is True:
            trace_length = 512
            tbase = np.arange(0, 5*trace_length, 5)*1e-9
            self.cosI = np.floor(127.*np.cos(2*np.pi*self.get('IF')*tbase))
            self.sinI = np.floor(127.*np.sin(2*np.pi*self.get('IF')*tbase))
            self.CBox.sig0_integration_weights(self.cosI)
            self.CBox.sig1_integration_weights(self.sinI)

        self.LO.on()
        # Changes are now incorporated in the awg seq
        self._awg_seq_parameters_changed = False

        self.CBox.set('nr_samples', 1)  # because using integrated avg

        # self.CBox.set('acquisition_mode', 'idle') # aded with xiang

    def probe(self, demodulation_mode=0, **kw):
        '''
        Starts acquisition and returns the data
            'COMP' : returns data as a complex point in the I-Q plane in Volts
        '''
        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set('acquisition_mode', 'integration averaging')
        self.CBox.demodulation_mode(demodulation_mode)
        # d = self.CBox.get_integrated_avg_results()
        # quick fix for spec units. Need to properrly implement it later
        # after this, output is in mV
        scale_factor_dacmV = 1000.*0.75/128.
        # scale_factor_integration = 1./float(self.IF()*self.CBox.nr_samples()*5e-9)
        scale_factor_integration = 1./(64.*self.CBox.integration_length())
        factor = scale_factor_dacmV*scale_factor_integration
        d = np.double(self.CBox.get_integrated_avg_results())*np.double(factor)
        # print(d)
        dat = d[0][0]+1j*d[1][0]

        # self.CBox.set('acquisition_mode', 'idle') # aded with xiang
        return dat


        # return s21

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


class LO_modulated_Heterodyne(HeterodyneInstrument):
    '''
    Heterodyne instrument for pulse modulated LO.
    Inherits functionality for the HeterodyneInstrument

    AWG is used for modulating signal and triggering the CBox
    CBox is used for acquisition.
    '''
    shared_kwargs = ['RF', 'LO', 'CBox', 'AWG']

    def __init__(self, name,  LO, CBox, AWG,
                 single_sideband_demod=False, **kw):
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, **kw)
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
        self._awg_seq_parameters_changed = True
        self._mod_amp_changed = True
        # internally used to reload awg sequence implicitly

    def prepare(self, get_t_base=True):
        '''
        This function needs to be overwritten for the ATS based version of this
        driver

        Sets parameters in the ATS_CW and turns on the sources.
        if optimize == True it will optimze the acquisition time for a fixed
        t_int.
        '''
        # only uploads a seq to AWG if something changed
        if ((self._awg_seq_filename not in self.AWG.get('setup_filename') or
                self._awg_seq_parameters_changed) and
                not self._disable_auto_seq_loading):
            self.seq_name = st_seqs.generate_and_upload_marker_sequence(
                500e-9, 20e-6, RF_mod=True,
                IF=self.get('IF'), mod_amp=0.5)

        self.AWG.run()
        if get_t_base is True:
            trace_length = self.CBox.get('nr_samples')
            tbase = np.arange(0, 5*trace_length, 5)*1e-9
            self.cosI = np.cos(2*np.pi*self.get('IF')*tbase)
            self.sinI = np.sin(2*np.pi*self.get('IF')*tbase)
        self.LO.on()
        # Changes are now incorporated in the awg seq
        self._awg_seq_parameters_changed = False

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
        # Split up in here to prevent unneedy
        if self._mod_amp_changed:
            self.AWG.set('ch3_amp', self.get('mod_amp'))
            self.AWG.set('ch4_amp', self.get('mod_amp'))
        if self._awg_seq_parameters_changed:
            self.prepare()
        self.CBox.set('acquisition_mode', 0)
        self.CBox.set('acquisition_mode', 4)
        d = self.CBox.get_integrated_avg_results()
        dat = d[0][0]+1j*d[1][0]
        return dat

    def _do_set_mod_amp(self, val):
        self._mod_amp = val
        self._mod_amp_changed = True

    def _do_get_mod_amp(self):
        return self._mod_amp

    def do_set_IF(self, val):
        if val != self._IF:
            self._awg_seq_parameters_changed = True
        self._IF = val
