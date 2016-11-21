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
    '''
    This is a virtual instrument for a homodyne source

    Instrument is CBox and UHFQC compatible

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
    shared_kwargs = ['RF', 'LO', 'AWG']

    def __init__(self, name,  RF, LO, AWG, acquisition_instr='CBox',
                 single_sideband_demod=False, **kw):
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, **kw)

        self.LO = LO
        self.RF = RF
        self.AWG = AWG

        self.add_parameter('frequency',
                           label='Heterodyne frequency',
                           units='Hz',
                           get_cmd=self.do_get_frequency,
                           set_cmd=self.do_set_frequency,
                           vals=vals.Numbers(9e3, 40e9))
        self.add_parameter('f_RO_mod', parameter_class=ManualParameter,
                           vals=vals.Numbers(-600e6, 600e6),
                           label='Intermodulation frequency',
                           units='Hz', initial_value=10e6)
        self.add_parameter('RF_power', label='RF power',
                           units='dBm',
                           set_cmd=self.do_set_RF_power,
                           get_cmd=self.do_get_RF_power)
        self.add_parameter('single_sideband_demod',
                           label='Single sideband demodulation',
                           parameter_class=ManualParameter)
        self.add_parameter('acquisition_instr',
                           set_cmd=self._do_set_acquisition_instr,
                           get_cmd=self._do_get_acquisition_instr,
                           vals=vals.Strings())
        self.add_parameter('nr_averages',
                            parameter_class=ManualParameter,
                           initial_value=1024,
                           vals=vals.Numbers(min_value=0, max_value=1e6))

        self.set('single_sideband_demod', single_sideband_demod)
        self._awg_seq_filename = 'Heterodyne_marker_seq_RF_mod'
        self._disable_auto_seq_loading = False
        self.acquisition_instr(acquisition_instr)

    def set_sources(self, status):
        self.RF.set('status', status)
        self.LO.set('status', status)

    def do_set_frequency(self, val):
        self._frequency = val
        # this is the definition agreed upon in issue 131
        self.RF.set('frequency', val)
        self.LO.set('frequency', val-self.f_RO_mod.get())

    def do_get_frequency(self):
        freq = self.RF.frequency()
        LO_freq = self.LO.frequency()
        if LO_freq != freq-self.f_RO_mod():
            logging.warning('f_RO_mod between RF and LO is not set correctly')
        return freq


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
                IF=self.get('f_RO_mod'), mod_amp=0.5)

        self.AWG.run()

        if get_t_base is True:
            if 'CBox' in self.acquisition_instr():
                trace_length = 512
                tbase = np.arange(0, 5*trace_length, 5)*1e-9
                self.cosI = np.floor(127.*np.cos(2*np.pi*self.get('f_RO_mod')*tbase))
                self.sinI = np.floor(127.*np.sin(2*np.pi*self.get('f_RO_mod')*tbase))
                self._acquisition_instr.sig0_integration_weights(self.cosI)
                self._acquisition_instr.sig1_integration_weights(self.sinI)
                self._acquisition_instr.set('nr_samples', 1)  # because using integrated avg
                self._acquisition_instr.nr_averages(int(self.nr_averages()))

            elif 'UHFQC' in self.acquisition_instr():
                print("f_RO_mod", self.get('f_RO_mod'))
                #self._acquisition_instr.prepare_SSB_weight_and_rotation(IF=self.get('f_RO_mod'),
                #                                        weight_function_I=0, weight_function_Q=1)
                self._acquisition_instr.prepare_DSB_weight_and_rotation(IF=self.get('f_RO_mod'),
                                                        weight_function_I=0, weight_function_Q=1)
                self._acquisition_instr.quex_rl_source(2) #this sets the result to integration and rotation outcome
                self._acquisition_instr.quex_rl_length(1) #only one sample to average over
                self._acquisition_instr.quex_rl_avgcnt(int(np.log2(self.nr_averages())))
                print("preparing UHFQ with avg", self._acquisition_instr.quex_rl_avgcnt())
                self._acquisition_instr.quex_wint_length(int(RO_length*(1.8e9)))
                # Configure the result logger to not do any averaging
                # The AWG program uses userregs/0 to define the number o iterations in the loop
                self._acquisition_instr.awgs_0_userregs_0(int(self.nr_averages()))
                self._acquisition_instr.awgs_0_userregs_1(0)#0 for rl, 1 for iavg
                self._acquisition_instr.awgs_0_single(1)


        self.LO.on()
        # Changes are now incorporated in the awg seq
        self._awg_seq_parameters_changed = False


        # self.CBox.set('acquisition_mode', 'idle') # aded with xiang

    def probe(self, demodulation_mode=0, **kw):
        '''
        Starts acquisition and returns the data
            'COMP' : returns data as a complex point in the I-Q plane in Volts
        '''
        if 'CBox' in self.acquisition_instr():
            self._acquisition_instr.set('acquisition_mode', 'idle')
            self._acquisition_instr.set('acquisition_mode', 'integration averaging')
            self._acquisition_instr.demodulation_mode(demodulation_mode)
            # d = self.CBox.get_integrated_avg_results()
            # quick fix for spec units. Need to properrly implement it later
            # after this, output is in mV
            scale_factor_dacmV = 1000.*0.75/128.
            # scale_factor_integration = 1./float(self.f_RO_mod()*self.CBox.nr_samples()*5e-9)
            scale_factor_integration = 1./(64.*self._acquisition_instr.integration_length())
            factor = scale_factor_dacmV*scale_factor_integration
            d = np.double(self._acquisition_instr.get_integrated_avg_results())*np.double(factor)
            # print(np.size(d))
            dat = d[0][0]+1j*d[1][0]
        elif 'UHFQC' in self.acquisition_instr():
            t0 = time.time()
            self._acquisition_instr.awgs_0_enable(1)
            try:
                temp = self._acquisition_instr.awgs_0_enable()
            except:
                temp = self._acquisition_instr.awgs_0_enable()
            del temp

            while self._acquisition_instr.awgs_0_enable() == 1:
                time.sleep(0.01)
            channels=[0,1]
            data = ['']*len(channels)
            for i, channel in enumerate(channels):
                dataset = eval("self._acquisition_instr.quex_rl_data_{}()".format(channel))
                data[i] = dataset[0]['vector']
            dat=data[0]+1j*data[1]
            t1 = time.time()
            print("time for UHFQC polling", t1-t0)
        return dat


        # return s21

    def get_demod_array(self):
        return self.cosI, self.sinI

    def demodulate_data(self, dat):
        '''
        Returns a complex point in the IQ plane by integrating and demodulating
        the data. Demodulation is done based on the 'f_RO_mod' and
        'single_sideband_demod' parameters of the Homodyne instrument.
        '''
        if self._f_RO_mod != 0:
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

    def _do_get_acquisition_instr(self):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily
        return self._acquisition_instr.name

    def _do_set_acquisition_instr(self, acquisition_instr):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily

        self._acquisition_instr = self.find_instrument(acquisition_instr)


class LO_modulated_Heterodyne(HeterodyneInstrument):
    '''
    Heterodyne instrument for pulse modulated LO.
    Inherits functionality for the HeterodyneInstrument

    AWG is used for modulating signal and triggering the UHFQC or
    CBox is used for acquisition.
    '''
    shared_kwargs = ['RF', 'LO', 'AWG']

    def __init__(self, name,  LO, AWG, acquisition_instr='CBox',
                 single_sideband_demod=False, **kw):
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, **kw)
        self.LO = LO
        self.AWG = AWG
        self._awg_seq_filename = 'Heterodyne_marker_seq_RF_mod'
        self.add_parameter('frequency',
                           label='Heterodyne frequency',
                           units='Hz',
                           get_cmd=self.do_get_frequency,
                           set_cmd=self.do_set_frequency,
                           vals=vals.Numbers(9e3, 40e9))
        self.add_parameter('f_RO_mod',
                           set_cmd=self.do_set_f_RO_mod,
                           get_cmd=self.do_get_f_RO_mod,
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

        self.add_parameter('acquisition_instr',
                           set_cmd=self._do_set_acquisition_instr,
                           get_cmd=self._do_get_acquisition_instr,
                           vals=vals.Strings())

        self.add_parameter('nr_averages',
                            parameter_class=ManualParameter,
                           initial_value=1024,
                           vals=vals.Numbers(min_value=0, max_value=1e6))
        # Negative vals should be done by setting the f_RO_mod negative

        self._f_RO_mod = 0  # ensures that awg_seq_par_changed flag is True
        self._mod_amp = .5
        self._frequency = None
        self.set('f_RO_mod', -10e6)
        self.set('mod_amp', .5)
        self._disable_auto_seq_loading = False
        self._awg_seq_parameters_changed = True
        self._mod_amp_changed = True
        self.acquisition_instr(acquisition_instr)
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
                IF=self.get('f_RO_mod'), mod_amp=0.5)

        self.AWG.run()

        if get_t_base is True:
            trace_length = self.CBox.get('nr_samples')
            tbase = np.arange(0, 5*trace_length, 5)*1e-9
            self.cosI = np.cos(2*np.pi*self.get('f_RO_mod')*tbase)
            self.sinI = np.sin(2*np.pi*self.get('f_RO_mod')*tbase)
        self.LO.on()
        # Changes are now incorporated in the awg seq
        self._awg_seq_parameters_changed = False

        self.CBox.set('nr_samples', 1)  # because using integrated avg

    def do_set_frequency(self, val):
        self._frequency = val
        # this is the definition agreed upon in issue 131
        # AWG modulation ensures that signal ends up at RF-frequency
        self.LO.set('frequency', val-self.f_RO_mod.get())

    def do_get_frequency(self):
        LO_freq = self.LO.get('frequency')
        freq = LO_freq + self._f_RO_mod
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

    def do_set_f_RO_mod(self, val):
        if val != self._f_RO_mod:
            self._awg_seq_parameters_changed = True
        self._f_RO_mod = val

    def _do_get_acquisition_instr(self):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily
        return self._acquisition_instr.name


    def _do_set_acquisition_instr(self, acquisition_instr):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily

        self._acquisition_instr = self.find_instrument(acquisition_instr)


