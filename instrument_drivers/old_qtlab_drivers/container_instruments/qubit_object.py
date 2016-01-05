from modules.analysis import analysis_toolbox as a_tools
from instrument import Instrument
import logging
import numpy as np
import h5py
import types
import qt
import time
from modules.measurement import sweep_functions as swf
from modules.measurement import CBox_sweep_functions as CB_swf
from modules.measurement import detector_functions as det
from modules.measurement import composite_detector_functions as cdet
from modules.analysis import measurement_analysis as MA
from modules.measurement import AWG_sweep_functions as awg_swf
from modules.measurement import calibration_toolbox as cal_tools
from modules.analysis import fitting_models as fit_mods
import imp


class qubit_object(Instrument):
    '''
    Instrument used for storing qubit parameters.
    '''

    def __init__(self, name, reset=False, **kw):

        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['Container', 'Qubit-Object'])

        #  timestamp list to keep track of experiments on this qubit
        self.add_parameter('timestamps', tags='logging',
                           type=list, flags=Instrument.FLAG_SET)

        # Qubit parameters
        self.add_parameter('E_c', units='GHz', tags='Qubit parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('E_c_stderr', units='GHz', tags='Qubit parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('E_j', units='GHz', tags='Qubit parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('E_j_stderr', units='GHz', tags='Qubit parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('f_max', units='GHz', tags='Qubit parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('f_max_stderr', units='GHz',
                           tags='Qubit_parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('dac_sweet_spot', units='mV',
                           tags='Qubit_parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('dac_sweet_spot_stderr', units='mV',
                           tags='Qubit_parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('dac_flux_coefficient', units='mV',
                           tags='Qubit_parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('dac_flux_coefficient_stderr', units='mV',
                           tags='Qubit_parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        # qubit asymmetry parameter for qubits with asymmetric SQUIDs
        self.add_parameter('asymmetry',
                           tags='Qubit_parameter',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('qubit_suffix',
                           tags='Qubit_parameter',
                           type=str, flags=Instrument.FLAG_GETSET)

        # Current quantities
        self.add_parameter('current_frequency', units='GHz',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('current_dac_value', units='mV',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('current_RO_frequency', units='GHz',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('pulse_amplitude_I', units='Volt',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('pulse_amplitude_Q', units='Volt',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)

        self.add_parameter('spec_pulse_amp', units=' V',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('pulse_phase_I', units='DEG',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('pulse_phase_Q', units='DEG',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('close_drag_switch',
                           tags='Current',
                           type=bool, flags=Instrument.FLAG_GETSET)
        self.add_parameter('gauss_width', units='ns',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('sideband_modulation_frequency',
                           flag=Instrument.FLAG_GETSET,
                           type=float, units='GHz', maxval=.5)
        self.add_parameter('pulse_amp_control',
                           tagss='Sample setup',
                           type=str, flags=Instrument.FLAG_GETSET)
        self.add_parameter('dac_channel',
                           tags='Sample setup',
                           type=int, flags=Instrument.FLAG_GETSET)
        self.add_parameter('freq_calc',
                           tags='Current',
                           type=str, flags=Instrument.FLAG_GETSET)
        self.add_parameter('duplexer_output_channel',
                           tags='Sample setup',
                           type=int, flags=Instrument.FLAG_GETSET)
        self.add_parameter('RF_source',
                           tags='Sample setup',
                           type=str, flags=Instrument.FLAG_GETSET)
        self.add_parameter('RF_CW_power', units='dBm',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('RF_TD_power', units='dBm',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('qubit_source',
                           tags='Sample setup',
                           type=str, flags=Instrument.FLAG_GETSET)
        self.add_parameter('source_power', units='dBm',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('qubit_drive',
                           tags='Sample setup',
                           type=str, flags=Instrument.FLAG_GETSET)
        self.add_parameter('drive_power', units='dBm',
                           tags='Current',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('AWG_source',
                           tags='Current',
                           type=str, flags=Instrument.FLAG_GETSET)
        self.add_parameter('AWG_ch_I',
                           tags='Sample setup',
                           type=int, flags=Instrument.FLAG_GETSET)
        self.add_parameter('AWG_ch_Q',
                           tags='Sample setup',
                           type=int, flags=Instrument.FLAG_GETSET)
        self.add_parameter('data_acquistion',
                           tags='Sample setup',
                           type=str, flags=Instrument.FLAG_GETSET)
        self.add_parameter('t_int', units='ns',
                           tags='Sample setup',
                           type=int, flags=Instrument.FLAG_GETSET)

        self.sideband_modulation_frequency = 0
        # placeholder to allow initialization of current freq and SB
        self.set_AWG_source('AWG')
        self.IVVI = qt.instruments['IVVI']
        self.Flux_Control = qt.instruments['Flux_Control']
        self.HM = qt.instruments['HM']
        self.TD_Meas = qt.instruments['TD_Meas']
        self.Pulsed_Spec = qt.instruments['Pulsed_Spec']
        if 'Duplexer' in qt.instruments.get_instrument_names():
            self.Dupl = qt.instruments['Duplexer']
        if 'CBox' in qt.instruments.get_instrument_names():
            self.CBox = qt.instruments['CBox']
            self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.set_freq_calc('dac')
        self.qubit_readout = self
        self.data_acquistion = 'ATS'
        self.set_asymmetry(0)

    def reload_modules(self):
        imp.reload(MA)
        imp.reload(det)
        imp.reload(cdet)
        imp.reload(swf)
        imp.reload(awg_swf)
        imp.reload(CB_swf)
        imp.reload(cal_tools)
        imp.reload(fit_mods)

    def calculate_frequency_dac(self, dac=None):
        '''
        returns calculated qubit frequency in GHz based on the given dac in mV
        Uses the parameters as set in the qubit object.
        '''
        if dac is None:
            dac = self.IVVI.get_dac(self.dac_channel)

        f_max = self.get_f_max()
        E_c = self.get_E_c()
        dac_flux_coefficient = self.get_dac_flux_coefficient()
        dac_sweet_spot = self.get_dac_sweet_spot()
        asymmetry = self.get_asymmetry()

        calculated_frequency = fit_mods.QubitFreqDac(dac_voltage=dac,
                                                     f_max=f_max,
                                                     E_c=E_c,
                                                     dac_sweet_spot=dac_sweet_spot,
                                                     dac_flux_coefficient=dac_flux_coefficient,
                                                     asymmetry=asymmetry)

        return calculated_frequency

    def calculate_frequency_flux(self, flux):
        '''
        returns calculated qubit frequency in GHz based on the given dac in mV
        Uses the parameters as set in the qubit object.
        '''

        return self.calculate_frequency_dac(flux + self.get_dac_sweet_spot())

    def calculate_dac_frequency(self, frequency):
        '''
        returns calculated qubit dac value for a given frequency
        '''
        f_max = self.get_f_max()
        E_c = self.get_E_c()
        dac_flux_coefficient = self.get_dac_flux_coefficient()
        dac_sweet_spot = self.get_dac_sweet_spot()

        calculated_dac_value_plus = 1/dac_flux_coefficient*np.arccos(
            np.power((frequency + E_c)/(f_max + E_c), 2)) + dac_sweet_spot
        calculated_dac_value_minus = -1/dac_flux_coefficient*np.arccos(
            np.power((frequency + E_c)/(f_max + E_c), 2)) + dac_sweet_spot

        if np.abs(calculated_dac_value_plus) > \
                np.abs(calculated_dac_value_minus):
            calculated_dac_value = calculated_dac_value_minus
        else:
            calculated_dac_value = calculated_dac_value_plus

        return calculated_dac_value

    def calculate_flux_frequency(self, frequency):
        '''
        returns calculated qubit flux value for a given frequency
        '''
        # f_max = self.get_f_max()
        # E_c = self.get_E_c()
        # dac_flux_coefficient = self.get_dac_flux_coefficient()

        # calculated_flux_value = 1/dac_flux_coefficient*np.arccos(
        #     np.power((frequency + E_c)/(f_max + E_c), 2))

        return self.calculate_dac_frequency(frequency) \
            - self.get_dac_sweet_spot()

    def calculate_flux_slope(self, flux_val):
        slope = self.calculate_dac_slope(flux_val + self.get_dac_sweet_spot())
        return slope

    def calculate_dac_slope(self, dac_voltage):
        f_max = self.get_f_max()
        E_c = self.get_E_c()
        dac_flux_coefficient = self.get_dac_flux_coefficient()
        dac_sweet_spot = self.get_dac_sweet_spot()

        # slope = -1*(f_max+E_c)*dac_flux_coefficient*np.sin(
        #     2 * dac_flux_coefficient * (dac_voltage-dac_sweet_spot)) \
        #     / (4 * np.power(np.abs(np.cos(dac_flux_coefficient
        #        * (dac_voltage-dac_sweet_spot))), 3./2.))
        slope = -(f_max+E_c)*dac_flux_coefficient * \
            np.sin(dac_flux_coefficient*(dac_voltage-dac_sweet_spot)) / \
            2 * np.sqrt(np.abs(np.cos(
                dac_flux_coefficient*(dac_voltage-dac_sweet_spot))))

        return slope

    def prepare_for_TD(self, drive_frequency=None, qubit_readout=None,
                       leave_source_on=False,
                       Navg=None, **kw):
        if drive_frequency is None:
            drive_frequency = self.current_frequency
            generator_frequency = self.generator_frequency
        else:
            generator_frequency = drive_frequency - \
                self.sideband_modulation_frequency

        self.qubit_readout = qubit_readout
        if self.qubit_readout is None:
            self.qubit_readout = self

        if leave_source_on is not True:
            self.qubit_source_instrument.off()
        # self.qubit_RF_instrument.set_power(self.RF_TD_power)

        TD_Meas = qt.instruments['TD_Meas']
        if Navg is not None:
            TD_Meas.set_Navg(Navg)
        try:
            self.get_t_int()
        except:
            logging.warning('Qubit t_int not set. Using value from TD_Meas.')
        else:
            self.TD_Meas.set_t_int(self.t_int)


        if not TD_Meas.get_multiplex():
            TD_Meas.set_f_readout(
                self.qubit_readout.get_current_RO_frequency()*1e9)
            TD_Meas.set_RF_source(self.qubit_readout.get_RF_source())
            TD_Meas.set_RF_power(self.qubit_readout.get_RF_TD_power())
        self.qubit_drive_instrument.set_frequency(
            (generator_frequency)*1e9)
        self.qubit_drive_instrument.set_power(self.drive_power)
        self.qubit_drive_instrument.on()

        if self.pulse_amp_control is 'AWG':
            print('setting pulse amps')
            amp_I = kw.get('drive_amplitude', self.pulse_amplitude_I)
            amp_Q = kw.get('drag_amplitude', self.pulse_amplitude_Q)
            print(amp_I, amp_Q)
            cmd1 = "self.AWG.set_ch" + str(self.AWG_ch_I) + \
                "_amplitude(amp_I)"
            exec(cmd1)
            cmd2 = "self.AWG.set_ch" + str(self.AWG_ch_Q) + \
                "_amplitude(amp_Q)"
            exec(cmd2)

        elif self.pulse_amp_control is 'Duplexer':
            self.setup_Duplexer(**kw)
        elif self.pulse_amp_control is 'CBox':
            self.CBox_lut_man.set_lut_mapping(['I', 'X180', 'Y180', 'X90',
                                              'Y90', 'Block', 'X180_delayed'])
            self.CBox_lut_man.set_amp180(self.pulse_amplitude_I*1000)
            self.CBox_lut_man.set_amp90(self.pulse_amplitude_I*1000/2.0)

    def prepare_for_CW(self, RF_power=None, source_power=None, **kw):

        disable_sources = kw.pop('disable_sources', True)
        # option exists to do a sweep without turning of the sources to e.g. extract the dispersive shift.
        if disable_sources:
            self.qubit_drive_instrument.off()

        if source_power is None:
            source_power = self.source_power
        self.qubit_source_instrument.set_power(source_power)

        if 'pulsemod_state' in list(self.qubit_source_instrument.get_parameters().keys()):
            self.qubit_source_instrument.set_pulsemod_state('OFF')
        self.HM.set_RF_source(self.RF_source)
        self.HM.set_frequency(self.current_RO_frequency*1e9)
        if RF_power is None:
            RF_power = self.RF_CW_power
        self.HM.set_RF_power(RF_power)
        if 'pulsemod_state' in list(self.qubit_RF_instrument.get_parameters().keys()):
            self.qubit_RF_instrument.set_pulsemod_state('OFF')
        self.HM.on()

        if self.get_pulse_amp_control() == 'Duplexer':
            self.Dupl.set_all_switches_to('OFF')
            duplexer_channel = self.get_duplexer_output_channel()
            self.Dupl.set_switch(3, duplexer_channel, 'ON')
            self.Dupl.set_attenuation(3, duplexer_channel, 40000)

    def prepare_for_Pulsed_Spec(self, RF_power=None, pulsed_excitation=False, drive_power=None, **kw):
        self.qubit_drive_instrument.off()
        if drive_power is None:
            drive_power = self.drive_power
        self.Pulsed_Spec.set_RF_power(self.RF_TD_power)
        self.Pulsed_Spec.set_f_readout(self.current_RO_frequency*1e9)
        try:
            self.get_t_int()
        except:
            logging.warning('Qubit t_int not set. Using value from Pulsed_Spec.')
        else:
            self.Pulsed_Spec.set_t_int(self.t_int)
        self.qubit_drive_instrument.set_power(self.drive_power)
        self.qubit_drive_instrument.on()

        if self.pulse_amp_control is 'AWG':
            cmd1 = "self.AWG.set_ch" + str(self.AWG_ch_I) + \
                "_amplitude(kw.get('drive_amplitude',self.spec_pulse_amp))"
            exec(cmd1)
            cmd2 = "self.AWG.set_ch" + str(self.AWG_ch_Q) + \
                "_amplitude(kw.get('drive_amplitude',self.spec_pulse_amp))"
            exec(cmd2)
        elif self.pulse_amp_control is 'Duplexer':
            self.setup_Duplexer(**kw)

    def get_msmt_label_suffix(self, mode, label=None, **kw):
        msmt_label = ''
        if mode == 'CW':
            RF_power = self.qubit_RF_instrument.get_power()
            source_power = self.qubit_source_instrument.get_power()
            if RF_power != self.RF_CW_power:
                msmt_label += '_RFpow{}'.format(RF_power)
            if source_power != self.source_power:
                msmt_label += '_Spow{}'.format(source_power)
        else:
            if self.get_pulse_amp_control() is 'Duplexer':
                output_channel = self.get_duplexer_output_channel()
                drive_ampl = self.Dupl.get_attenuation(1, output_channel)
                drag_ampl = self.Dupl.get_attenuation(2, output_channel)
                phase_I = self.Dupl.get_phase(1, output_channel)
                phase_Q = self.Dupl.get_phase(2, output_channel)
                if drive_ampl != self.pulse_amplitude_I:
                    msmt_label += '_drv{}'.format(drive_ampl)
                if drag_ampl != self.pulse_amplitude_Q:
                    msmt_label += '_drag{}'.format(drag_ampl)
                if phase_I != self.pulse_phase_I:
                    msmt_label += '_phiI{}'.format(phase_I)
                if phase_Q != self.pulse_phase_Q:
                    msmt_label += '_phiQ{}'.format(phase_Q)
        msmt_label += '_{}'.format(self.get_name())
        if self.qubit_readout.get_name() != self.get_name():
            msmt_label += '_{}'.format(self.qubit_readout.get_name())
        if label is not None:
            msmt_label += '_{}'.format(label)
        return msmt_label

    def find_resonator_frequency(self, f_start=None, f_stop=None,
                                 f_span=4e-3, f_step=1e-4,
                                 update_qubit=True,
                                 suppress_print_statements=True,
                                 MC_name='MC',
                                 use_min=False,
                                 use_FWHM=False,
                                 fitting_model='hanger', **kw):
        '''
        Sweeps the readout generator to calibrate the readout frequency.

        args:
            f_start, f_stop  : If None, current RO frequency is used
            f_span=10e-3 (10 MHz)   : span around center frequency.
                                    Only used when f_start is not given
            f_step=5e-5         : frequency stepsize (GHz)
            use_min=False         : Whether to return min freq or fitted freq
            fitting_model='hanger': Fitting model ('hanger' or 'lorentzian')

        return {'f_resonator', 'f_resonator_stderr',
                'quality_factor', 'quality_factor_stderr',
                'freqs', 'data'}
               All frequencies in GHz

        '''
        if f_start is None:
            f_start = self.current_RO_frequency - f_span/2
            f_stop = self.current_RO_frequency + f_span/2
        sweep_points = np.arange(f_start, f_stop, f_step)

        self.prepare_for_CW(**kw)
        self.HM.on()

        MC = qt.instruments[MC_name]
        MC.set_sweep_function(swf.HM_frequency_GHz())
        MC.set_detector_function(det.HomodyneDetector())
        MC.set_sweep_points(sweep_points)
        self.AWG.start()
        msmt_name = 'Resonator_Scan{}'.format(self.get_msmt_label_suffix('CW',
                                              **kw))
        data = MC.run(name=msmt_name,
                      suppress_print_statements=suppress_print_statements,
                      **kw)

        hm_a = MA.Homodyne_Analysis(auto=True, label='Resonator_Scan',
                                    fitting_model=fitting_model)
        if use_min is False:
            fit_results = hm_a.fit_results
            fitted_cavity_frequency_param = fit_results.params['f0']
            f_resonator = fitted_cavity_frequency_param.value
            f_resonator_stderr = fitted_cavity_frequency_param.stderr

            fitted_quality_factor_param = fit_results.params['Q']
            quality_factor = fitted_quality_factor_param.value
            quality_factor_stderr = fitted_quality_factor_param.stderr
        elif use_min is True:
            f_resonator = hm_a.min_frequency
            f_resonator_stderr = 0
            quality_factor = 0
            quality_factor_stderr = 0
        if use_FWHM:
            fit_results = hm_a.fit_results
            fitted_cavity_frequency_param = fit_results.params['f0']
            f_resonator = fitted_cavity_frequency_param.value
            f_resonator_stderr = fitted_cavity_frequency_param.stderr
            fitted_quality_factor_param = fit_results.params['Q']
            f_resonator = f_resonator + f_resonator/fitted_quality_factor_param
            print("FWHM",  f_resonator/fitted_quality_factor_param)
            print("f_resonator", f_resonator)
        if fitting_model == 'lorentzian':

            use_max = kw.get('use_max', False)
            idx = np.argmax(data[0])
            f_resonator_max = sweep_points[idx]
            relation = np.abs(f_resonator-f_resonator_max)
            if (relation > 0.0025) or (use_max == True):
                f_resonator = f_resonator_max
                print('Resonator Fit fixed BY IMPOSING MAX')


        if True:  # chi_squared<critical_chi_squared: #Not implemented yet
            if not suppress_print_statements:
                print('Found resonator frequency to %s +/- (%s) Hz ' \
                    % (f_resonator, f_resonator_stderr))
                print('Found quality factor to be  %.0f +/- (%.0f) ' \
                    % (quality_factor, quality_factor_stderr))

            if update_qubit:
                self.set_current_RO_frequency(f_resonator)

            result = {'f_resonator': f_resonator,
                      'f_resonator_stderr': f_resonator_stderr,
                      'quality_factor': quality_factor,
                      'quality_factor_stderr': quality_factor_stderr,
                      'freqs': sweep_points,
                      'data': data}

            return result
        else:
            logging.error('Error: Fit did not converge!' +
                          ' \n Resonator frequency not found.')

    def resonator_power_sweep(self,
                              power_start, power_stop, power_stepsize,
                              freq_span,  freq_stepsize, center_freq=None,
                              MC_name='MC', **kw):
        '''
        A 2D sweep around the current_RO_frequency at varied RF_power.
        This function does not perform any fancy analysis yet.

        Ideally in the future this function extracts Delta from the high power
        and low power fits.

        At low power and at high power the shift in resonator frequency
        d_omega = g^2 /Delta
        This gives a prediction for the qubit freq as:
        Delta = f_cav_high_power - f_qubit

        g is known as a design parameter from maxwell simulations of the sample.

        '''
        MC = qt.instruments[MC_name]

        self.prepare_for_CW(**kw)

        if center_freq is None:
            center_freq = self.current_RO_frequency
        freq_start = center_freq - freq_span/2
        freq_end = center_freq + freq_span/2

        freq_sweep_points = np.arange(freq_start, freq_end, freq_stepsize)
        power_sweep_points = np.arange(power_start, power_stop, power_stepsize)
        MC.set_sweep_function(swf.HM_frequency_GHz())
        MC.set_sweep_points(freq_sweep_points)
        MC.set_sweep_function_2D(swf.HM_power_dBm())
        MC.set_sweep_points_2D(power_sweep_points)

        MC.set_detector_function(det.HomodyneDetector())
        self.HM.set_sources('On')
        MC.run_2D('Resonator_Powersweep_'+self.get_name())
        self.HM.set_sources('Off')

        tta = MA.TwoD_Analysis(auto=True, normalize=True)

        return True

    def resonator_dac_sweep(self,
                            dac_start, dac_stop, dac_stepsize,
                            freq_stepsize,
                            freq_span=None,
                            freq_start=None, freq_stop=None, center_freq=None,
                            dac_channel=None,
                            MC_name='MC',
                            **kw):
        '''
        Function under construction.

        The dac sweep is used to track the resonator while moving the respectve
        qubit. There are two distinct cases

        1. Qubit located above the resonator:
            the qubit will cross the resonator causing the fit to lose it.
            Between the qubit arches it will converge again.
            By estimating the center between two points where the track is lost
            or by fitting the (known) functional form the center of the arch
            and thereby the qubit sweet spot can be determined.
        2. Qubit located below the resonator:
            The qubit will not cross the resonator but will move more closely to
            it. In this case the functional form can also be fit and used to
            estimate the qubit sweet spot.
        '''
        MC = qt.instruments[MC_name]

        if dac_channel is None:
            dac_channel = self.get_dac_channel()

        if center_freq is None:
            center_freq = self.current_RO_frequency
        if freq_start == None:
            freq_start = center_freq - freq_span/2
            freq_stop = center_freq + freq_span/2

        freq_sweep_points = np.arange(freq_start, freq_stop, freq_stepsize)
        dac_sweep_points = np.arange(dac_start, dac_stop, dac_stepsize)
        dac_val_before_exeriment = self.IVVI.get_dac(dac_channel)

        MC.set_sweep_function(swf.HM_frequency_GHz())
        MC.set_sweep_points(freq_sweep_points)
        MC.set_sweep_function_2D(swf.Bias_Dac_mV(
                                 dac_channel=dac_channel))
        MC.set_sweep_points_2D(dac_sweep_points)

        MC.set_detector_function(det.HomodyneDetector())

        self.prepare_for_CW(**kw)
        self.HM.on()

        MC.run_2D('Resonator_Dacsweep_'+self.get_name())
        self.HM.set_sources('Off')
        self.IVVI.set_dac(dac_channel, dac_val_before_exeriment)

        tta = MA.TwoD_Analysis(auto=True)

        return True


    def find_frequency_spec(self,
                            MC_name='MC',
                            f_span=0.05,
                            f_step=0.001,
                            f_start=None, f_stop=None,
                            update_qubit=True,
                            suppress_print_statements=True,
                            source_power=None,
                            freq_calc=None,
                            pulsed=False, use_AWG=False, **kw):
        '''
        Performs a spectroscopy measurement to find the qubit frequency.

        source_power, can be set to None, 'qubit' and value (float).
            None uses what is set in the source
            'qubit' , sets the value stored in the qubit object
            value, sets the source power to the value specified
        f_start=None
            If set to "None" uses the qubit object to calculate the value.
            if set to a value scans to "f_stop"

        feq_calc=None, determines the method used to calculate the freq.
            "current", uses "self.current_frequency"
            "flux", uses "self.calculate_frequency_flux"
            "dac", uses "self.calculate_frequency_dac"

        By default, uses AWG sequence named 'Spec_5014'
        '''
        MC = qt.instruments[MC_name]
        if f_start is None:
            if freq_calc is None:
                freq_calc = self.freq_calc
            if freq_calc is 'current':
                calculated_frequency = self.current_frequency
            if freq_calc is 'flux':
                dac_channel = self.dac_channel
                flux_val = self.Flux_Control.get_flux(dac_channel)
                calculated_frequency = self.calculate_frequency_flux(flux_val)
            elif freq_calc is 'dac':
                dac_channel = self.dac_channel
                dac_val = self.IVVI.get_dac(dac_channel)
                calculated_frequency = self.calculate_frequency_dac(dac_val)
            f_start = calculated_frequency - f_span/2.
            f_stop = calculated_frequency + f_span/2.

        # self.qubit_RF_instrument.set_power(self.RF_TD_power) # in prepare
        # source = self.qubit_drive
        # source.set_power(self.get_drive_power())
        # source.on()
        if not pulsed:
            self.prepare_for_CW(**kw)
        else:
            self.prepare_for_Pulsed_Spec(**kw)

        self.AWG.start()

        if f_stop is None:
            raise ValueError('f_stop not specified')
        sweep_points = np.arange(f_start, f_stop, f_step)

        if pulsed is False:
            source = self.qubit_source_instrument
            if source_power is None:
                source.set_power(self.get_source_power())
            else:
                source.set_power(source_power)
            source.on()
            try:
                source.set_pulsemod_state('OFF')
            except:
                print('this was a stupid overnight fix')
            print('source', source)
            MC.set_sweep_function(swf.Source_frequency_GHz(Source=source))
            MC.set_sweep_points(sweep_points)
            MC.set_detector_function(det.HomodyneDetector())
        elif pulsed is True:
            if use_AWG is True:
                source = self.qubit_drive_instrument
                source.set_power(self.get_drive_power())
                source.on()
                print('source', source)
                print('sideband', self.sideband_modulation_frequency)
                print('AWG is used!')
                MC.set_sweep_function(
                    swf.Source_frequency_modulated_GHz(
                        Source=source,
                        modulation_freq=self.sideband_modulation_frequency))
                MC.set_sweep_points(sweep_points)
                MC.set_detector_function(det.PulsedSpectroscopyDetector())

            else:
                drive = self.qubit_drive_instrument
                drive.off()
                source = self.qubit_source_instrument
                if source_power is None:
                    source.set_power(self.get_source_power())
                else:
                    source.set_power(source_power)
                source.on()
                try:
                    source.set_pulsemod_state('ON')
                except:
                    print('this is an ad hoc fix')
                MC.set_sweep_function(swf.Source_frequency_GHz(Source=source))
                MC.set_sweep_points(sweep_points)
                MC.set_detector_function(det.PulsedSpectroscopyDetector())


        msmt_name = 'Qubit_Scan{}'.format(self.get_msmt_label_suffix('CW',
                                          **kw))
        # FIXME CW only if pulsed is false
        data = MC.run(name=msmt_name,
                      suppress_print_statements=suppress_print_statements,
                      **kw)

        self.AWG.stop()
        source.off()

        so_a = MA.Qubit_Spectroscopy_Analysis(auto=True, label='Qubit_Scan',
                                              close_file=False)
        f_qubit, f_qubit_stderr = so_a.get_frequency_estimate()
        peaks = so_a.peaks
        qubit_linewidth = so_a.get_linewidth_estimate()
        so_a.finish()

        if not suppress_print_statements:
            print('Qubit frequency estimated to: "%s" GHz' % f_qubit)

        if update_qubit:
            self.set_current_frequency(f_qubit)

        result = {'f_qubit': f_qubit, 'f_qubit_stderr': f_qubit_stderr,
                  'qubit_linewidth': qubit_linewidth,
                  'freqs': sweep_points, 'data': data, 'peaks': peaks}
        return result

    def find_frequency_12(self,
                          MC_name='MC',
                          f_span=5e-3,
                          f_step=0.5e-3,
                          f_start=None,
                          f_stop=None,
                          anharm_estimate=None,
                          f_01=None,
                          f_01_detuning=0.,
                          suppress_print_statements=True,
                          source_power=None,
                          source12=None,
                          source12_extra_power=6,
                          freq_calc=None,
                          pulsed=False, use_AWG=False, **kw):
        '''
        Performs a two-tone spectroscopy measurement to find the frequency of the
        qubit's 1-2 transition.

        source_power, can be set to None, 'qubit' and value (float).
            None uses what is set in the qubit
            value, sets the source power to the value specified
        f_start=None
            If set to "None" uses the qubit object to calculate the value.
            if set to a value scans to "f_stop"

        feq_calc=None, determines the method used to calculate the freq.
            "current", uses "self.current_frequency"
            "flux", uses "self.calculate_frequency_flux"
            "dac", uses "self.calculate_frequency_dac"

        By default, uses AWG sequence named 'Spec_5014'
        '''
        MC = qt.instruments[MC_name]
        if f_01 is None:
            if freq_calc is None:
                freq_calc = self.freq_calc
            if freq_calc is 'current':
                f_01 = self.current_frequency
            if freq_calc is 'flux':
                dac_channel = self.dac_channel
                flux_val = self.Flux_Control.get_flux(dac_channel)
                f_01 = self.calculate_frequency_flux(flux_val)
            elif freq_calc is 'dac':
                f_01 = self.calculate_frequency_dac()
        if f_start is None:
            if anharm_estimate is None:
                raise ValueError('anharm_estimate must be specified if f_start is not specified')
            f_12 = f_01-anharm_estimate
            f_start = f_12 - f_span/2.
            f_stop = f_12 + f_span/2.

        if not pulsed:
            self.prepare_for_CW(**kw)
        else:
            self.prepare_for_Pulsed_Spec(**kw)
        if source12 is None:
            source12 = self.qubit_drive_instrument
        else:
            source12 = qt.instruments[source12]

        self.AWG.start()

        if f_stop is None:
            raise ValueError('f_stop not specified')
        sweep_points = np.arange(f_start, f_stop, f_step)

        if pulsed is False:
            source = self.qubit_source_instrument
            self.qubit_drive_instrument.off()
            if source_power is None:
                source.set_power(self.get_source_power())
            else:
                source.set_power(source_power)
            if 'pulsemod_state' in list(source.get_parameters().keys()):
                self.source.set_pulsemod_state('OFF')
            source.set_frequency((f_01+f_01_detuning)*1e9)
            source.on()
            MC.set_sweep_function(swf.Source_frequency_GHz(Source=source12))
            MC.set_sweep_points(sweep_points)
            MC.set_detector_function(det.HomodyneDetector())
        elif pulsed is True:
            if use_AWG is True:
                print('pulsed-AWG mode has not yet been bug-tested')
                source = self.qubit_drive_instrument
                self.qubit_source_instrument.off()
                source.set_power(self.get_drive_power())
                source.set_frequency((f_01+f_01_detuning-self.sideband_modulation_frequency)*1e9)
                source.on()
                MC.set_sweep_function(
                    swf.Source_frequency_modulated_GHz(
                        Source=source12,
                        modulation_freq=self.sideband_modulation_frequency))
                MC.set_sweep_points(sweep_points)
                MC.set_detector_function(det.PulsedSpectroscopyDetector())

            else:
                source = self.qubit_source_instrument
                self.qubit_drive_instrument.off()
                if 'pulsemod_state' in list(source.get_parameters().keys()):
                    self.source.set_pulsemod_state('OFF')
                if source_power is None:
                    source.set_power(self.get_source_power())
                else:
                    source.set_power(source_power)
                source.set_frequency((f_01+f_01_detuning)*1e9)
                source.on()
                MC.set_sweep_function(swf.Source_frequency_GHz(Source=source12))
                MC.set_sweep_points(sweep_points)
                MC.set_detector_function(det.PulsedSpectroscopyDetector())

        source12.set_power(source.get_power()+source12_extra_power)
        source12.on()

        msmt_name = 'Three_Tone_Scan{}'.format(self.get_msmt_label_suffix('CW',
                                          **kw))
        # FIXME CW only if pulsed is false
        data = MC.run(name=msmt_name,
                      suppress_print_statements=suppress_print_statements,
                      **kw)

        self.AWG.stop()
        source.off()
        source12.off()

        so_a = MA.Qubit_Spectroscopy_Analysis(auto=True, label='Three_Tone_Scan',
                                              close_file=False)
        f_12, f_12_stderr = so_a.get_frequency_estimate()
        peaks = so_a.peaks
        f_12_linewidth = so_a.get_linewidth_estimate()
        so_a.finish()

        if not suppress_print_statements:
            print('f_12 frequency estimated to: "%s" GHz' % f_12)

        result = {'f_12': f_12, 'f_12_stderr': f_12_stderr,
                  'f_12_linewidth': f_12_linewidth,
                  'freqs': sweep_points, 'data': data, 'peaks': peaks}
        return result

    def measure_E_c(self,
                   MC_name='MC',
                   f_01_span=0.03,
                   f_01_step=0.1e-3,
                   f_12_span=0.02,
                   f_12_step=0.5e-3,
                   f_01=None,
                   f_01_detuning=0.,
                   anharm_estimate=None,
                   f_12=None,
                   suppress_print_statements=True,
                   source_power=None,
                   source12=None,
                   source12_extra_power=6,
                   freq_calc=None,
                   pulsed=False, use_AWG=False, **kw):
        '''
        Performs a two-tone 2D spectroscopy measurement to measure the f_01/f_12
        "cross" and, from the frequency of the qubit's 1-2 transition, estimate
        the E_c and E_j

        source_power, can be set to None, 'qubit' and value (float).
            None uses what is set in the qubit
            value, sets the source power to the value specified
        f_start=None
            If set to "None" uses the qubit object to calculate the value.
            if set to a value scans to "f_stop"

        freq_calc=None, determines the method used to calculate the freq.
            "current", uses "self.current_frequency"
            "flux", uses "self.calculate_frequency_flux"
            "dac", uses "self.calculate_frequency_dac"

        By default, uses AWG sequence named 'Spec_5014'
        '''
        MC = qt.instruments[MC_name]
        if f_01 is None:
            if freq_calc is None:
                freq_calc = self.freq_calc
            if freq_calc is 'current':
                f_01 = self.current_frequency
            if freq_calc is 'flux':
                dac_channel = self.dac_channel
                flux_val = self.Flux_Control.get_flux(dac_channel)
                f_01 = self.calculate_frequency_flux(flux_val)
            elif freq_calc is 'dac':
                f_01 = self.calculate_frequency_dac()
        f_01_start = f_01 - f_01_span/2.
        f_01_stop = f_01 + f_01_span/2.
        f_01_sweeppoints = np.arange(f_01_start, f_01_stop, f_01_step)
        if f_12 is None:
            if anharm_estimate is None:
                raise ValueError('anharm_estimate must be specified if f_12 is not specified')
            f_12 = f_01-anharm_estimate
        f_12_start = f_12 - f_12_span/2.
        f_12_stop = f_12 + f_12_span/2.
        f_12_sweeppoints = np.arange(f_12_start, f_12_stop, f_12_step)

        if not pulsed:
            self.prepare_for_CW(**kw)
        else:
            self.prepare_for_Pulsed_Spec(**kw)
        if source12 is None:
            source12 = self.qubit_drive_instrument
        else:
            source12 = qt.instruments[source12]

        self.AWG.start()

        if pulsed is False:
            source = self.qubit_source_instrument
            self.qubit_drive_instrument.off()
            if source_power is None:
                source.set_power(self.get_source_power())
            else:
                source.set_power(source_power)
            if 'pulsemod_state' in list(source.get_parameters().keys()):
                self.source.set_pulsemod_state('OFF')
            source.set_frequency((f_01+f_01_detuning)*1e9)
            source.on()
            MC.set_sweep_function(swf.Source_frequency_GHz(Source=source))
            MC.set_sweep_points(f_01_sweeppoints)
            MC.set_sweep_function_2D(swf.Source_frequency_GHz(Source=source12))
            MC.set_sweep_points_2D(f_12_sweeppoints)
            MC.set_detector_function(det.HomodyneDetector())
        elif pulsed is True:
            if use_AWG is True:
                print('pulsed-AWG mode has not yet been bug-tested')
                source = self.qubit_drive_instrument
                self.qubit_source_instrument.off()
                source.set_power(self.get_drive_power())
                source.set_frequency((f_01+f_01_detuning-self.sideband_modulation_frequency)*1e9)
                source.on()
                MC.set_sweep_function(
                    swf.Source_frequency_modulated_GHz(
                        Source=source,
                        modulation_freq=self.sideband_modulation_frequency))
                MC.set_sweep_points(f_01_sweeppoints)
                MC.set_sweep_function_2D(
                    swf.Source_frequency_modulated_GHz(
                        Source=source12,
                        modulation_freq=self.sideband_modulation_frequency))
                MC.set_sweep_points_2D(f_12_sweeppoints)
                MC.set_detector_function(det.PulsedSpectroscopyDetector())

            else:
                source = self.qubit_source_instrument
                self.qubit_drive_instrument.off()
                if 'pulsemod_state' in list(source.get_parameters().keys()):
                    self.source.set_pulsemod_state('OFF')
                if source_power is None:
                    source.set_power(self.get_source_power())
                else:
                    source.set_power(source_power)
                source.set_frequency((f_01+f_01_detuning)*1e9)
                source.on()
                MC.set_sweep_function(swf.Source_frequency_GHz(Source=source))
                MC.set_sweep_points(f_01_sweeppoints)
                MC.set_sweep_function_2D(swf.Source_frequency_GHz(Source=source12))
                MC.set_sweep_points_2D(f_12_sweeppoints)
                MC.set_detector_function(det.PulsedSpectroscopyDetector())

        source12.set_power(source.get_power()+source12_extra_power)
        source12.on()

        msmt_name = 'Three_Tone_Measure_E_c{}'.format(self.get_msmt_label_suffix('CW',
                                          **kw))
        # FIXME CW only if pulsed is false
        data = MC.run_2D(name=msmt_name,
                      suppress_print_statements=suppress_print_statements,
                      **kw)

        self.AWG.stop()
        source.off()
        source12.off()

        # so_a = MA.Qubit_Spectroscopy_Analysis(auto=True, label='Three_Tone_Scan',
        #                                       close_file=False)
        # f_12, f_12_stderr = so_a.get_frequency_estimate()
        # peaks = so_a.peaks
        # f_12_linewidth = so_a.get_linewidth_estimate()
        # so_a.finish()

        # if not suppress_print_statements:
        #     print 'f_12 frequency estimated to: "%s" GHz' % f_12

        # result = {'f_12': f_12, 'f_12_stderr': f_12_stderr,
        #           'f_12_linewidth': f_12_linewidth,
        #           'freqs': sweep_points, 'data': data, 'peaks': peaks}
        # return result
        return

    def spectroscopy_power_sweep(self,
                                 power_start, power_stop, power_step,
                                 f_step, f_start, f_stop, f_span=0.001,
                                 f_calc=None,
                                 pulsed=False,
                                 MC_name='MC', **kw):
        '''
        A 2D sweep around the current_frequency at varied source_power.
        This function does not perform any fancy analysis yet.

        Ideally it would set the qubit frequency to the frequecy at the lowest
        power where it still has some signal.

        The pulsed mode of this function has not been tested yet.
        '''
        MC = qt.instruments[MC_name]

        if f_start is None:
            if f_calc is None:
                f_calc = self.f_calc
            if f_calc is 'current':
                calculated_frequency = self.current_frequency
            if f_calc is 'flux':
                dac_channel = self.dac_channel
                flux_val = self.Flux_Control.get_flux(dac_channel)
                calculated_frequency = self.calculate_frequency_flux(flux_val)
            elif f_calc is 'dac':
                dac_channel = self.dac_channel
                dac_val = self.IVVI.get_dac(dac_channel)
                calculated_frequency = self.calculate_frequency_dac(dac_val)
            f_start = calculated_frequency - f_span/2.
            f_stop = calculated_frequency + f_span/2.

        if not pulsed:
            self.prepare_for_CW(**kw)
        else:
            self.prepare_for_Pulsed_Spec(**kw)

        self.AWG.start()

        if f_stop is None:
            raise ValueError('f_stop not specified')

        freq_sweep_points = np.arange(f_start, f_stop, f_step)
        power_sweep_points = np.arange(power_start, power_stop, power_step)

        if pulsed is False:
            source = self.qubit_source_instrument
            source.set_power(self.get_source_power())
            source.on()
            MC.set_sweep_function(swf.Source_frequency_GHz(Source=source))
            MC.set_sweep_points(freq_sweep_points)
            MC.set_sweep_function_2D(swf.Source_power_dBm(Source=source))
            MC.set_sweep_points_2D(power_sweep_points)
            MC.set_detector_function(det.HomodyneDetector())
        elif pulsed is True:
            source = self.qubit_drive_instrument
            source.set_power(self.get_drive_power())
            source.on()
            print('source', source)
            print('sideband', self.sideband_modulation_frequency)
            MC.set_sweep_function(
                swf.Source_frequency_modulated_GHz(
                    Source=source,
                    modulation_freq=self.sideband_modulation_frequency))
            MC.set_sweep_points(freq_sweep_points)
            MC.set_sweep_function_2D(swf.Source_power_dBm(Source=source))
            MC.set_sweep_points_2D(power_sweep_points)
            MC.set_detector_function(det.PulsedSpectroscopyDetector())

        self.HM.set_sources('On')
        MC.run_2D('Spectroscopy_Powersweep_'+self.get_name(),**kw)
        self.HM.set_sources('Off')

        tta = MA.TwoD_Analysis(auto=True, normalize=True)

        return True

    def find_frequency_Ramsey(self,
                              stepsizes=[10, 30, 100, 200, 500],
                              update_qubit=True,
                              suppress_print_statements=False, **kw):
        '''
        Optimizes the qubit frequency and measures T2 star.
        It does this by performing a Ramsey experiment with an artificial added
        phase for increasing ranges of interpulse delay.
        By fitting a frequency to the oscillation a detuning is extracted,
        giving an estimate for the qubit frequency.
        This is used in the next timestamp.
        When the range is larger than 3.5 * T2star the next timestep is not
        executed.
        '''

        frequency = self.current_frequency
        for k, stepsize in enumerate(stepsizes):
            TD_a, applied_detuning = self.measure_Ramsey(
                drive_frequency=frequency, stepsize=stepsize, **kw)
            detuning = TD_a.total_detuning - abs(applied_detuning*1e-9)
            if self.pulse_amp_control == 'CBox':
                print('Frequency was %.4f GHz' % frequency)
                frequency -= detuning
                print('Fitted detuning was: %.4f GHz' % TD_a.total_detuning)
                print('Applied detuning was: %.4f GHz' % (applied_detuning*1e-9))
                print('Frequency set to %.4f GHz' % frequency)
            else:
                # Detuning is set in the different direction in the FPGA seq
                # print 'adding detuning to freq'
                print('Frequency was %.4f GHz' % frequency)
                frequency += detuning
                print('Fitted detuning was: %.4f GHz' % TD_a.total_detuning)
                print('Applied detuning was: %.4f GHz' % (applied_detuning*1e-9))
                print('Frequency set to %.4f GHz' % frequency)



            T2_star = TD_a.T2_star
            next_stepsize = stepsizes[min(k+1, len(stepsizes)-1)]

            if next_stepsize*60 > T2_star*3:  # all in ns
                if not suppress_print_statements:
                    print('range={} > 3*tau={}, breaking loop'.format(
                        next_stepsize*60, 3*T2_star))
                break

            if not suppress_print_statements:
                print('Frequency estimated to {:.9} GHz, ' \
                    'Moving to next stepsize.'.format(frequency))

        if not suppress_print_statements:
            print('Frequency calibration done.')
            print('Frequency estimated to {:.9} GHz'.format(frequency))

        if update_qubit:
            self.set_current_frequency(frequency)

        result = {'frequency': frequency,
                  'frequency_stderr': TD_a.detuning_stderr,
                  'T2_star': T2_star,
                  'T2_star_stderr': TD_a.T2_star_stderr,
                  'stepsize': stepsize}
        return result

    def find_resonator_qubit_frequency(self, **kw):
        '''
        arguments prepended with "resonator_" get passed to
        "find_resonator_frequency"
        arguments prepended with "qubit_" get passed to "find_frequency_spec"

        Look at individual docstrings to see what input argumetns are required.
        '''
        print('Measuring resonator')
        resonator_kw = {key.split('resonator_')[1]: val
                        for key, val in list(kw.items()) if 'resonator_' in key}
        resonator_kw.update(kw)

        resonator_res = self.find_resonator_frequency(**resonator_kw)

        self.HM.set_frequency(self.current_RO_frequency*1e9)

        print('Measuring qubit')
        qubit_kw = {key.split('qubit_')[1]: val
                    for key, val in list(kw.items()) if 'qubit_' in key}
        qubit_kw.update(kw)
        qubit_res = self.find_frequency_spec(**qubit_kw)

        result = {}
        result['resonator_freqs'] = resonator_res.pop('freqs')
        result['resonator_data'] = resonator_res.pop('data')
        result['f_resonator'] = resonator_res.pop('f_resonator')
        result['qubit_freqs'] = qubit_res.pop('freqs')
        result['qubit_data'] = qubit_res.pop('data')
        result['f_qubit'] = qubit_res.pop('f_qubit')
        result = dict(list(result.items()) + list(resonator_res.items()) +
                      list(qubit_res.items()))
        return result

    def tracked_spectroscopy(self, dac_range, dac_channel=None,
                             qubit_initial_frequency=None,
                             pulsed=False,
                             qubit_init_factor=3,
                             qubit_stepsize=2e-3,
                             qubit_span=3e-2,
                             resonator_stepsize=1.5e-4,
                             resonator_span=1e-2,
                             MC_name='MC',
                             resonator_use_min=False,
                             resonator_use_max=False,
                             fitting_model='hanger'):
        '''
        Lot's to improved on suggestions should be placed in issue #118
        '''
        if dac_channel is None:
            dac_channel = self.get_dac_channel()

        if qubit_initial_frequency is None:
            # Tracked spec is used when the calculation curves are not known.
            qubit_initial_frequency = self.current_frequency

        resonator_initial_frequency = self.get_current_RO_frequency()
        MC = qt.instruments[MC_name]
        MC.set_sweep_function(swf.Bias_Dac_mV(dac_channel=dac_channel))
        MC.set_sweep_points(dac_range)
        MC.set_detector_function(cdet.Tracked_Qubit_Spectroscopy(
            qubit=self,
            qubit_initial_frequency=qubit_initial_frequency,  # None
            qubit_stepsize=qubit_stepsize,  # .5e-4
            qubit_span=qubit_span,
            qubit_init_factor=qubit_init_factor,
            resonator_initial_frequency=resonator_initial_frequency,
            resonator_stepsize=resonator_stepsize,
            resonator_span=resonator_span,
            sweep_points=dac_range,
            pulsed=pulsed,
            resonator_use_min=resonator_use_min,
            resonator_use_max=resonator_use_max,
            fitting_model=fitting_model))
        MC.run(print_sweep_points=True)
        MA.MeasurementAnalysis(auto=True)

    def measure_Rabi(self, prepare=True, MC_name='MC', drive_amplitude=None,
                     debug_mode=False, **kw):
        if prepare:
            self.prepare_for_TD(**kw)
        MC = qt.instruments[MC_name]

        if self.pulse_amp_control == 'CBox':
            self.CBox.set_awg_mode(0, 0)
            self.CBox.set_awg_mode(1, 0)
            self.AWG.set_setup_filename('FPGA_Codeword_Rabi_5014',
                                        force_load=False)
            MC.set_sweep_function(CB_swf.Lut_man_amp180_90(
                                  reload_pulses=True))
            if drive_amplitude is None:
                pulse_amp = self.pulse_amplitude_I
            else:
                pulse_amp = drive_amplitude
                # To make it compatible with the calibration
            # Sweep points need to be set explicitly for a CBox Rabi.
            Rabi_segments = 20
            NoCalPoints = 4
            Rabi_points = np.linspace(-1.5*pulse_amp,
                                      1.5*pulse_amp,
                                      Rabi_segments)
            cal_points = [0]*int(NoCalPoints/2) + [pulse_amp]*int(NoCalPoints/2)

            # Reversion is to make low freq drift visible
            even_Rabi_points = Rabi_points[::2]
            reverted_odd_Rabi_points = Rabi_points[-1::-2]

            sweep_points = np.concatenate((even_Rabi_points,
                                          reverted_odd_Rabi_points,
                                          cal_points), axis=0)
            MC.set_sweep_points(sweep_points)
        else:
            NoCalPoints = 10
            is_Duplexer = self.pulse_amp_control is 'Duplexer'

            MC.set_sweep_function(awg_swf.Rabi(Duplexer=is_Duplexer,
                                  gauss_width=self.gauss_width,
                                  qubit_suffix=self.qubit_suffix))

        if self.TD_Meas.get_multiplex():
            MC.set_detector_function(det.TimeDomainDetector_multiplexed())
        elif (self.data_acquistion == 'CBox' and
              self.pulse_amp_control == 'CBox'):
            MC.set_detector_function(
                det.QuTechCBox_integrated_average_single_trace_Detector())
        elif (self.data_acquistion == 'CBox' and
              self.pulse_amp_control != 'CBox'):
            MC.set_detector_function(
                det.QuTechCBox_integrated_average_Detector())
        elif (self.data_acquistion == 'ATS' and
              self.pulse_amp_control == 'CBox'):
            MC.set_detector_function(
                det.timedomain_single_trace_detector())
        else:
            MC.set_detector_function(det.TimeDomainDetector())

        msmt_name = 'Rabi{}'.format(self.get_msmt_label_suffix('TD', **kw))
        MC.run(name=msmt_name, suppress_print_statements=True,
               debug_mode=debug_mode)
        self.TD_a = MA.Rabi_Analysis(auto=True, NoCalPoints=NoCalPoints)
        return self.TD_a

    def measure_T1(self, prepare=True, MC_name='MC', stepsize=1000,
                   debug_mode=False, **kw):
        if prepare:
            self.prepare_for_TD(**kw)
        MC = qt.instruments[MC_name]
        is_Duplexer = self.pulse_amp_control is 'Duplexer'
        MC.set_sweep_function(awg_swf.T1(Duplexer=is_Duplexer,
                              gauss_width=self.gauss_width,
                              qubit_suffix=self.qubit_suffix,
                              stepsize=stepsize, **kw))
        if self.TD_Meas.get_multiplex():
            MC.set_detector_function(det.TimeDomainDetector_multiplexed_cal)
        elif self.data_acquistion == 'CBox':
            MC.set_detector_function(det.QuTechCBox_integrated_average_Detector())
        else:
            MC.set_detector_function(det.TimeDomainDetector())
        msmt_name = 'T1_{}{}'.format(stepsize, self.get_msmt_label_suffix('TD', **kw))
        MC.run(name=msmt_name, suppress_print_statements=True,
               debug_mode=debug_mode)

        self.TD_a = MA.T1_Analysis(auto=True)
        return self.TD_a

    def measure_Echo(self, prepare=True, MC_name='MC', stepsize=100,
                     debug_mode=False, **kw):
        if prepare:
            self.prepare_for_TD(**kw)
        MC = qt.instruments[MC_name]
        is_Duplexer = self.pulse_amp_control is 'Duplexer'
        MC.set_sweep_function(awg_swf.Echo(Duplexer=is_Duplexer,
                              gauss_width=self.gauss_width,
                              qubit_suffix=self.qubit_suffix,
                              stepsize=stepsize))
        if self.TD_Meas.get_multiplex():
            MC.set_detector_function(det.TimeDomainDetector_multiplexed_cal)
        elif self.data_acquistion == 'CBox':
            MC.set_detector_function(det.QuTechCBox_integrated_average_Detector)
        else:
            MC.set_detector_function(det.TimeDomainDetector())
        msmt_name = 'Echo_{}{}'.format(stepsize,
                                       self.get_msmt_label_suffix('TD', **kw))
        MC.run(name=msmt_name, suppress_print_statements=True,
               debug_mode=debug_mode)
        self.TD_a = MA.Ramsey_Analysis(auto=True, label='Echo')
        return self.TD_a

    def measure_Ramsey(self, prepare=True, MC_name='MC', stepsize=100,
                       debug_mode=False, **kw):
        if prepare:
            self.prepare_for_TD(**kw)
        MC = qt.instruments[MC_name]
        f_old = self.qubit_drive_instrument.get_frequency()

        # Detuning is chosen such that it shows 4 oscillations
        # It is set in the tektronix sequences.

        if self.pulse_amp_control == 'CBox':
            # Adds 5ns per stepsize
            applied_detuning = (self.sideband_modulation_frequency*1e9*5e-9) / \
                ((stepsize+5)*1e-9)
            print('Adding artificial detuning of "%.4f" MHz' % (
                applied_detuning*1e-6))

            MC.set_sweep_function(CB_swf.Ramsey_tape(reload_pulses=True,
                                  stepsize=stepsize))
        else:
            applied_detuning = 4./(60*stepsize*1e-9)
            print('Adding artificial detuning of "%.4f" MHz' % (
                applied_detuning*1e-6))
            is_Duplexer = self.pulse_amp_control is 'Duplexer'

            MC.set_sweep_function(awg_swf.Ramsey(Duplexer=is_Duplexer,
                                  gauss_width=self.gauss_width,
                                  qubit_suffix=self.qubit_suffix,
                                  stepsize=stepsize))

        if self.TD_Meas.get_multiplex():
            MC.set_detector_function(det.TimeDomainDetector_multiplexed_cal)
        elif self.data_acquistion == 'CBox':
            MC.set_detector_function(
                det.QuTechCBox_integrated_average_Detector)
        else:
            MC.set_detector_function(det.TimeDomainDetector())
        msmt_name = 'Ramsey_{}{}'.format(
            stepsize, self.get_msmt_label_suffix('TD', **kw))
        MC.run(name=msmt_name, suppress_print_statements=True,
               debug_mode=debug_mode)
        self.TD_a = MA.Ramsey_Analysis(auto=True)
        self.qubit_drive_instrument.set_frequency(f_old)
        return self.TD_a, applied_detuning

    def measure_AllXY(self, prepare=True, MC_name='MC', sequence='AllXY',
                      debug_mode=False, subsequence_sufix='', **kw):
        MC = qt.instruments[MC_name]
        if prepare:
            self.prepare_for_TD(**kw)
        MC = qt.instruments[MC_name]
        if self.pulse_amp_control == 'CBox':
            MC.set_sweep_function(CB_swf.AllXY_tape())
        else:
            is_Duplexer = self.pulse_amp_control is 'Duplexer'
            MC.set_sweep_function(awg_swf.AllXY(Duplexer=is_Duplexer,
                                  gauss_width=self.gauss_width,
                                  qubit_suffix=self.qubit_suffix,
                                  subsequence_sufix=subsequence_sufix,
                                  sequence=sequence))
        if self.TD_Meas.get_multiplex():
            MC.set_detector_function(det.TimeDomainDetector_multiplexed_cal)

        elif self.data_acquistion == 'CBox':
            MC.set_detector_function(det.QuTechCBox_integrated_average_Detector())
        else:
            MC.set_detector_function(det.TimeDomainDetector())
        msmt_name = 'AllXY{}'.format(self.get_msmt_label_suffix('TD', **kw))
        MC.run(name=msmt_name, suppress_print_statements=True,
               debug_mode=debug_mode)
        self.TD_a = MA.AllXY_Analysis(auto=True)
        return self.TD_a

    def measure_RB(self, seed, mode="", num_Cliffords=700,
                   prepare=True, MC_name='MC', debug_mode=False, **kw):
        '''
        Measures the randomized benchmarking sequence on a qubit.
        docstring stil incomplete... (FIXME)
        '''
        if prepare:
            self.prepare_for_TD(**kw)
        MC = qt.instruments[MC_name]
        is_Duplexer = self.pulse_amp_control is 'Duplexer'
        MC.set_sweep_function(awg_swf.RB(seed=seed,
                              num_Cliffords=num_Cliffords,
                              Duplexer=is_Duplexer,
                              mode=mode,
                              gauss_width=self.gauss_width,
                              qubit_suffix=self.qubit_suffix,
                              **kw))
        if self.TD_Meas.get_multiplex():
            MC.set_detector_function(det.TimeDomainDetector_multiplexed_cal)
        elif self.data_acquistion == 'CBox':
            MC.set_detector_function(det.QuTechCBox_integrated_average_Detector)
        else:
            MC.set_detector_function(det.TimeDomainDetector())
        msmt_name = 'RB{}{}_{}{}'.format(
            mode, num_Cliffords, seed, self.get_msmt_label_suffix('TD', **kw))
        MC.run(name=msmt_name, suppress_print_statements=True,
               debug_mode=debug_mode)
        self.TD_a = MA.MeasurementAnalysis(auto=True)
        return self.TD_a

    def measure_drive_detuning(self, prepare=True, MC_name='MC',
                               debug_mode=False, **kw):
        if prepare:
            self.prepare_for_TD(**kw)
        MC = qt.instruments[MC_name]
        if self.pulse_amp_control == 'CBox':
            MC.set_sweep_function(CB_swf.flipping_sequence())
        else:
            is_Duplexer = self.pulse_amp_control is 'Duplexer'
            MC.set_sweep_function(awg_swf.PiX360(
                                  Duplexer=is_Duplexer,
                                  gauss_width=self.gauss_width,
                                  qubit_suffix=self.qubit_suffix))
        if self.TD_Meas.get_multiplex():
            MC.set_detector_function(det.TimeDomainDetector_multiplexed_cal)
        elif self.data_acquistion == 'CBox':
            MC.set_detector_function(
                det.QuTechCBox_integrated_average_Detector)
        else:
            MC.set_detector_function(det.TimeDomainDetector())
        msmt_name = 'DriveDetuning{}'.format(
            self.get_msmt_label_suffix('TD', **kw))
        MC.run(name=msmt_name, suppress_print_statements=True,
               debug_mode=debug_mode)
        self.TD_a = MA.DriveDetuning_Analysis(auto=True)
        return self.TD_a

    def measure_drag_detuning(self, prepare=True, MC_name='MC',
                              debug_mode=False, **kw):
        if prepare:
            self.prepare_for_TD(**kw)
        MC = qt.instruments[MC_name]
        if self.pulse_amp_control is 'CBox':
            MC.set_sweep_function(CB_swf.drag_detuning())
        else:
            is_Duplexer = self.pulse_amp_control is 'Duplexer'
            MC.set_sweep_function(awg_swf.DragDetuning(
                                  Duplexer=is_Duplexer,
                                  gauss_width=self.gauss_width,
                                  qubit_suffix=self.qubit_suffix))
        if self.TD_Meas.get_multiplex():
            MC.set_detector_function(det.TimeDomainDetector_multiplexed_cal)
        elif self.data_acquistion == 'CBox':
            MC.set_detector_function(det.QuTechCBox_integrated_average_Detector)
        else:
            MC.set_detector_function(det.TimeDomainDetector())
        msmt_name = 'DragDetuning{}'.format(self.get_msmt_label_suffix('TD', **kw))
        MC.run(name=msmt_name, suppress_print_statements=True,
               debug_mode=debug_mode)
        self.TD_a = MA.OnOff_Analysis(auto=True, label='Drag')
        return self.TD_a

    def measure_Sequence(self, sequence_name, prepare=True,
                         add_filename_tags=True, NoSegments=None,
                         MC_name='MC', debug_mode=False, msmt_subname='',
                         qubit_suffix=None, **kw):
        '''
        Measures the AWG sequence set with "sequence_name=". The gauss_width
        and AWG type get added to the name automatically.
        '''
        if qubit_suffix is None:
            qubit_suffix = self.qubit_suffix
        if prepare:
            self.prepare_for_TD(**kw)
        MC = qt.instruments[MC_name]
        is_Duplexer = self.pulse_amp_control is 'Duplexer'
        MC.set_sweep_function(awg_swf.AWG_Sweep_File(sequence_name,
                              Duplexer=is_Duplexer,
                              gauss_width=self.gauss_width,
                              NoSegments=NoSegments,
                              qubit_suffix=qubit_suffix,
                              add_filename_tags=add_filename_tags,
                              cal_points=kw.get('cal_points', 10)))
        if self.TD_Meas.get_multiplex():
            if 'cal_points' in list(kw.keys()):
                MC.set_detector_function(det.TimeDomainDetector_multiplexed_cal)
            else:
                MC.set_detector_function(det.TimeDomainDetector_multiplexed)
        else:
            if 'cal_points' in list(kw.keys()):
                MC.set_detector_function(det.TimeDomainDetector())
            elif self.data_acquistion == 'CBox':
                MC.set_detector_function(det.QuTechCBox_integrated_average_Detector)
            else:
                MC.set_detector_function(det.TimeDomainDetector())
        msmt_name = sequence_name + msmt_subname + self.get_msmt_label_suffix('TD', **kw)
        MC.run(name=msmt_name, suppress_print_statements=True,
               debug_mode=debug_mode, gauss_width=self.gauss_width,
               qubit_suffix=qubit_suffix)
        self.TD_a = MA.MeasurementAnalysis(auto=True)
        return self.TD_a

    def measure_SSRO_fidelity(self, MC_name='MC', NoSamples=5000, no_fits=False,
                              add_filename_tags=True, nr_measurements=1,
                              qubit_suffix=""):
        '''
        Currently for CBox and ATS
        NoSamples only works for CBox. If you want to change the number of shots
        when measuring with the ATS, you should set TD_Meas NoSweeps
        '''
        imp.reload(MA)
        self.nr_measurements=nr_measurements
        self.prepare_for_TD()
        MC = qt.instruments[MC_name]
        MC.set_sweep_function(awg_swf.OnOff(qubit_suffix=qubit_suffix,
                              gauss_width=self.gauss_width,
                              nr_segments=2, nr_measurements=self.nr_measurements))
        # Because not relevant cause box is driving and this exists
        # self.gauss_width))
        if self.data_acquistion == 'CBox':
            MC.set_detector_function(
            det.QuTechCBox_AlternatingShots_Streaming_Detector(
                NoSamples=NoSamples))
        elif self.data_acquistion == 'ATS':
            self.TD_Meas.set_shot_mode(True)

            print(self.TD_Meas.get_shot_mode())
            if self.TD_Meas.get_multiplex():
                MC.set_detector_function(det.TimeDomainDetector_multiplexed_cal)
            else:
                MC.set_detector_function(det.TimeDomainDetector())
        else:
            raise Exception('data_acquistion "%s" not recognized' %
                            self.data_acquistion)

        if nr_measurements == 1:
            MC.run('SSRO_char')
        if nr_measurements == 2:
            MC.run('SSRO_char_2')

        t0 =time.time()
        ana = MA.SSRO_Analysis(auto=True, close_file=True,
                               label='SSRO_char',no_fits=no_fits)
        print('analyzing took %.2f' %((time.time() -t0)))

        self.TD_Meas.set_shot_mode(False)
        return

    def calibrate_drag_amplitude(self, initial_drag_amplitude=None,
                                 drag_amplitude_ranges=[40000, 10000, 2000],
                                 points_per_sweep=5,
                                 calibrate_drive_amplitude=True,
                                 update=True, **kw):

        if initial_drag_amplitude is None:
            initial_drag_amplitude = self.pulse_amplitude_Q

        if self.TD_Meas.get_multiplex():
            raise NameError('Timedomain is set to multiplexed mode')
        if self.get_pulse_amp_control() != 'Duplexer':
            raise NameError('Currently only Duplexer mode is supported')

        best_drag_amplitude = initial_drag_amplitude
        for k, drag_amplitude_range in enumerate(drag_amplitude_ranges):
            if calibrate_drive_amplitude:
                self.calibrate_drive_amplitude_flipping(Navg=1,
                    drag_amplitude=best_drag_amplitude)
            contrast_arr = np.zeros(points_per_sweep)
            drag_amplitudes = np.linspace(max(0, best_drag_amplitude -
                                            drag_amplitude_range / 2),
                                        min(50000, best_drag_amplitude +
                                            drag_amplitude_range / 2),
                                        points_per_sweep)
            for kk, drag_amplitude in enumerate(drag_amplitudes):
                print('\nMeasuring drag {}'.format(drag_amplitude))
                ma = self.measure_drag_detuning(drag_amplitude=drag_amplitude,
                                                **kw)
                contrast_arr[kk] = ma.contrast
            best_drag_amplitude = drag_amplitudes[np.argmin(contrast_arr)]
            print('Best drag amplitude found at {}'.format(best_drag_amplitude))
        print('Finished calibrating drag amplitude')
        if update:
            print('Updating qubit drag amplitude')
            self.set_pulse_amplitude_Q(best_drag_amplitude)

    def calibrate_drive_amplitude(self, **kw):
        self.calibrate_drive_amplitude_Rabi(**kw)
        self.calibrate_drive_amplitude_flipping(**kw)

    def calibrate_drive_amplitude_Rabi(self, max_iterations=5,
                                       desired_accuracy=.05,
                                       update_qubit=True,
                                       suppress_print_statements=False,
                                       initial_pulse_amplitude=None,
                                       **kw):
        '''
        Calibrates the amplitude of AWG pulses based on a predefined Rabi
        sequence.
        '''
        if self.TD_Meas.get_multiplex():
            raise NameError('Timedomain is set to multiplexed mode')

        if initial_pulse_amplitude is None:
            pulse_amplitude_I = self.pulse_amplitude_I
            pulse_amplitude_Q = self.pulse_amplitude_Q
        else:
            pulse_amplitude_I = initial_pulse_amplitude
            pulse_amplitude_Q = initial_pulse_amplitude

        max_AWG_amp = 1.5  # Hardcoded in the AWG; safety for the IQ mixers.

        for i in range(max_iterations):
            self.set_pulse_amplitude_I(pulse_amplitude_I)
            self.set_pulse_amplitude_Q(pulse_amplitude_Q)

            TD_a = self.measure_Rabi(**kw)

            drive_scaling_factor = TD_a.drive_scaling_factor
            if not suppress_print_statements:
                print('drive_scaling_factor {:.3}'.format(drive_scaling_factor))

            if self.pulse_amp_control == 'AWG':
                pulse_amplitude_I *= drive_scaling_factor
                pulse_amplitude_Q *= drive_scaling_factor
                if pulse_amplitude_Q > max_AWG_amp:
                    pulse_amplitude_I = 1.5
                    pulse_amplitude_Q = 1.5
                    logging.warning('Trying to set to high pulse_amplitude' )
                    break
            elif self.pulse_amp_control == 'Duplexer':
                pulse_amplitude_I = self.Dupl.calculate_attenuation(
                    pulse_amplitude_I,
                    drive_scaling_factor)
            elif self.pulse_amp_control == 'CBox':
                pulse_amplitude_I *= drive_scaling_factor
                if pulse_amplitude_I > 1000:
                    pulse_amplitude_I = 1000
                    logging.warning('Trying to set to high pulse_amplitude' )
                    break

            if abs(1-drive_scaling_factor) < desired_accuracy:
                break

        if abs(1-drive_scaling_factor) > desired_accuracy:
            logging.error('Rabi Amplitude Calibration did not converge')
        elif not suppress_print_statements:
            print('Rabi Amplitude Calibration converged to: "%s" and "%s"' % (
                pulse_amplitude_I, pulse_amplitude_Q))

        if update_qubit:
            self.set_pulse_amplitude_I(pulse_amplitude_I)
            self.set_pulse_amplitude_Q(pulse_amplitude_Q)
            if self.pulse_amp_control == 'CBox':
                self.CBox_lut_man.set_amp180(pulse_amplitude_I*1000)
                self.CBox_lut_man.set_amp90(pulse_amplitude_I*1000/2.0)

        return pulse_amplitude_I, pulse_amplitude_Q

    def calibrate_drive_amplitude_flipping(self, max_iterations=5,
                                           desired_accuracy=.001,
                                           update_qubit=True,
                                           suppress_print_statements=False,
                                           **kw):
        '''
        Calibrates the amplitude of AWG pulses based on a predefined Rabi
        sequence.
        '''

        if self.TD_Meas.get_multiplex():
            raise NameError('Timedomain is set to multiplexed mode')

        if not suppress_print_statements:
            print('Accurately calibrating drive amplitude using '\
                'PiX360 pulse sequence')

        self.prepare_for_TD(**kw)
        for k in range(max_iterations):
            if self.pulse_amp_control == 'Duplexer':
                drive_ampl = self.Dupl.get_attenuation(
                    1, self.duplexer_output_channel)
            else:
                pulse_amplitude_I = self.pulse_amplitude_I
                drive_ampl = pulse_amplitude_I

            TD_a = self.measure_drive_detuning(**kw)
            drive_scaling_factor = TD_a.drive_scaling_factor

            if not suppress_print_statements:
                print('drive_scaling_factor', drive_scaling_factor)

            # Check if drive_scaling_factor is within boundaries
            if drive_scaling_factor > 1.1:
                drive_scaling_factor = 1.1
                if not suppress_print_statements:
                    print('Qubit drive scaling %.3f ' % drive_scaling_factor \
                        + 'is too high, capping at 1.1')
            elif drive_scaling_factor < 0.9:
                drive_scaling_factor = 0.9
                if not suppress_print_statements:
                    print('Qubit drive scaling %.3f ' % drive_scaling_factor \
                        + 'is too low, capping at 0.9')

            if (self.pulse_amp_control == 'AWG' or
                    self.pulse_amp_control == 'CBox'):
                self.pulse_amplitude_I *= drive_scaling_factor
                self.pulse_amplitude_Q *= drive_scaling_factor
            elif self.pulse_amp_control == 'Duplexer':
                pulse_amplitude_I = self.Dupl.calculate_attenuation(
                    drive_ampl, drive_scaling_factor)
                self.Dupl.set_attenuation(
                    1, self.duplexer_output_channel,
                    pulse_amplitude_I)

            if abs(drive_scaling_factor - 1) < desired_accuracy:
                if not suppress_print_statements:
                    print('within threshold')
                break

        # If converged?
        print('Drive calibration set to {}'.format(drive_ampl))
        if update_qubit:
            self.set_pulse_amplitude_I(drive_ampl)

    def calibrate_mixer_offsets(self, AWG_nr=0, **kw):
        '''
        Calibrates the mixer offsets by minimizing the leakage signal
        at the generator_frequency.
        '''

        # Turn on sources
        self.qubit_drive_instrument.set_frequency(self.generator_frequency*1e9)
        self.qubit_drive_instrument.on()
        if self.pulse_amp_control == 'AWG':
            Ch_I_offset, Ch_Q_offset = cal_tools.mixer_carrier_cancellation(
                frequency=self.generator_frequency,
                AWG_name=self.get_AWG_source(),
                voltage_grid=[.1, 0.05, 0.02, 0.01],
                x_tol=0.001,
                AWG_channel1=self.AWG_ch_I,
                AWG_channel2=self.AWG_ch_Q)
        elif self.pulse_amp_control == 'CBox':
            Ch_I_offset, Ch_Q_offset = cal_tools.mixer_carrier_cancellation(
                frequency=self.generator_frequency,
                AWG_name='AWG', pulse_amp_control=self.pulse_amp_control,
                voltage_grid=[100, 50, 20],
                x_tol=1,
                AWG_channel1=1,
                AWG_channel2=0)
            self.CBox.set_dac_offset(AWG_nr, 1, Ch_I_offset)
            self.CBox.set_dac_offset(AWG_nr, 0, Ch_Q_offset)

        print('Mixer offset calibration converged to: Ch_I_offset = %.3f, Ch_Q_offset = %.3f' %(
            Ch_I_offset, Ch_Q_offset))
        self.qubit_drive_instrument.off()

    def calibrate_mixer_skewness(self,
                                 estimated_IQ_phase_skewness=0,
                                 estimated_QI_amp_ratio=1,
                                 mixers=[1, 2], **kw):
        pulse_amp_control = self.get_pulse_amp_control()
        allowed_pulse_amp_controls = ['CBox', 'Duplexer', 'AWG']
        if pulse_amp_control not in allowed_pulse_amp_controls:
            raise NameError('Currently only CBox and Duplexer mode supported')
        phase_min = np.zeros(2)
        ampl_min = np.zeros(2)
        power_min = np.zeros(2)

        self.qubit_drive_instrument.on()
        if pulse_amp_control == 'Duplexer':
            for k, mixer in enumerate(mixers):
                print('Calibrating mixer {}'.format(mixer))

                self.Dupl.set_all_switches_to('off')
                self.Dupl.set_switch(mixer, self.get_duplexer_output_channel(),
                                     'on')
                self.Dupl.set_attenuation(mixer,
                                          self.get_duplexer_output_channel(),
                                          50000)
                AWG_channel = 2*mixer - 1

                phase_min[k], ampl_min[k] = \
                    cal_tools.mixer_skewness_calibration(
                        self.generator_frequency,
                        estimated_IQ_phase_skewness=estimated_IQ_phase_skewness,
                        estimated_QI_amp_ratio=estimated_QI_amp_ratio,
                        sideband_frequency=self.get_sideband_modulation_frequency(),
                        AWG_channel=AWG_channel,
                        AWG_name=self.get_AWG_source(),
                        pulse_amp_control=pulse_amp_control,
                        drive_name=self.get_qubit_drive(), **kw)
            for k, mixer in enumerate([1, 2]):
                print('\n' + '*'*30)
                print('Final values mixer {}: phase: {}, amplitude: {}'\
                    .format(mixer, phase_min[k], ampl_min[k]))

            self.qubit_drive_instrument.off()
            return phase_min, ampl_min

        elif pulse_amp_control is 'CBox' or pulse_amp_control is 'AWG':
            print('Calibrating mixer using CBox')
            phi, alpha = \
                cal_tools.mixer_skewness_calibration_adaptive(
                    source=self.qubit_drive_instrument,
                    generator_frequency=self.generator_frequency,
                    sideband_frequency=self.get_sideband_modulation_frequency(),
                    pulse_amp_control=pulse_amp_control, **kw)

            print('\n' + '*'*30)
            print('Final values correction matrix: phi: {}, alpha: {}'\
                .format(phi, alpha))
            self.qubit_drive_instrument.off()
            return phi, alpha

    def move_to_dac_value(self, dac_value):
        self.IVVI.set_dac(self.dac_channel, dac_value)
        self.set_current_dac_value(dac_value)

    def tune_to_frequency_spec(self, target_frequency,
                               threshold_frequency=1e-3, max_iterations=4,
                               freq_calc=None, **kw):
        suppress_print_statements = kw.get('suppress_print_statements', False)
        if isinstance(target_frequency, types.InstanceType):
            target_frequency = target_frequency.get_current_frequency()
        if freq_calc is None:
            freq_calc = self.freq_calc
        if freq_calc is 'flux':
            flux_val = self.calculate_flux_frequency(target_frequency)
            print('Setting flux to', flux_val)
            self.Flux_Control.set_flux(self.dac_channel, flux_val)
        elif freq_calc is 'dac':
            dac_val = self.calculate_dac_frequency(target_frequency)
            self.IVVI.set_dac(self.dac_channel, dac_val)

        for k in range(max_iterations):
            result = self.find_resonator_qubit_frequency(**kw)

            f_difference = target_frequency - result['f_qubit']
            if not suppress_print_statements:
                print('Current frequency difference: %.9f GHz' % f_difference)
            if abs(f_difference) < threshold_frequency:
                break
            if freq_calc is 'flux':
                flux_slope = self.calculate_flux_slope(flux_val)
                flux_val += f_difference / flux_slope
                self.Flux_Control.set_flux(self.dac_channel, flux_val)
            elif freq_calc is 'dac':
                dac_slope = self.calculate_dac_slope(dac_val)
                dac_val += f_difference / dac_slope
                self.IVVI.set_dac(self.dac_channel, dac_val)
        if abs(f_difference) < threshold_frequency:
            print('Frequency tuned to within threshold frequency')
        else:
            print('Failed to tune within threshold frequency')
        return result

    def tune_to_frequency_Ramsey(self,
                                 target_frequency,
                                 frequency_threshold=3e-5,
                                 max_iterations=5,
                                 stepsizes=[10, 30, 100, 200, 500],
                                 update_qubit=True,
                                 suppress_print_statements=False,
                                 **kw):
        '''
        Routine that tunes qubit to a certain frequency with it's flux
        channel.
        '''

        def tune_flux(qubit, detuning):
            flux_channel = qubit.get_dac_channel()
            flux = self.Flux_Control.get_flux(flux_channel)
            flux_slope = qubit.calculate_flux_slope(flux)

            flux_difference = float(detuning) / flux_slope
            self.Flux_Control.set_flux(flux_channel, flux - flux_difference)
            if not suppress_print_statements:
                print('target frequency:', target_frequency)
                print('frequency estimate:', frequency)
                print('detuning: {} Hz'.format(detuning*1e9))
                print('flux_difference:', flux_difference)

        if self.TD_Meas.get_multiplex():
            raise NameError('Timedomain is set to multiplexed mode')
        # frequency = target_frequency
        # for k, stepsize in enumerate(stepsizes):

        if isinstance(target_frequency, types.InstanceType):
            target_frequency = target_frequency.get_current_frequency()
        stepsize_idx = 0
        stepsize = stepsizes[stepsize_idx]
        max_stepsize = stepsizes[-1]

        for k in range(max_iterations):
            TD_a = self.measure_Ramsey(drive_frequency=target_frequency,
                                       stepsize=stepsize, **kw)
            detuning = TD_a.detuning
            frequency = target_frequency + detuning
            T2_star = TD_a.T2_star
            next_stepsize = stepsizes[min(k+1, len(stepsizes)-1)]

            if (abs(detuning) < frequency_threshold) & (stepsize==max_stepsize):
                if not suppress_print_statements:
                    print('Detuning {} Hz is smaller than threshold {}'.format(
                        detuning * 1e9, frequency_threshold))
                break

            if next_stepsize*60 > T2_star*3:  # all in ns
                if not suppress_print_statements:
                    print('Step size {} ns is at maximum'.format(stepsize))
                max_stepsize = stepsize
            elif max_stepsize == stepsize:
                max_stepsize = next_stepsize

            if stepsize < max_stepsize:
                stepsize_idx += 1
                stepsize = stepsizes[stepsize_idx]
            # print 'flux_diffence', flux_difference
            # breaker
            tune_flux(self, detuning)

        if not suppress_print_statements:
            print('Frequency tuning done.')
            print('target frequency:', target_frequency)
            print('frequency estimate:', frequency)
            print('detuning: {} Hz'.format(detuning * 1e9))

        if update_qubit:
            self.set_current_frequency(frequency)

        result = {'frequency': frequency,
                  'frequency_stderr': TD_a.detuning_stderr,
                  'T2_star': T2_star,
                  'T2_star_stderr': TD_a.T2_star_stderr,
                  'stepsize': stepsize}
        return result

    def optimize_readout_frequency(self,
                                   suppress_print_statements=False,
                                   Navg=3,
                                   f_step=0.0001,
                                   f_range=0.002,
                                   update_qubit=True,
                                   quadrature='I',
                                   MC_name='MC',
                                   **kw):

        if self.TD_Meas.get_multiplex():
            raise NameError('Timedomain is set to multiplexed mode')
        MC = qt.instruments[MC_name]

        TD_Meas = qt.instruments['TD_Meas']
        old_Navg = TD_Meas.get_Navg()
        self.prepare_for_TD(Navg=Navg, **kw)

        f_curr = self.get_current_RO_frequency()

        frequencies = np.arange(f_curr-f_range/2, f_curr+f_range/2+1e-6, f_step)

        is_Duplexer = self.pulse_amp_control is 'Duplexer'
        if self.data_acquistion == 'CBox':
            MC.set_detector_function(
                cdet.SSRO_Fidelity_Detector_CBox(
                    Duplexer=is_Duplexer,
                    gauss_width=self.gauss_width,
                    qubit_suffix=self.qubit_suffix,
                    measurement_name=self.get_name()))

        else:
            MC.set_detector_function(
                cdet.SSRO_Fidelity_Detector_ATS(Duplexer=is_Duplexer,
                                              gauss_width=self.gauss_width,
                                              qubit_suffix=self.qubit_suffix,
                                              measurement_name=self.get_name()))

        MC.set_sweep_function(swf.TD_RO_frequency_GHz())
        MC.set_sweep_points(frequencies)

        MC.run(name='Readout_Frequency_Optimization',
               debug_mode=True, mon_start=3, suppress_print_statements=True,
               print_sweep_points=True)


        TD_Meas.set_Navg(old_Navg)

        anal = MA.MeasurementAnalysis(auto=True, close_file=True,
                                      label='Readout_Frequency_Optimization')

        if update_qubit is True:
            max_index = np.argmax(anal.measured_values[0])
            optimum_frequency = anal.sweep_points[max_index]
            self.set_current_RO_frequency(optimum_frequency)

        return [anal.sweep_points, anal.measured_values[0]]

    def optimize_t_int(self,
                       suppress_print_statements=False,
                       Navg=3,
                       t_int_step=250,
                       t_int_start=500,
                       t_int_stop=4000,
                       update_qubit=True,
                       quadrature='I',
                       MC_name='MC',
                       **kw):

        if self.TD_Meas.get_multiplex():
            raise NameError('Timedomain is set to multiplexed mode')
        MC = qt.instruments[MC_name]

        TD_Meas = qt.instruments['TD_Meas']
        old_Navg = TD_Meas.get_Navg()
        self.prepare_for_TD(Navg=Navg, **kw)

        t_ints = np.arange(t_int_start, t_int_stop+1e-3, t_int_step)

        is_Duplexer = self.pulse_amp_control is 'Duplexer'
        if self.data_acquistion == 'CBox':
            MC.set_detector_function(
                cdet.SSRO_Fidelity_Detector_CBox(
                    Duplexer=is_Duplexer,
                    gauss_width=self.gauss_width,
                    qubit_suffix=self.qubit_suffix,
                    measurement_name=self.get_name()))

        else:
            MC.set_detector_function(
                cdet.SSRO_Fidelity_Detector_ATS(Duplexer=is_Duplexer,
                                              gauss_width=self.gauss_width,
                                              qubit_suffix=self.qubit_suffix,
                                              measurement_name=self.get_name()))

        MC.set_sweep_function(swf.TD_t_int())
        MC.set_sweep_points(t_ints)

        MC.run(name='Integration_Time_Optimization',
               debug_mode=True, mon_start=3, suppress_print_statements=True,
               print_sweep_points=True)


        TD_Meas.set_Navg(old_Navg)

        ma = MA.MeasurementAnalysis(auto=True, close_file=True,
                                      label='Integration_Time_Optimization')

        if update_qubit is True:
            max_index = np.argmax(ma.measured_values[0])
            optimum_t_int = ma.sweep_points[max_index]
            self.set_t_int(optimum_t_int)

        return [ma.sweep_points, ma.measured_values[0], optimum_t_int]

    def optimize_readout_power(self,
                               suppress_print_statements=False,
                               Navg=2,
                               power_range=10,
                               power_step=0.5,
                               update_qubit=True,
                               quadrature='I',
                               MC_name='MC',
                               **kw):
        if self.TD_Meas.get_multiplex():
            raise NameError('Timedomain is set to multiplexed mode')
        MC = qt.instruments[MC_name]
        TD_Meas = qt.instruments['TD_Meas']
        old_Navg = TD_Meas.get_Navg()

        self.prepare_for_TD(Navg=Navg, **kw)

        power_current = self.RF_TD_power

        powers = np.arange(power_current-power_range/2,
                           power_current+power_range/2+0.001,
                           power_step)

        is_Duplexer = self.pulse_amp_control is 'Duplexer'

        if self.data_acquistion == 'CBox':
            MC.set_detector_function(
                cdet.SSRO_Fidelity_Detector_CBox(
                    Duplexer=is_Duplexer,
                    gauss_width=self.gauss_width,
                    qubit_suffix=self.qubit_suffix,
                    measurement_name=self.get_name()))

        else:
            MC.set_detector_function(
                cdet.SSRO_Fidelity_Detector_ATS(Duplexer=is_Duplexer,
                                              gauss_width=self.gauss_width,
                                              qubit_suffix=self.qubit_suffix,
                                              measurement_name=self.get_name()))
        MC.set_sweep_function(swf.TD_RO_power_dBm())
        MC.set_sweep_points(powers)

        MC.run(name='Readout_Power_Optimization', debug_mode=True,
               mon_start=3, suppress_print_statements=True,
               print_sweep_points=True)

        TD_Meas.set_Navg(old_Navg)

        anal = MA.MeasurementAnalysis(auto=True, close_file=True,
                                      label='Readout_Power_Optimization')

        if update_qubit is True:
            max_index = np.argmax(anal.measured_values[0])
            optimum_power = anal.sweep_points[max_index]
            self.set_RF_TD_power(optimum_power)

        return [anal.sweep_points, anal.measured_values[0]]

    def set_curve_parameters_from_data(self, timestamp=None):
        if timestamp is None:
            data_folder = a_tools.latest_data()
        else:
            data_folder = a_tools.data_from_time(timestamp)
        filepath = a_tools.measurement_filename(data_folder)
        with h5py.File(filepath, 'r') as data_file:
            params = {'E_c': 0, 'dac_flux_coefficient': 0,
                      'dac_sweet_spot': 0, 'f_max': 0, 'asymmetry': 0}
            for param in list(params.keys()):
                params[param] = data_file[
                    'Analysis']['Fitted Params curve'][param].attrs['value']

            self.set_E_c(params['E_c'])
            self.set_dac_flux_coefficient(params['dac_flux_coefficient'])
            self.set_dac_sweet_spot(params['dac_sweet_spot'])
            self.set_f_max(params['f_max'])
            self.set_asymmetry(params['asymmetry'])

    def setup_Duplexer(self, close_other_channels=True,
                       drive_amplitude=None, drag_amplitude=None,
                       phase_I=None, phase_Q=None, **kw):
        if close_other_channels:
            self.Dupl.set_all_switches_to('OFF')

        self.Dupl.set_switch(1, self.duplexer_output_channel, 'EXT')
        if self.close_drag_switch:
            self.Dupl.set_switch(2, self.duplexer_output_channel, 'OFF')
        else:
            self.Dupl.set_switch(2, self.duplexer_output_channel, 'EXT')

        if drive_amplitude is None:
            drive_amplitude = self.pulse_amplitude_I
        if drag_amplitude is None:
            drag_amplitude = self.pulse_amplitude_Q
        if phase_I is None:
            phase_I = self.pulse_phase_I
        if phase_Q is None:
            phase_Q = self.pulse_phase_Q

        self.Dupl.set_attenuation(1, self.duplexer_output_channel,
                                  drive_amplitude)
        self.Dupl.set_attenuation(2, self.duplexer_output_channel,
                                  drag_amplitude)
        self.Dupl.set_phase(1, self.duplexer_output_channel, phase_I)
        self.Dupl.set_phase(2, self.duplexer_output_channel, phase_Q)

    def AWG_set_offset(val, which_offset):
        AWG_obj=qt.instruments[self.AWG_source]
        if which_offset == 1:
            AWG_obj.set_ch1_offset(val)
        if which_offset == 2:
            AWG_obj.set_ch2_offset(val)
        if which_offset == 3:
            AWG_obj.set_ch3_offset(val)
        if which_offset == 4:
            AWG_obj.set_ch4_offset(val)

    def AWG_get_offset(which_offset):
        AWG_obj=qt.instruments[self.AWG_source]
        if which_offset == 1:
            res=AWG_obj.get_ch1_offset()
        if which_offset == 2:
            res=AWG_obj.get_ch1_offset()
        if which_offset == 3:
            res=AWG_obj.get_ch1_offset()
        if which_offset == 4:
            res=AWG_obj.get_ch1_offset()
        return res

    def AWG_set_amplitude(val, which_ch):
        AWG_obj = qt.instruments[self.AWG_source]
        if which_ch == 1:
            AWG_obj.set_ch1_amplitude(val)
        if which_ch == 2:
            AWG_obj.set_ch2_amplitude(val)
        if which_ch == 3:
            AWG_obj.set_ch3_amplitude(val)
        if which_ch == 4:
            AWG_obj.set_ch4_amplitude(val)

    def AWG_get_amplitude(which_ch):
        AWG_obj = qt.instruments[self.AWG_source]
        if which_ch == 1:
            res = AWG_obj.get_ch1_amplitude()
        if which_ch == 2:
            res = AWG_obj.get_ch1_amplitude()
        if which_ch == 3:
            res = AWG_obj.get_ch1_amplitude()
        if which_ch == 4:
            res = AWG_obj.get_ch1_amplitude()
        return res
    def do_get_E_c(self):
        return self.E_c

    def do_set_E_c(self, E_c):
        self.E_c = E_c

    def do_get_E_c_stderr(self):
        return self.E_c_stderr

    def do_set_E_c_stderr(self, E_c_stderr):
        self.E_c_stderr = E_c_stderr

    def do_get_E_j(self):
        return self.E_j

    def do_set_E_j(self, E_j):
        self.E_j = E_j

    def do_get_E_j_stderr(self):
        return self.E_j_stderrls

    def do_set_E_j_stderr(self, E_j_stderr):
        self.E_j_stderr = E_j_stderr

    def do_get_T1(self):
        return self.T1

    def do_set_T1(self, T1):
        self.T1 = T1

    def do_get_T1_stderr(self):
        return self.T1_stderr

    def do_set_T1_stderr(self, T1_stderr):
        self.T1_stderr = T1_stderr

    def do_get_T2_star(self):
        return self.T2_star

    def do_set_T2_star(self, T2_star):
        self.T2_star = T2_star

    def do_get_T2_star_stderr(self):
        return self.T2_star_stderr

    def do_set_T2_star_stderr(self, T2_star_stderr):
        self.T2_star_stderr = T2_star_stderr

    def do_get_T2_echo(self):
        return self.T2_echo

    def do_set_T2_echo(self, T2_echo):
        self.T2_echo = T2_echo

    def do_get_T2_echo_stderr(self):
        return self.T2_echo_stderr

    def do_set_T2_echo_stderr(self, T2_echo_stderr):
        self.T2_echo_stderr = T2_echo_stderr

    def do_get_f_max(self):
        return self.f_max

    def do_set_f_max(self, f_max):
        self.f_max = f_max

    def do_get_f_max_stderr(self):
        return self.f_max_stderr

    def do_set_f_max_stderr(self, f_max_stderr):
        self.f_max_stderr = f_max_stderr

    def do_get_dac_sweet_spot(self):
        return self.dac_sweet_spot

    def do_set_dac_sweet_spot(self, dac_sweet_spot):
        self.dac_sweet_spot = dac_sweet_spot

    def do_get_dac_sweet_spot_stderr(self):
        return self.dac_sweet_spot_stderr

    def do_set_dac_sweet_spot_stderr(self, dac_sweet_spot_stderr):
        self.dac_sweet_spot_stderr = dac_sweet_spot_stderr

    def do_get_dac_flux_coefficient(self):
        return self.dac_flux_coefficient

    def do_set_dac_flux_coefficient(self, dac_flux_coefficient):
        self.dac_flux_coefficient = dac_flux_coefficient

    def do_get_dac_flux_coefficient_stderr(self):
        return self.dac_flux_coefficient_stderr

    def do_set_dac_flux_coefficient_stderr(self, dac_flux_coefficient_stderr):
        self.dac_flux_coefficient_stderr = dac_flux_coefficient_stderr

    def do_get_asymmetry(self):
        return self.asymmetry

    def do_set_asymmetry(self, asymmetry):
        self.asymmetry = asymmetry

    def do_get_flux_zero(self):
        return self.flux_zero

    def do_set_flux_zero(self, flux_zero):
        self.flux_zero = flux_zero

    def do_get_flux_zero_stderr(self):
        return self.flux_zero_stderr

    def do_set_flux_zero_stderr(self, flux_zero_stderr):
        self.flux_zero_stderr = flux_zero_stderr

    def do_get_current_frequency(self):
        return self.current_frequency

    def do_set_current_frequency(self, current_frequency):
        self.current_frequency = current_frequency
        self.generator_frequency = current_frequency - \
            self.sideband_modulation_frequency

    def do_get_current_dac_value(self):
        return self.current_dac_value

    def do_set_current_dac_value(self, current_dac_value):
        self.current_dac_value = current_dac_value

    def do_get_current_RO_frequency(self):
        return self.current_RO_frequency

    def do_set_current_RO_frequency(self, current_RO_frequency):
        self.current_RO_frequency = current_RO_frequency
        self.HM.set_frequency(current_RO_frequency*1.e9)

    def do_get_dac_channel(self):
        return self.dac_channel

    def do_set_dac_channel(self, dac_channel):
        self.dac_channel = dac_channel

    def do_get_freq_calc(self):
        return self.freq_calc

    def do_set_freq_calc(self, freq_calc):
        self.freq_calc = freq_calc

    def do_get_duplexer_output_channel(self):
        return self.duplexer_output_channel

    def do_set_duplexer_output_channel(self, duplexer_output_channel):
        self.duplexer_output_channel = duplexer_output_channel

    def do_set_AWG_source(self, AWG_name):
        self.AWG_source = AWG_name
        self.AWG = qt.instruments[AWG_name]

    def do_get_AWG_source(self):
        return self.AWG_source

    def do_get_qubit_source(self):
        return self.qubit_source

    def do_set_qubit_source(self, qubit_source):
        self.qubit_source = qubit_source
        self.qubit_source_instrument = qt.instruments[qubit_source]

    def do_get_RF_source(self):
        return self.RF_source

    def do_set_RF_source(self, RF_source):
        self.RF_source = RF_source
        self.qubit_RF_instrument = qt.instruments[RF_source]

    def do_get_qubit_drive(self):
        return self.qubit_drive

    def do_set_qubit_drive(self, qubit_drive):
        self.qubit_drive = qubit_drive
        self.qubit_drive_instrument = qt.instruments[qubit_drive]

    def do_get_pulse_amp_control(self):
        return self.pulse_amp_control

    def do_set_pulse_amp_control(self, pulse_amp_control):
        self.pulse_amp_control = pulse_amp_control

    def do_get_source_power(self):
        return self.source_power

    def do_set_source_power(self, source_power):
        self.source_power = source_power
        try:
            self.qubit_source_instrument.set_power(source_power)
        except:
            logging.warning('Currently no qubit source instrument set.')

    def do_get_drive_power(self):
        return self.drive_power

    def do_set_drive_power(self, drive_power):
        self.drive_power = drive_power
        try:
            self.qubit_drive_instrument.set_power(drive_power)
        except:
            logging.warning('Currently no qubit drive instrument set.')

    def do_get_RF_CW_power(self):
        return self.RF_CW_power

    def do_set_RF_CW_power(self, RF_CW_power):
        self.RF_CW_power = RF_CW_power
        self.HM.set_RF_power(RF_CW_power)

    def do_get_RF_TD_power(self):
        return self.RF_TD_power

    def do_set_RF_TD_power(self, RF_TD_power):
        self.RF_TD_power = RF_TD_power
        self.TD_Meas.set_RF_power(RF_TD_power)

    def do_get_pulse_amplitude_I(self):
        return self.pulse_amplitude_I

    def do_set_pulse_amplitude_I(self, pulse_amplitude_I):
        self.pulse_amplitude_I = pulse_amplitude_I

    def do_get_pulse_amplitude_Q(self):
        return self.pulse_amplitude_Q

    def do_set_pulse_amplitude_Q(self, pulse_amplitude_Q):
        self.pulse_amplitude_Q = pulse_amplitude_Q

    def do_get_spec_pulse_amp(self):
        return self.spec_pulse_amp

    def do_set_spec_pulse_amp(self, spec_pulse_amp):
        self.spec_pulse_amp = spec_pulse_amp

    def do_get_pulse_phase_I(self):
        return self.pulse_phase_I

    def do_set_pulse_phase_I(self, pulse_phase_I):
        self.pulse_phase_I = pulse_phase_I

    def do_get_pulse_phase_Q(self):
        return self.pulse_phase_Q

    def do_set_pulse_phase_Q(self, pulse_phase_Q):
        self.pulse_phase_Q = pulse_phase_Q

    def do_get_close_drag_switch(self):
        return self.close_drag_switch

    def do_set_close_drag_switch(self, close_drag_switch):
        self.close_drag_switch = close_drag_switch

    def do_get_gauss_width(self):
        return self.gauss_width

    def do_set_gauss_width(self, gauss_width):
        self.gauss_width = gauss_width

    def do_get_sideband_modulation_frequency(self):
        return self.sideband_modulation_frequency

    def do_set_sideband_modulation_frequency(self,
                                             sideband_modulation_frequency):
        '''
        Sets the sideband modulation used for pulsing.
        The definition used is the following
            driving_freq = generator_frequency + sideband_frequency
        '''
        self.sideband_modulation_frequency = sideband_modulation_frequency
        self.generator_frequency = self.current_frequency - \
            self.sideband_modulation_frequency

    def do_set_timestamps(self, timestamps):
        self.timestamps = timestamps

    def do_get_timestamps(self):
        return self.timestamps

    def append_timestamps(self, timestamp):
        self.timestamps.append(timestamp)

    def do_get_qubit_suffix(self):
        return self.qubit_suffix

    def do_set_qubit_suffix(self, qubit_suffix):
        self.qubit_suffix = qubit_suffix

    def do_get_AWG_ch_I(self):
        return self.AWG_ch_I

    def do_set_AWG_ch_I(self, ch_val):
        self.AWG_ch_I = ch_val

    def do_get_AWG_ch_Q(self):
        return self.AWG_ch_Q

    def do_set_AWG_ch_Q(self, ch_val):
        self.AWG_ch_Q = ch_val

    def do_get_data_acquistion(self):
        return self.data_acquistion

    def do_set_data_acquistion(self, data_acquistion):
        self.data_acquistion = data_acquistion

    def do_get_t_int(self):
        return self.t_int

    def do_set_t_int(self, t_int):
        self.t_int = t_int
