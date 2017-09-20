import logging
import numpy as np

from pycqed.analysis import measurement_analysis as ma
from pycqed.measurement import detector_functions as det
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
    import Qubit

class Qubit_TuneUp(Qubit):

    def __init__(self, name, **kw):

        super().__init__(name, **kw)

    ######################################
    ## Single-qubit tune-up measurements##
    ######################################

    # RESONATOR SPECTROSCOPY

    def find_resonator_frequency(self, freqs=None, update=False, MC=None,
                                 close_fig=True, fitting_model='hanger', **kw):
        """
        Finds the resonator frequency by performing a heterodyne experiment
        if freqs == None it will determine a default range dependent on the
        last known frequency of the resonator.
        WARNING: Does not automatically update the RO resonator parameters.
        Set update=True if you want this!
        """

        if not update:
            logging.warning("Does not automatically update the RO "
                            "resonator parameters. "
                            "Set update=True if you want this!")
        if np.any(freqs<500e6):
            logging.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if freqs is None:
            if self.f_RO_resonator() != 0 and self.Q_RO_resonator() != 0:
                fmin = self.f_RO_resonator()*(1-10/self.Q_RO_resonator())
                fmax = self.f_RO_resonator()*(1+10/self.Q_RO_resonator())
                freqs = np.linspace(fmin, fmax, 100)
            else:
                raise ValueError("Unspecified frequencies for find_resonator_"
                                 "frequency and no previous value exists")

        if MC is None:
            MC = self.MC

        self.measure_resonator_spectroscopy(freqs, MC, analyze=False)
        label = 'resonator_scan' + self.msmt_suffix
        HA = ma.Homodyne_Analysis(qb_name=self.name,
                                  label=label, close_fig=close_fig,
                                  fitting_model=fitting_model,**kw)
        f0 = HA.fit_res.params['f0'].value
        df0 = HA.fit_res.params['f0'].stderr
        Q = HA.fit_res.params['Q'].value
        dQ = HA.fit_res.params['Q'].stderr
        if f0 > max(freqs) or f0 < min(freqs):
            logging.warning('extracted frequency outside of range of scan')
        elif df0 > f0:
            logging.warning('resonator frequency uncertainty greater than '
                            'value')
        elif dQ > Q:
            logging.warning('resonator Q factor uncertainty greater than '
                            'value')
        elif update:  # don't update if there was trouble
            self.f_RO_resonator(f0)
            self.Q_RO_resonator(Q)
            self.heterodyne.frequency(f0)
        return f0

    def measure_resonator_spectroscopy(self, freqs=None, MC=None,
                                       analyze=True, close_fig=True):
        """ Varies the frequency of the microwave source to the resonator and
        measures the transmittance """
        if freqs is None:
            raise ValueError("Unspecified frequencies for measure_resonator_"
                             "spectroscopy")
        if np.any(freqs<500e6):
            logging.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if MC is None:
            MC = self.MC

        previous_freq = self.heterodyne.frequency()

        self.prepare_for_continuous_wave()
        MC.set_sweep_function(pw.wrap_par_to_swf(
            self.heterodyne.frequency))
        MC.set_sweep_points(freqs)
        demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
            else 'double'
        MC.set_detector_function(det.Heterodyne_probe(
            self.heterodyne,
            trigger_separation=self.heterodyne.trigger_separation(),
            demod_mode=demod_mode))
        MC.run(name='resonator_scan'+self.msmt_suffix)

        self.heterodyne.frequency(previous_freq)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)


    # QUBIT SPECTROSCOPY

    def find_frequency(self, freqs, method='cw_spectroscopy', update=False,
                       MC=None, close_fig=True, analyze_ef=False, analyze=True,
                       **kw):
        """
        WARNING: Does not automatically update the qubit frequency parameter.
        Set update=True if you want this!

        Args:
            method:                   the spectroscopy type; options: 'pulsed',
                                      'spectrsocopy'
            update:                   whether to update the relevant qubit
                                      parameters with the found frequency(ies)
            MC:                       the measurement control object
            close_fig:                whether or not to close the figure
            analyze_ef:               whether or not to also look for the gf/2

        Keyword Args:
            interactive_plot:        (default=False)
                whether to plot with plotly or not
            analyze_ef:              (default=False)
                whether to look for another f_ge/2 peak/dip
            percentile:              (default=20)
                percentile of the data that is considered background noise
            num_sigma_threshold:     (default=5)
                used to define the threshold above(below) which to look for
                peaks(dips); threshold = background_mean +
                num_sigma_threshold * background_std
            window_len              (default=3)
                filtering window length; uses a_tools.smooth
            analysis_window         (default=10)
                how many data points (calibration points) to remove before
                sending data to peak_finder; uses a_tools.cut_edges,
                data = data[(analysis_window//2):-(analysis_window//2)]
            amp_only                (default=False)
                whether only I data exists
            save_name               (default='Source Frequency')
                figure name with which it will be saved
            auto                    (default=True)
                automatically perform the entire analysis upon call
            label                   (default=none?)
                label of the analysis routine
            folder                  (default=working folder)
                working folder
            NoCalPoints             (default=4)
                number of calibration points
            print_fit_results       (default=True)
                print the fit report
            print_frequency         (default=False)
                whether to print the f_ge and f_gf/2
            make_fig          {default=True)
                    whether or not to make a figure
            show                    (default=True)
                show the plots
            show_guess              (default=False)
                plot with initial guess values
            close_file              (default=True)
                close the hdf5 file

        Returns:
            the peak frequency(ies).
        """
        if not update:
            logging.warning("Does not automatically update the qubit "
                            "frequency parameter. "
                            "Set update=True if you want this!")
        if np.any(freqs<500e6):
            logging.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if MC is None:
            MC = self.MC

        if freqs is None:
            f_span = kw.get('f_span', 100e6)
            f_mean = kw.get('f_mean', self.f_qubit())
            nr_points = kw.get('nr_points', 100)
            if f_mean == 0:
                logging.warning("find_frequency does not know where to "
                                "look for the qubit. Please specify the "
                                "f_mean or the freqs function parameter.")
                return 0
            else:
                freqs = np.linspace(f_mean - f_span/2, f_mean + f_span/2,
                                    nr_points)

        if 'pulse' not in method.lower():
            self.measure_spectroscopy(freqs, pulsed=False, MC=MC,
                                      close_fig=close_fig)
            label = 'spectroscopy'
        else:
            self.measure_spectroscopy(freqs, pulsed=True, MC=MC,
                                      close_fig=close_fig)
            label = 'pulsed-spec'

        if analyze_ef:
            label = 'high_power_' + label

        if analyze:
            amp_only = hasattr(self.heterodyne, 'RF')
            SpecA = ma.Qubit_Spectroscopy_Analysis(
                qb_name=self.name,
                analyze_ef=analyze_ef,
                label=label,
                amp_only=amp_only,
                close_fig=close_fig,**kw)
            self.f_qubit(SpecA.fitted_freq)
            f0 = SpecA.fitted_freq
            if update:
                self.f_qubit(f0)
            if analyze_ef:
                f0_ef = 2*SpecA.fitted_freq_gf_over_2 - f0
                if update:
                    self.f_ef_qubit(f0_ef)
            if analyze_ef:
                return f0, f0_ef
            else:
                return f0
        else:
            return

    def measure_spectroscopy(self, freqs=None, pulsed=False, MC=None,
                             analyze=True, close_fig=True):
        """ Varies qubit drive frequency and measures the resonator
        transmittance """
        if freqs is None:
            raise ValueError("Unspecified frequencies for measure_spectroscopy")
        if np.any(freqs<500e6):
            logging.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if MC is None:
            MC = self.MC

        if not pulsed:

            self.heterodyne.frequency(self.f_RO())
            self.prepare_for_continuous_wave()
            self.cw_source.on()

            MC.set_sweep_function(pw.wrap_par_to_swf(self.cw_source.frequency))
            MC.set_sweep_points(freqs)
            demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
                else 'double'
            MC.set_detector_function(det.Heterodyne_probe(
                self.heterodyne,
                trigger_separation=self.heterodyne.trigger_separation(),
                demod_mode=demod_mode))
            MC.run(name='spectroscopy'+self.msmt_suffix)

            self.cw_source.off()

        else:
            self.prepare_for_pulsed_spec()

            spec_pars = self.get_spec_pars()
            RO_pars = self.get_RO_pars()

            self.cw_source.on()

            sq.Pulsed_spec_seq(spec_pars, RO_pars)

            self.AWG.start()

            MC.set_sweep_function(self.cw_source.frequency)
            MC.set_sweep_points(freqs)
            demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
                else 'double'
            MC.set_detector_function(det.Heterodyne_probe(
                self.heterodyne,
                trigger_separation=self.heterodyne.trigger_separation(),
                demod_mode=demod_mode))
            MC.run(name='pulsed-spec' + self.msmt_suffix)

            self.cw_source.off()


        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)


    # RABI MEASUREMENT

    def find_amplitudes(self, rabi_amps=None, label=None, for_ef=False,
                        update=False, MC=None, close_fig=True, cal_points=True,
                        no_cal_points=None, upload=True, last_ge_pulse=True,
                        analyze=True, **kw):

        """
            Finds the pi and pi/2 pulse amplitudes from the fit to a Rabi
            experiment. Uses the Rabi_Analysis(_new)
            class from measurement_analysis.py
            WARNING: Does not automatically update the qubit amplitudes.
            Set update=True if you want this!

            Analysis script for the Rabi measurement:
            1. The I and Q data are rotated and normalized based on the calibration
                points. In most analysis routines, the latter are typically 4:
                2 X180 measurements, and 2 identity measurements, which get
                averaged resulting in one X180 point and one identity point.
                However, the default for Rabi is 2 (2 identity measurements)
                because we typically do Rabi in order to find the correct amplitude
                for an X180 pulse. However, if a previous such value exists, this
                routine also accepts 4 cal pts. If X180_ef pulse was also
                previously calibrated, this routine also accepts 6 cal pts.
            2. The normalized data is fitted to a cosine function.
            3. The pi-pulse and pi/2-pulse amplitudes are calculated from the fit.
            4. The normalized data, the best fit results, and the pi and pi/2
                pulses are plotted.

            The ef analysis assumes the the e population is zero (because of the
            ge X180 pulse at the end).

            Arguments:
                rabi_amps:          amplitude sweep points for the
                                    Rabi experiment
                label:              label of the analysis routine
                for_ef:             find amplitudes for the ef transition
                update:             update the qubit amp180 and amp90 parameters
                MC:                 the measurement control object
                close_fig:          close the resulting figure?
                cal_points          whether to used calibration points of not
                no_cal_points       number of calibration points to use; if it's
                                    the first time rabi is run
                                    then 2 cal points (two I pulses at the end)
                                    should be used for the ge Rabi,
                                    and 4 (two I pulses and 2 ge X180 pulses at
                                    the end) for the ef Rabi
                last_ge_pulse       whether to map the population to the ground
                                    state after each run of the Rabi experiment
                                    on the ef level
            Keyword arguments:
                other keyword arguments. The Rabi sweep parameters 'amps_mean',
                 'amps_span', and 'nr_poinys' should be passed here. This will
                 result in a sweep over rabi_amps = np.linspace(amps_mean -
                 amps_span/2, amps_mean + amps_span/2, nr_points)

                auto              (default=True)
                    automatically perform the entire analysis upon call
                print_fit_results (default=True)
                    print the fit report
                make_fig          {default=True)
                    whether or not to make a figure
                show              (default=True)
                    show the plots
                show_guess        (default=False)
                    plot with initial guess values
                show_amplitudes   (default=True)
                    print the pi&piHalf pulses amplitudes
                plot_amplitudes   (default=True)
                    plot the pi&piHalf pulses amplitudes
                no_of_columns     (default=1)
                    number of columns in your paper; figure sizes will be adjusted
                    accordingly (1 col: figsize = ( 7in , 4in ) 2 cols: figsize =
                    ( 3.375in , 2.25in ), PRL guidelines)

            Returns:
                pi and pi/2 pulses amplitudes + their stderr as a dictionary with
                keys 'piPulse', 'piHalfPulse', 'piPulse_std', 'piHalfPulse_std'.
            """

        if not update:
            logging.warning("Does not automatically update the qubit pi and "
                            "pi/2 amplitudes. "
                            "Set update=True if you want this!")

        if MC is None:
            MC = self.MC

        if (cal_points) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if for_ef==False,'
                            'or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if not cal_points:
            no_cal_points = 0

        #how many times to apply the Rabi pulse
        n = kw.get('n',1)

        if rabi_amps is None:
            amps_span = kw.get('amps_span', 1.)
            amps_mean = kw.get('amps_mean', self.amp180())
            nr_points = kw.get('nr_points', 30)
            if amps_mean == 0:
                logging.warning("find_amplitudes does not know over which "
                                "amplitudes to do Rabi. Please specify the "
                                "amps_mean or the amps function parameter.")
                return 0
            else:
                rabi_amps = np.linspace(amps_mean - amps_span/2, amps_mean +
                                        amps_span/2, nr_points)

        if label is None:
            if for_ef:
                label = 'Rabi_2nd' + self.msmt_suffix
            else:
                label = 'Rabi' + self.msmt_suffix

        #Perform Rabi
        if for_ef is False:
            self.measure_rabi(amps=rabi_amps, n=n, MC=MC,
                              close_fig=close_fig,
                              label=label,
                              cal_points=cal_points,
                              no_cal_points=no_cal_points,
                              upload=upload)
        else:
            self.measure_rabi_2nd_exc(amps=rabi_amps, n=n, MC=MC,
                                      close_fig=close_fig, label=label,
                                      cal_points=cal_points,
                                      last_ge_pulse=last_ge_pulse,
                                      no_cal_points=no_cal_points,
                                      upload=upload)

        #get pi and pi/2 amplitudes from the analysis results
        if analyze:
            RabiA = ma.Rabi_Analysis(label=label, qb_name=self.name,
                                     NoCalPoints=no_cal_points,
                                     close_fig=close_fig, for_ef=for_ef,
                                     last_ge_pulse=last_ge_pulse, **kw)

            rabi_amps = RabiA.rabi_amplitudes   #This is a dict with keywords
            #'piPulse',  'piPulse_std',
            #'piHalfPulse', 'piHalfPulse_std

            amp180 = rabi_amps['piPulse']
            amp90 = rabi_amps['piHalfPulse']

            if update:
                if for_ef is False:
                    self.amp180(amp180)
                    self.amp90_scale(amp90/amp180)
                else:
                    self.amp180_ef(amp180)
                    self.amp90_scale_ef(amp90/amp180)
        else:
            return

    def measure_rabi(self, amps=None, MC=None, analyze=True,
                     close_fig=True, cal_points=True, no_cal_points=2,
                     upload=True, label=None,  n=1):

        """
        Varies the amplitude of the qubit drive pulse and measures the readout
        resonator transmission.

        Args:
            amps            the array of drive pulse amplitudes
            MC              the MeasurementControl object
            analyse         whether to create a (base) MeasurementAnalysis
                            object for this measurement; offers possibility to
                            manually analyse data using the classes in
                            measurement_analysis.py
            close_fig       whether or not to close the default analysis figure
            cal_points      whether or not to use calibration points
            no_cal_points   how many calibration points to use
            upload          whether or not to upload the sequence to the AWG
            label           the measurement label
            n               the number of times the drive pulses with the same
                            amplitude should be repeated in each measurement
        """

        if amps is None:
            raise ValueError("Unspecified amplitudes for measure_rabi")

        # Define the measurement label
        if label is None:
            label = 'Rabi-n{}'.format(n) + self.msmt_suffix

        # Prepare the physical instruments for a time domain measurement
        self.prepare_for_timedomain()

        # Define the MeasurementControl object for this measurement
        if MC is None:
            MC = self.MC

        # Specify the sweep function, the sweep points,
        # and the detector function, and run the measurement
        MC.set_sweep_function(awg_swf.Rabi(pulse_pars=self.get_drive_pars(),
                                           RO_pars=self.get_RO_pars(), n=n,
                                           cal_points=cal_points,
                                           no_cal_points=no_cal_points,
                                           upload=upload))
        MC.set_sweep_points(amps)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        # Create a MeasurementAnalysis object for this measurement
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_rabi_2nd_exc(self, amps=None, n=1, MC=None, analyze=True,
                             label=None, last_ge_pulse=True,
                             close_fig=True, cal_points=True, no_cal_points=4,
                             upload=True):

        if amps is None:
            raise ValueError("Unspecified amplitudes for measure_rabi")

        if label is None:
            label = 'Rabi_2nd_exc-n{}'.format(n) + self.msmt_suffix

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.Rabi_2nd_exc(
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            last_ge_pulse=last_ge_pulse,
            amps=amps, n=n, upload=upload,
            cal_points=cal_points, no_cal_points=no_cal_points))
        MC.set_sweep_points(amps)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)


    # RAMSEY MEASUREMENT

    # Ramsey for one artificial_detuning
    def find_frequency_T2_ramsey(self, times, for_ef=False, artificial_detuning=0,
                                 update=False, MC=None,
                                 cal_points=True, close_fig=True, upload=True,
                                 last_ge_pulse=True, label=None,
                                 no_cal_points=None, analyze=True, **kw):

        """
        Finds the real qubit frequency and the dephasing rate T2* from the fit
        to a Ramsey experiment.
        Uses the Ramsey_Analysis class from measurement_analysis.py
        The ef analysis assumes the the e population is zero (because of the ge
        X180 pulse at the end).

        WARNING: Does not automatically update the qubit freq and T2_star
        parameters. Set update=True if you want this!

        :param times                    array of times over which to sweep in
                                        the Ramsey measurement
        :param artificial_detuning:     difference between drive frequency and
                                        qubit frequency estimated from
                                        qubit spectroscopy
        :param update:                  update the qubit amp180 and amp90
                                        parameters
        :param MC:                      the measurement control object
        :param close_fig:               close the resulting figure?
        :param kw:                      other keyword arguments. The Rabi sweep
                                        time delays array 'times',
                                        or the parameter 'times_mean' should be
                                        passed here (in seconds)
        :return:                        the real qubit frequency
                                        (=self.f_qubit()+artificial_detuning-
                                        fitted_freq)
                                        + stddev, the dephasing rate T2* +
                                        stddev
        """
        if not update:
            logging.warning("Does not automatically update the qubit frequency "
                            "and T2_star parameters. "
                            "Set update=True if you want this!")

        if artificial_detuning == 0:
            logging.warning('Artificial_detuning=0; qubit driven at "%s" '
                            'estimated with '
                            'spectroscopy' %self.f_qubit())
        if np.abs(artificial_detuning)<1e3:
            logging.warning('The artificial detuning is too small. The units '
                            'should be Hz.')

        if np.any(times>1e-3):
            logging.warning('Some of the values in the times array might be too '
                            'large.The units should be seconds.')

        if (cal_points is True) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if for_ef==False,'
                            'or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if cal_points is False:
            no_cal_points = 0

        if MC is None:
            MC = self.MC

        if label is None:
            if for_ef:
                label = 'Ramsey_2nd' +self.msmt_suffix
            else:
                label = 'Ramsey' + self.msmt_suffix

        if times is None:
            times_span = kw.get('times_span', 5e-6)
            times_mean = kw.get('times_mean', 2.5e-6)
            nr_points = kw.get('nr_points', 50)
            if times_mean == 0:
                logging.warning("find_frequency_T2_ramsey does not know over "
                                "which times to do Ramsey. Please specify the "
                                "times_mean or the times function parameter.")
                return 0
            else:
                times = np.linspace(times_mean - times_span/2,
                                    times_mean + times_span/2,
                                    nr_points)
        #Perform Ramsey
        if for_ef is False:
            self.measure_ramsey(times=times,
                                artificial_detuning=artificial_detuning,
                                MC=MC,
                                cal_points=cal_points,
                                close_fig=close_fig, upload=upload, label=label)
            #Needed for analysis
            qubit_frequency_spec = self.f_qubit()

        else:
            self.measure_ramsey_2nd_exc(times=times,
                                        artificial_detuning=artificial_detuning,
                                        MC=MC, cal_points=cal_points,
                                        close_fig=close_fig, upload=upload,
                                        last_ge_pulse=last_ge_pulse,
                                        no_cal_points=no_cal_points, label=label)
            #Needed for analysis
            qubit_frequency_spec = self.f_ef_qubit()

        if analyze:

            RamseyA = ma.Ramsey_Analysis(auto=True, qb_name=self.name,
                                         NoCalPoints=no_cal_points, label=label,
                                         for_ef=for_ef, last_ge_pulse=last_ge_pulse,
                                         artificial_detuning=artificial_detuning,
                                         **kw)

            #get new freq and T2* from analysis results
            new_qubit_freq = RamseyA.qubit_frequency    #value
            fitted_freq = RamseyA.ramsey_freq           #dict
            T2_star = RamseyA.T2_star                   #dict

            print('New qubit frequency = {:.10f} \t stderr = {:.10f}'.format(
                new_qubit_freq,RamseyA.ramsey_freq['freq_stderr']))
            print('T2_Star = {:.5f} \t stderr = {:.5f}'.format(
                T2_star['T2_star'],T2_star['T2_star_stderr']))

            if update:
                if for_ef:
                    self.f_ef_qubit(new_qubit_freq)
                    self.T2_star_ef(T2_star['T2_star'])
                else:
                    self.f_qubit(new_qubit_freq)
                    self.T2_star(T2_star['T2_star'])

            return new_qubit_freq, fitted_freq, T2_star

        else:
            return

    def measure_ramsey(self, times=None, artificial_detuning=0, label='',
                       MC=None, analyze=True, close_fig=True, cal_points=True,
                       upload=True):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detuning is None:
            logging.warning('Artificial detuning is 0.')
        if np.abs(artificial_detuning) < 1e3:
            logging.warning('The artificial detuning is too small. The units'
                            'should be Hz.')
        if np.any(times > 1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # Define the measurement label
        if label == '':
            label = 'Ramsey' + self.msmt_suffix

        Rams_swf = awg_swf.Ramsey(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            artificial_detuning=artificial_detuning, cal_points=cal_points,
            upload=upload)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_ramsey_2nd_exc(self, times=None, artificial_detuning=0, label=None,
                               MC=None, analyze=True, close_fig=True, cal_points=True,
                               n=1, upload=True, last_ge_pulse=True, no_cal_points=6):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detuning is None:
            logging.warning('Artificial detuning is 0.')
        if np.abs(artificial_detuning)<1e3:
            logging.warning('The artificial detuning is too small. The units'
                            'should be Hz.')
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        if label is None:
            label = 'Ramsey_2nd'+self.msmt_suffix

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        Rams_2nd_swf = awg_swf.Ramsey_2nd_exc(
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            artificial_detuning=artificial_detuning,
            cal_points=cal_points, n=n, upload=upload,
            no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse)
        MC.set_sweep_function(Rams_2nd_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    # Ramsey for two artificial_detunings
    def calibrate_ramsey(self, times, for_ef=False,
                         artificial_detunings=None, update=False, label=None,
                         MC=None, cal_points=True, no_cal_points=None,
                         close_fig=True, upload=True, last_ge_pulse=True, **kw):

        """
        Finds the real qubit frequency and the dephasing rate T2* from the fit
        to a Ramsey experiment.
        Uses the Ramsey_Analysis class from measurement_analysis.py
        The ef analysis assumes the the e population is zero (because of the ge
        X180 pulse at the end).

        WARNING: Does not automatically update the qubit freq and T2_star
        parameters. Set update=True if you want this!

        :param times                    array of times over which to sweep in
                                        the Ramsey measurement
        :param artificial_detuning:     difference between drive frequency and
                                        qubit frequency estimated from
                                        qubit spectroscopy
        :param update:                  update the qubit amp180 and amp90
                                        parameters
        :param MC:                      the measurement control object
        :param close_fig:               close the resulting figure?
        :param kw:                      other keyword arguments. The Rabi sweep
                                        time delays array 'times',
                                        or the parameter 'times_mean' should be
                                        passed here (in seconds)
        :return:                        the real qubit frequency
                                        + stddev, the dephasing rate T2* +
                                        stddev
        """
        if not update:
            logging.warning("Does not automatically update the qubit frequency "
                            "and T2_star parameters. "
                            "Set update=True if you want this!")

        if artificial_detunings is None:
            logging.warning('Artificial_detuning=0; qubit driven at "%s" '
                            'estimated with '
                            'spectroscopy' %self.f_qubit())
        if np.any(np.asarray(np.abs(artificial_detunings))<1e3):
            logging.warning('The artificial detuning is too small.')
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.')

        if (cal_points is True) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if for_ef==False,'
                            'or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if cal_points is False:
            no_cal_points = 0

        if MC is None:
            MC = self.MC

        if times is None:
            logging.warning("find_frequency_T2_ramsey does not know over "
                            "which times to do Ramsey. Please specify the "
                            "times_mean or the times function parameter.")

        if label is None:
            if for_ef:
                label = 'Ramsey_mult_det_2nd' +self.msmt_suffix
            else:
                label = 'Ramsey_mult_det' + self.msmt_suffix

        # Each time value must be repeated len(artificial_detunings) times to
        # correspond to the logic in Ramsey_seq_multiple_detunings sequence
        len_art_det = len(artificial_detunings)
        temp_array = np.zeros((times.size-no_cal_points)*len_art_det)
        for i in range(len(artificial_detunings)):
            np.put(temp_array,list(range(i,temp_array.size,len_art_det)),times)
        times =  np.append(temp_array,times[-no_cal_points::])

        #Perform Ramsey
        if for_ef is False:
            self.measure_ramsey_multiple_detunings(times=times,
                                                   artificial_detunings=artificial_detunings,
                                                   MC=MC,
                                                   label=label,
                                                   cal_points=cal_points,
                                                   close_fig=close_fig, upload=upload)

        else:
            self.measure_ramsey_2nd_exc_multiple_detunings(times=times,
                                                           artificial_detunings=artificial_detunings,
                                                           cal_points=cal_points, no_cal_points=no_cal_points,
                                                           close_fig=close_fig, upload=upload,
                                                           last_ge_pulse=last_ge_pulse, MC=MC, label=label)

        # Analyze data if analyze==True
        if kw.pop('analyze',True):
            RamseyA = ma.Ramsey_Analysis_multiple_detunings(auto=True,
                                                            label=label,
                                                            qb_name=self.name,
                                                            NoCalPoints=no_cal_points,
                                                            for_ef=for_ef,
                                                            last_ge_pulse=last_ge_pulse,
                                                            artificial_detunings=artificial_detunings, **kw)

            #get new freq and T2* from analysis results
            new_qubit_freq = RamseyA.qubit_frequency    #value
            fitted_freq = RamseyA.ramsey_freq           #dict
            T2_star = RamseyA.T2_star                   #dict

            print('New qubit frequency = {:.10f} \t stderr = {:.10f}'.format(
                new_qubit_freq, RamseyA.ramsey_freq['freq_stderr']))
            print('T2_Star = {:.5f} \t stderr = {:.5f}'.format(
                T2_star['T2_star'],T2_star['T2_star_stderr']))

            if update:
                if for_ef:
                    self.f_ef_qubit(new_qubit_freq)
                    self.T2_star_ef(T2_star['T2_star'])
                else:
                    self.f_qubit(new_qubit_freq)
                    self.T2_star(T2_star['T2_star'])

            return new_qubit_freq, fitted_freq, T2_star
        else:
            return

    def measure_ramsey_multiple_detunings(self, times=None,
                                          artificial_detunings=None, label='',
                                          MC=None, analyze=True, close_fig=True,
                                          cal_points=True, upload=True):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detunings is None:
            logging.warning('Artificial detuning is 0.')
        uniques = np.unique(times[range(len(artificial_detunings))])
        if uniques.size>1:
            raise ValueError("The values in the times array are not repeated "
                             "len(artificial_detunings) times.")
        if np.any(np.asarray(np.abs(artificial_detunings))<1e3):
            logging.warning('The artificial detuning is too small. The units '
                            'should be Hz.')
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # Define the measurement label
        if label == '':
            label = 'Ramsey_mult_det' + self.msmt_suffix

        Rams_swf = awg_swf.Ramsey_multiple_detunings(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            artificial_detunings=artificial_detunings, cal_points=cal_points,
            upload=upload)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_ramsey_2nd_exc_multiple_detunings(self, times=None,
                                                  artificial_detunings=None, label=None,
                                                  MC=None, analyze=True, close_fig=True,
                                                  cal_points=True, n=1, upload=True,
                                                  last_ge_pulse=True, no_cal_points=6):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detunings is None:
            logging.warning('Artificial detunings were not given.')
        if np.any(np.asarray(np.abs(artificial_detunings))<1e3):
            logging.warning('The artificial detuning is too small. The units '
                            'should be Hz.')
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        if label is None:
            label = 'Ramsey_mult_det_2nd'+self.msmt_suffix

        Rams_2nd_swf = awg_swf.Ramsey_2nd_exc_multiple_detunings(
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            artificial_detunings=artificial_detunings,
            cal_points=cal_points, n=n, upload=upload,
            no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse)
        MC.set_sweep_function(Rams_2nd_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)


    # DRAG PULSE CALIBRATION MEASUREMENT
    # as described in Baur,M. PhD Thesis(2012)

    def find_qscale(self, qscales, label=None, for_ef=False, update=False,
                    MC=None, close_fig=True, last_ge_pulse=True, upload=False,
                    cal_points=True, no_cal_points=None, **kw):

        '''
        Performs the QScale calibration measurement ( (xX)-(xY)-(xmY) ) and
        extracts the optimal QScale parameter
        from the fits (ma.QScale_Analysis).
        WARNING: Does not automatically update the qubit qscale parameter. Set
        update=True if you want this!

        ma.QScale_Analysis:
        1. The I and Q data are rotated and normalized based on the calibration
            points. In most
            analysis routines, the latter are typically 4: 2 X180 measurements,
            and 2 identity measurements, which get averaged resulting in one
            X180 point and one identity point.
        2. The data points for the same qscale value are extracted (every other
            3rd point because the sequence
            used for this measurement applies the 3 sets of pulses
            ( (xX)-(xY)-(xmY) ) consecutively for each qscale value).
        3. The xX data is fitted to a lmfit.models.ConstantModel(), and the
            other 2 to an lmfit.models.LinearModel().
        4. The data and the resulting fits are all plotted on the same graph
            (self.make_figures).
        5. The optimal qscale parameter is obtained from the point where the 2
            linear fits intersect.

        Other possible  input parameters:
            qscales
                array of qscale values over which to sweep...
            or qscales_mean and qscales_span
                ...or the mean qscale value and the span around it
                (defaults to 3) as kw. Then the script will construct the sweep
                points as np.linspace(qscales_mean - qscales_span/2,
                qscales_mean + qscales_span/2, nr_points)

        Keyword parameters:
            label             (default=none?)
                label of the analysis routine
            for_ef            (default=False)
                whether to obtain the drag_qscale_ef parameter
            update            (default=True)
                whether or not to update the qubit drag_qscale parameter with
                the found value
            MC                (default=self.MC)
                the measurement control object
            close_fig         (default=True)
                close the resulting figure
            last_ge_pulse     (default=True)
                whether to apply an X180 ge pulse at the end

            Keyword parameters:
                qscale_mean       (default=self.drag_qscale()
                    mean of the desired qscale sweep values
                qscale_span       (default=3)
                    span around the qscale mean
                nr_points         (default=30)
                    number of sweep points between mean-span/2 and mean+span/2
                auto              (default=True)
                    automatically perform the entire analysis upon call
                folder            (default=working folder)
                    Working folder
                NoCalPoints       (default=4)
                    Number of calibration points
                cal_points        (default=[[-4, -3], [-2, -1]])
                    The indices of the calibration points
                show              (default=True)
                    show the plot
                show_guess        (default=False)
                    plot with initial guess values
                plot_title        (default=measurementstring)
                    the title for the plot as a string
                xlabel            (default=self.xlabel)
                    the label for the x axis as a string
                ylabel            (default=r'$F|1\rangle$')
                    the label for the x axis as a string
                close_file        (default=True)
                    close the hdf5 file

        Returns:
            the optimal DRAG QScale parameter + its stderr as a dictionary with
            keys 'qscale' and 'qscale_std'.
        '''

        if not update:
            logging.warning("Does not automatically update the qubit qscale "
                            "parameter. "
                            "Set update=True if you want this!")

        if (cal_points) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if for_ef==False,'
                            'or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if not cal_points:
            no_cal_points = 0

        if MC is None:
            MC = self.MC

        if label is None:
            label = 'QScale' + self.msmt_suffix

        if qscales is None:
            logging.warning("find_qscale does not know over which "
                            "qscale values to sweep. Please specify the "
                            "qscales_mean or the qscales function"
                            " parameter.")

        # Each qscale value must be repeated 3 times to correspoond to the
        # logic in QScale sequence
        temp_array = np.zeros(3*(qscales.size-no_cal_points))
        np.put(temp_array,list(range(0,temp_array.size,3)),qscales)
        np.put(temp_array,list(range(1,temp_array.size,3)),qscales)
        np.put(temp_array,list(range(2,temp_array.size,3)),qscales)
        qscales = np.append(temp_array,qscales[-no_cal_points::])

        #Perform the qscale calibration measurement
        if for_ef:
            # Run measuremet
            self.measure_qscale_2nd_exc(qscales=qscales, MC=MC, upload=upload,
                                        close_fig=close_fig, label=label,
                                        last_ge_pulse=last_ge_pulse,
                                        cal_points=cal_points,
                                        no_cal_points=no_cal_points)
        else:
            self.measure_qscale(qscales=qscales, MC=MC, upload=upload,
                                close_fig=close_fig, label=label)

        # Perform analysis and extract the optimal qscale parameter
        # Returns the optimal qscale parameter
        if kw.pop('analyze',True):
            QscaleA = ma.QScale_Analysis(auto=True, qb_name=self.name,
                                         label=label,
                                         NoCalPoints=no_cal_points,
                                         for_ef=for_ef,
                                         last_ge_pulse=last_ge_pulse, **kw)

            Qscale_dict = QscaleA.optimal_qscale #dictionary of value, stderr
            Qscale_value = Qscale_dict['qscale']

            if update:
                if for_ef:
                    self.motzoi_ef(Qscale_value)
                else:
                    self.motzoi(Qscale_value)

            return Qscale_dict
        else:
            return

    def measure_qscale(self, qscales=None, MC=None, analyze=True, upload=True,
                       close_fig=True, label=None, cal_points=True):

        if qscales is None:
            raise ValueError("Unspecified qscale values for measure_qscale")
        uniques = np.unique(qscales[range(3)])
        if uniques.size>1:
            raise ValueError("The values in the qscales array are not repeated "
                             "3 times.")

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        if label is None:
            label = 'QScale'+self.msmt_suffix

        MC.set_sweep_function(awg_swf.QScale(qscales=qscales,
                                             pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
                                             upload=upload, cal_points=cal_points))
        MC.set_sweep_points(qscales)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_qscale_2nd_exc(self, qscales=None, MC=None, analyze=True,
                               upload=True, close_fig=True, label=None,
                               cal_points=True, no_cal_points=6,
                               last_ge_pulse=True):

        if qscales is None:
            raise ValueError("Unspecified qscale values for"
                             " measure_qscale_2nd_exc")
        uniques = np.unique(qscales[range(3)])
        if uniques.size>1:
            raise ValueError("The values in the qscales array are not repeated "
                             "3 times.")

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        if label is None:
            label = 'QScale_2nd_exc'+self.msmt_suffix

        MC.set_sweep_function(awg_swf.QScale_2nd_exc(
            qscales=qscales,
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            upload=upload, cal_points=cal_points, no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse))
        MC.set_sweep_points(qscales)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)


    # T1 MEASUREMENT

    def find_T1(self, times, label=None, for_ef=False, update=False, MC=None,
                cal_points=True, no_cal_points=None, close_fig=True,
                last_ge_pulse=True, upload=True, **kw):

        """
        Finds the relaxation time T1 from the fit to an exponential
        decay function.
        WARNING: Does not automatically update the qubit T1 parameter.
        Set update=True if you want this!

        Routine:
            1. Apply pi pulse to get population in the excited state.
            2. Wait for different amounts of time before doing a measurement.

        Uses the T1_Analysis class from measurement_analysis.py.
        The ef analysis assumes the the e population is zero (because of the
        ge X180 pulse at the end).

        Arguments:
            times:                   array of times to wait before measurement
            label:                   label of the analysis routine
            for_ef:                  find T1 for the 2nd excitation (ef)
            update:                  update the qubit T1 parameter
            MC:                      the measurement control object
            close_fig:               close the resulting figure?

        Keyword Arguments:
            other keyword arguments. The the parameters times_mean, times_span,
            nr_points should be passed here. These are an alternative to
            passing the times array.

            auto              (default=True)
                automatically perform the entire analysis upon call
            print_fit_results (default=True)
                print the fit report
            make_fig          (default=True)
                whether to make the figures or not
            show_guess        (default=False)
                plot with initial guess values
            show_T1           (default=True)
                print the T1 and T1_stderr
            no_of_columns     (default=1)
                number of columns in your paper; figure sizes will be adjusted
                accordingly  (1 col: figsize = ( 7in , 4in ) 2 cols:
                figsize = ( 3.375in , 2.25in ), PRL guidelines)

        Returns:
            the relaxation time T1 + standard deviation as a dictionary with
            keys: 'T1', and 'T1_std'

        ! Specify either the times array or the times_mean value (defaults to
        5 micro-s) and the span around it (defaults to 10 micro-s) as kw.
        Then the script will construct the sweep points as
        np.linspace(times_mean - times_span/2, times_mean + times_span/2,
        nr_points)
        """

        if not update:
            logging.warning("Does not automatically update the qubit "
                            "T1 parameter. Set update=True if you want this!")
        if np.any(times>1e-3):
            logging.warning('Some of the values in the times array might be too '
                            'large.The units should be seconds.')

        if (cal_points) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if for_ef==False,'
                            'or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if not cal_points:
            no_cal_points = 0

        if MC is None:
            MC = self.MC

        if label is None:
            if for_ef:
                label = 'T1_2nd' + self.msmt_suffix
            else:
                label = 'T1' + self.msmt_suffix

        if times is None:
            times_span = kw.get('times_span', 10e-6)
            times_mean = kw.get('times_mean', 5e-6)
            nr_points = kw.get('nr_points', 50)
            if times_mean == 0:
                logging.warning("find_T1 does not know how long to wait before"
                                "doing the read out. Please specify the "
                                "times_mean or the times function parameter.")
                return 0
            else:
                times = np.linspace(times_mean - times_span/2, times_mean +
                                    times_span/2, nr_points)

        #Perform measurement
        if for_ef:
            self.measure_T1_2nd_exc(times=times, MC=MC,
                                    close_fig=close_fig,
                                    cal_points=cal_points,
                                    no_cal_points=no_cal_points,
                                    last_ge_pulse=last_ge_pulse,
                                    upload=upload)

        else:
            self.measure_T1(times=times, MC=MC,
                            close_fig=close_fig,
                            cal_points=cal_points,
                            upload=upload)

        #Extract T1 and T1_stddev from ma.T1_Analysis
        if kw.pop('analyze',True):
            T1_Analysis = ma.T1_Analysis(label=label, qb_name=self.name,
                                         NoCalPoints=no_cal_points,
                                         for_ef=for_ef,
                                         last_ge_pulse=last_ge_pulse, **kw)
            T1_dict = T1_Analysis.T1
            T1_value = T1_dict['T1']

            if update:
                if for_ef:
                    self.T1_ef(T1_value)
                else:
                    self.T1(T1_value)

            return T1_dict
        else:
            return

    def measure_T1(self, times=None, MC=None, analyze=True, upload=True,
                   close_fig=True, cal_points=True, label=None):

        if times is None:
            raise ValueError("Unspecified times for measure_T1")
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        # Define the measurement label
        if label is None:
            label = 'T1' + self.msmt_suffix

        MC.set_sweep_function(awg_swf.T1(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            upload=upload, cal_points=cal_points))
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_T1_2nd_exc(self, times=None, MC=None, analyze=True, upload=True,
                           close_fig=True, cal_points=True, no_cal_points=6,
                           label=None, last_ge_pulse=True):

        if times is None:
            raise ValueError("Unspecified times for measure_T1_2nd_exc")
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()

        if label is None:
            label = 'T1_2nd'+self.msmt_suffix

        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.T1_2nd_exc(
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            upload=upload,
            cal_points=cal_points,
            no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse))
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)
