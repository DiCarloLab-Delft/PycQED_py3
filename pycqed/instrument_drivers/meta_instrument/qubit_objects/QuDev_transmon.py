import logging
import numpy as np

from qcodes.instrument.parameter import ManualParameter

from pycqed.measurement import detector_functions as det
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.analysis import measurement_analysis as ma

from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
    import Qubit


class QuDev_transmon(Qubit):
    def __init__(self, name, MC, heterodyne_instr, cw_source, **kw):
        super().__init__(name, **kw)

        self.MC = MC
        self.heterodyne_instr = heterodyne_instr
        self.cw_source = cw_source

        self.add_parameter('f_RO_resonator', label='RO resonator frequency',
                           units='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('Q_RO_resonator', label='RO resonator Q factor',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('optimal_acquisition_delay', label='Optimal '
                           'acquisition delay', units='s', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('f_qubit_spectroscopy', label='Qubit frequency '
                           'from spectroscopy', units='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('kappa_qubit_spectroscopy',
                           label='Width of qubit from spectroscopy',
                           units='Hz', initial_value=0,
                           parameter_class=ManualParameter)


    def prepare_for_continuous_wave(self):
        # heterodyne instrument is a separate instrument and should always be
        # prepared for cw experiments
        pass

    def measure_heterodyne_spectroscopy(self, freqs=None, MC=None,
                                        analyze=True, close_fig=True):
        """ Varies the frequency of the microwave source to the resonator and
        measures the transmittance """
        if freqs is None:
            raise ValueError("Unspecified frequencies for measure_heterodyne_"
                             "spectroscopy")

        if MC is None:
            MC = self.MC

        previous_freq = self.heterodyne_instr.frequency()

        self.prepare_for_continuous_wave()
        MC.set_sweep_function(pw.wrap_par_to_swf(
            self.heterodyne_instr.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(det.Heterodyne_probe(
            self.heterodyne_instr, trigger_separation=5e-6,
            demod_mode='single'))
        MC.run(name='resonator_scan'+self.msmt_suffix)

        self.heterodyne_instr.frequency(previous_freq)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_homodyne_acqusition_delay(self, delays=None, MC=None,
                                          analyze=True, close_fig=True):
        """
        Varies the delay between the homodyne modulation signal and
        acquisition. Measures the transmittance.
        """
        if delays is None:
            raise ValueError("Unspecified delays for measure_homodyne_"
                             "acquisition_delay")

        if MC is None:
            MC = self.MC

        # set number of averages to 1 due to a readout bug
        previous_nr_averages = self.heterodyne_instr.nr_averages()
        self.heterodyne_instr.nr_averages(1)
        previous_delay = self.heterodyne_instr.acquisition_delay()

        self.prepare_for_continuous_wave()
        MC.set_sweep_function(pw.wrap_par_to_swf(
            self.heterodyne_instr.acquisition_delay))
        MC.set_sweep_points(delays)
        MC.set_detector_function(det.Heterodyne_probe(
            self.heterodyne_instr, trigger_separation=5e-6,
            demod_mode='single'))
        MC.run(name='acquisition_delay_scan'+self.msmt_suffix)

        self.heterodyne_instr.acquisition_delay(previous_delay)
        self.heterodyne_instr.nr_averages(previous_nr_averages)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_spectroscopy(self, freqs=None, MC=None, analyze=True,
                             close_fig=True, update=False):
        """ Varies qubit drive frequency and measures the resonator
        transmittance """
        if freqs is None:
            raise ValueError("Unspecified frequencies for "
                                 "measure_spectroscopy and no previous value")

        if MC is None:
            MC = self.MC

        self.prepare_for_continuous_wave()
        self.cw_source.on()

        MC.set_sweep_function(pw.wrap_par_to_swf(self.cw_source.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(det.Heterodyne_probe(
            self.heterodyne_instr, trigger_separation=2.8e-6,
            demod_mode='single'))
        MC.run(name='spectroscopy'+self.msmt_suffix)

        self.cw_source.off()

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_rabi(self):
        raise NotImplementedError()

    def measure_T1(self):
        raise NotImplementedError()

    def measure_ramsey(self):
        raise NotImplementedError()

    def measure_echo(self):
        raise NotImplementedError()

    def measure_allxy(self):
        raise NotImplementedError()

    def measure_ssro(self):
        raise NotImplementedError()

    def find_resonator_frequency(self, update=True, freqs=None, MC=None,
                                 close_fig=True):
        """
        Finds the resonator frequency by performing a heterodyne experiment
        if freqs == None it will determine a default range dependent on the
        last known frequency of the resonator.
        """
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

        self.measure_heterodyne_spectroscopy(freqs, MC, analyze=False)

        HA = ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig,
                                  fitting_model='lorentzian')
        f0 = HA.fit_results.params['f0'].value*1e9
        df0 = HA.fit_results.params['f0'].stderr*1e9
        Q = HA.fit_results.params['Q'].value
        dQ = HA.fit_results.params['Q'].stderr
        if f0 > max(freqs) or f0 < min(freqs):
            logging.warning('exracted frequency outside of range of scan')
        elif df0 > f0:
            logging.warning('resonator frequency uncertainty greater than '
                            'value')
        elif dQ > Q:
            logging.warning('resonator Q factor uncertainty greater than '
                            'value')
        elif update:  # don't update if there was trouble
            self.f_RO_resonator(f0)
            self.Q_RO_resonator(Q)
            self.heterodyne_instr.frequency(f0)
        return f0

    def find_homodyne_acqusition_delay(self, delays=None, update=True, MC=None,
                                         close_fig=True):
        """
        Finds the acquisition delay for a homodyne experiment that corresponds
        to maximal signal strength.
        """
        if delays is None:
            delays = np.linspace(0,1e-6,100)

        if MC is None:
            MC = self.MC

        self.measure_homodyne_acqusition_delay(delays, MC, analyze=False)

        DA = ma.Acquisition_Delay_Analysis(label=self.msmt_suffix,
                                           close_fig=close_fig)
        d = DA.max_delay

        if update:
            self.optimal_acquisition_delay(d)
            self.heterodyne_instr.acquisition_delay(d)
        return d