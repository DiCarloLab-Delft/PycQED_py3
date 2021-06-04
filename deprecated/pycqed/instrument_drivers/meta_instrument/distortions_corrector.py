"""
N.B. I have started a v2 of this instrument that should be forward compatible
with real time linear distortions in Dec 2017 -MAR

This file contains the Distortions_corrector.

An object used to correct distortions using an interactive procedure
involving repeated measurements.
"""
import pycqed.analysis.fitting_models as fm
import pycqed.measurement.kernel_functions as kf
import numpy as np
import scipy.linalg
import scipy.interpolate as sc_intpl
from pycqed.analysis import fitting_models as fit_mods
import lmfit
import os.path
import datetime
import json
import logging
import PyQt5
from qcodes.plots.pyqtgraph import QtPlot


class Distortion_corrector():

    def __init__(self, kernel_object, square_amp: float=1,
                 nr_plot_points: int=1000,
                 sampling_rate: float=1e9,
                 auto_save_plots: bool=True):
        '''
        Instantiates an object.

        Args:
            kernel_object (Instrument):
                    kernel object instrument that handles applying kernels to
                    flux pulses.
            square_amp (float):
                    Amplitude of the square pulse that is applied. This is
                    needed for correct normalization of the step response.
            nr_plot_points (int):
                    Number of points of the waveform that are plotted. Can be
                    changed in self.nr_plot_points.
        '''
        # Initialize instance variables
        # Plotting
        self._y_min = 0
        self._y_max = 1
        self._stop_idx = -1
        self._start_idx = 0
        self._t_start_loop = 0  # sets x range for plotting during loop
        self._t_stop_loop = 30e-6
        self.nr_plot_points = nr_plot_points
        self.sampling_rate = sampling_rate

        self.square_amp = square_amp

        # Kernels
        self.kernel_length = 0
        self.kernel_combined_dict = {}
        self.new_kernel_dict = {}

        self.kernel_object = kernel_object

        # Files
        self.filename = ''
        # where traces and plots are saved
        self.data_dir = self.kernel_object.kernel_dir()
        self._iteration = 0
        self.auto_save_plots = auto_save_plots

        # Data
        self.waveform = []
        self.time_pts = []
        self.new_step = []

        # Fitting
        self.known_fit_models = ['exponential', 'double_exponential',
                                 'triple_exponential', 'high-pass',
                                 'damped_osc', 'spline']
        self.fit_model = None
        self.fit_res = None

        # Default fit model used in the interactive loop
        self._fit_model_loop = 'exponential'

        self._loop_helpstring = str(
            'h:      Print this help.\n'
            'q:      Quit the loop.\n'
            'p <pars>:\n'
            '        Print the parameters of the last fit if pars not given.\n'
            '        If pars are given in the form of JSON string, \n'
            '        e.g., {"parA": a, "parB": b} the parameters of the last\n'
            '        fit are updated with those provided.'
            's <filename>:\n'
            '        Save the current plot to "filename.png".\n'
            'model <name>:\n'
            '        Choose the fit model that is used.\n'
            '        Available models:\n'
            '           ' + str(self.known_fit_models) + '\n'
            'xrange <min> <max>:\n'
            '        Set the x-range of the plot to (min, max). The points\n'
            '        outside this range are not plotted. The number of\n'
            '        points plotted in the given interval is fixed to\n'
            '        self.nr_plot_points (default=1000).\n'
            'square_amp <amp> \n'
            '        Set the square_amp used to normalize measured waveforms.\n'
            '        If amp = "?" the current square_amp is printed.')

        # Make window for plots
        self.vw = QtPlot(window_title='Distortions', figsize=(600, 400))

    def load_kernel_file(self, filename):
        '''
        Loads kernel dictionary (containing kernel and metadata) from a JSON
        file. This function looks only in the directory
        self.kernel_object.kernel_dir() for the file.
        Returns a dictionary of the kernel and metadata.
        '''
        with open(os.path.join(self.kernel_object.kernel_dir(),
                               filename)) as infile:
            data = json.load(infile)
        return data

    def save_kernel_file(self, kernel_dict, filename):
        '''
        Saves kernel dictionary (containing kernel and metadata) to a JSON
        file in the directory self.kernel_object.kernel_dir().
        '''
        directory = self.kernel_object.kernel_dir()
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, filename),
                  'w') as outfile:
            json.dump(kernel_dict, outfile, indent=True, sort_keys=True)

    def save_plot(self, filename):
        try:
            directory = self.kernel_object.kernel_dir()
            if not os.path.exists(directory):
                os.makedirs(directory)
            # FIXME: saving disabled as it is currently broken.
            # self.vw.save(os.path.join(self.kernel_object.kernel_dir(),
            #                           filename))
        except Exception as e:
            logging.warning('Could not save plot.')

    def open_new_correction(self, kernel_length, AWG_sampling_rate, name):
        '''
        Opens a new correction with name 'filename', i.e. initializes the
        combined kernel to a Dirac delta and empties kernel_list of the
        kernel object associated with self.

        Args:
            kernel_length (float):
                    Length of the corrections kernel in s.
            AWG_sampling_rate (float):
                    Sampling rate of the AWG generating the flux pulses in Hz.
            name (string):
                    Name for the new kernel. The files will be named after
                    this, but with different suffixes (e.g. '_combined.json').
        '''
        self.kernel_length = int(kernel_length * AWG_sampling_rate)
        self.filename = name
        self._iteration = 0

        # Initialize kernel to Dirac delta
        init_ker = np.zeros(self.kernel_length)
        init_ker[0] = 1
        self.kernel_combined_dict = {
            'metadata': {},  # dictionary of kernel dictionaries
            'kernel': list(init_ker),
            'iteration': 0
        }
        self.save_kernel_file(self.kernel_combined_dict,
                              '{}_combined.json'.format(self.filename))

        # Configure kernel object
        self.kernel_object.add_kernel_to_kernel_list(
            '{}_combined.json'.format(self.filename))

    def resume_correction(self, filename):
        '''
        Loads combined kernel from the specified file and prepares for adding
        new corrections to that kernel.
        '''
        # Remove '_combined.json' from filename
        self.filename = '_'.join(filename.split('_')[:-1])
        self.kernel_combined_dict = self.load_kernel_file(filename)
        self._iteration = self.kernel_combined_dict['iteration']
        self.kernel_length = len(self.kernel_combined_dict['kernel'])

        # Configure kernel object
        self.kernel_object.kernel_list([])
        self.kernel_object.add_kernel_to_kernel_list(filename)

    def empty_kernel_list(self):
        self.kernel_object.kernel_list([])

    def measure_trace(self, verbose=True):
        raise NotImplementedError(
            'Base class is not attached to physical instruments and does not '
            'implement measurements.')

    def fit_exp_model(self, start_time_fit, end_time_fit):
        '''
        Fits an exponential of the form
            A * exp(-t/tau) + offset
        to the last trace that was measured (self.waveform).
        The fit model and result are saved in self.fit_model and self.fit_res,
        respectively. The new predistortion kernel and information about the
        fit is stored in self.new_kernel_dict.

        Args:
            start_time_fit (float): start of the fitted interval
            end_time_fit (float):   end of the fitted interval
        '''
        self._start_idx = np.argmin(np.abs(self.time_pts - start_time_fit))
        self._stop_idx = np.argmin(np.abs(self.time_pts - end_time_fit))

        # Prepare the fit model
        self.fit_model = lmfit.Model(fm.ExpDecayFunc)
        self.fit_model.set_param_hint('offset',
                                      value=self.waveform[self._stop_idx],
                                      vary=True)
        self.fit_model.set_param_hint('amplitude',
                                      value=(self.waveform[self._start_idx] -
                                             self.waveform[self._stop_idx]),
                                      vary=True)
        self.fit_model.set_param_hint('tau',
                                      value=end_time_fit-start_time_fit,
                                      vary=True)
        self.fit_model.set_param_hint('n', value=1, vary=False)
        params = self.fit_model.make_params()

        # Do the fit
        fit_res = self.fit_model.fit(
            data=self.waveform[self._start_idx:self._stop_idx],
            t=self.time_pts[self._start_idx:self._stop_idx],
            params=params)

        self.fitted_waveform = fit_res.eval(
            t=self.time_pts[self._start_idx:self._stop_idx])

        # Analytic form of the predistorted square pulse (input that creates a
        # square pulse at the output)
        offset = fit_res.best_values['offset']
        amp = fit_res.best_values['amplitude']
        tau = fit_res.best_values['tau']

        # Check if parameters are physical and print warnings if not
        if tau < 0:
            print('Warning: unphysical tau = {} (expect tau > 0).'
                  .format(tau))
        if offset < 0:
            print('Warning: unphysical offset = {} (expect offset > 0)'.format(
                offset))

        # Save the results
        self.fit_res = fit_res
        new_ker = kf.decay_kernel(
            amp=amp, tau=tau, offset=offset,
            length=self.kernel_object.corrections_length(),
            sampling_rate=self.sampling_rate)

        self.new_kernel_dict = {
            'name': self.filename + '_' + str(self._iteration),
            'filter_params': {
                'b0': 1 / (offset + amp),
                'b1': 1 / (tau * (offset + amp)),
                'a1': -offset / (tau * (offset + amp))
            },
            'fit': {
                'model': 'exponential',
                'amp': amp,
                'tau': tau,
                'offset': offset
            },
            'kernel': list(new_ker)
        }

    def fit_double_exp_model(self, start_time_fit, end_time_fit):
        '''
        Fit a double exponential decay of the form
            C + A1 * exp(-t/tau1) + A2 * exp(-t/tau2)
        to last measured trace (self.waveform). The predistortion is
        calculated only from the exponential with the longest decay constant.
        The fit model and result are saved in self.fit_model and self.fit_res,
        respectively. The new predistortion kernel and information about the
        fit is stored in self.new_kernel_dict.

        Args:
            start_time_fit (float): start of the fitted interval
            end_time_fit (float):   end of the fitted interval
        '''
        self._start_idx = np.argmin(np.abs(self.time_pts - start_time_fit))
        self._stop_idx = np.argmin(np.abs(self.time_pts - end_time_fit))

        # Prepare the fit model
        self.fit_model = lmfit.Model(fm.DoubleExpDecayFunc)
        self.fit_model.set_param_hint('offset',
                                      value=self.waveform[self._stop_idx],
                                      vary=True)
        self.fit_model.set_param_hint('amp1',
                                      value=(self.waveform[self._start_idx] -
                                             self.waveform[self._stop_idx])/2,
                                      vary=True)
        self.fit_model.set_param_hint('tau1',
                                      value=end_time_fit-start_time_fit,
                                      vary=True)
        self.fit_model.set_param_hint('amp2',
                                      value=(self.waveform[self._start_idx] -
                                             self.waveform[self._stop_idx])/4,
                                      vary=True)
        self.fit_model.set_param_hint('tau2',
                                      value=(end_time_fit-start_time_fit)/2,
                                      vary=True)

        self.fit_model.set_param_hint('n', value=1, vary=False)
        params = self.fit_model.make_params()

        # Do the fit
        fit_res = self.fit_model.fit(
            data=self.waveform[self._start_idx:self._stop_idx],
            t=self.time_pts[self._start_idx:self._stop_idx],
            params=params)
        self.fitted_waveform = fit_res.eval(
            t=self.time_pts[self._start_idx:self._stop_idx])

        # Find exponential with largest decay constant
        maxInd = np.argmax([fit_res.best_values['tau{}'.format(i+1)]
                            for i in range(2)]) + 1

        # Analytic form of the predistorted square pulse (input that creates a
        # square pulse at the output)
        C = fit_res.best_values['offset']
        A = fit_res.best_values['amp{}'.format(maxInd)]
        a = A / C
        aTilde = a / (a + 1)
        tau = fit_res.best_values['tau{}'.format(maxInd)]
        tauTilde = tau * (a + 1)

        tPts = np.arange(0, self.kernel_length/self.sampling_rate,
                         1/self.sampling_rate)
        predist_step = (1 - aTilde * np.exp(-tPts/tauTilde)) / C

        # Check if parameters are physical and print warnings if not
        if tau < 0:
            print('Warning: unphysical tau = {} (expect tau > 0).'
                  .format(tau))
        if C < 0:
            print('Warning: unphysical C = {} (expect C > 0)'.format(C))

        # Save the results
        self.fit_res = fit_res
        self.new_step = predist_step
        # Take every 5th point because sampling rate of AWG is 1 GHz and
        # sampling rate of scope is 5 GHz.
        # TODO: un-hardcode this
        new_ker = kf.kernel_from_kernel_stepvec(predist_step)

        self.new_kernel_dict = {
            'name': self.filename + '_' + str(self._iteration),
            'filter_params': {
                'b0': 1 / (C + A),
                'b1': 1 / (tau * (C + A)),
                'a1': -C / (tau * (C + A))
            },
            'fit': {
                'model': 'double_exponential',
                'A': A,
                'tau': tau,
                'C': C
            },
            'kernel': list(new_ker)
        }

    def fit_triple_exp_model(self, start_time_fit, end_time_fit):
        '''
        Fit a triple exponential decay of the form
            C + A1 * exp(-t/tau1) + A2 * exp(-t/tau2) + A3 * exp(-t/tau3)
        to last measured trace (self.waveform). The predistortion is
        calculated only from the exponential with the longest decay constant.
        The fit model and result are saved in self.fit_model and self.fit_res,
        respectively. The new predistortion kernel and information about the
        fit is stored in self.new_kernel_dict.

        Args:
            start_time_fit (float): start of the fitted interval
            end_time_fit (float):   end of the fitted interval
        '''
        self._start_idx = np.argmin(np.abs(self.time_pts - start_time_fit))
        self._stop_idx = np.argmin(np.abs(self.time_pts - end_time_fit))

        # Prepare the fit model
        self.fit_model = lmfit.Model(fm.TripleExpDecayFunc)
        self.fit_model.set_param_hint('offset',
                                      value=self.waveform[self._stop_idx],
                                      vary=True)
        self.fit_model.set_param_hint('amp1',
                                      value=(self.waveform[self._start_idx] -
                                             self.waveform[self._stop_idx])/2,
                                      vary=True)
        self.fit_model.set_param_hint('tau1',
                                      value=end_time_fit-start_time_fit,
                                      vary=True)
        self.fit_model.set_param_hint('amp2',
                                      value=(self.waveform[self._start_idx] -
                                             self.waveform[self._stop_idx])/4,
                                      vary=True)
        self.fit_model.set_param_hint('tau2',
                                      value=(end_time_fit-start_time_fit)/2,
                                      vary=True)
        self.fit_model.set_param_hint('amp3',
                                      value=(self.waveform[self._start_idx] -
                                             self.waveform[self._stop_idx])/4,
                                      vary=True)
        self.fit_model.set_param_hint('tau3',
                                      value=(end_time_fit-start_time_fit)/2,
                                      vary=True)

        self.fit_model.set_param_hint('n', value=1, vary=False)
        params = self.fit_model.make_params()

        # Do the fit
        fit_res = self.fit_model.fit(
            data=self.waveform[self._start_idx:self._stop_idx],
            t=self.time_pts[self._start_idx:self._stop_idx],
            params=params)
        self.fitted_waveform = fit_res.eval(
            t=self.time_pts[self._start_idx:self._stop_idx])

        # Find exponential with largest decay constant
        maxInd = np.argmax([fit_res.best_values['tau{}'.format(i+1)]
                            for i in range(3)]) + 1

        # Analytic form of the predistorted square pulse (input that creates a
        # square pulse at the output)
        C = fit_res.best_values['offset']
        A = fit_res.best_values['amp{}'.format(maxInd)]
        a = A / C
        aTilde = a / (a + 1)
        tau = fit_res.best_values['tau{}'.format(maxInd)]
        tauTilde = tau * (a + 1)

        tPts = np.arange(0, self.kernel_length/self.sampling_rate,
                         1/self.sampling_rate)
        predist_step = (1 - aTilde * np.exp(-tPts/tauTilde)) / C

        # Check if parameters are physical and print warnings if not
        if tau < 0:
            print('Warning: unphysical tau = {} (expect tau > 0).'
                  .format(tau))
        if C < 0:
            print('Warning: unphysical C = {} (expect C > 0)'.format(C))

        # Save the results
        self.fit_res = fit_res
        self.new_step = predist_step
        # Take every 5th point because sampling rate of AWG is 1 GHz and
        # sampling rate of scope is 5 GHz.
        # TODO: un-hardcode this
        new_ker = kf.kernel_from_kernel_stepvec(predist_step)

        self.new_kernel_dict = {
            'name': self.filename + '_' + str(self._iteration),
            'filter_params': {
                'b0': 1 / (C + A),
                'b1': 1 / (tau * (C + A)),
                'a1': -C / (tau * (C + A))
            },
            'fit': {
                'model': 'triple_exponential',
                'A': A,
                'tau': tau,
                'C': C
            },
            'kernel': list(new_ker)
        }

    def fit_high_pass(self, start_time_fit, end_time_fit):
        '''
        Fits a model for a simple RC high-pass
            exp(-t/tau), tau = RC
        to the last trace that was measured (self.waveform).
        The fit model and result are saved in self.fit_model and self.fit_res,
        respectively. The new predistortion kernel and information about the
        fit is stored in self.new_kernel_dict.

        Args:
            start_time_fit (float): start of the fitted interval
            end_time_fit (float):   end of the fitted interval
        '''
        self._start_idx = np.argmin(np.abs(self.time_pts - start_time_fit))
        self._stop_idx = np.argmin(np.abs(self.time_pts - end_time_fit))

        # Prepare the fit model: exponential, where only tau is varied
        self.fit_model = lmfit.Model(fm.ExpDecayFunc)
        self.fit_model.set_param_hint('tau',
                                      value=end_time_fit-start_time_fit,
                                      vary=True)
        self.fit_model.set_param_hint('offset',
                                      value=0,
                                      vary=False)
        self.fit_model.set_param_hint('amplitude',
                                      value=1,
                                      vary=True)
        self.fit_model.set_param_hint('n', value=1, vary=False)
        params = self.fit_model.make_params()

        # Do the fit
        fit_res = self.fit_model.fit(
            data=self.waveform[self._start_idx:self._stop_idx],
            t=self.time_pts[self._start_idx:self._stop_idx],
            params=params)
        self.fitted_waveform = fit_res.eval(
            t=self.time_pts[self._start_idx:self._stop_idx])

        # Analytic form of the predistorted square pulse (input that creates a
        # square pulse at the output)
        tau = fit_res.best_values['tau']

        tPts = np.arange(0, self.kernel_length/self.sampling_rate,
                         1/self.sampling_rate)
        predist_step = tPts/tau + 1

        # Check if parameters are physical and print warnings if not
        if tau < 0:
            print('Warning: unphysical tau = {} (expect tau > 0).'
                  .format(tau))

        # Save the results
        self.fit_res = fit_res
        self.new_step = predist_step

        new_ker = kf.kernel_from_kernel_stepvec(predist_step)

        self.new_kernel_dict = {
            'name': self.filename + '_' + str(self._iteration),
            'filter_params': {
                'b0': 1,
                'b1': 1/tau,  # incorrect because not units of #samples
                'a1': 0  # 'a0': 1 by definition
            },
            'fit': {
                'model': 'high-pass',
                'tau': tau
            },
            'kernel': list(new_ker)
        }

    def fit_damped_osc(self, start_time_fit: float, end_time_fit: float):
        '''
        Fits a model for an underdamped oscillation (RLC circuit)
            A * exp(-t/tau) * cos(omega * t) + C
        to the last trace that was measured (self.waveform).
        The fit model and result are saved in self.fit_model and self.fit_res,
        respectively. The new predistortion kernel and information about the
        fit is stored in self.new_kernel_dict.

        Args:
            start_time_fit (float): start of the fitted interval
            end_time_fit (float):   end of the fitted interval
        '''
        self._start_idx = np.argmin(np.abs(self.time_pts - start_time_fit))
        self._stop_idx = np.argmin(np.abs(self.time_pts - end_time_fit))

        # Prepare the fit model: exponentially damped oscillation.
        self.fit_model = lmfit.Model(fm.ExpDampOscFunc)
        self.fit_model.set_param_hint('amplitude',
                                      value=1,
                                      vary=True)
        self.fit_model.set_param_hint('frequency',
                                      value=1 / (end_time_fit-start_time_fit),
                                      vary=True)
        self.fit_model.set_param_hint('tau',
                                      value=end_time_fit - start_time_fit,
                                      vary=True)
        self.fit_model.set_param_hint('exponential_offset',
                                      value=self.waveform[self._stop_idx],
                                      vary=True)

        self.fit_model.set_param_hint('phase',
                                      value=0,
                                      vary=False)
        self.fit_model.set_param_hint('n',
                                      value=1,
                                      vary=False)
        self.fit_model.set_param_hint('oscillation_offset',
                                      value=0,
                                      vary=False)
        params = self.fit_model.make_params()

        # Do the fit
        fit_res = self.fit_model.fit(
            data=self.waveform[self._start_idx:self._stop_idx],
            t=self.time_pts[self._start_idx:self._stop_idx],
            params=params)
        self.fitted_waveform = fit_res.eval(
            t=self.time_pts[self._start_idx:self._stop_idx])

        # Analytic form of the predistorted square pulse (input that creates a
        # square pulse at the output)
        A = fit_res.best_values['amplitude']
        omega = 2 * np.pi * fit_res.best_values['frequency']
        tau = fit_res.best_values['tau']
        C = fit_res.best_values['exponential_offset']
        l = (omega * tau)**2

        tPts = np.arange(0, self.kernel_length*1e-9,
                         1e-9)  # TODO: sampling rate
        predist_step = (1 + tPts / tau * (1 + l) + l *
                        (np.exp(-tPts / tau) - 1)) / A

        # Check if parameters are physical and print warnings
        if tau < 0:
            print('Warning: unphysical tau = {} (expect tau > 0).'
                  .format(tau))
        if np.abs(C) > 1e-3:
            print('Warning: unphysical offset C = {} (expect C = 0). '
                  'Offset is ignored in kernel inversion.'.format(C))
        if A < 0:
            print('Warning: unphysical amplitude A = {} (expect A > 0).'
                  .format(A))

        # Save the results
        self.fit_res = fit_res
        self.new_step = predist_step
        new_ker = kf.kernel_from_kernel_stepvec(predist_step)

        self.new_kernel_dict = {
            'name': self.filename + '_' + str(self._iteration),
            'filter_params': {
                'b0': 1 / A,
                'b1': 2 / (tau * A),
                'b2': (1 + l) / (A * tau**2),
                'a1': 1 / tau
            },
            'fit': {
                'model': 'damped_osc',
                'A': A,
                'omega': omega,
                'tau': tau,
                'C': C
            },
            'kernel': list(new_ker)
        }

    def fit_spline(self, start_time_fit, end_time_fit, s=0.001,
                   weight_tau='inf'):
        '''
        Fit the data using a spline interpolation.
        The fit model and result are saved in self.fit_model and self.fit_res,
        respectively. The new predistortion kernel and information about the
        fit is stored in self.new_kernel_dict.

        Args:
            start_time_fit (float):
                    Start of the fitted interval.
            end_time_fit (float):
                    End of the fitted interval.
            s (float):
                    Smoothing condition for the spline. See documentation on
                    scipy.interpolate.splrep for more information.
            weight_tau (float or 'auto'):
                    The points are weighted by a decaying exponential with
                    time constant weight_tau.
                    If this is 'auto' the time constant is chosen to be
                    end_time_fit.
                    If this is 'inf' all weights are set to 1.
                    Smaller weight means the spline can have a larger
                    distance from this point. See documentation on
                    scipy.interpolate.splrep for more information.
        '''
        self._start_idx = np.argmin(np.abs(self.time_pts - start_time_fit))
        self._stop_idx = np.argmin(np.abs(self.time_pts - end_time_fit))

        if weight_tau == 'auto':
            weight_tau = end_time_fit

        if weight_tau == 'inf':
            splWeights = np.ones(self._stop_idx - self._start_idx)
        else:
            splWeights = np.exp(
                -self.time_pts[self._start_idx:self._stop_idx] / weight_tau)

        splTuple = sc_intpl.splrep(
            x=self.time_pts[self._start_idx:self._stop_idx],
            y=self.waveform[self._start_idx:self._stop_idx],
            w=splWeights,
            s=s)
        splStep = sc_intpl.splev(
            self.time_pts[self._start_idx:self._stop_idx],
            splTuple, ext=3)

        # Pad step response with avg of last 10 points (assuming the user has
        # chosen the range such that the response has become flat)
        splStep = np.concatenate((splStep,
                                  np.ones(self.kernel_length - len(splStep)) *
                                  np.mean(splStep[-10:])))

        self.fit_res = None
        self.fit_model = None
        self.fitted_waveform = splStep[:self._stop_idx-self._start_idx]

        # Calculate the kernel and invert it.
        h = np.empty_like(splStep)
        h[0] = splStep[0]
        h[1:] = splStep[1:] - splStep[:-1]

        filterMatrix = np.zeros((len(h), len(h)))
        for n in range(len(h)):
            for m in range(n+1):
                filterMatrix[n, m] = h[n - m]

        new_ker = scipy.linalg.inv(filterMatrix)[:, 0]
        self.new_step = np.convolve(new_ker,
                                    np.ones(len(splStep)))[:len(splStep)]
        self.new_kernel_dict = {
            'name': self.filename + '_' + str(self._iteration),
            'filter_params': {},
            'fit': {
                'model': 'spline',
                's': s,
                'weight_tau': weight_tau
            },
            'kernel': list(new_ker)
        }

    def plot_trace(self, start_time=0, stop_time=10e-6, nr_plot_pts=4000,
                   save_y_range=True):
        '''
        Plot last trace that was measured (self.waveform).
        Args:
            start_time (float): Start of the plotted interval.
            stop_time (float):  End of the plotted interval.
            save_y_range (bool):
                                Keep the current y-range of the plot.
        '''
        start_idx = np.argmin(np.abs(self.time_pts - start_time))
        stop_idx = np.argmin(np.abs(self.time_pts - stop_time))
        step = max(
            int(len(self.time_pts[start_idx:stop_idx]) // nr_plot_pts), 1)

        # Save the y-range of the plot if a window is open.
        err = False
        try:
            x_range, y_range = self.vw.subplots[0].getViewBox().viewRange()
        except Exception as e:
            print(e)
            err = True

        plot_t_pts = self.time_pts[:len(self.waveform)]

        # Plot
        self.vw.clear()
        self.vw.add(x=plot_t_pts[start_idx:stop_idx:step],
                    y=self.waveform[start_idx:stop_idx:step],
                    symbol='o')
        self.vw.add(x=[start_time, stop_time],
                    y=[1]*2,
                    color=(150, 150, 150))
        self.vw.add(x=[start_time, stop_time],
                    y=[0]*2,
                    color=(150, 150, 150))
        self.vw.add(x=[start_time, stop_time],
                    y=[-1]*2,
                    color=(150, 150, 150))

        # Set the y-range to previous value
        if save_y_range and not err:
            self.vw.subplots[0].setYRange(y_range[0], y_range[1])

        # Labels need to be set in the end, else they don't show sometimes
        self.vw.subplots[0].getAxis('bottom').setLabel('t', 's')
        self.vw.subplots[0].getAxis('left').setLabel('normalized amp', '')

    def plot_fit(self, start_time=0, stop_time=10e-6, save_y_range=True,
                 nr_plot_pts=4000):
        '''
        Plot last trace that was measured (self.waveform) and the latest fit.
        Args:
            start_time (float): Start of the plotted interval.
            stop_time (float):  End of the plotted interval.
            save_y_range (bool):
                                Keep the current y-range of the plot.
        '''
        self.plot_trace(start_time=start_time, stop_time=stop_time,
                        save_y_range=save_y_range, nr_plot_pts=nr_plot_pts)
        self.vw.add(x=self.time_pts[self._start_idx:self._stop_idx],
                    y=self.fitted_waveform,
                    color='#ff7f0e')

        # Labels need to be set in the end, else they don't show sometimes
        self.vw.subplots[0].getAxis('bottom').setLabel('t', 's')
        self.vw.subplots[0].getAxis('left').setLabel('amp', 'V')

    def test_new_kernel(self):
        '''
        Save the new kernel self.new_kernel_dict to its own file and add it to
        the kernel list of the kernel object.
        '''
        # Save new kernel to file
        full_name = '{}_{}.json'.format(self.filename, self._iteration)
        self.save_kernel_file(self.new_kernel_dict, full_name)

        # Add new kernel to the kernel list in the kernel object
        self.kernel_object.add_kernel_to_kernel_list(full_name)

    def apply_new_kernel(self):
        '''
        Includes the latest distortion in the combined kernel by convolving
        it. The correction number (self._iteration) is incremented, such that
        the kernel file for the latest distortion is not overwritten anymore.
        '''
        # Integrate the new kernel into kernel_combined
        ker_combined = np.convolve(
            self.kernel_combined_dict['kernel'],
            self.new_kernel_dict['kernel'])[:self.kernel_length]
        self.kernel_combined_dict['kernel'] = list(ker_combined)

        self.kernel_combined_dict['metadata'][self.new_kernel_dict['name']] = {
            'fit': self.new_kernel_dict['fit'],
            'filter_params': self.new_kernel_dict['filter_params'],
            'name': self.new_kernel_dict['name']
        }

        # Remove the last correction from the kernel list in the kernel
        # object, since it is now included in the combined kernel.
        test_kernel_name = '{}_{}.json'.format(self.filename, self._iteration)
        if test_kernel_name in self.kernel_object.kernel_list():
            self.kernel_object.remove_kernel_from_kernel_list(
                '{}_{}.json'.format(self.filename, self._iteration))

        self.kernel_combined_dict['iteration'] += 1
        self._iteration += 1  # This correction is considered completed.

        # Overwrite the file where kernel_combined is saved
        self.save_kernel_file(self.kernel_combined_dict,
                              '{}_combined.json'.format(self.filename))
        # Have to manually set _config_chaned in the kernel object, because
        # we're just overwriting the same file and kernel_list is not
        # necessarily changed.
        self.kernel_object._config_changed = True

    def discard_new_kernel(self):
        '''
        Removes a the last kernel that was added from the distortions.
        '''
        # Remove kernel from kernel list
        test_kernel_name = '{}_{}.json'.format(self.filename, self._iteration)
        if test_kernel_name in self.kernel_object.kernel_list():
            self.kernel_object.remove_kernel_from_kernel_list(
                '{}_{}.json'.format(self.filename, self._iteration))

    def recalculate_combined_kernel(self):
        '''
        Recalculates the combined kernel from its components. This can be used
        for example to change the length of the combined kernel, as the
        individual kernels are stored in full length (length of measured
        trace).
        '''
        # Initialize combined kernel to dirac delta
        init_ker = np.zeros(int(self.kernel_length))
        init_ker[0] = 1
        self.kernel_combined_dict['kernel'] = list(init_ker)

        # Convolve the kernels
        for kName in self.kernel_combined_dict['metadata'].keys():
            fname = '{}.json'.format(kName)
            ker_dict = self.load_kernel_file(fname)

            ker_combined = np.convolve(
                self.kernel_combined_dict['kernel'],
                ker_dict['kernel'])[:self.kernel_length]
            self.kernel_combined_dict['kernel'] = list(ker_combined)

        # Save and upload the new combined kernel
        self.save_kernel_file(self.kernel_combined_dict,
                              '{}_combined.json'.format(self.filename))
        # Have to manually set _config_chaned in the kernel object, because
        # we're just overwriting the same file and kernel_list is not
        # changed.
        self.kernel_object._config_changed = True

    def interactive_loop(self):
        '''
        Starts interactive loop to iteratively add corrections.
        '''
        # Loop:
        # 1. Measure trace and plot
        # 2. Fit and plot
        # 3. Test correction and plot
        #   -> discard: back to 2.
        #   -> approve: continue with 4.
        # 4. Apply correction
        #   -> quit?
        #   -> back to 2.
        print('********\n'
              'Interactive room-temperature distortion corrections\n'
              '********\n'
              'At any prompts you may use these commands:\n'
              + self._loop_helpstring)

        while True:
            inp = input('New kernel? ([y]/n) ')
            if inp in ['y', 'n', '']:
                break
        if inp == 'y' or inp == '':
            filename = input('File name: (default: yymmdd_RT_corr) ')
            if filename == '':
                filename = '{}_RT_corr'.format(
                    datetime.date.today().strftime('%y%m%d'))
            self.open_new_correction(
                kernel_length=self.kernel_object.corrections_length(),
                AWG_sampling_rate=self.sampling_rate,
                name=filename)
        else:
            # Continue working with current kernel; nothing to do
            pass

        # 1. Measure trace and plot
        self.measure_trace()

        # Set up initial plot range
        self._t_start_loop = 0
        self._t_stop_loop = self.time_pts[-1]

        self.plot_trace(self._t_start_loop, self._t_stop_loop,
                        save_y_range=False, nr_plot_pts=self.nr_plot_points)

        # LOOP STARTS HERE
        self._fit_model_loop = 'exponential'  # Default fit model used
        while True:
            print('\n-- Correction number {} --'.format(self._iteration))
            print('Current fit model: {}'.format(self._fit_model_loop))
            # 2. Fit and plot
            repeat = True
            while repeat:
                inp = input('Fit range: ')
                repeat, quit = self._handle_interactive_input(inp, 'any')
                if not quit and not repeat:
                    try:
                        inp = inp.split(' ')
                        fit_start = float(inp[0])
                        fit_stop = float(inp[1])
                    except Exception as e:
                        print('input format: "t_start t_stop"')
                        repeat = True
            if quit:
                # Exit loop
                break

            if self._fit_model_loop == 'exponential':
                self.fit_exp_model(fit_start, fit_stop)
            elif self._fit_model_loop == 'double_exponential':
                self.fit_double_exp_model(fit_start, fit_stop)
            elif self._fit_model_loop == 'triple_exponential':
                self.fit_triple_exp_model(fit_start, fit_stop)
            elif self._fit_model_loop == 'high-pass':
                self.fit_high_pass(fit_start, fit_stop)
            elif self._fit_model_loop == 'damped_osc':
                self.fit_damped_osc(fit_start, fit_stop)
            elif self._fit_model_loop == 'spline':
                self.fit_spline(fit_start, fit_stop)

            self.plot_fit(self._t_start_loop, self._t_stop_loop,
                          nr_plot_pts=self.nr_plot_points)

            repeat = True
            while repeat:
                inp = input('Accept? ([y]/n) ').strip()
                repeat, quit = self._handle_interactive_input(inp,
                                                              ['y', 'n', ''])
            if quit:
                # Exit loop
                break
            elif inp != 'y' and inp != '':
                # Go back to 2.
                continue

            # Fit was accepted -> save plot
            if self.auto_save_plots:
                self.save_plot('fit_{}.png'.format(self._iteration))

            # 3. Test correction and plot
            # Save last data, in case new distortion is rejected.
            previous_t = self.time_pts
            previous_wave = self.waveform

            print('Testing new correction.')
            self.test_new_kernel()
            self.measure_trace()
            self.plot_trace(self._t_start_loop, self._t_stop_loop,
                            nr_plot_pts=self.nr_plot_points)

            repeat = True
            while repeat:
                inp = input('Accept? ([y]/n) ').strip()
                repeat, quit = self._handle_interactive_input(inp,
                                                              ['y', 'n', ''])
            if quit:
                # Exit loop
                break
            elif inp != 'y' and inp != '':
                print('Discarding new correction.')
                self.discard_new_kernel()
                self.time_pts = previous_t
                self.waveform = previous_wave
                self.plot_trace(self._t_start_loop, self._t_stop_loop,
                                nr_plot_pts=self.nr_plot_points)
                # Go back to 2.
                continue

            # Correction was accepted -> save plot
            if self.auto_save_plots:
                self.save_plot('trace_{}.png'.format(self._iteration))

            # 4. Apply correction
            print('Applying new correction.')
            self.apply_new_kernel()

    def _handle_interactive_input(self, inp, valid_inputs):
        '''
        Handles input from user in an interactive loop session. Takes
        action in special cases.
        Args:
            inp (string):   Input given by the user.
            valid_inputs (list of strings or 'any'):
                            List of inputs that are accepted. Any input is
                            accepted if this is 'any'.

        Returns:
            repeat (bool):  Should the input prompt be repeated.
            quit (bool):    Should the loop be exited.
        '''
        repeat = True
        quit = False

        inp_elements = inp.split(' ')
        if (inp_elements[0].lower() == 'xrange'
                and len(inp_elements) == 3):
            self._t_start_loop = float(inp_elements[1])
            self._t_stop_loop = float(inp_elements[2])
            if len(self.vw.traces) == 4:  # 3 grey lines + 1 data trace
                # Only data plotted
                self.plot_trace(self._t_start_loop, self._t_stop_loop,
                                nr_plot_pts=self.nr_plot_points)
            else:
                # Fit also plotted
                self.plot_fit(self._t_start_loop, self._t_stop_loop,
                              nr_plot_pts=self.nr_plot_points)

        elif inp_elements[0] == 'h':
            print(self._loop_helpstring)

        elif inp_elements[0] == 'q':
            self.print_summary()
            quit = True
            repeat = False

        elif inp_elements[0] == 'p':
            if len(inp_elements) == 1:
                try:
                    for param, val in self.new_kernel_dict['fit'].items():
                        print('{} = {}'.format(param, val))
                except KeyError:
                    print('No fit has been done yet!')
            else:
                self._update_latest_params(json_string=inp[1:])

        elif (inp_elements[0] == 's' and len(inp_elements == 2)):
            self.save_plot('{}.png'.format(inp_elements[1]))
            print('Current plot saved.')

        elif (inp_elements[0] == 'model' and len(inp_elements) == 2):
            if inp_elements[1] in self.known_fit_models:
                self._fit_model_loop = str(inp_elements[1])
                print('Using fit model "{}".'.format(self._fit_model_loop))
            else:
                print('Model "{}" unknown. Please choose from {}.'
                      .format(inp_elements[1], self.known_fit_models))

        elif (inp_elements[0] == 'square_amp' and len(inp_elements) == 2):
            if inp_elements[1] == '?':
                print('square_amp is {}'.format(self.square_amp))
            else:
                try:
                    square_amp = float(inp_elements[1])
                    self._set_square_amp(square_amp)
                except ValueError:
                    print('Square_amp can only be set to a float')

        elif valid_inputs != 'any':
            if inp not in valid_inputs:
                print('Valid inputs: {}'.format(valid_inputs))
            else:
                repeat = False

        else:
            # Any input ok
            repeat = False

        return repeat, quit

    def _update_latest_params(self, json_string):
        """
        Uses a JSON formatted string to update the parameters of the
        latest fit.

        For each model does the following
            1. update the 'fit' dict
            2. update the 'filter_params' dict
            3. recalculate the kernel
            4. calculate the new "fit"
            5. Plot the new "fit"

        Currently only supported for the high-pass and exponential model.
        """
        try:
            par_dict = json.loads(json_string)
        except Exception as e:
            print(e)
            return

        # 1. update the 'fit' dict
        self.new_kernel_dict['fit'].update(par_dict)

        if self._fit_model_loop == 'high-pass':
            tau = self.new_kernel_dict['fit']['tau']
            # 2. update the filter_params' dict
            self.new_kernel_dict['filter_params'] = {
                'b0': 1,
                'b1': 1/tau,
                'a1': 0}
            # 3. recalculate the kernel
            tPts = np.arange(0, self.kernel_length/self.sampling_rate,
                             1/self.sampling_rate)
            predist_step = tPts/tau + 1
            self.new_kernel_dict['kernel'] = \
                list(kf.kernel_from_kernel_stepvec(predist_step))
            # 4. calculate the fit
            self.fitted_waveform = self.fit_res.eval(
                t=self.time_pts[self._start_idx:self._stop_idx], tau=tau)

        elif self._fit_model_loop == 'exponential':
            amp = self.new_kernel_dict['fit']['amp']
            tau = self.new_kernel_dict['fit']['tau']
            offset = self.new_kernel_dict['fit']['offset']

            # 2. update the 'filter_params' dict
            self.new_kernel_dict['filter_params'] = {
                'b0': 1 / (offset + amp),
                'b1': 1 / (tau * (offset + amp)),
                'a1': -offset / (tau * (offset + amp))}
            # 3. recalculate the new kernel
            self.new_kernel_dict['kernel'] = list(kf.decay_kernel(
                amp=amp, tau=tau, offset=offset,
                length=self.kernel_object.corrections_length(),
                sampling_rate=self.sampling_rate))
            # 4. calculate the fit
            self.fitted_waveform = self.fit_res.eval(
                t=self.time_pts[self._start_idx:self._stop_idx],
                offset=offset, amplitude=amp, tau=tau)

        elif self._fit_model_loop == 'double_exponential':
            print('Updating params not implemented for double_exponential')
        elif self._fit_model_loop == 'triple_exponential':
            print('Updating params not implemented for triple_exponential')
        elif self._fit_model_loop == 'damped_osc':
            print('Updating params not implemented for damped_osc')
        elif self._fit_model_loop == 'spline':
            print('Updating params not implemented for spline')

        # The fit results still have to be updated
        self.plot_fit(self._t_start_loop, self._t_stop_loop,
                      nr_plot_pts=self.nr_plot_points)

    def print_summary(self):
        '''
        Prints a summary of all corrections that have been applied.
        '''
        print('\n********\n'
              'Summary of corrected distortions:\n'
              '********')
        print('Number of applied corrections: {}'.format(self._iteration))
        i = 0
        for key, item in self.kernel_combined_dict['metadata'].items():
            print('{}: {}'.format(i, key))
            print('Fit parameters:')
            for param, val in item['fit'].items():
                print('\t{} = {}'.format(param, val))
            i += 1

        print('End of summary.')
        print('********')

    def detect_edge_and_normalize_wf(self, waveform, edge_level=0.1):
        """
        Detects the edge of a waveform and cuts of the leading zeros as well
        as normalizes the waveform to "self.square_amp".

        args:
            waveform      (array) : the raw_waveform
            edge_level    (float) : fractional change to declare an edge
        Returns:
            norm_waveform (array) : shifted and normalized waveform
        """
        # Normalize waveform and find rising edge
        edge_idx = -1
        abs_edge_change = (np.max(waveform) - np.min(waveform))*edge_level

        for i in range(len(waveform) - 1):
            if waveform[i+1] - waveform[i] > abs_edge_change:
                edge_idx = i
                break
        if edge_idx < 0:
            # This is an important error but should not crash the
            # process
            logging.warning('Failed to find rising edge.')
            self.edge_idx = 0
            return waveform
        norm_waveform = (waveform -
                         np.mean(waveform[:edge_idx-1])) / self.square_amp
        # FIXME: this should be returned and other methods should use this
        self.edge_idx = edge_idx
        return norm_waveform

    def _set_square_amp(self, square_amp: float):
        old_square_amp = self.square_amp
        self.square_amp = square_amp
        if len(self.waveform) > 0:
            self.waveform = self.waveform*old_square_amp/self.square_amp
        self.plot_trace(self._t_start_loop, self._t_stop_loop,
                        nr_plot_pts=self.nr_plot_points)
        print('Updated square amp from {} to {}'.format(old_square_amp,
                                                        square_amp))


class Dummy_distortion_corrector(Distortion_corrector):

    def measure_trace(self, verbose=True):
        sampling_rate = 5e9
        # Generate some dummy square wave
        self.raw_waveform = np.concatenate([np.zeros(100), np.ones(50000),
                                            np.zeros(1000)])
        noise = np.random.rand(len(self.raw_waveform)) * 0.02
        self.raw_waveform += noise

        self.raw_waveform = np.convolve(
            self.raw_waveform,  self.kernel_object.get_decay_kernel_1())

        self.raw_time_pts = np.arange(len(self.raw_waveform))/sampling_rate
        # Normalize waveform and find rising edge
        self.waveform = self.detect_edge_and_normalize_wf(self.raw_waveform)
        self.time_pts = np.arange(len(self.waveform))/sampling_rate


class RT_distortion_corrector_AWG8(Distortion_corrector):

    def __init__(self, flux_lutman, measure_scope_trace, square_amp: float,
                 nr_plot_points: int=1000, ):
        '''
        Instantiates an object.
        Note: Sampling rate of the scope is assumed to be 5 GHz. Sampling rate
        of the AWG is assumed to be 1 GHz.

        Args:
            flux_lutman (Instrument):
                    Lookup table manager for the AWG.
            oscilloscope (Instrument):
                    Oscilloscope instrument.

            nr_plot_points (int):
                    Number of points of the waveform that are plotted. Can be
                    changed in self.nr_plot_points.
        '''
        self.flux_lutman = flux_lutman
        self.measure_scope_trace = measure_scope_trace
        super().__init__(
            kernel_object=flux_lutman.instr_distortion_kernel.get_instr(),
            square_amp=square_amp, sampling_rate=2.4e9,
            nr_plot_points=nr_plot_points)

        self.raw_waveform = []
        self.raw_time_pts = []

    def measure_trace(self, verbose=True):
        '''
        Measure a trace with the oscilloscope.
        Raw data is saved to self.raw_time_pts and self.raw_waveform.
        Data clipped to start at the rising edge is saved to self.time_pts
        and self.waveform.

        N.B. This measure trace method makes two assumptions
            1. The scope is properly configured.
            2. The CCLight is running the correct program that triggers the
               AWG8.
        '''
        # Upload waveform
        self.flux_lutman.load_waveform_onto_AWG_lookuptable(
            'square', regenerate_waveforms=True)
        if verbose:
            print('Measuring trace...')
        self.raw_time_pts, self.raw_waveform = self.measure_scope_trace()

        # Normalize waveform and find rising edge
        self.waveform = self.detect_edge_and_normalize_wf(self.raw_waveform,
                                                          edge_level=0.05)
        self.time_pts = self.raw_time_pts - self.raw_time_pts[self.edge_idx]


class RT_distortion_corrector_QWG(Distortion_corrector):
    '''
    Class for applying corrections for room-temperature distortions to flux
    pulses.
    '''

    # TODO:
    #   - check TODOs in this file
    #   - waveform length will become parameter in QWG_flux_lutman
    #       -> adapt this class to handle that
    def __init__(self, AWG_lutman, measure_scope_trace, square_amp: float,
                 nr_plot_points: int=1000):
        '''
        Instantiates an object.
        Note: Sampling rate of the scope is assumed to be 5 GHz. Sampling rate
        of the AWG is assumed to be 1 GHz.

        Args:
            AWG_lutman (Instrument):
                    Lookup table manager for the AWG.
            oscilloscope (Instrument):
                    Oscilloscope instrument.

            nr_plot_points (int):
                    Number of points of the waveform that are plotted. Can be
                    changed in self.nr_plot_points.
        '''
        self.AWG_lutman = AWG_lutman
        self.measure_scope_trace = measure_scope_trace
        super().__init__(kernel_object=AWG_lutman.F_kernel_instr.get_instr(),
                         square_amp=square_amp,
                         nr_plot_points=nr_plot_points)

        self.raw_waveform = []
        self.raw_time_pts = []

    def measure_trace(self, verbose=True):
        '''
        Measure a trace with the oscilloscope.
        Raw data is saved to self.raw_time_pts and self.raw_waveform.
        Data clipped to start at the rising edge is saved to self.time_pts
        and self.waveform.
        '''
        # Upload waveform
        self.AWG_lutman.QWG.get_instr().stop()
        self.AWG_lutman.load_pulse_onto_AWG_lookuptable('square')
        self.AWG_lutman.QWG.get_instr().start()

        if verbose:
            print('Measuring trace...')
        self.raw_time_pts, self.raw_waveform = self.measure_scope_trace()

        # Measurement should be implemented using measurement_control
        # self.data_dir should be set to data dir of last measurement
        # a = ma.MeasurementAnalysis()
        # self.data_dir = a.folder

        # Normalize waveform and find rising edge
        self.waveform = self.detect_edge_and_normalize_wf(self.raw_waveform)
        # Sampling rate hardcoded to 5 GHz
        self.time_pts = np.arange(len(self.waveform)) / 5e9


class RT_distortion_corrector_5014(Distortion_corrector):
    '''
    Class for applying corrections for room-temperature distortions to flux
    pulses.
    '''

    # TODO:
    #   - check TODOs in this file
    #   - waveform length will become parameter in QWG_flux_lutman
    #       -> adapt this class to handle that
    def __init__(self, kernel, AWG, oscilloscope, square_amp: float,
                 nr_plot_points: int=1000):
        '''
        Instantiates an object.
        Note: Sampling rate of the scope is assumed to be 5 GHz. Sampling rate
        of the AWG is assumed to be 1 GHz.

        Args:
            AWG_lutman (Instrument):
                    Lookup table manager for the AWG.
            oscilloscope (Instrument):
                    Oscilloscope instrument.
            square_amp (float):
                    Amplitude of the square pulse that is applied. This is
                    needed for correct normalization of the step response.
            nr_plot_points (int):
                    Number of points of the waveform that are plotted. Can be
                    changed in self.nr_plot_points.
        '''

        self.scope = oscilloscope
        self.AWG = AWG
        super().__init__(kernel_object=kernel, square_amp=square_amp,
                         nr_plot_points=nr_plot_points)

        self.raw_waveform = []
        self.raw_time_pts = []

    def measure_trace(self, verbose=True, upload=True):
        '''
        Measure a trace with the oscilloscope.
        Raw data is saved to self.raw_time_pts and self.raw_waveform.
        Data clipped to start at the rising edge is saved to self.time_pts
        and self.waveform.
        '''
        # Upload waveform
        if upload:
            from pycqed.measurement.pulse_sequences import fluxing_sequences

            distortions_dict = {
                'ch_list': ['ch3'],
                'ch3': self.kernel_object.get_corrections_kernel()
            }
            self.AWG.stop()
            pulse_pars = {'pulse_type': 'SquarePulse',
                          'pulse_delay': .1e-6,
                          'channel': 'ch3',
                          'amplitude': self.square_amp,
                          'length': 8e-6,
                          'dead_time_length': 2e-6}
            fluxing_sequences.single_pulse_seq(pulse_pars=pulse_pars,
                                               distortion_dict=distortions_dict)
            self.AWG.start()
            self.AWG.run()

        if verbose:
            print('Measuring trace...')
        self.raw_time_pts, self.raw_waveform = self.scope.measure_trace()

        # Measurement should be implemented using measurement_control
        # self.data_dir should be set to data dir of last measurement
        # a = ma.MeasurementAnalysis()
        # self.data_dir = a.folder

        # Normalize waveform and find rising edge
        self.waveform = self.detect_edge_and_normalize_wf(self.raw_waveform)
        # Sampling rate hardcoded to 5 GHz
        self.time_pts = np.arange(len(self.waveform)) / 5e9


class Cryo_distortion_corrector(Distortion_corrector):

    def __init__(self, time_pts, qubit, demodulate: bool=False,
                 f_demod: float=0, square_amp: float=1,
                 nr_plot_points: int=4000,
                 chunk_size: int=32):
        '''
        Instantiate an object.
        This class uses qubit.measure_ram_z to measure the step response.

        Args:
            time_pts (array):
                    Time points at which the Ram-Z should be measured
                    (x-axis).
            qubit (Instrument):
                    Qubit object on which the experiment is performed.
            demodulate (bool):
                    Whether to demodulate the signal in the analysis.
            f_demod (float):
                    Demodulation frequency used in the analysis.
            square_amp (float):
                    Amplitude of the square pulse used to normalize the step
                    response.
            nr_plot_points (int):
                    Number of points of the waveform that are plotted. Can be
                    changed in self.nr_plot_points.
            chunk_size (int):
                    Chunk size used in the measurement.
        '''
        # TODO: add filename initialization
        self.AWG_lutman = qubit.flux_LutMan.get_instr()
        self.qubit = qubit
        super().__init__(
            kernel_object=self.AWG_lutman.F_kernel_instr.get_instr(),
            square_amp=square_amp,
            nr_plot_points=nr_plot_points)
        self.time_pts = time_pts
        self.phases = []

        self.chunk_size = chunk_size
        self.V_per_phi0 = self.qubit.V_per_phi0()
        self.V_0 = self.qubit.V_offset()
        self.f_demod = f_demod
        self.demodulate = demodulate

    def measure_trace(self, verbose=True):
        '''
        Measures the step response using two Ram-Z experiments with 90 deg
        phase difference between them.
        '''
        self.AWG_lutman.QWG.get_instr().stop()
        self.AWG_lutman.load_pulse_onto_AWG_lookuptable('square')
        self.AWG_lutman.QWG.get_instr().start()

        a = self.qubit.measure_ram_z(
            self.time_pts,
            chunk_size=self.chunk_size,
            demodulate=self.demodulate,
            f_demod=self.f_demod,
            flux_amp_analysis=self.square_amp)

        self.waveform = a.step_response
        self.phases = a.phases

    def fit_spline_phase(self, start_time_fit: float, end_time_fit: float,
                         s: float=0.001, weight_tau='auto',
                         disable_inversion: bool=False):
        '''
        Fit the phase data using a spline interpolation.

        Args:
            start_time_fit (float):
                    Start of the fitted interval.
            end_time_fit (float):
                    End of the fitted interval.
            s (float):
                    Smoothing condition for the spline. See documentation on
                    scipy.interpolate.splrep for more information.
            weight_tau (float or 'auto'):
                    The points are weighted by a decaying exponential with
                    time constant weight_tau.
                    If this is 'auto' the time constant is chosen to be
                    end_time_fit.
                    If this is 'inf' all weights are set to 1.
                    Smaller weight means the spline can have a larger
                    distance from this point. See documentation on
                    scipy.interpolate.splrep for more information.
        '''
        self._start_idx = np.argmin(np.abs(self.time_pts - start_time_fit))
        self._stop_idx = np.argmin(np.abs(self.time_pts - end_time_fit))

        if weight_tau == 'auto':
            weight_tau = end_time_fit

        if weight_tau == 'inf':
            splWeights = np.ones(self._stop_idx - self._start_idx)
        else:
            splWeights = np.exp(
                -self.time_pts[self._start_idx:self._stop_idx] / weight_tau)

        splTuple = sc_intpl.splrep(
            x=self.time_pts[self._start_idx:self._stop_idx],
            y=self.phases[self._start_idx:self._stop_idx],
            w=splWeights,
            s=s)
        splObj = sc_intpl.BSpline(*splTuple)
        splPhase = sc_intpl.splev(
            self.time_pts[self._start_idx:self._stop_idx], splObj)
        splDetun = sc_intpl.splev(
            self.time_pts[self._start_idx:self._stop_idx],
            splObj.derivative()) / (360.0) \
            + self.f_demod * self.demodulate
        splStep = np.array([fit_mods.Qubit_freq_to_dac(
            frequency=self.qubit.f_max() - df,
            f_max=self.qubit.f_max(),
            E_c=self.qubit.E_c(),
            dac_sweet_spot=self.qubit.V_offset(),
            V_per_phi0=self.qubit.V_per_phi0(),
            asymmetry=0) for df in splDetun]) / self.square_amp

        # Pad step response with avg of last 10 points (assuming the user has
        # chosen the range such that the response has become flat)
        splStep = np.concatenate(
            (splStep,
             np.ones(self.kernel_length - len(splPhase)) *
             np.mean(splStep[-10:])))

        self.fit_res = None
        self.fit_model = None
        self.fitted_phases = splPhase[:self._stop_idx-self._start_idx]
        self.fitted_waveform = splStep[:self._stop_idx-self._start_idx]

        if not disable_inversion:
            # Calculate the kernel and invert it.
            h = np.empty_like(splStep)
            h[0] = splStep[0]
            h[1:] = splStep[1:] - splStep[:-1]

            filterMatrix = np.zeros((len(h), len(h)))
            for n in range(len(h)):
                for m in range(n+1):
                    filterMatrix[n, m] = h[n - m]

            new_ker = scipy.linalg.inv(filterMatrix)[:, 0]
            self.new_step = np.convolve(new_ker,
                                        np.ones(len(splStep)))[:len(splStep)]
            self.new_kernel_dict = {
                'name': self.filename + '_' + str(self._iteration),
                'filter_params': {},
                'fit': {
                    'model': 'spline_phase',
                    's': s,
                    'weight_tau': weight_tau
                },
                'kernel': list(new_ker)
            }

    def plot_phase(self, start_time: float=0, stop_time: float=1e-6,
                   save_y_range: bool=True, nr_plot_pts: int=4000):
        origWaveform = self.waveform
        self.waveform = self.phases
        self.plot_trace(start_time=start_time,
                        stop_time=stop_time,
                        save_y_range=save_y_range,
                        nr_plot_pts=nr_plot_pts)
        self.vw.subplots[0].getAxis('bottom').setLabel('t', 's')
        self.vw.subplots[0].getAxis('left').setLabel('phase', 'deg')

        self.waveform = origWaveform

    def plot_fit_phase(self, start_time: float=0, stop_time: float=1e-6,
                       save_y_range: bool=True, nr_plot_pts: int=4000):
        origWaveform = self.waveform
        self.waveform = self.phases
        self.plot_trace(start_time=start_time,
                        stop_time=stop_time,
                        save_y_range=save_y_range,
                        nr_plot_pts=nr_plot_pts)
        self.vw.add(x=self.time_pts[self._start_idx:self._stop_idx],
                    y=self.fitted_phases,
                    color='#ff7f0e')
        # Labels need to be set in the end, else they don't show sometimes
        self.vw.subplots[0].getAxis('bottom').setLabel('t', 's')
        self.vw.subplots[0].getAxis('left').setLabel('phase', 'deg')

        self.waveform = origWaveform
