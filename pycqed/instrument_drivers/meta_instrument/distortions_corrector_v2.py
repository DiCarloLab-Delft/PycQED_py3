"""
N.B. this is a v2 of the distortions_corrector started in Dec 2017 -MAR

This file contains the Distortions_corrector.

An object used to correct distortions using an interactive procedure
involving repeated measurements.
"""

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from qcodes.utils import validators as vals
import pycqed.analysis.fitting_models as fm
import pycqed.measurement.kernel_functions_ZI as kf
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


class Distortion_corrector(Instrument):

    def __init__(self, name,
                 nr_plot_points: int=1000,
                 sampling_rate: float=2.4e9,
                 auto_save_plots: bool=True, **kw):
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
                    changed in self.cfg_nr_plot_points().
        '''
        super().__init__(name, **kw)
        # Initialize instance variables
        # Plotting
        self._y_min = 0
        self._y_max = 1
        self._stop_idx = -1
        self._start_idx = 0
        self._t_start_loop = 0  # sets x range for plotting during loop
        self._t_stop_loop = 30e-6
        self.add_parameter('cfg_nr_plot_points',
                           initial_value=nr_plot_points,
                           parameter_class=ManualParameter)
        self.sampling_rate = sampling_rate
        self.add_parameter('cfg_sampling_rate',
                           initial_value=sampling_rate,
                           parameter_class=ManualParameter)
        self.add_parameter('instr_dist_kern',
                           parameter_class=InstrumentRefParameter)


        # Files
        self.filename = ''
        # where traces and plots are saved
        # self.data_dir = self.kernel_object.kernel_dir()
        self._iteration = 0
        self.auto_save_plots = auto_save_plots

        # Data
        self.waveform = []
        self.time_pts = []
        self.new_step = []

        # Fitting
        self.known_fit_models = ['exponential', 'high-pass', 'spline']
        self.fit_model = None
        self.edge_idx = None
        self.fit_res = None
        self.predicted_waveform = None

        # Default fit model used in the interactive loop
        self._fit_model_loop = 'exponential'

        self._loop_helpstring = str(
            'h:      Print this help.\n'
            'q:      Quit the loop.\n'
            'm:      Remeasures the trace. \n'
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
            '        self.cfg_nr_plot_points() (default=1000).\n'
            'square_amp <amp> \n'
            '        Set the square_amp used to normalize measured waveforms.\n'
            '        If amp = "?" the current square_amp is printed.')

        # Make window for plots
        self.vw = QtPlot(window_title=name, figsize=(600, 400))

    # def load_kernel_file(self, filename):
    #     '''
    #     Loads kernel dictionary (containing kernel and metadata) from a JSON
    #     file. This function looks only in the directory
    #     self.kernel_object.kernel_dir() for the file.
    #     Returns a dictionary of the kernel and metadata.
    #     '''
    #     with open(os.path.join(self.kernel_object.kernel_dir(),
    #                            filename)) as infile:
    #         data = json.load(infile)
    #     return data

    # def save_kernel_file(self, kernel_dict, filename):
    #     '''
    #     Saves kernel dictionary (containing kernel and metadata) to a JSON
    #     file in the directory self.kernel_object.kernel_dir().
    #     '''
    #     directory = self.kernel_object.kernel_dir()
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     with open(os.path.join(directory, filename),
    #               'w') as outfile:
    #         json.dump(kernel_dict, outfile, indent=True, sort_keys=True)

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

    # def open_new_correction(self, kernel_length, AWG_sampling_rate, name):
    #     '''
    #     Opens a new correction with name 'filename', i.e. initializes the
    #     combined kernel to a Dirac delta and empties kernel_list of the
    #     kernel object associated with self.

    #     Args:
    #         kernel_length (float):
    #                 Length of the corrections kernel in s.
    #         AWG_sampling_rate (float):
    #                 Sampling rate of the AWG generating the flux pulses in Hz.
    #         name (string):
    #                 Name for the new kernel. The files will be named after
    #                 this, but with different suffixes (e.g. '_combined.json').
    #     '''
    #     self.kernel_length = int(kernel_length * AWG_sampling_rate)
    #     self.filename = name
    #     self._iteration = 0

    #     # Initialize kernel to Dirac delta
    #     init_ker = np.zeros(self.kernel_length)
    #     init_ker[0] = 1
    #     self.kernel_combined_dict = {
    #         'metadata': {},  # dictionary of kernel dictionaries
    #         'kernel': list(init_ker),
    #         'iteration': 0
    #     }
    #     self.save_kernel_file(self.kernel_combined_dict,
    #                           '{}_combined.json'.format(self.filename))

    #     # Configure kernel object
    #     self.kernel_object.add_kernel_to_kernel_list(
    #         '{}_combined.json'.format(self.filename))

    # def resume_correction(self, filename):
    #     '''
    #     Loads combined kernel from the specified file and prepares for adding
    #     new corrections to that kernel.
    #     '''
    #     # Remove '_combined.json' from filename
    #     self.filename = '_'.join(filename.split('_')[:-1])
    #     self.kernel_combined_dict = self.load_kernel_file(filename)
    #     self._iteration = self.kernel_combined_dict['iteration']
    #     self.kernel_length = len(self.kernel_combined_dict['kernel'])

    #     # Configure kernel object
    #     self.kernel_object.kernel_list([])
    #     self.kernel_object.add_kernel_to_kernel_list(filename)

    # def empty_kernel_list(self):
    #     self.kernel_object.kernel_list([])

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
        self.fit_model = lmfit.Model(fm.gain_corr_ExpDecayFunc)
        self.fit_model.set_param_hint('gc',
                                      value=self.waveform[self._stop_idx],
                                      vary=True)
        self.fit_model.set_param_hint('amp',
                                      value=(self.waveform[self._start_idx] -
                                             self.waveform[self._stop_idx]),
                                      vary=True)
        self.fit_model.set_param_hint('tau',
                                      value=end_time_fit-start_time_fit,
                                      vary=True)
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
        amp = fit_res.best_values['amp']
        tau = fit_res.best_values['tau']

        # Check if parameters are physical and print warnings if not
        if tau < 0:
            print('Warning: unphysical tau = {} (expect tau > 0).'
                  .format(tau))

        # Save the results
        self.fit_res = fit_res
        self.predicted_waveform = kf.exponential_decay_correction(
            self.waveform, tau=tau, amp=amp,
            sampling_rate=self.scope_sampling_rate)



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

        tau = fit_res.best_values['tau']

        # Check if parameters are physical and print warnings if not
        if tau < 0:
            print('Warning: unphysical tau = {} (expect tau > 0).'
                  .format(tau))
        # Save the fit results and predicted correction
        self.fit_res = fit_res

        self.predicted_waveform = kf.bias_tee_correction(
            self.waveform, tau=tau, sampling_rate=self.scope_sampling_rate)



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

    def plot_trace(self, start_time=-.5e-6, stop_time=10e-6, nr_plot_pts=4000,
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
                    symbol='o', symbolSize=5, name='Measured waveform')

        if self.predicted_waveform is not None:
            start_idx = np.argmin(np.abs(self.time_pts - start_time))
            stop_idx = np.argmin(np.abs(self.time_pts - stop_time))
            step = max(
                int(len(self.time_pts[start_idx:stop_idx]) // nr_plot_pts), 1)
            self.vw.add(x=self.time_pts[start_idx:stop_idx:step],
                        y=self.predicted_waveform[start_idx:stop_idx:step],
                        name='Predicted waveform')

        self.vw.add(x=[start_time, stop_time],
                    y=[self.waveform[stop_idx]]*2,
                    color=(150, 150, 150))
        self.vw.add(x=[start_time, stop_time],
                    y=[0]*2,
                    color=(150, 150, 150))
        self.vw.add(x=[start_time, stop_time],
                    y=[-self.waveform[stop_idx]]*2,
                    color=(150, 150, 150))



        # Set the y-range to previous value
        if save_y_range and not err:
            self.vw.subplots[0].setYRange(y_range[0], y_range[1])

        # Labels need to be set in the end, else they don't show sometimes
        self.vw.subplots[0].getAxis('bottom').setLabel('t', 's')
        self.vw.subplots[0].getAxis('left').setLabel('Amplitude', 'V')

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
                    color = '#2ca02c',
                    name='Fit')

        # Labels need to be set in the end, else they don't show sometimes
        self.vw.subplots[0].getAxis('bottom').setLabel('t', 's')
        self.vw.subplots[0].getAxis('left').setLabel('amp', 'V')

    def test_new_kernel(self):
        '''
        Save the new kernel self.new_kernel_dict to its own file and add it to
        the kernel list of the kernel object.
        '''

        self._iteration
        dist_kern = self.instr_dist_kern.get_instr()

        if self._fit_model_loop == 'high-pass':
            tau = self.fit_res.best_values['tau']
            model = {'model': 'high-pass', 'params': {'tau':tau}}
            dist_kern.set('filter_model_{:02}'.format(self._iteration), model)
        elif self._fit_model_loop == 'exponential':
            tau = self.fit_res.best_values['tau']
            amp = self.fit_res.best_values['amp']
            model = {'model': 'exponential', 'params':{'tau':tau, 'amp':amp}}
            dist_kern.set('filter_model_{:02}'.format(self._iteration), model)
        else:
            raise NotImplementedError

    def apply_new_kernel(self):
        '''
        The correction number (self._iteration) is incremented, such that
        the kernel file for the latest distortion is not overwritten anymore.
        '''
        self._iteration += 1  # This correction is considered completed.

    def discard_new_kernel(self):
        '''
        Removes a the last kernel that was added from the distortions.
        '''
        dist_kern = self.instr_dist_kern.get_instr()
        dist_kern.set('filter_model_{:02}'.format(self._iteration), {})


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
        if inp == 'y':
            print('Resetting all kernels in kernel object')
            self.instr_dist_kern.get_instr().reset_kernels()
            self._iteration = 0
        else:
            # Continue working with current kernel; determine how many filters
            # already exist.
            self._iteration = self.instr_dist_kern.get_instr().get_first_empty_filter()
            print('Starting from iteration {}'.format(self._iteration))


        # 1. Measure trace and plot
        self.measure_trace()

        # Set up initial plot range
        self._t_start_loop = 0
        self._t_stop_loop = self.time_pts[-1]

        self.plot_trace(self._t_start_loop, self._t_stop_loop,
                        save_y_range=False, nr_plot_pts=self.cfg_nr_plot_points())

        # LOOP STARTS HERE
        # Default fit model used, high-pass is typically the first model
        self._fit_model_loop = 'high-pass'
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
            elif self._fit_model_loop == 'high-pass':
                self.fit_high_pass(fit_start, fit_stop)
            elif self._fit_model_loop == 'spline':
                self.fit_spline(fit_start, fit_stop)

            self.plot_fit(self._t_start_loop, self._t_stop_loop,
                          nr_plot_pts=self.cfg_nr_plot_points())

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
                            nr_plot_pts=self.cfg_nr_plot_points())

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
                                nr_plot_pts=self.cfg_nr_plot_points())
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
                                nr_plot_pts=self.cfg_nr_plot_points())
            else:
                # Fit also plotted
                self.plot_fit(self._t_start_loop, self._t_stop_loop,
                              nr_plot_pts=self.cfg_nr_plot_points())

        elif inp_elements[0] == 'm':
            # Remeasures the trace
            print('Remeasuring trace')
            self.measure_trace()
            self.plot_trace(self._t_start_loop, self._t_stop_loop,
                save_y_range=False, nr_plot_pts=self.cfg_nr_plot_points())


        elif inp_elements[0] == 'h':
            print(self._loop_helpstring)

        elif inp_elements[0] == 'q':
            self.print_summary()
            quit = True
            repeat = False

        elif inp_elements[0] == 'p':
            if len(inp_elements) == 1:
                try:
                    # for param, val in self.new_kernel_dict['fit'].items():
                    #     print('{} = {}'.format(param, val))
                    print(self.fit_res.best_values)
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
        self.fit_res.best_values.update(par_dict)
        self.fitted_waveform = self.fit_res.eval(
            t=self.time_pts[self._start_idx:self._stop_idx],
            tau=self.fit_res.best_values['tau'])

        if self._fit_model_loop == 'high-pass':
            self.predicted_waveform = kf.bias_tee_correction(
                self.waveform,  tau=self.fit_res.best_values['tau'],
                 sampling_rate=self.scope_sampling_rate)

        elif self._fit_model_loop == 'exponential':
            self.predicted_waveform = kf.exponential_decay_correction(
                self.waveform, tau=self.fit_res.best_values['tau'],
                amp=self.fit_res.best_values['amp'],
                sampling_rate=self.scope_sampling_rate)

        # The fit results still have to be updated
        self.plot_fit(self._t_start_loop, self._t_stop_loop,
                      nr_plot_pts=self.cfg_nr_plot_points())

    def print_summary(self):
        '''
        Prints a summary of all corrections that have been applied.
        '''
        self.instr_dist_kern.get_instr().print_overview()


    def _set_square_amp(self, square_amp: float):
        old_square_amp = self.square_amp
        self.square_amp = square_amp
        if len(self.waveform) > 0:
            self.waveform = self.waveform*old_square_amp/self.square_amp
        self.plot_trace(self._t_start_loop, self._t_stop_loop,
                        nr_plot_pts=self.cfg_nr_plot_points())
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

    def __init__(self, name, measure_scope_trace,
                 nr_plot_points: int=1000, **kw):
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
                    changed in self.cfg_nr_plot_points().
        '''
        super().__init__(name, sampling_rate=2.4e9,
                         nr_plot_points=nr_plot_points, **kw)

        self.add_parameter('instr_flux_lutman',
                           parameter_class=InstrumentRefParameter)

        self.measure_scope_trace = measure_scope_trace

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
        self.instr_flux_lutman.get_instr().load_waveform_onto_AWG_lookuptable(
            'square', regenerate_waveforms=True)
        if verbose:
            print('Measuring trace...')
        self.raw_time_pts, self.waveform = self.measure_scope_trace()

        # Find rising edge
        if self.edge_idx == None:
            # this is because finding the edge is usally most robust in the
            # beginning
            self.edge_idx = detect_edge(self.waveform, edge_level=0.02)
        self.time_pts = self.raw_time_pts - self.raw_time_pts[self.edge_idx]
        self.scope_sampling_rate = 1/(self.time_pts[1]-self.time_pts[0])


class RT_distortion_corrector_QWG(Distortion_corrector):

    def __init__(self, name, measure_scope_trace,
                 nr_plot_points: int=1000, **kw):
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
                    changed in self.cfg_nr_plot_points().
        '''
        super().__init__(name, sampling_rate=1e9,
                         nr_plot_points=nr_plot_points, **kw)

        self.add_parameter('instr_flux_lutman',
                           parameter_class=InstrumentRefParameter)

        self.measure_scope_trace = measure_scope_trace

        self.raw_waveform = []
        self.raw_time_pts = []
        self._edge_for_trace = 0.05

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
        self.instr_flux_lutman.get_instr().load_waveform_onto_AWG_lookuptable(
            'square', regenerate_waveforms=True)
        if verbose:
            print('Measuring trace...')
        self.raw_time_pts, self.waveform = self.measure_scope_trace()

        # Find rising edge
        if self.edge_idx == None:
            # this is because finding the edge is usally most robust in the
            # beginning
            self.edge_idx = detect_edge(self.waveform, edge_level=self._edge_for_trace)
        self.time_pts = self.raw_time_pts - self.raw_time_pts[self.edge_idx]
        self.scope_sampling_rate = 1/(self.time_pts[1]-self.time_pts[0])


# def detect_edge(y, edge_level=0.1):
#     """
#     Trivial edge detection algortihm
#     """
#     edge_idx = -1
#     abs_edge_change = (np.max(y) - np.min(y))*edge_level
#     for i in range(len(y) - 1):
#         if (y[i+1] - y[i]) > abs_edge_change:
#             edge_idx = i
#             print('edge detected at idx:', edge_idx)
#             break
#     if edge_idx < 0:
#         # This is an important error but should not crash the
#         # process
#         logging.warning('Failed to find rising edge.')
#         edge_idx = 0
#     return edge_idx


def detect_edge(y, edge_level=0.10):
    """
    Detects the first crossing of some threshold and returns the index
    """

    th = y > edge_level*np.max(y)
    # marks all but the first occurence of True to False
    th[1:][th[:-1] & th[1:]] = False
    return np.where(th)[0][0]