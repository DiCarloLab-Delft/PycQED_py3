import pycqed.analysis.fitting_models as fm
import pycqed.measurement.kernel_functions as kf
import pycqed.instrument_drivers.physical_instruments.RTO1024 as RTO
import pycqed.measurement.detector_functions as det
import pycqed.analysis.measurement_analysis as ma
import pycqed.measurement.measurement_control as mc
import numpy as np
import numpy.linalg
import scipy.interpolate as sc_intpl
import lmfit
import os.path
import datetime
import json
import logging
from qcodes.plots.pyqtgraph import QtPlot


class Distortion_corrector():
    def __init__(self, kernel_object, nr_plot_points=1000):
        '''
        Instantiates an object.

        Args:
            kernel_object (Instrument):
                    kernel object instrument that handles applying kernels to
                    flux pulses.
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

        # Kernels
        self.kernel_length = 0
        self.kernel_combined_dict = {}
        self.new_kernel_dict = {}

        self.ker_obj = kernel_object

        # Files
        self.filename = ''
        self.kernel_dir = self.ker_obj.kernel_dir()
        self.data_dir = self.kernel_dir  # where traces and plots are saved
        self._iteration = 0

        # Data
        self.waveform = []
        self.time_pts = []
        self.new_step = []

        # Fitting
        self.known_fit_models = ['exponential', 'high-pass', 'spline']
        self.fit_model = None
        self.fit_res = None

        # Default fit model used in the interactive loop
        self._fit_model_loop = 'exponential'

        self._loop_helpstring = str(
            'h:      Print this help.\n'
            'q:      Quit the loop.\n'
            'p:      Print the parameters of the last fit.\n'
            's <filename>:\n'
            '        Save the current plot to "filename.png".\n'
            'model <name>:\n'
            '        Choose the fit model that is used.\n'
            'xrange <min> <max>:\n'
            '        Set the x-range of the plot to (min, max). The points\n'
            '        outside this range are not plotted. The number of\n'
            '        points plotted in the given interval is fixed to\n'
            '        self.nr_plot_points (default=1000).')

        # Make window for plots
        self.vw = QtPlot(window_title='Distortions', figsize=(600, 400))

    def load_kernel_file(self, filename):
        '''
        Loads kernel dictionary (containing kernel and metadata) from a JSON
        file. This function looks only in the directory self.kernel_dir for
        the file.
        Returns a dictionary of the kernel and metadata.
        '''
        with open(os.path.join(self.kernel_dir, filename)) as infile:
            data = json.load(infile)
        return data

    def save_kernel_file(self, kernel_dict, filename):
        '''
        Saves kernel dictionary (containing kernel and metadata) to a JSON
        file in the directory self.kernel_dir.
        '''
        with open(os.path.join(self.kernel_dir, filename), 'w') as outfile:
            json.dump(kernel_dict, outfile)

    def save_plot(self, filename):
        try:
            self.vw.save(os.path.join(self.data_dir, filename))
        except Exception as e:
            logging.warning('Could not save plot.')

    def open_new_correction(self, kernel_length, AWG_sampling_rate,
                            name='{}_corr'.format(
                                datetime.date.today().strftime('%y%m%d'))):
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
        # TODO: kernel list should not be emptied for cryo corrections
        self.ker_obj.kernel_list([])
        self.ker_obj.add_kernel_to_kernel_list(
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
        self.ker_obj.kernel_list([])
        self.ker_obj.add_kernel_to_kernel_list(filename)

    def measure_trace(self, verbuse=True):
        raise NotImplementedError(
            'Base class is not attached to physical instruments and does not '
            'implement measurements.')

    def fit_exp_model(self, start_time_fit, end_time_fit):
        '''
        Fits an exponential of the form
            A * exp(-t/tau) + C
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
                                      value=self.waveform[self._start_idx] -
                                      self.waveform[self._stop_idx],
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
        C = fit_res.best_values['offset']
        A = fit_res.best_values['amplitude']
        a = A / C
        aTilde = a / (a + 1)
        tau = fit_res.best_values['tau']
        tauTilde = tau * (a + 1)

        tPts = np.arange(0, self.kernel_length*1e-9, 1e-9)  # TODO AWG sampling rate
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
                'model': 'exponential',
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
                                      vary=False)
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

        tPts = np.arange(0, self.kernel_length*1e-9, 1e-9)  # TODO AWG sampling rate
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
                'b1': 1/tau,
                'a1': 0
            },
            'fit': {
                'model': 'high-pass',
                'tau': tau
            },
            'kernel': list(new_ker)
        }

    def fit_spline(self, start_time_fit, end_time_fit, s=0.001):
        '''
        Fit the data using a spline interpolation.

        Args:
            start_time_fit (float):
                    Start of the fitted interval.
            end_time_fit (float):
                    End of the fitted interval.
            s (float):
                    Smoothing condition for the spline. See documentation on
                    scipy.interpolate.splrep for more information.
        '''
        self._start_idx = np.argmin(np.abs(self.time_pts - start_time_fit))
        self._stop_idx = np.argmin(np.abs(self.time_pts - end_time_fit))

        splTuple = sc_intpl.splrep(
            x=self.time_pts[self._start_idx:self._stop_idx],
            y=self.waveform[self._start_idx:self._stop_idx],
            w=np.exp(-self.time_pts[self._start_idx:self._stop_idx] /
                     self.time_pts[self._stop_idx]),
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

        new_ker = numpy.linalg.inv(filterMatrix)[:, 0]
        self.new_step = np.convolve(new_ker,
                                    np.ones(len(splStep)))[:len(splStep)]
        self.new_kernel_dict = {
            'name': self.filename + '_' + str(self._iteration),
            'filter_params': {},
            'fit': {
                'model': 'spline',
                's': s
            },
            'kernel': list(new_ker)
        }

    def plot_trace(self, start_time=0, stop_time=10e-6, no_plot_pts=1000,
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
            int(len(self.time_pts[start_idx:stop_idx]) // no_plot_pts), 1)

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

    def plot_fit(self, start_time=0, stop_time=10e-6, save_y_range=True):
        '''
        Plot last trace that was measured (self.waveform) and the latest fit.
        Args:
            start_time (float): Start of the plotted interval.
            stop_time (float):  End of the plotted interval.
            save_y_range (bool):
                                Keep the current y-range of the plot.
        '''
        self.plot_trace(start_time=start_time, stop_time=stop_time,
                        save_y_range=save_y_range)
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
        self.ker_obj.add_kernel_to_kernel_list(full_name)

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

        self.kernel_combined_dict['metadata'][self.new_kernel_dict['name']] =\
            {
                'fit': self.new_kernel_dict['fit'],
                'filter_params': self.new_kernel_dict['filter_params'],
                'name': self.new_kernel_dict['name']
        }

        # Remove the last correction from the kernel list in the kernel
        # object, since it is now included in the combined kernel.
        test_kernel_name = '{}_{}.json'.format(self.filename, self._iteration)
        if test_kernel_name in self.ker_obj.kernel_list():
            self.ker_obj.remove_kernel_from_kernel_list(
                '{}_{}.json'.format(self.filename, self._iteration))

        self.kernel_combined_dict['iteration'] += 1
        self._iteration += 1  # This correction is considered completed.

        # Overwrite the file where kernel_combined is saved
        self.save_kernel_file(self.kernel_combined_dict,
                              '{}_combined.json'.format(self.filename))
        # Have to manually set _config_chaned in the kernel object, because
        # we're just overwriting the same file and kernel_list is not
        # necessarily changed.
        self.ker_obj._config_changed = True

    def discard_new_kernel(self):
        '''
        Removes a the last kernel that was added from the distortions.
        '''
        # Remove kernel from kernel list
        test_kernel_name = '{}_{}.json'.format(self.filename, self._iteration)
        if test_kernel_name in self.ker_obj.kernel_list():
            self.ker_obj.remove_kernel_from_kernel_list(
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
        self.ker_obj._config_changed = True

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
                kernel_length=self.ker_obj.corrections_length(),
                AWG_sampling_rate=self.AWG_lutman.sampling_rate(),
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
                        save_y_range=False)

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
            elif self._fit_model_loop == 'high-pass':
                self.fit_high_pass(fit_start, fit_stop)
            elif self._fit_model_loop == 'spline':
                self.fit_spline(fit_start, fit_stop)

            self.plot_fit(self._t_start_loop, self._t_stop_loop)

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
            self.save_plot('fit_{}.png'.format(self._iteration))

            # 3. Test correction and plot
            # Save last data, in case new distortion is rejected.
            previous_t = self.time_pts
            previous_wave = self.waveform

            print('Testing new correction.')
            self.test_new_kernel()
            self.measure_trace()
            self.plot_trace(self._t_start_loop, self._t_stop_loop)

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
                self.plot_trace(self._t_start_loop, self._t_stop_loop)
                # Go back to 2.
                continue

            # Correction was accepted -> save plot
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
                self.plot_trace(self._t_start_loop, self._t_stop_loop)
            else:
                # Fit also plotted
                self.plot_fit(self._t_start_loop, self._t_stop_loop)

        elif inp_elements[0] == 'h':
            print(self._loop_helpstring)

        elif inp_elements[0] == 'q':
            self.print_summary()
            quit = True
            repeat = False

        elif inp_elements[0] == 'p':
            try:
                for param, val in self.new_kernel_dict['fit'].items():
                    print('{} = {}'.format(param, val))
            except KeyError:
                print('No fit has been done yet!')

        elif (inp_elements[0] == 's' and len(inp_elements == 2)):
            self.save_plot('{}.png'.format(inp_elements[1]))
            print('Current plot saved.')

        elif (inp_elements[0] == 'model' and len(inp_elements == 2)):
            if inp_elements[1] in self.known_fit_models:
                self._fit_model_loop = str(inp_elements[1])
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


class RT_distortion_corrector(Distortion_corrector):
    '''
    Class for applying corrections for room-temperature distortions to flux
    pulses.
    '''

    # TODO:
    #   - check TODOs in this file
    #   - waveform length will become parameter in QWG_flux_lutman
    #       -> adapt this class to handle that
    def __init__(self, AWG_lutman, oscilloscope, square_amp: float,
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
        self.AWG_lutman = AWG_lutman
        self.scope = oscilloscope
        self.square_amp = square_amp  # Normalization for step
        super().__init__(kernel_object=AWG_lutman.F_kernel_instr.get_instr(),
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
        self.raw_time_pts, self.raw_waveform = \
            self.scope.measure_trace()

        # Measurement should be implemented using measurement_control
        # self.data_dir should be set to data dir of last measurement
        # a = ma.MeasurementAnalysis()
        # self.data_dir = a.folder

        # Normalize waveform and find rising edge
        waveform = self.raw_waveform / self.square_amp
        edge_idx = -1
        for i in range(len(waveform) - 1):
            if waveform[i+1] - waveform[i] > 0.01:
                edge_idx = i
                break
        if edge_idx < 0:
            raise ValueError('Failed to find rising edge.')

        self.waveform = waveform[edge_idx:]
        # Sampling rate hardcoded to 5 GHz
        self.time_pts = np.arange(len(self.waveform)) / 2.5e9


class Cryo_distortion_corrector(Distortion_corrector):
    def __init__(self, time_pts, qubit, V_per_phi0,
                 V_0=0, f_demod=0, square_amp=1, nr_plot_points=4000,
                 demodulate=False, chunk_size=32):
        '''
        Instantiate an object.
        This class uses qubit.measure_ram_z to measure the step response.

        Args:
            time_pts (array):
                    Time points at which the Ram-Z should be measured
                    (x-axis).
            quibt (Instrument):
                    Qubit object on which the experiment is performed.
            AWG_lutman (Instrument):
                    Lookup table manager for the AWG.
            nr_plot_points (int):
                    Number of points of the waveform that are plotted. Can be
                    changed in self.nr_plot_points.
        '''
        # TODO: add filename initialization
        self.AWG_lutman = qubit.flux_LutMan.get_instr()
        self.qubit = qubit
        super().__init__(kernel_object=self.AWG_lutman.F_kernel_instr.get_instr(),
                         nr_plot_points=nr_plot_points)
        self.time_pts = time_pts

        self.chunk_size=chunk_size
        self.V_per_phi0 = V_per_phi0
        self.V_0 = V_0
        self.f_demod = f_demod
        self.square_amp = square_amp
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

        self.data_dir = a.folder

        self.waveform = a.step_response
