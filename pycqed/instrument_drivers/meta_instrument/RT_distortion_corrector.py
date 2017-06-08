import pycqed.analysis.fitting_models as fm
import pycqed.measurement.kernel_functions as kf
import pycqed.instrument_drivers.physical_instruments.RTO1024 as RTO
import numpy as np
import lmfit
from qcodes.plots.pyqtgraph import QtPlot
import os.path
import datetime
import json


class RT_distortion_corrector():
    '''
    Class for applying corrections for room-temperature distortions to flux
    pulses.
    '''

    def __init__(self,
                 AWG_lutman=None,
                 oscilloscope=None,
                 square_amp=1,
                 no_plot_points=1000,
                 init_instr=True):
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
            no_plot_points (int):
                    Number of points of the waveform that are plotted. Can be
                    changed in self.no_plot_points.
            init_instr (bool):
                    Determines whether instruments (AWG_lutman, oscilloscope,
                    kernel object) are initialized. Can be used to instantiate
                    an object even when no instruments are connected.
        '''
        self._instr_initialized = False
        if init_intsr:
            if AWG_lutman is None or oscilloscope is None:
                raise ValueError('AWG_lutman or oscilloscope not defined. '
                                 'Give AWG_lutman and oscilloscope arguments '
                                 'or set init_instr=False.')
            else:
                self.init_instruments(AWG_lutman, scope)

        # Initialize instance variables
        # Plotting
        self._y_min = 0
        self._y_max = 1
        self._stop_idx = -1
        self._start_idx = 0
        self._t_start_loop = 0  # sets x range for plotting during loop
        self._t_stop_loop = 30e-6
        self.no_plot_points = no_plot_points

        # Kernels
        self.kernel_length = 0
        self.kernel_combined_dict = {}
        self.new_kernel_dict = {}

        # Files
        self.filename = ''
        # self.kernel_dir is initialized in self.init_instruemnts, since it
        # takes its value from the kernel_object.

        # Data
        self.raw_waveform = []
        self.raw_time_pts = []
        self.waveform = []
        self.time_pts = []
        self.new_step = []
        self.square_amp = square_amp  # Normalization for step

        # Fitting
        self.known_fit_models = ['exponential', 'high-pass']
        self.fit_model = None
        self.fit_res = None

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
            '        self.no_plot_points (default=1000).')

        # Make window for plots
        self.vw = QtPlot(window_title='Distortions', figsize=(600, 400))

    def init_instruments(self, AWG_lutman, scope):
        '''
        Attach and initialize instruments.
        self.ker_obj is set to the kernel_object instrument of AWG_lutman.
        '''
        self.AWG_lutman = AWG_lutman
        self.ker_obj = AWG_lutman.F_kernel_instr.get_instr()
        self.scope = oscilloscope
        self.scope.prepare_measurement(-10e-6, 40e-6)
        self.kernel_dir = self.ker_obj.kernel_dir()
        self._instr_initialized = True

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

    def open_new_correction(self, kernel_length,
                            filename='{}_RT_corr'.format(
                                datetime.date.today().strftime('%y%m%d'))):
        '''
        Opens a new correction with name 'filename', i.e. initialized the
        combined kernel to a Dirac delta, empties kernel_list of the
        kernel object associated with self, and uploads an undistorted
        square pulse.
        '''
        if not self._instr_initialized:
            raise RuntimeError('Instruments are not initialized.')

        self.kernel_length = int(kernel_length * 1e9)  # AWG SR: 1 GHz
        self.filename = filename
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

        # Start with new kernel list
        self.ker_obj.kernel_list([])
        self.ker_obj.add_kernel_to_kernel_list(
            '{}_combined.json'.format(filename))

        # Load initial square pulse without distortions
        self.AWG_lutman.QWG.get_instr().stop()
        self.AWG_lutman.generate_standard_pulses()
        self.AWG_lutman.load_pulse_onto_AWG_lookuptable('Square_flux_pulse')
        self.AWG_lutman.QWG.get_instr().start()

    def resume_correction(self, filename):
        '''
        Loads combined kernel from the specified file and prepares for adding
        new corrections to that kernel.
        '''
        if not self._instr_initialized:
            raise RuntimeError('Instruments are not initialized.')

        # Remove '_combined.json' from filename
        self.filename = '_'.join(filename.split('_')[:-1])
        self.kernel_combined_dict = self.load_kernel_file(filename)
        self._iteration = self.kernel_combined_dict['iteration']
        self.kernel_length = len(self.kernel_combined_dict['kernel'])

        self.ker_obj.kernel_list([])
        self.ker_obj.add_kernel_to_kernel_list(filename)

        self.AWG_lutman.QWG.get_instr().stop()
        self.AWG_lutman.generate_standard_pulses()
        self.AWG_lutman.load_pulse_onto_AWG_lookuptable('Square_flux_pulse')
        self.AWG_lutman.QWG.get_instr().start()

    def measure_trace(self, verbose=True):
        '''
        Measure a trace with the oscilloscope.
        Raw data is saved to self.raw_time_pts and self.raw_waveform.
        Data clipped to start at the rising edge is saved to self.time_pts
        and self.waveform.
        '''
        if not self._instr_initialized:
            raise RuntimeError('Instruments are not initialized.')

        if verbose:
            print('Measuring trace...')
        self.raw_time_pts, self.raw_waveform = \
            self.scope.measure_trace()
        waveform = self.raw_waveform / self.square_amp
        edge_idx = -1
        for i in range(len(waveform) - 1):
            if waveform[i+1] - waveform[i] > 0.01:
                edge_idx = i
                break
        if edge_idx < 0:
            raise ValueError('Failed to find rising edge.')

        self.waveform = waveform[edge_idx:]
        # Sampling rate hardcoded to 10 GHz
        self.time_pts = np.arange(len(self.waveform)) / 5e9

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

        # Analytic form of the predistorted square pulse (input that creates a
        # square pulse at the output)
        C = fit_res.best_values['offset']
        A = fit_res.best_values['amplitude']
        a = A / C
        aTilde = a / (a + 1)
        tau = fit_res.best_values['tau']
        tauTilde = tau * (a + 1)
        predist_step = (1 - aTilde * np.exp(-self.time_pts/tauTilde)) / C

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
        new_ker = kf.kernel_from_kernel_stepvec(predist_step[::5])

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

        # Analytic form of the predistorted square pulse (input that creates a
        # square pulse at the output)
        tau = fit_res.best_values['tau']
        predist_step = self.time_pts/tau + 1

        # Check if parameters are physical and print warnings if not
        if tau < 0:
            print('Warning: unphysical tau = {} (expect tau > 0).'
                  .format(tau))

        # Save the results
        self.fit_res = fit_res
        self.new_step = predist_step
        # Take every 5th point because sampling rate of AWG is 1 GHz and
        # sampling rate of scope is 5 GHz.
        # TODO: un-hardcode this
        new_ker = kf.kernel_from_kernel_stepvec(predist_step[::5])

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

        # Plot
        self.vw.clear()
        self.vw.add(x=self.time_pts[start_idx:stop_idx:step],
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
        self.vw.subplots[0].getAxis('left').setLabel('amp', 'V')

    def plot_fit(self, start_time=0, stop_time=10e-6, save_y_range=True):
        '''
        Plot last trace that was measured (self.waveform) and the latest fit.
        Args:
            start_time (float): Start of the plotted interval.
            stop_time (float):  End of the plotted interval.
            save_y_range (bool):
                                Keep the current y-range of the plot.
        '''
        yPts = self.fit_model.eval(
            t=self.time_pts[self._start_idx:self._stop_idx],
            **self.fit_res.best_values)
        self.plot_trace(start_time=start_time, stop_time=stop_time,
                        save_y_range=save_y_range)
        self.vw.add(x=self.time_pts[self._start_idx:self._stop_idx],
                    y=yPts,
                    color='#ff7f0e')

        # Labels need to be set in the end, else they don't show sometimes
        self.vw.subplots[0].getAxis('bottom').setLabel('t', 's')
        self.vw.subplots[0].getAxis('left').setLabel('amp', 'V')

    def test_new_kernel(self):
        # Save the new kernel to file and add that to the kernel list in the
        # kernel object. Upload new waveform.
        # Kernel list should be [kernel_combined, new_kernel]
        if not self._instr_initialized:
            raise RuntimeError('Instruments are not initialized.')

        # Save new kernel to file
        full_name = '{}_{}.json'.format(self.filename, self._iteration)
        self.save_kernel_file(self.new_kernel_dict, full_name)

        # Add new kernel to the kernel list in the kernel object
        self.ker_obj.add_kernel_to_kernel_list(full_name)

        # Upload waveform
        self.AWG_lutman.QWG.get_instr().stop()
        self.AWG_lutman.load_pulse_onto_AWG_lookuptable('Square_flux_pulse')
        self.AWG_lutman.QWG.get_instr().start()

    def apply_new_kernel(self):
        '''
        Includes the latest distortion in the combined kernel by convolving
        it. The correction number (self._iteration) is incremented, such that
        the kernel file for the latest distortion is not overwritten anymore.
        '''
        if not self._instr_initialized:
            raise RuntimeError('Instruments are not initialized.')

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

        # Upload waveform
        self.AWG_lutman.QWG.get_instr().stop()
        self.AWG_lutman.load_pulse_onto_AWG_lookuptable('Square_flux_pulse')
        self.AWG_lutman.QWG.get_instr().start()

    def discard_new_kernel(self):
        '''
        Removes a the last kernel that was added from the distortions.
        '''
        if not self._instr_initialized:
            raise RuntimeError('Instruments are not initialized.')

        # Remove kernel from kernel list
        test_kernel_name = '{}_{}.json'.format(self.filename, self._iteration)
        if test_kernel_name in self.ker_obj.kernel_list():
            self.ker_obj.remove_kernel_from_kernel_list(
                '{}_{}.json'.format(self.filename, self._iteration))

        # Upload waveform
        self.AWG_lutman.QWG.get_instr().stop()
        self.AWG_lutman.load_pulse_onto_AWG_lookuptable('Square_flux_pulse')
        self.AWG_lutman.QWG.get_instr().start()

    def recalculate_combined_kernel(self):
        '''
        Recalculates the combined kernel from its components. This can be used
        for example to change the length of the combined kernel, as the
        individual kernels are stored in full length (length of measured
        trace).
        '''
        if not self._instr_initialized:
            raise RuntimeError('Instruments are not initialized.')

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

        self.AWG_lutman.QWG.get_instr().stop()
        self.AWG_lutman.load_pulse_onto_AWG_lookuptable('Square_flux_pulse')
        self.AWG_lutman.QWG.get_instr().start()

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
        if not self._instr_initialized:
            raise RuntimeError('Instruments are not initialized.')

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
                filename=filename)
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
        model = 'exponential'  # Default fit model used
        while True:
            print('\n-- Correction number {} --'.format(self._iteration))
            print('Current fit model: {}'.format(model))
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

            if model == 'exponential':
                self.fit_exp_model(fit_start, fit_stop)
            elif model == 'high-pass':
                self.fit_high_pass(fit_stop, fit_stop)

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
            self.vw.save(os.path.join(self.kernel_dir,
                                      'fit_{}.png'.format(self._iteration)))

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
            self.vw.save(os.path.join(self.kernel_dir,
                                      'trace_{}.png'.format(self._iteration)))

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
            self.vw.save(os.path.join(self.kernel_dir,
                                      '{}.png'.format(inp_elements[1])))
            print('Current plot saved.')

        elif (inp_elements[0] == 'model' and len(inp_elements == 2)):
            if inp_elements[1] in self.known_fit_models:
                model = str(inp_elements[1])
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
