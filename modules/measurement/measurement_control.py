try:
    import msvcrt  # used on windows to catch keyboard input
except:
    print('Warning: Could not import msvcrt (used for detecting keystrokes)')

import types
from modules.measurement import hdf5_data as h5d
from modules.utilities import general
import logging
import time
import sys
import os
import numpy as np
from scipy.optimize import fmin_powell
from modules.utilities.general import dict_to_ordered_tuples


class MeasurementControl:
    '''
    New version of Measurement Control that allows for adaptively determining
    data points.
    '''
    def __init__(self, name, **kw):
        # self.add_parameter('sweep_function_names',
        #                    flags=Instrument.FLAG_GETSET, type=list)
        # self.add_parameter('detector_function_name',
        #                    flags=Instrument.FLAG_GETSET, type=str)

        # Parameters for logging
        # self.add_parameter('git_hash',
        #                    flags=Instrument.FLAG_GET, type=str)
        # self.add_parameter('measurement_begintime',
        #                    flags=Instrument.FLAG_GET, type=str)
        # self.add_parameter('measurement_endtime',
        #                    flags=Instrument.FLAG_GET, type=str)

        # self.add_parameter('sweep_points',
        #                    flags=Instrument.FLAG_GETSET, type=list)

        # self.add_parameter('measurement_name',
        #                    flags=Instrument.FLAG_GETSET, type=str)
        # self.add_parameter('optimization_method',
        #                    flags=Instrument.FLAG_GETSET, type=str)

        self.get_git_hash()
        # self.Plotmon = qt.instruments['Plotmon']
        # if self.Plotmon is None:
        #     logging.warning('Measurement Control could not connect to Plotmon')

    ##############################################
    # Functions used to control the measurements #
    ##############################################

    def run(self, name=None, mode='1D', **kw):
        '''
        Core of the Measurement control.
        Starts by preparing
        '''
        self.catch_previous_human_abort()
        self.set_measurement_name(name)
        self.print_measurement_start_msg()
        with h5d.Data(name=self.get_measurement_name()) as self.data_object:
            self.start_mflow()
            # Such that it is also saved if the measurement fails
            # (might want to overwrite again at the end)
            self.save_instrument_settings(self.data_object)

            self.create_experimentaldata_dataset()
            if mode == '1D':
                self.measure()
            elif mode == '2D':
                self.measure_2D()
            elif mode == 'adaptive':
                self.measure_soft_adaptive()
            else:
                raise ValueError('mode %s not recognized' %mode)

            # Commented out because not functioning correctly and it causes major
            # timing hangups for nested measurements (issue #152)
            # self.save_instrument_logging(self.data_object)

    def measure(self, *kw):
        if (self.sweep_functions[0].sweep_control !=
                self.detector_function.detector_control):
                # FIXME only checks first sweepfunction
            raise Exception('Sweep and Detector functions not of the same type.'
                            + 'Aborting measurement')
            print(self.sweep_function.sweep_control)
            print(self.detector_function.detector_control)

        for sweep_function in self.sweep_functions:
            sweep_function.prepare()

        if self.sweep_functions[0].sweep_control == 'soft':
            self.detector_function.prepare()
            self.measure_soft_static()
        if self.sweep_functions[0].sweep_control == 'hard':
            self.iteration = 0
            if len(self.sweep_functions) == 1:
                self.detector_function.prepare(
                    sweep_points=self.get_sweep_points())
                self.measure_hard()
            elif len(self.sweep_functions) == 2:
                self.detector_function.prepare(
                    sweep_points=self.get_sweep_points()[0: self.xlen, 0])
                self.complete = False
                for j in range(self.ylen):
                    if not self.complete:  # added specifically for 2D hard sweeps
                        for i, sweep_function in enumerate(self.sweep_functions):
                            x = self.get_sweep_points()[
                                self.iteration*self.xlen]
                            if i != 0:
                                sweep_function.set_parameter(x[i])
                        self.measure_hard()
            else:
                raise Exception('hard measurements have not been generalized to N-D yet')
        for sweep_function in self.sweep_functions:
            sweep_function.finish()
        self.detector_function.finish()

        self.get_measurement_endtime()
        return

    def measure_soft_static(self):
        for i, sweep_point in enumerate(self.sweep_points):
            self.measurement_function(sweep_point)
            self.print_progress_static_soft_sweep(i)

    def measure_soft_adaptive(self, method=None):
        '''
        Uses the adaptive function and keywords for that function as
        specified in self.af_pars()
        '''
        self.save_optimization_settings()
        adaptive_function = self.af_pars.pop('adaptive_function')
        print('Adaptive function passed: %s' % adaptive_function)

        for sweep_function in self.sweep_functions:
            sweep_function.prepare()
        self.detector_function.prepare()

        if adaptive_function == 'Powell':
            print('Optimizing using scipy.fmin_powell')
            adaptive_function = fmin_powell
        if type(adaptive_function) == types.FunctionType:
            try:
                adaptive_function(self.optimization_function, **self.af_pars)
            except StopIteration:
                print('Reached f_termination: %s' % (self.f_termination))
        else:
            raise Exception('optimization function: "%s" not recognized'
                            % adaptive_function)

        for sweep_function in self.sweep_functions:
            sweep_function.finish()
        self.detector_function.finish()

        self.get_measurement_endtime()
        return

    def measure_hard(self):
        '''
        ToDo: integrate soft averaging into MC
        '''
        self.check_keyboard_interupt()
        # note, checking after the data comes in is pointless in hard msmt
        new_data = np.array(self.detector_function.get_values()).T

        if len(np.shape(new_data)) == 1:
            single_col = True
            shape_new_data = (len(new_data), 1)
        else:
            single_col = False
            shape_new_data = np.shape(new_data)

        # resizing only for 1 set of new_data  now... needs to improve
        shape_new_data = (shape_new_data[0], shape_new_data[1]+1)

        datasetshape = self.dset.shape

        self.iteration = datasetshape[0]/shape_new_data[0] + 1
        start_idx = shape_new_data[0]*(self.iteration-1)
        new_datasetshape = (shape_new_data[0]*self.iteration, datasetshape[1])
        self.dset.resize(new_datasetshape)
        len_new_data = shape_new_data[0]
        if single_col:
            self.dset[start_idx:,
                      len(self.sweep_functions)] = new_data
        else:
            self.dset[start_idx:,
                      len(self.sweep_functions):] = new_data

        sweep_len = len(self.get_sweep_points().T)
        # Only add sweep points if these make sense (i.e. same shape as new_data)
        if sweep_len == len_new_data:  # 1D sweep
            self.dset[:, 0] = self.get_sweep_points().T
        elif hasattr(self, 'xlen'):  # 2D sweep
            # always add for a 2D sweep
            relevant_swp_points = self.get_sweep_points()[
                start_idx:start_idx+len_new_data:]
            self.dset[start_idx:, 0:len(self.sweep_functions)] = \
                relevant_swp_points

        for i in range(len(self.detector_function.value_names)):
            self.update_plotmon(
                mon_nr=i+1, x_ind=0,
                y_ind=-len(self.detector_function.value_names)+i)
        if hasattr(self, 'TwoD_array'):
            self.update_plotmon_2D_hard()
            self.print_progress_static_2D_hard()

        return new_data

    def measurement_function(self, x):
        '''
        Core measurement function used for soft sweeps
        '''
        self.check_keyboard_interupt()

        if np.size(x) == 1:
            x = [x]
        if np.size(x) != len(self.sweep_functions):
            raise ValueError('size of x "%s" not equal to # sweep functions' % x)
        for i, sweep_function in enumerate(self.sweep_functions[::-1]):
            sweep_function.set_parameter(x[::-1][i])
            # x[::-1] changes the order in which the parameters are set, so
            # it is first the outer sweep point and then the inner.This
            # is generally not important except for specifics: f.i. the phase
            # of an agilent generator is reset to 0 when the frequency is set.

        datasetshape = self.dset.shape
        self.iteration = datasetshape[0] + 1
        vals = self.detector_function.acquire_data_point(
            iteration=self.iteration)

        # Resizing dataset and saving
        new_datasetshape = (self.iteration, datasetshape[1])
        self.dset.resize(new_datasetshape)
        savable_data = np.append(x, vals)
        self.dset[self.iteration-1, :] = savable_data
        # update plotmon
        if len(self.sweep_functions) == 1:
            for i in range(len(self.detector_function.value_names)):
                self.update_plotmon(mon_nr=i+1, x_ind=0, y_ind=i+1)
        elif len(self.sweep_functions) == 2:
            for i in range(len(self.detector_function.value_names)):
                self.update_plotmon(mon_nr=i+1, x_ind=0, y_ind=i+1)
        elif len(self.sweep_functions) <= 4:
            for i in range(len(self.sweep_functions)):
                self.update_plotmon(
                    mon_nr=i+1, x_ind=i,
                    y_ind=-len(self.detector_function.value_names))
        else:
            for i in range(4):
                self.update_plotmon(
                    mon_nr=i+1, x_ind=i,
                    y_ind=-len(self.detector_function.value_names))
        if hasattr(self, 'TwoD_array'):
            self.update_plotmon_2D()
        return vals

    def optimization_function(self, x):
        '''
        A wrapper around the measurement function.
        It takes the following actions based on parameters specified
        in self.af_pars:
        - Rescales the function using the "x_scale" parameter, default is 1
        - Inverts the measured values if "minimize"==False
        - Compares measurement value with 'f_termination' and raises an
        exception, that gets caught outside of the optimization loop, if
        the measured value is smaller than this f_termination.

        Measurement function with scaling to correct physical value
        '''
        if hasattr(self.x_scale, '__iter__'):  # to check if
            for i in range(len(x)):
                x[i] = float(x[i])/float(self.x_scale[i])
        elif self.x_scale != 1:  # only rescale if needed
            for i in range(len(x)):
                x[i] = float(x[i])/float(self.x_scale[i])
        if self.minimize_optimization:
            vals = self.measurement_function(x)
            if (vals < self.f_termination) & (self.f_termination is not None):
                raise StopIteration()
        else:
            vals = self.measurement_function(x)
            # when maximizing interrupt when larger than condition before
            # inverting
            if (vals > self.f_termination) & (self.f_termination is not None):
                raise StopIteration()
            vals = np.multiply(-1, vals)

        # print 'Completed iteration %d' % self.iteration
        # measurement function ensures this self.iteration is correct
        # print 'Measured with parameters: ', x
        # print 'Measured value: ', vals
        self.update_plotmon(
            mon_nr=5, x_ind=None,
            y_ind=-len(self.detector_function.value_names))
        # to check if vals is an array with multiple values
        if hasattr(vals, '__iter__'):
            if len(vals) > 1:
                vals = vals[0]
        return vals

    def finish(self):
        '''
        Deletes arrays to clean up memory
        (Note better way to do is also overload the remove function and make
        sure all attributes are removed.
        '''
        try:
            del(self.TwoD_array)
        except AttributeError:
            pass
        try:
            del(self.dset)
        except AttributeError:
            pass
        try:
            del(self.sweep_points)
        except AttributeError:
            pass

    def check_keyboard_interupt(self):
        try:  # Try except statement is to make it work on non windows pc
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == 'q':
                    filedir = os.path.dirname(self.data_object.filepath)
                    self.data_object.close()
                    filedirparts = os.path.split(filedir)
                    newdirname = filedirparts[1][:6]+'_X'+filedirparts[1][6:]
                    newfiledir = os.path.join(filedirparts[0], newdirname)
                    os.rename(filedir, newfiledir)
                    raise KeyboardInterrupt('Human interupt q')
        except:
            pass

    ###################
    # 2D-measurements #
    ###################

    def run_2D(self, name=None, **kw):
        self.run(name=name, mode='2D', **kw)

    def measure_2D(self, **kw):
        '''
        Sweeps over two parameters set by sweep_function and sweep_function_2D.
        The outer loop is set by sweep_function_2D, the inner loop by the
        sweep_function.

        Soft(ware) controlled sweep functions require soft detectors.
        Hard(ware) controlled sweep functions require hard detectors.
        '''

        if np.size(self.get_sweep_points()[0]) == 2:
            self.measure(**kw)
        elif np.size(self.get_sweep_points()[0]) == 1:
            print('Reshaping sweep points')
            self.xlen = len(self.get_sweep_points())
            self.ylen = len(self.sweep_points_2D)
            self.preallocate_2D_plot()

            # create inner loop pts
            sweep_pts_0 = np.tile(self.get_sweep_points(), self.ylen)
            # create outer loop
            sweep_pts_1 = np.repeat(self.sweep_points_2D, self.xlen)
            c = np.column_stack((sweep_pts_0, sweep_pts_1))
            self.set_sweep_points(c)
            self.measure(**kw)
            del self.TwoD_array
        return

    def set_sweep_function_2D(self, sweep_function):
        if len(self.sweep_functions) != 1:
            raise KeyError('Specify sweepfunction 1D before specifying sweep_function 2D')
        else:
            self.sweep_functions.append(sweep_function)
            self.sweep_function_names.append(
                str(sweep_function.__class__.__name__))

    def set_sweep_points_2D(self, sweep_points_2D):
        self.sweep_points_2D = sweep_points_2D

    ###########
    # Plotmon #
    ###########

    def update_plotmon(self, mon_nr, x_ind=0, y_ind=-1):
        y = self.dset[:, y_ind]
        if x_ind is None:
            x = np.arange(len(y))
        else:
            x = self.dset[:, x_ind]
        data = [x, y]
        if hasattr(self, 'Plotmon'):
            self.Plotmon.plot2D(mon_nr, data)

    def preallocate_2D_plot(self):
        '''
        Preallocates a data array to be used for the update_plotmon_2D command.

        Made to work with at most 2 2D arrays (as this is how the labview code
        works). It should be easy to extend this function for more vals.
        '''
        if len(self.detector_function.value_names) == 1:
            self.TwoD_array = np.empty((self.xlen,
                                       self.ylen))
        else:
            self.TwoD_array = np.empty(
                [self.xlen,
                 self.ylen,
                 len(self.detector_function.value_names)])

        self.x_step = self.get_sweep_points()[1] - self.get_sweep_points()[0]
        self.y_step = self.sweep_points_2D[1] - self.sweep_points_2D[0]
        # The - self. _step/2.0 is to make center of tile lie on the value
        self.x_start = self.get_sweep_points()[0] - self.x_step/2.0
        self.y_start = self.sweep_points_2D[0] - self.y_step/2.0

        self.TwoD_array[:] = np.NAN

    def update_plotmon_2D(self):
        '''
        Adds latest measured value to the TwoD_array and sends it
        to the plotmon.
        Note that the plotmon only supports evenly spaced lattices.

        Made to work with at most 2 2D arrays (as this is how the labview code
        works). It should be easy to extend this function for more vals.
        '''
        i = self.iteration-1
        x_ind = i % self.xlen
        y_ind = i / self.xlen
        if len(self.detector_function.value_names) == 1:
            z_ind = len(self.sweep_functions)
            self.TwoD_array[x_ind, y_ind] = self.dset[i, z_ind]
            if self.Plotmon is not None:
                self.Plotmon.plot3D(1, data=self.TwoD_array,
                                    axis=(self.x_start, self.x_step,
                                          self.y_start, self.y_step))
        else:
            for j in range(2):
                z_ind = len(self.sweep_functions) + j
                self.TwoD_array[x_ind, y_ind, j] = self.dset[i, z_ind]
                if self.Plotmon is not None:
                    self.Plotmon.plot3D(j+1, data=self.TwoD_array[:, :, j],
                                        axis=(self.x_start, self.x_step,
                                              self.y_start, self.y_step))

    def update_plotmon_2D_hard(self):
        '''
        Adds latest datarow to the TwoD_array and send it
        to the plotmon.
        Note that the plotmon only supports evenly spaced lattices.

        Made to work with at most 2 2D arrays (as this is how the labview code
        works). It should be easy to extend this function for more vals.
        '''
        i = self.iteration-1
        y_ind = i
        if len(self.detector_function.value_names) == 1:
            z_ind = len(self.sweep_functions)
            self.TwoD_array[:, y_ind] = self.dset[i*self.xlen:(i+1)*self.xlen,
                                                  z_ind]
            if self.Plotmon is not None:
                self.Plotmon.plot3D(1, data=self.TwoD_array,
                                    axis=(self.x_start, self.x_step,
                                          self.y_start, self.y_step))
        else:
            for j in range(2):
                z_ind = len(self.sweep_functions) + j
                self.TwoD_array[:, y_ind, j] = self.dset[
                    i*self.xlen:(i+1)*self.xlen, z_ind]
                if self.Plotmon is not None:
                    self.Plotmon.plot3D(j+1, data=self.TwoD_array[:, :, j],
                                        axis=(self.x_start, self.x_step,
                                              self.y_start, self.y_step))

    ##################################
    # Small helper/utility functions #
    ##################################

    def get_data_object(self):
        '''
        Used for external functions to write to a datafile.
        This is used in time_domain_measurement as a hack and is not
        recommended.
        '''
        return self.data_object

    def create_experimentaldata_dataset(self):
        data_group = self.data_object.create_group('Experimental Data')
        self.dset = data_group.create_dataset(
            'Data', (0, len(self.sweep_functions) +
                     len(self.detector_function.value_names)),
            maxshape=(None, len(self.sweep_functions) +
                      len(self.detector_function.value_names)))

        column_names = []
        sweep_par_names = []
        sweep_par_units = []

        for sweep_function in self.sweep_functions:
            column_names.append(sweep_function.parameter_name+' (' +
                                sweep_function.unit+')')
            sweep_par_names.append(sweep_function.parameter_name)
            sweep_par_units.append(sweep_function.unit)

        for i, val_name in enumerate(self.detector_function.value_names):
            column_names.append(val_name+' (' +
                                self.detector_function.value_units[i] + ')')

        self.dset.attrs['column_names'] = h5d.encode_to_utf8(column_names)

        # Added to tell analysis how to extract the data
        data_group.attrs['datasaving_format'] = h5d.encode_to_utf8('Version 2')
        data_group.attrs['sweep_parameter_names'] = h5d.encode_to_utf8(sweep_par_names)
        data_group.attrs['sweep_parameter_units'] = h5d.encode_to_utf8(sweep_par_units)

        data_group.attrs['value_names'] = h5d.encode_to_utf8(self.detector_function.value_names)
        data_group.attrs['value_units'] = h5d.encode_to_utf8(self.detector_function.value_units)


    def save_optimization_settings(self):
        '''
        Saves the parameters used for optimization
        '''
        opt_sets_grp = self.data_object.create_group('Optimization settings')
        param_list = dict_to_ordered_tuples(self.af_pars)
        for (param, val) in param_list:
            opt_sets_grp.attrs[param] = str(val)

    def save_instrument_settings(self, *args):
        pass
        # if len(args) == 0:
        #     data_object = self.data_object
        # else:
        #     data_object = args[0]
        # set_grp = data_object.create_group('Instrument settings')
        # inslist = dict_to_ordered_tuples(qt.instruments.get_instruments())
        # for (iname, ins) in inslist:
        #     instrument_grp = set_grp.create_group(iname)
        #     parameter_list = dict_to_ordered_tuples(ins.get_parameters())
        #     for (param, popts) in parameter_list:
        #         val = ins.get(param, query=False)
        #         instrument_grp.attrs[param] = str(val)

    def init_instrument_changelog(self):

        self._init_time = time.clock()
        self._instrument_changelog = {}  #

        qt.instruments.connect('instrument-changed', self.add_change_to_log)

    def add_change_to_log(self, inslist, instr, changed_pars):
        '''
        function to connect to instrument-changed signal
        '''
        logentry = {}
        logentry[instr] = changed_pars
        ts = time.clock()-self._init_time
        self._instrument_changelog[self.get_datetimestamp()+(
            '%.4f' % (ts-int(ts)))[1:]] = logentry
        # the last line is to append subseconds to the key

    def get_instrument_changelog(self):
        return self._instrument_changelog

    def disconnect_instrument_logging(self):
        try:
            qt.instruments.disconnect_by_func(self.add_change_to_log)
        except:
            print('Trying to stop instrument logging but logging already stopped...')
            print(repr(sys.exc_info()[1]))

    def save_instrument_logging(self, *args):
        if len(args) == 0:
            data = self.data
        else:
            data = args[0]
        length = sum([len(list(x[1].values())[0]) for x in list(self._instrument_changelog.items())])
        dataset = data.create_dataset("Instrument_changelog",
                                      (length, 4), dtype="S25")
        k = 0
        for date_time, instrument_dict in list(self._instrument_changelog.items()):
            for instrument, params in list(instrument_dict.items()):
                for param, param_value in list(params.items()):
                    dataset[k] = [date_time, instrument, param,
                                  str(param_value)]
        #             print date_time, instrument, param, param_value
                    k += 1

    def start_mflow(self):
        '''
        Checks if measurement is running
        '''
        self.get_measurement_begintime()
        # Commented out because not functioning correctly and it causes major
        # timing hangups for nested measurements (issue #152)
        # self.init_instrument_changelog()
        self.get_git_hash()
        # if not qt.flow.is_measuring():
        #     qt.mstart()
        # else:
        #     pass

    # def stop_mflow(self):
    #     '''
    #     stops measurement flow
    #     '''
    #     if qt.flow.is_measuring():
    #         qt.mend()
    #     else:
    #         print('Warning: Tried to stop measurement ' + \
    #             'but it was already stopped')
        # Commented out because not functioning correctly and it causes major
        # timing hangups for nested measurements (issue #152)
        # self.disconnect_instrument_logging()

    def print_progress_static_soft_sweep(self, i):
        percdone = (i+1)*1./len(self.sweep_points)*100
        elapsed_time = time.time() - self.begintime
        scrmes = "{percdone}% completed, elapsed time: "\
            "{t_elapsed} s, time left: {t_left} s".format(
                percdone=int(percdone),
                t_elapsed=round(elapsed_time, 1),
                t_left=round((100.-percdone)/(percdone) *
                             elapsed_time, 1) if
                percdone != 0 else '')
        if percdone != 100:
            end_char = '\r'
        else:
            end_char = '\n'
        print(scrmes, end=end_char)


    def print_progress_static_2D_hard(self):
        acquired_points = self.dset.shape[0]
        total_nr_pts = self.sweep_points.shape[0]
        if acquired_points == total_nr_pts:
            self.complete = True
        elif acquired_points > total_nr_pts:
            self.complete = True
            logging.warning(
                'Warning nr of acq points is larger nr of sweep points')
            logging.warning('Acq pts: %s, total_nr_pts: %s' % (
                            acquired_points, total_nr_pts))

        percdone = acquired_points*1./total_nr_pts*100
        elapsed_time = time.time() - self.begintime
        scrmes = "{percdone}% completed, elapsed time: "\
            "{t_elapsed} s, time left: {t_left} s".format(
                percdone=int(percdone),
                t_elapsed=round(elapsed_time, 1),
                t_left=round((100.-percdone)/(percdone) *
                             elapsed_time, 1) if
                percdone != 0 else '')
        sys.stdout.write(60*'\b'+scrmes)

    def print_measurement_start_msg(self):
        if len(self.sweep_functions) == 1:
            print('\n' + '*'*78)
            print('Starting measurement: %s' % self.get_measurement_name())
            print('Sweep function: %s' % self.get_sweep_function_names()[0])
            print('Detector function: %s' % self.get_detector_function_name())
        else:
            print('\n' + '*'*78)
            print('Starting measurement: %s' % self.get_measurement_name())
            for i, sweep_function in enumerate(self.sweep_functions):
                print('Sweep function %d: %s' % (
                    i, self.sweep_function_names[i]))
            print('Detector function: %s' % self.get_detector_function_name())

    def catch_previous_human_abort(self):
        try:
            pass
            # qt.msleep()
        except ValueError:
            print('Human abort from previuos stop. Continuing measurement')

    def get_datetimestamp(self):
        return time.strftime('%Y%m%d_%H%M%S', time.localtime())

    ####################################
    # Non-parameter get/set functions  #
    ####################################

    def set_sweep_function(self, sweep_function):
        '''
        Used if only 1 sweep function is set.
        '''
        self.sweep_functions = [sweep_function]
        self.set_sweep_function_names(
            [str(sweep_function.__class__.__name__)])

    def get_sweep_function(self):
        return self.sweep_functions[0]

    def set_sweep_functions(self, sweep_functions):
        '''
        Used to set an arbitrary number of sweep functions.
        '''
        self.sweep_functions = sweep_functions

        sweep_function_names = []
        for swf in sweep_functions:
            sweep_function_names.append(str(swf.__class__.__name__))
            # input str(swf.__class__.__name__)
        self.set_sweep_function_names(sweep_function_names)

    def get_sweep_functions(self):
        return self.sweep_functions

    def set_sweep_function_names(self, swfname):
        self.sweep_function_names = swfname

    def get_sweep_function_names(self):
        return self.sweep_function_names

    def set_detector_function(self, detector_function):
        self.detector_function = detector_function
        self.set_detector_function_name(str(
            detector_function.__class__.__name__))

    def get_detector_function(self):
        return self.detector_function

    def set_detector_function_name(self, dfname):
        self._dfname = dfname

    def get_detector_function_name(self):
        return self._dfname

    ################################
    # Parameter get/set functions  #
    ################################

    def get_git_hash(self):
        self.git_hash = general.get_git_revision_hash()
        return self.git_hash

    def get_measurement_begintime(self):
        self.begintime = time.time()
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def get_measurement_endtime(self):
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def set_sweep_points(self, sweep_points):
        self.sweep_points = np.array(sweep_points)
        # line below is because some sweep funcs have their own sweep points attached
        self.sweep_functions[0].sweep_points = np.array(sweep_points)

    def get_sweep_points(self):
        return self.sweep_functions[0].sweep_points

    def set_adaptive_function_parameters(self, adaptive_function_parameters):
        self.af_pars = adaptive_function_parameters

        # scaling should not be used if a "direc" argument is available
        # in the adaptive function itself, if not specified equals 1
        self.x_scale = self.af_pars.pop('x_scale', 1)
        # Determines if the optimization will minimize or maximize
        self.minimize_optimization = self.af_pars.pop('minimize', True)
        self.f_termination = self.af_pars.pop('f_termination', None)
        print(self.f_termination)

    def get_adaptive_function_parameters(self):
        return self.af_pars

    def set_measurement_name(self, measurement_name):
        if measurement_name == 'None':
            self.measurement_name = 'Measurement'
        else:
            self.measurement_name = measurement_name

    def get_measurement_name(self):
        return self.measurement_name

    def set_optimization_method(self, optimization_method):
        self.optimization_method = optimization_method

    def get_optimization_method(self):
        return self.optimization_method
