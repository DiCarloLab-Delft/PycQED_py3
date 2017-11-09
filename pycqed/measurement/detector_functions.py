'''
Module containing a collection of detector functions used by the
Measurement Control.
'''
import numpy as np
import logging
import time
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.fit_toolbox import functions as fn
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control import sequence
from qcodes.instrument.parameter import _BaseParameter
from pycqed.instrument_drivers.virtual_instruments.pyqx import qasm_loader as ql


class Detector_Function(object):

    '''
    Detector_Function class for MeasurementControl
    '''

    def __init__(self, **kw):
        self.name = self.__class__.__name__
        self.set_kw()
        self.value_names = ['val A', 'val B']
        self.value_units = ['arb. units', 'arb. units']

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s' % (key, kw[key]))

    def get_values(self):
        pass

    def prepare(self, **kw):
        pass

    def finish(self, **kw):
        pass


###############################################################################
###############################################################################
####################             None Detector             ####################
###############################################################################
###############################################################################

class None_Detector(Detector_Function):

    def __init__(self, **kw):
        super(None_Detector, self).__init__()
        self.detector_control = 'soft'
        self.set_kw()
        self.name = 'None_Detector'
        self.value_names = ['None']
        self.value_units = ['None']

    def acquire_data_point(self, **kw):
        '''
        Returns something random for testing
        '''
        return np.random.random()


class Hard_Detector(Detector_Function):

    def __init__(self, **kw):
        super().__init__()
        self.detector_control = 'hard'

    def prepare(self, sweep_points=None):
        pass

    def finish(self):
        pass


class Soft_Detector(Detector_Function):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.detector_control = 'soft'

    def acquire_data_point(self, **kw):
        return np.random.random()

    def prepare(self):
        pass
##########################################################################
##########################################################################
####################     Hardware Controlled Detectors     ###############
##########################################################################
##########################################################################


class Dummy_Detector_Hard(Hard_Detector):

    def __init__(self, delay=0, noise=0, **kw):
        super(Dummy_Detector_Hard, self).__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.value_names = ['distance', 'Power']
        self.value_units = ['m', 'W']
        self.delay = delay
        self.noise = noise
        self.times_called = 0

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points

    def get_values(self):
        x = self.sweep_points
        noise = self.noise * (np.random.rand(2, len(x)) - .5)
        data = np.array([np.sin(x / np.pi),
                         np.cos(x/np.pi)])
        data += noise
        time.sleep(self.delay)
        # Counter used in test suite to test how many times data was acquired.
        self.times_called += 1

        return data


class QX_Hard_Detector(Hard_Detector):

    def __init__(self, qxc, qasm_filenames, p_error=0.004,
                 num_avg=128, **kw):
        super().__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.value_names = []
        self.value_units = []
        self.times_called = 0
        self.__qxc = qxc
        self.num_avg = num_avg
        self.num_files = len(qasm_filenames)
        self.p_error = p_error
        self.delay = 1
        self.current = 0
        self.randomizations = []

        for i in range(self.__qxc.get_nr_qubits()):
            self.value_names.append("q"+str(i))
            self.value_units.append('|1>')

        # load files
        logging.info("QX_RB_Hard_Detector : loading qasm files...")
        for i, file_name in enumerate(qasm_filenames):
            t1 = time.time()
            qasm = ql.qasm_loader(file_name, qxc.get_nr_qubits())
            qasm.load_circuits()
            circuits = qasm.get_circuits()
            self.randomizations.append(circuits)
            # create the circuits on the server
            t1 = time.time()

            for c in circuits:
                circuit_name = c[0] + "{}".format(i)
                self.__qxc.create_circuit(circuit_name, c[1])
            t2 = time.time()
            logging.info("[+] qasm loading time :", t2-t1)

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points
        self.circuits = self.randomizations[self.current]

    def get_values(self):
        # x = self.sweep_points
        # only serves to initialize the arrays
        # data = np.array([np.sin(x / np.pi), np.cos(x/np.pi)])
        i = 0
        qubits = self.__qxc.get_nr_qubits()

        data = np.zeros((qubits, len(self.sweep_points)))

        for c in self.circuits:
            self.__qxc.send_cmd("reset_measurement_averaging")
            circuit_name = c[0] + "{}".format(self.current)
            self.__qxc.run_noisy_circuit(circuit_name, self.p_error,
                                         "depolarizing_channel", self.num_avg)
            for n in range(qubits):
                f = self.__qxc.get_measurement_average(n)
                data[n][i] = f
            # data[1][i] = f
            i = i + 1
        self.current = int((self.current + 1) % self.num_files)
        return (1-np.array(data))


class Dummy_Shots_Detector(Hard_Detector):

    def __init__(self, max_shots=10, **kw):
        super().__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.value_names = ['shots']
        self.value_units = ['m']
        self.max_shots = max_shots
        self.times_called = 0

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points
        self.times_called += 1

    def get_values(self):
        x = self.sweep_points
        dat = x[:self.max_shots]
        return dat


class Sweep_pts_detector(Detector_Function):

    """
    Returns the sweep points, used for testing purposes
    """

    def __init__(self, params, chunk_size=80):
        self.detector_control = 'hard'
        self.value_names = []
        self.value_units = []
        self.chunk_size = chunk_size
        self.i = 0
        for par in params:
            self.value_names += [par.name]
            self.value_units += [par.units]

    def prepare(self, sweep_points):
        self.i = 0
        self.sweep_points = sweep_points

    def get_values(self):
        return self.get()

    def acquire_data_point(self):
        return self.get()

    def get(self):
        print('passing chunk {}'.format(self.i))
        start_idx = self.i*self.chunk_size
        end_idx = start_idx + self.chunk_size
        self.i += 1
        time.sleep(.2)
        if len(np.shape(self.sweep_points)) == 2:
            return self.sweep_points[start_idx:end_idx, :].T
        else:
            return self.sweep_points[start_idx:end_idx]


class ZNB_VNA_detector(Hard_Detector):

    def __init__(self,  VNA, **kw):
        '''
        Detector function for the Rohde & Schwarz ZNB VNA
        '''
        super(ZNB_VNA_detector, self).__init__()
        self.VNA = VNA
        self.value_names = ['ampl', 'phase',
                            'real', 'imag', ]
        self.value_units = ['', 'radians',
                            '', '']

    def get_values(self):
        '''
        Start a measurement, wait untill the end and retrive data.
        Return real and imaginary transmission coefficients +
        amplitude (linear) and phase (deg or radians)
        '''
        self.VNA.start_sweep_all()  # start a measurement
        # wait untill the end of measurement before moving on
        self.VNA.wait_to_continue()
        # for visualization on the VNA screen (no effect on data)
        self.VNA.autoscale_trace()
        # get data and process them
        real_data, imag_data = self.VNA.get_real_imaginary_data()

        complex_data = np.add(real_data, 1j*imag_data)
        ampl_linear = np.abs(complex_data)
        # ampl_dB = 20*np.log10(ampl_linear)
        phase_radians = np.arctan2(imag_data, real_data)

        return ampl_linear, phase_radians, real_data, imag_data


# Detectors for QuTech Control box modes
class CBox_input_average_detector(Hard_Detector):

    def __init__(self, CBox, AWG, nr_averages=1024, nr_samples=512, **kw):
        super(CBox_input_average_detector, self).__init__()
        self.CBox = CBox
        self.value_names = ['Ch0', 'Ch1']
        self.value_units = ['mV', 'mV']
        self.AWG = AWG
        scale_factor_dacmV = 1000.*0.75/128.
        scale_factor_integration = 1./(64.*self.CBox.integration_length())
        self.factor = scale_factor_dacmV*scale_factor_integration
        self.nr_samples = nr_samples
        self.nr_averages = nr_averages

    def get_values(self):
        if self.AWG is not None:
            self.AWG.start()
        data = np.double(self.CBox.get_input_avg_results()) * \
            np.double(self.factor)
        return data

    def prepare(self, sweep_points):
        self.CBox.acquisition_mode(0)
        if self.AWG is not None:
            self.AWG.stop()
        self.CBox.nr_averages(int(self.nr_averages))
        self.CBox.nr_samples(int(self.nr_samples))
        self.CBox.acquisition_mode('input averaging')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.CBox.acquisition_mode(0)


class CBox_integrated_average_detector(Hard_Detector):

    def __init__(self, CBox, AWG, seg_per_point=1, normalize=False, rotate=False,
                 nr_averages=1024, integration_length=1e-6, **kw):
        '''
        Integration average detector.
        Defaults to averaging data in a number of segments equal to the
        nr of sweep points specificed.

        seg_per_point allows you to use more than 1 segment per sweeppoint.
        this is for example useful when doing a MotzoiXY measurement in which
        there are 2 datapoints per sweep point.
        Normalize/Rotate adds a third measurement with the rotated/normalized data.
        '''
        super().__init__(**kw)
        self.CBox = CBox
        if rotate or normalize:
            self.value_names = ['F|1>', 'F|1>']
            self.value_units = ['', '']
        else:
            self.value_names = ['I', 'Q']
            self.value_units = ['a.u.', 'a.u.']
        self.AWG = AWG
        self.seg_per_point = seg_per_point
        self.rotate = rotate
        self.normalize = normalize
        self.cal_points = kw.get('cal_points', None)
        self.nr_averages = nr_averages
        self.integration_length = integration_length

    def get_values(self):
        succes = False
        i = 0
        while not succes:
            try:
                self.AWG.stop()
                self.CBox.set('acquisition_mode', 'idle')
                self.CBox.set('acquisition_mode', 'integration averaging')
                if self.AWG is not None:
                    self.AWG.start()
                # does not restart AWG tape in CBox as we don't use it anymore
                data = self.CBox.get_integrated_avg_results()
                succes = True
            except Exception as e:
                logging.warning('Exception caught retrying')
                logging.warning(e)
                self.CBox.set('acquisition_mode', 'idle')
                if self.AWG is not None:
                    self.AWG.stop()
                # commented because deprecated
                # self.CBox.restart_awg_tape(0)
                # self.CBox.restart_awg_tape(1)
                # self.CBox.restart_awg_tape(2)

                self.CBox.set('acquisition_mode', 'integration averaging')
                # Is needed here to ensure data aligns with seq elt
                if self.AWG is not None:
                    self.AWG.start()
            i += 1
            if i > 20:
                break
        if self.rotate or self.normalize:
            return self.rotate_and_normalize(data)
        else:
            return data

    def rotate_and_normalize(self, data):
        """
        Rotates and normalizes
        """
        if self.cal_points is None:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=data,
                    cal_zero_points=list(range(-4, -2)),
                    cal_one_points=list(range(-2, 0)))
        else:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.measured_values[0:2],
                    cal_zero_points=self.cal_points[0],
                    cal_one_points=self.cal_points[1])
        return self.corr_data, self.corr_data

    def prepare(self, sweep_points):
        self.CBox.set('nr_samples', self.seg_per_point*len(sweep_points))
        if self.AWG is not None:
            self.AWG.stop()  # needed to align the samples
        self.CBox.nr_averages(int(self.nr_averages))
        self.CBox.integration_length(int(self.integration_length/(5e-9)))
        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set('acquisition_mode', 'integration averaging')
        if self.AWG is not None:
            self.AWG.start()  # Is needed here to ensure data aligns with seq elt

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')


class CBox_single_integration_average_det(Soft_Detector):

    '''
    Detector used for acquiring single points of the CBox while externally
    triggered by the AWG.
    Soft version of the regular integrated avg detector.

    Has two acq_modes, 'IQ' and 'AmpPhase'
    '''

    def __init__(self, CBox, acq_mode='IQ', **kw):
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_single_integration_avg_det'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        if acq_mode == 'IQ':
            self.acquire_data_point = self.acquire_data_point_IQ
        elif acq_mode == 'AmpPhase':
            self.acquire_data_point = self.acquire_data_point_amp_ph
        else:
            raise ValueError('acq_mode must be "IQ" or "AmpPhase"')

    def acquire_data_point_IQ(self, **kw):
        success = False
        i = 0
        while not success:
            self.CBox.acquisition_mode('integration averaging')
            try:
                data = self.CBox.get_integrated_avg_results()
                success = True
            except Exception as e:
                logging.warning(e)
                logging.warning('Exception caught retrying')
            self.CBox.acquisition_mode('idle')
            i += 1
            if i > 10:
                break
        return data

    def acquire_data_point_amp_ph(self, **kw):
        data = self.acquire_data_point_IQ()
        S21 = data[0] + 1j * data[1]
        return abs(S21), np.angle(S21)/(2*np.pi)*360

    def prepare(self):
        self.CBox.set('nr_samples', 1)
        self.CBox.set('acquisition_mode', 'idle')

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')


class CBox_single_int_avg_with_LutReload(CBox_single_integration_average_det):

    '''
    Detector used for acquiring single points of the CBox while externally
    triggered by the AWG.
    Very similar to the regular integrated avg detector.
    '''

    def __init__(self, CBox, LutMan, reload_pulses='all', awg_nrs=[0], **kw):
        super().__init__(CBox, **kw)
        self.LutMan = LutMan
        self.reload_pulses = reload_pulses
        self.awg_nrs = awg_nrs

    def acquire_data_point(self, **kw):
        #
        # self.LutMan.load_pulse_onto_AWG_lookuptable('X180', 1)
        if self.reload_pulses == 'all':
            for awg_nr in self.awg_nrs:
                self.LutMan.load_pulses_onto_AWG_lookuptable(awg_nr)

        else:
            for pulse_name in self.reload_pulses:
                for awg_nr in self.awg_nrs:
                    self.LutMan.load_pulse_onto_AWG_lookuptable(
                        pulse_name, awg_nr)
        return super().acquire_data_point(**kw)


class CBox_integration_logging_det(Hard_Detector):

    def __init__(self, CBox, AWG, integration_length=1e-6, LutMan=None,
                 reload_pulses=False,
                 awg_nrs=None, **kw):
        '''
        If you want AWG reloading you should give a LutMan and specify
        on what AWG nr to reload default is no reloading of pulses.
        '''
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_integration_logging_detector'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        self.AWG = AWG

        self.LutMan = LutMan
        self.reload_pulses = reload_pulses
        self.awg_nrs = awg_nrs
        self.integration_length = integration_length

    def get_values(self):
        exception_mode = True
        if exception_mode:
            success = False
            i = 0
            while not success and i < 10:
                try:
                    d = self._get_values()
                    success = True
                except Exception as e:
                    logging.warning(
                        'Exception {} caught, retaking data'.format(e))
                    i += 1
        else:
            d = self._get_values()
        return d

    def _get_values(self):
        self.AWG.stop()
        self.CBox.set('acquisition_mode', 'idle')
        if self.awg_nrs is not None:
            for awg_nr in self.awg_nrs:
                self.CBox.restart_awg_tape(awg_nr)
                if self.reload_pulses:
                    self.LutMan.load_pulses_onto_AWG_lookuptable(awg_nr)
        self.CBox.set('acquisition_mode', 'integration logging')
        self.AWG.start()

        data = self.CBox.get_integration_log_results()

        self.CBox.set('acquisition_mode', 'idle')
        return data

    def prepare(self, sweep_points):
        self.CBox.integration_length(int(self.integration_length/(5e-9)))

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')
        self.AWG.stop()


class CBox_integration_logging_det_shots(Hard_Detector):

    def __init__(self, CBox, AWG, LutMan=None, reload_pulses=False,
                 awg_nrs=None, shots=8000, **kw):
        '''
        If you want AWG reloading you should give a LutMan and specify
        on what AWG nr to reload default is no reloading of pulses.
        '''
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_integration_logging_detector'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        self.AWG = AWG

        self.LutMan = LutMan
        self.reload_pulses = reload_pulses
        self.awg_nrs = awg_nrs
        self.repetitions = int(np.ceil(shots/8000))

    def get_values(self):
        d_0 = []
        d_1 = []
        for i in range(self.repetitions):
            exception_mode = True
            if exception_mode:
                success = False
                i = 0
                while not success and i < 10:
                    try:
                        d = self._get_values()
                        success = True
                    except Exception as e:
                        logging.warning(
                            'Exception {} caught, retaking data'.format(e))
                        i += 1
            else:
                d = self._get_values()
            h_point = len(d)/2
            d_0.append(d[:h_point])
            d_1.append(d[h_point:])
        d_all = np.concatenate(
            (np.array(d_0).flatten(), np.array(d_1).flatten()))

        return d_all

    def _get_values(self):
        self.AWG.stop()
        self.CBox.set('acquisition_mode', 'idle')
        if self.awg_nrs is not None:
            for awg_nr in self.awg_nrs:
                self.CBox.restart_awg_tape(awg_nr)
                if self.reload_pulses:
                    self.LutMan.load_pulses_onto_AWG_lookuptable(awg_nr)
        self.CBox.set('acquisition_mode', 'integration logging')
        self.AWG.start()

        data = self.CBox.get_integration_log_results()

        self.CBox.set('acquisition_mode', 'idle')
        return data

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')
        self.AWG.stop()


class CBox_state_counters_det(Soft_Detector):

    def __init__(self, CBox, **kw):
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_state_counters_detector'
        # A and B refer to the counts for the different weight functions
        self.value_names = ['no error A', 'single error A', 'double error A',
                            '|0> A', '|1> A',
                            'no error B', 'single error B', 'double error B',
                            '|0> B', '|1> B', ]
        self.value_units = ['#']*10

    def acquire_data_point(self):
        success = False
        i = 0
        while not success and i < 10:
            try:
                data = self._get_values()
                success = True
            except Exception as e:
                logging.warning('Exception {} caught, retaking data'.format(e))
                i += 1
        return data

    def _get_values(self):

        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set('acquisition_mode', 'integration logging')

        data = self.CBox.get_qubit_state_log_counters()
        self.CBox.set('acquisition_mode', 'idle')
        return np.concatenate(data)  # concatenates counters A and B

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')


class CBox_single_qubit_event_s_fraction(CBox_state_counters_det):

    '''
    Child of the state counters detector
    Returns fraction of event type s by using state counters 1 and 2
    Rescales the measured counts to percentages.
    '''

    def __init__(self, CBox):
        super(CBox_state_counters_det, self).__init__()
        self.CBox = CBox
        self.name = 'CBox_single_qubit_event_s_fraction'
        self.value_names = ['frac. err.', 'frac. 2 or more', 'frac. event s']
        self.value_units = ['%', '%', '%']

    def prepare(self, **kw):
        self.nr_shots = self.CBox.log_length.get()

    def acquire_data_point(self):
        d = super().acquire_data_point()
        data = [
            d[1]/self.nr_shots*100,
            d[2]/self.nr_shots*100,
            (d[1]-d[2])/self.nr_shots*100]
        return data


class CBox_single_qubit_frac1_counter(CBox_state_counters_det):

    '''
    Based on the shot counters, returns the fraction of shots that corresponds
    to a specific state.
    Note that this is not corrected for RO-fidelity.

    Also note that depending on the side of the RO the F|1> and F|0> could be
    inverted
    '''

    def __init__(self, CBox):
        super(CBox_state_counters_det, self).__init__()
        self.detector_control = 'soft'
        self.CBox = CBox
        self.name = 'CBox_single_qubit_frac1_counter'
        # A and B refer to the counts for the different weight functions
        self.value_names = ['Frac_1']
        self.value_units = ['']

    def acquire_data_point(self):
        d = super().acquire_data_point()
        data = d[4]/(d[3]+d[4])
        return data


class CBox_digitizing_shots_det(CBox_integration_logging_det):

    """docstring for  CBox_digitizing_shots_det"""

    def __init__(self, CBox, AWG, threshold,
                 LutMan=None, reload_pulses=False, awg_nrs=None):
        super().__init__(CBox, AWG, LutMan, reload_pulses, awg_nrs)
        self.name = 'CBox_digitizing_shots_detector'
        self.value_names = ['Declared state']
        self.value_units = ['']
        self.threshold = threshold

    def get_values(self):
        dat = super().get_values()
        d = dat[0]
        # comparing 8000 vals with threshold takes 3.8us
        # converting to int 10.8us and to float 13.8us, let's see later if we
        # can cut that.
        return (d > self.threshold).astype(int)


##############################################################################
##############################################################################
####################     Software Controlled Detectors     ###################
##############################################################################
##############################################################################


class Dummy_Detector_Soft(Soft_Detector):

    def __init__(self, delay=0, **kw):
        self.set_kw()
        self.delay = delay
        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['mV', 'mV']
        self.i = 0
        # self.x can be used to set x value externally
        self.x = None

    def acquire_data_point(self, **kw):
        if self.x is None:
            x = self.i/15.
        self.i += 1
        time.sleep(self.delay)
        return np.array([np.sin(x/np.pi), np.cos(x/np.pi)])


class QX_Detector(Soft_Detector):

    def __init__(self, qxc, delay=0, **kw):
        self.set_kw()
        self.delay = delay
        self.detector_control = 'soft'
        self.name = 'QX_Detector'
        self.value_names = ['F']  # ['F', 'F']
        self.value_units = ['Error Rate']  # ['mV', 'mV']
        self.__qxc = qxc
        self.__cnt = 0

    def acquire_data_point(self, **kw):
        circuit_name = ("circuit%i" % self.__cnt)
        errors = 0

        executions = 1000
        p_error = 0.001+self.__cnt*0.003
        '''
        for i in range(0,executions):
            self.__qxc.run_noisy_circuit(circuit_name,p_error)
            m0 = self.__qxc.get_measurement(0)
            # m1 = self.__qxc.get_measurement(1)
            if int(m0) != 0 :
                errors += 1
            # print("[+] measurement outcome : %s %s" % (m0,m1))
        # x = self.__cnt/15.
        '''
        print("[+] p error  :", p_error)
        # print("[+] errors   :",errors)
        # f = (executions-errors)/executions
        self.__qxc.send_cmd("reset_measurement_averaging")
        self.__qxc.run_noisy_circuit(
            circuit_name, p_error, "depolarizing_channel", executions)
        f = self.__qxc.get_measurement_average(0)
        print("[+] fidelity :", f)
        self.__qxc.send_cmd("reset_measurement_averaging")

        self.__cnt = self.__cnt+1
        return f


class Function_Detector(Soft_Detector):
    """
    Defines a detector function that wraps around an user-defined function.
    Inputs are:
        get_function (fun) : function used for acquiring values
        value_names (list) : names of the elements returned by the function
        value_units (list) : units of the elements returned by the function
        result_keys (list) : keys of the dictionary returned by the function
                             if not None
        msmt_kw   (dict)   : kwargs for the get_function, dict items can be
            values or parameters. If they are parameters the output of the
            get method will be used for each get_function evaluation.

    The input function get_function must return a dictionary.
    The contents(keys) of this dictionary are going to be the measured
    values to be plotted and stored by PycQED
    """

    def __init__(self, get_function, value_names=None,
                 value_units=None, msmt_kw={}, result_keys=None, **kw):
        super().__init__()
        self.get_function = get_function
        self.result_keys = result_keys
        self.value_names = value_names
        self.value_units = value_units
        self.msmt_kw = msmt_kw
        if self.value_names is None:
            self.value_names = result_keys
        if self.value_units is None:
            self.value_units = ['a.u.'] * len(self.value_names)

    def acquire_data_point(self, **kw):
        measurement_kwargs = {}
        # If an entry has a get method that will be used to set the value.
        # This makes parameters work in this context.
        for key, item in self.msmt_kw.items():
            if isinstance(item, _BaseParameter):
                value = item.get()
            else:
                value = item
            measurement_kwargs[key] = value

        # Call the function
        result = self.get_function(**measurement_kwargs)
        if self.result_keys is None:
            return result
        else:
            results = [result[key] for key in self.result_keys]
            if len(results) == 1:
                return results[0]  # for a single entry we don't want a list
            return results


class Detect_simulated_hanger_Soft(Soft_Detector):

    def __init__(self, **kw):
        self.set_kw()

        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['mV', 'mV']

    def acquire_data_point(self, **kw):
        f = self.source.get_frequency()
        f0 = 5.e9
        Q = 10000.
        Qe = 12000.
        theta = 0.2
        A = 50.
        Inoise = np.random.randn()
        Qnoise = np.random.randn()

        IQ = fn.disp_hanger_S21_complex(*(f, f0, Q, Qe, A, theta))
        return IQ.real+Inoise, IQ.imag+Qnoise


class Heterodyne_probe(Soft_Detector):

    def __init__(self, HS, threshold=1.75, trigger_separation=10e-6,
                 demod_mode='double', RO_length=2000e-9, **kw):
        super().__init__(**kw)
        self.HS = HS
        self.name = 'Heterodyne probe'
        self.value_names = ['|S21|', 'S21 angle']  # , 'Re{S21}', 'Im{S21}']
        self.value_units = ['V', 'deg']
        self.first = True
        self.last_frequency = 0.
        self.threshold = threshold
        self.last = 1.
        self.HS.trigger_separation(trigger_separation)
        if 'double' in demod_mode:
            HS.single_sideband_demod(False)
        else:
            HS.single_sideband_demod(True)

        self.trigger_separation = trigger_separation
        self.demod_mode = demod_mode
        self.RO_length = RO_length

    def prepare(self):
        self.HS.RO_length(self.RO_length)
        self.HS.trigger_separation(self.trigger_separation)

        self.HS.prepare()

    def acquire_data_point(self, **kw):
        passed = False
        c = 0
        while(not passed):
            S21 = self.HS.probe()
            cond_a = ((abs(S21)/self.last) >
                      self.threshold) or ((self.last/abs(S21)) > self.threshold)
            cond_b = self.HS.frequency() >= self.last_frequency
            if cond_a and cond_b:
                passed = False
            else:
                passed = True
            if self.first or c > 3:
                passed = True
            # if not passed:
            #     print('retrying HS probe')
            c += 1
        self.last_frequency = self.HS.frequency()
        self.first = False
        self.last = abs(S21)
        return abs(S21), np.angle(S21)/(2*np.pi)*360,  # S21.real, S21.imag

    def finish(self):
        self.HS.finish()


class Heterodyne_probe_soft_avg(Soft_Detector):

    def __init__(self, HS, threshold=1.75, Navg=10, **kw):
        super().__init__(**kw)
        self.HS = HS
        self.name = 'Heterodyne probe'
        self.value_names = ['|S21|', 'S21 angle']  # , 'Re{S21}', 'Im{S21}']
        self.value_units = ['mV', 'deg']  # , 'a.u.', 'a.u.']
        self.first = True
        self.last_frequency = 0.
        self.threshold = threshold
        self.last = 1.
        self.Navg = Navg

    def prepare(self):
        self.HS.prepare()

    def acquire_data_point(self, **kw):
        accum_real = 0.
        accum_imag = 0.
        for i in range(self.Navg):
            measure = self.acquire_single_data_point(**kw)
            accum_real += measure[0]
            accum_imag += measure[1]
        S21 = (accum_real+1j*accum_imag)/float(self.Navg)

        return abs(S21), np.angle(S21)/(2*np.pi)*360

    def acquire_single_data_point(self, **kw):
        passed = False
        c = 0
        while(not passed):
            S21 = self.HS.probe()
            cond_a = (
                abs(S21)/self.last > self.threshold) or (self.last/abs(S21) > self.threshold)
            cond_b = self.HS.frequency() > self.last_frequency
            if cond_a and cond_b:
                passed = False
            else:
                passed = True
            if self.first or c > 3:
                passed = True
            if not passed:
                print('retrying HS probe')
            c += 1
        self.last_frequency = self.HS.frequency()
        self.first = False
        self.last = abs(S21)
        return S21.real, S21.imag

    def finish(self):
        self.HS.finish()


class Signal_Hound_fixed_frequency(Soft_Detector):

    def __init__(self, signal_hound, frequency=None, Navg=1, delay=0.1,
                 prepare_for_each_point=False, **kw):
        super().__init__()
        self.frequency = frequency
        self.name = 'SignalHound_fixed_frequency'
        self.value_names = ['Power']
        self.value_units = ['dBm']
        self.delay = delay
        self.SH = signal_hound
        if frequency is not None:
            self.SH.set('frequency', frequency)
        self.Navg = Navg
        self.prepare_for_each_point = prepare_for_each_point

    def acquire_data_point(self, **kw):
        if self.prepare_for_each_point:
            self.SH.prepare_for_measurement()
        time.sleep(self.delay)
        return self.SH.get_power_at_freq(Navg=self.Navg)

    def prepare(self, **kw):
        self.SH.prepare_for_measurement()

    def finish(self, **kw):
        self.SH.abort()


class SH_mixer_skewness_det(Soft_Detector):

    '''
    Based on the "Signal_Hound_fixed_frequency" detector.
    generates an AWG seq to measure sideband transmission

    Inputs:
        frequency       (Hz)
        QI_amp_ratio    (parameter)
        IQ_phase        (parameter)
        SH              (instrument)
        f_mod           (Hz)

    '''

    def __init__(self, frequency, QI_amp_ratio, IQ_phase, SH,
                 I_ch, Q_ch,
                 station,
                 Navg=1, delay=0.1, f_mod=10e6, verbose=False, **kw):
        super(SH_mixer_skewness_det, self).__init__()
        self.SH = SH
        self.frequency = frequency
        self.name = 'SignalHound_mixer_skewness_det'
        self.value_names = ['Power']
        self.value_units = ['dBm']
        self.delay = delay
        self.SH.frequency.set(frequency)  # Accepts input in Hz
        self.Navg = Navg
        self.QI_amp_ratio = QI_amp_ratio
        self.IQ_phase = IQ_phase
        self.pulsar = station.pulsar
        self.f_mod = f_mod
        self.I_ch = I_ch
        self.Q_ch = Q_ch
        self.verbose = verbose

    def acquire_data_point(self, **kw):
        QI_ratio = self.QI_amp_ratio.get()
        skewness = self.IQ_phase.get()
        if self.verbose:
            print('QI ratio: %.3f' % QI_ratio)
            print('skewness: %.3f' % skewness)
        self.generate_awg_seq(QI_ratio, skewness, self.f_mod)
        self.pulsar.AWG.start()
        time.sleep(self.delay)
        return self.SH.get_power_at_freq(Navg=self.Navg)

    def generate_awg_seq(self, QI_ratio, skewness, f_mod):
        SSB_modulation_el = element.Element('SSB_modulation_el',
                                            pulsar=self.pulsar)
        cos_pulse = pulse.CosPulse(channel=self.I_ch, name='cos_pulse')
        sin_pulse = pulse.CosPulse(channel=self.Q_ch, name='sin_pulse')

        SSB_modulation_el.add(pulse.cp(cos_pulse, name='cos_pulse',
                                       frequency=f_mod, amplitude=0.15,
                                       length=1e-6, phase=0))
        SSB_modulation_el.add(pulse.cp(sin_pulse, name='sin_pulse',
                                       frequency=f_mod, amplitude=0.15 *
                                       QI_ratio,
                                       length=1e-6, phase=90+skewness))

        seq = sequence.Sequence('Sideband_modulation_seq')
        seq.append(name='SSB_modulation_el', wfname='SSB_modulation_el',
                   trigger_wait=False)
        self.pulsar.program_awgs(seq, SSB_modulation_el)

    def prepare(self, **kw):
        self.SH.prepare_for_measurement()

    def finish(self, **kw):
        self.SH.abort()

# ---------------------------------------------------------------------------
# CBox v3 detectors
# ---------------------------------------------------------------------------


class CBox_v3_integrated_average_detector(Hard_Detector):

    def __init__(self, CBox, seg_per_point=1, **kw):
        '''
        Integration average detector.
        Defaults to averaging data in a number of segments equal to the
        nr of sweep points specificed.

        seg_per_point allows you to use more than 1 segment per sweeppoint.
        this is for example useful when doing a MotzoiXY measurement in which
        there are 2 datapoints per sweep point.
        '''
        super().__init__(**kw)
        self.CBox = CBox
        self.name = 'CBox_integrated_average_detector'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        self.seg_per_point = seg_per_point

    def get_values(self):
        succes = False
        data = None
        i = 0
        while not succes:
            try:
                data = self.CBox.get_integrated_avg_results()
                succes = True
            except Exception as e:
                logging.warning('Exception caught retrying')
                logging.warning(e)
                self.CBox.set('acquisition_mode', 'idle')
                self.CBox.set('acquisition_mode', 'integration averaging mode')
            i += 1
            if i > 20:
                break

        return data

    def prepare(self, sweep_points):
        self.CBox.set('nr_samples', self.seg_per_point*len(sweep_points))
        self.CBox.run_mode(0)
        # integration average.
        self.CBox.acquisition_mode('integration averaging mode')
        self.CBox.run_mode(1)

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')


class CBox_v3_single_integration_average_det(Soft_Detector):

    '''
    Detector used for acquiring single points of the CBox while externally
    triggered by the AWG.
    Soft version of the regular integrated avg detector.

    Has two acq_modes, 'IQ' and 'AmpPhase'
    '''

    def __init__(self, CBox, acq_mode='IQ', **kw):
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_v3_single_integration_avg_det'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        if acq_mode == 'AmpPhase':
            self.acquire_data_point = self.acquire_data_point_amp_ph

    def acquire_data_point(self, **kw):
        success = False
        i = 0
        # import traceback  as tb
        # import sys
        # tb.print_tb(sys.last_traceback)
        while not success:
            print("acquiring")

            self.CBox.set('acquisition_mode', 'integration averaging mode')
            try:
                data = self.CBox.get_integrated_avg_results()
                print("detector function, data", data)
                success = True
            except Exception as e:
                logging.warning(e)
                logging.warning('Exception caught retrying')
            self.CBox.set('acquisition_mode', 'idle')
            i += 1
            if i > 20:
                break
        return data

    def acquire_data_point_amp_ph(self, **kw):
        data = self.acquire_data_point_IQ()
        S21 = data[0] + 1j * data[1]
        return abs(S21), np.angle(S21)/(2*np.pi)*360

    def prepare(self):
        self.CBox.run_mode(0)
        self.CBox.set('nr_samples', 1)
        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.run_mode(1)

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')


class CBox_v3_single_int_avg_with_LutReload(CBox_v3_single_integration_average_det):

    '''
    Detector used for acquiring single points of the CBox while externally
    triggered by the AWG.
    Very similar to the regular integrated avg detector.
    '''

    def __init__(self, CBox, LutMan, reload_pulses='all', awg_nrs=[2], **kw):
        super().__init__(CBox, **kw)
        self.LutMan = LutMan
        self.reload_pulses = reload_pulses
        self.awg_nrs = awg_nrs
        self.name = 'CBox_v3_single_int_avg_with_LutReload'

    def acquire_data_point(self, **kw):
        #
        # self.LutMan.load_pulse_onto_AWG_lookuptable('X180', 1)
        if self.reload_pulses == 'all':
            for awg_nr in self.awg_nrs:
                self.LutMan.load_pulses_onto_AWG_lookuptable(awg_nr)
        else:
            for pulse_name in self.reload_pulses:
                for awg_nr in self.awg_nrs:
                    self.LutMan.load_pulse_onto_AWG_lookuptable(
                        pulse_name, awg_nr)

        return super().acquire_data_point(**kw)

# --------------------------------------------
# Zurich Instruments UHFQC detector functions
# --------------------------------------------


class UHFQC_input_average_detector(Hard_Detector):

    '''
    Detector used for acquiring averaged input traces withe the UHFQC

    '''

    def __init__(self, UHFQC, AWG=None, channels=(0, 1),
                 nr_averages=1024, nr_samples=4096, **kw):
        super(UHFQC_input_average_detector, self).__init__()
        self.UHFQC = UHFQC
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = 'ch{}'.format(channel)
            self.value_units[i] = 'V'
        self.AWG = AWG
        self.nr_samples = nr_samples
        self.nr_averages = nr_averages

    def get_values(self):
        self.UHFQC.quex_rl_readout(0)  # resets UHFQC internal readout counters

        self.UHFQC.acquisition_arm()
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(samples=self.nr_sweep_points,
                                               arm=False, acquisition_time=0.01)
        data = np.array([data_raw[key]
                         for key in sorted(data_raw.keys())])  # *self.scaling_factor

        return data

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.quex_iavg_length(self.nr_samples)
        self.UHFQC.quex_iavg_avgcnt(int(np.log2(self.nr_averages)))
        self.UHFQC.awgs_0_userregs_1(1)  # 0 for rl, 1 for iavg
        self.UHFQC.awgs_0_userregs_0(
            int(self.nr_averages))  # 0 for rl, 1 for iavg
        self.nr_sweep_points = self.nr_samples
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='iavg')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class UHFQC_demodulated_input_avg_det(UHFQC_input_average_detector):
    '''
    Detector used for acquiring averaged input traces withe the UHFQC.
    Additionally trace are demoulated.
    '''

    def __init__(self, f_RO_mod, UHFQC,
                 real_imag=True, AWG=None, channels=(0, 1),
                 nr_averages=1024, nr_samples=4096, **kw):
        super().__init__(UHFQC=UHFQC, AWG=AWG, channels=channels,
                         nr_averages=nr_averages, nr_samples=nr_samples, **kw)
        self.f_mod = f_RO_mod
        self.real_imag = real_imag

    def get_values(self):
        data = super().get_values()
        timePts = np.arange(len(data[0])) / 1.8e9
        data_I_demod = (np.array(data[0]) * np.cos(2*np.pi*timePts*self.f_mod) -
                        np.array(data[1]) * np.sin(2*np.pi*timePts*self.f_mod))
        data_Q_demod = (np.array(data[0]) * np.sin(2*np.pi*timePts*self.f_mod) +
                        np.array(data[1]) * np.cos(2*np.pi*timePts*self.f_mod))
        ret_data = np.array([data_I_demod, data_Q_demod])

        if not self.real_imag:
            S21 = data[0] + 1j*data[1]
            amp = np.abs(S21)
            phase = np.angle(S21) / (2*np.pi) * 360
            ret_data = np.array([amp, phase])

        return ret_data


class UHFQC_integrated_average_detector(Hard_Detector):

    '''
    Detector used for integrated average results with the UHFQC

    '''

    def __init__(self, UHFQC, AWG=None,
                 integration_length: float=1e-6, nr_averages: int=1024,
                 channels: list=(0, 1, 2, 3), result_logging_mode: str='raw',
                 real_imag: bool=True,
                 seg_per_point: int =1, single_int_avg: bool =False,
                 chunk_size: int=None,
                 prepare_function=None, prepare_function_kwargs: dict=None,
                 **kw):
        """
        Args:
        UHFQC (instrument) : data acquisition device
        AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller

        integration_length (float): integration length in seconds
        nr_averages (int)         : nr of averages per data point

        result_logging_mode (str) :  options are
            - raw        -> returns raw data in V
            - lin_trans  -> applies the linear transformation matrix and
                            subtracts the offsets defined in the UFHQC.
                            This is typically used for crosstalk suppression
                            and normalization. Requires optimal weights.
            - digitized  -> returns fraction of shots based on the threshold
                            defined in the UFHQC. Requires optimal weights.

        real_imag (bool)     : if False returns data in polar coordinates
                                useful for e.g., spectroscopy
                                #FIXME -> should be named "polar"
        seg_per_point (int)  : number of segments per sweep point
        single_int_avg (bool): if True makes this a soft detector
        """

        super().__init__()
        self.UHFQC = UHFQC
        self.name = '{}_UHFQC_integrated_average'.format(result_logging_mode)
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = '{} w{}'.format(result_logging_mode,
                                                  channel)
        if result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.channels)
            self.scaling_factor = 1/(1.8e9*integration_length*nr_averages)
        elif result_logging_mode == 'lin_trans':
            self.value_units = ['a.u.']*len(self.channels)
            self.scaling_factor = 1/nr_averages

        elif result_logging_mode == 'digitized':
            self.value_units = ['frac']*len(self.channels)
            self.scaling_factor = 1

        self.single_int_avg = single_int_avg
        if self.single_int_avg:
            self.detector_control = 'soft'

        self.seg_per_point = seg_per_point

        self.AWG = AWG
        self.nr_averages = nr_averages
        self.integration_length = integration_length
        # 0/1/2 crosstalk supressed /digitized/raw
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        self.result_logging_mode_idx = res_logging_indices[result_logging_mode]
        self.result_logging_mode = result_logging_mode
        self.chunk_size = chunk_size

        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs
        self._set_real_imag(real_imag)

    def _set_real_imag(self, real_imag=False):
        """
        Function so that real imag can be changed after initialization
        """

        self.real_imag = real_imag
        if self.result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.channels)
        else:
            self.value_units = ['']*len(self.channels)

        if not self.real_imag:
            if len(self.channels) != 2:
                raise ValueError('Length of "{}" is not 2'.format(
                                 self.channels))
            self.value_names[0] = 'Magn'
            self.value_names[1] = 'Phase'
            self.value_units[1] = 'deg'

    def get_values(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.quex_rl_readout(1)  # resets UHFQC internal readout counters
        self.UHFQC.acquisition_arm()
        # starting AWG

        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(
            samples=self.nr_sweep_points, arm=False, acquisition_time=0.01)
        # the self.channels should be the same as data_raw.keys().
        # this is to be tested (MAR 26-9-2017)
        data = np.array([data_raw[key]
                         for key in sorted(data_raw.keys())])*self.scaling_factor

        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'quex_trans_offset_weightfunction_{}'.format(channel))
        if not self.real_imag:
            data = self.convert_to_polar(data)
        return data

    def convert_to_polar(self, data):
        if len(data) != 2:
            raise ValueError('Expect 2 channels for rotation. Got {}'.format(
                             len(data)))
        I = data[0]
        Q = data[1]
        S21 = I + 1j*Q
        data[0] = np.abs(S21)
        data[1] = np.angle(S21)/(2*np.pi)*360
        return data

    def acquire_data_point(self):
        return self.get_values()

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()

        # Determine the number of sweep points and set them
        if sweep_points is None or self.single_int_avg:
            # this case will be used when it is a soft detector
            # Note: single_int_avg overrides chunk_size
            # single_int_avg = True is equivalent to chunk_size = 1
            self.nr_sweep_points = self.seg_per_point
        else:
            self.nr_sweep_points = len(sweep_points)*self.seg_per_point
        # this sets the result to integration and rotation outcome
            if (self.chunk_size is not None and
                    self.chunk_size < self.nr_sweep_points):
                # Chunk size is defined and smaller than total number of sweep
                # points -> only acquire one chunk
                self.nr_sweep_points = self.chunk_size * self.seg_per_point

            if (self.chunk_size is not None and
                    self.chunk_size < self.nr_sweep_points):
                # Chunk size is defined and smaller than total number of sweep
                # points -> only acquire one chunk
                self.nr_sweep_points = self.chunk_size * self.seg_per_point

        self.UHFQC.awgs_0_userregs_0(
            int(self.nr_averages*self.nr_sweep_points))
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg (input avg)
        # The AWG program uses userregs/0 to define the number of iterations in
        # the loop

        self.UHFQC.quex_rl_length(self.nr_sweep_points)
        self.UHFQC.quex_rl_avgcnt(int(np.log2(self.nr_averages)))
        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))

        self.UHFQC.quex_rl_source(self.result_logging_mode_idx)
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

        # Optionally perform extra actions on prepare
        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class UHFQC_correlation_detector(UHFQC_integrated_average_detector):
    '''
    Detector used for correlation mode with the UHFQC.
    The argument 'correlations' is a list of tuples specifying which channels
    are correlated, and on which channel the correlated signal is output.
    For instance, 'correlations=[(0, 1, 3)]' will put the correlation of
    channels 0 and 1 on channel 3.
    '''

    def __init__(self, UHFQC, AWG=None, integration_length=1e-6,
                 nr_averages=1024, rotate=False, real_imag=True,
                 channels: list = [0, 1], correlations: list=[(0, 1)],
                 value_names=None,
                 seg_per_point=1, single_int_avg=False, thresholding=False,
                 **kw):
        super().__init__(
            UHFQC, AWG=AWG, integration_length=integration_length,
            nr_averages=nr_averages, real_imag=real_imag,
            channels=channels,
            seg_per_point=seg_per_point, single_int_avg=single_int_avg,
            result_logging_mode='raw',  # FIXME -> do the proper thing (MAR)
            **kw)
        self.correlations = correlations
        self.thresholding = thresholding

        if value_names is None:
            self.value_names = []
            for ch in channels:
                self.value_names += ['w{}'.format(ch)]
        else:
            self.value_names = value_names

        # Note that V^2 is in brackets to prevent confusion with unit prefixes
        if not thresholding:
            self.value_units = ['V']*len(self.value_names) + \
                               ['(V^2)']*len(self.correlations)
        else:
            self.value_units = ['fraction']*len(self.value_names) + \
                               ['normalized']*len(self.correlations)
        for corr in correlations:
            self.value_names += ['corr ({},{})'.format(corr[0], corr[1])]

        self.define_correlation_channels()

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()
        if sweep_points is None or self.single_int_avg:
            self.nr_sweep_points = self.seg_per_point
        else:
            self.nr_sweep_points = len(sweep_points)*self.seg_per_point

        self.UHFQC.quex_rl_length(self.nr_sweep_points)
        self.UHFQC.quex_rl_avgcnt(int(np.log2(self.nr_averages)))
        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))

        self.set_up_correlation_weights()

        # Configure the result logger to not do any averaging
        # The AWG program uses userregs/0 to define the number o iterations in
        # the loop
        self.UHFQC.awgs_0_userregs_0(
            int(self.nr_averages*self.nr_sweep_points))
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg

        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

    def define_correlation_channels(self):
        self.correlation_channels = []
        for corr in self.correlations:
            # Start by assigning channels
            if corr[0] not in self.channels or corr[1] not in self.channels:
                raise ValueError('Correlations should be in channels')

            correlation_channel = -1
            # 4 is the (current) max number of weights in the UHFQC (v5)
            for ch in range(4):
                if ch in self.channels:
                    # Disable correlation mode as this is used for normal
                    # acquisition
                    self.UHFQC.set('quex_corr_{}_mode'.format(ch), 0)

                if ch not in self.channels:
                    # selects the lowest available free channel
                    self.channels += [ch]
                    correlation_channel = ch
                    print('Using channel {} for correlation ({}, {}).'
                          .format(ch, corr[0], corr[1]))
                    # correlation mode is turned on in the
                    # set_up_correlation_weights method
                    break
            if correlation_channel < 0:
                raise ValueError('No free channel available for correlation.')

            self.correlation_channels += [correlation_channel]

    def set_up_correlation_weights(self):
        if self.thresholding:
            # correlations mode after threshold
            # NOTE: thresholds need to be set outside the detctor object.
            self.UHFQC.quex_rl_source(5)
        else:
            # correlations mode before threshold
            self.UHFQC.quex_rl_source(4)
        # Configure correlation mode
        for correlation_channel, corr in zip(self.correlation_channels,
                                             self.correlations):
            # Duplicate source channel to the correlation channel and select
            # second channel as channel to correlate with.
            copy_int_weights_real = \
                self.UHFQC.get('quex_wint_weights_{}_real'.format(corr[0]))[
                    0]['vector']
            copy_int_weights_imag = \
                self.UHFQC.get('quex_wint_weights_{}_imag'.format(corr[0]))[
                    0]['vector']
            self.UHFQC.set(
                'quex_wint_weights_{}_real'.format(correlation_channel),
                copy_int_weights_real)
            self.UHFQC.set(
                'quex_wint_weights_{}_imag'.format(correlation_channel),
                copy_int_weights_imag)
            # Enable correlation mode one the correlation output channel and
            # set the source to the second source channel
            self.UHFQC.set('quex_corr_{}_mode'.format(correlation_channel), 1)
            self.UHFQC.set('quex_corr_{}_source'.format(correlation_channel),
                           corr[1])

            # If thresholding is enabled, set the threshold for the correlation
            # channel.
            if self.thresholding:
                thresh_level = \
                    self.UHFQC.get('quex_thres_{}_level'.format(corr[0]))
                self.UHFQC.set(
                    'quex_thres_{}_level'.format(correlation_channel),
                    thresh_level)

    def get_values(self):
        # Slightly different way to deal with scaling factor
        self.scaling_factor = 1 / (1.8e9*self.integration_length)

        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.quex_rl_readout(1)  # resets UHFQC internal readout counters
        self.UHFQC.acquisition_arm()
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(samples=self.nr_sweep_points,
                                               arm=False,
                                               acquisition_time=0.01)

        data = []
        if self.thresholding:
            for key in sorted(data_raw.keys()):
                data.append(np.array(data_raw[key]))
        else:
            for key in sorted(data_raw.keys()):
                if key in self.correlation_channels:
                    data.append(np.array(data_raw[key]) *
                                (self.scaling_factor**2 / self.nr_averages))
                else:
                    data.append(np.array(data_raw[key]) *
                                (self.scaling_factor / self.nr_averages))

        return data


class UHFQC_integration_logging_det(Hard_Detector):

    '''
    Detector used for integrated average results with the UHFQC

    '''

    def __init__(self, UHFQC, AWG=None,
                 integration_length: float=1e-6,
                 nr_shots: int=4094,
                 channels: list=(0, 1),
                 result_logging_mode: str='raw', **kw):
        """
        Args:
        UHFQC (instrument) : data acquisition device
        AWG   (instrument) : device responsible for starting and stopping
                             the experiment, can also be a central controller.
        integration_length (float): integration length in seconds
        nr_shots (int)     : nr of shots (max is 4095)
        channels (list)    : index (channel) of UHFQC weight functions to use

        result_logging_mode (str):  options are
            - raw        -> returns raw data in V
            - lin_trans  -> applies the linear transformation matrix and
                            subtracts the offsets defined in the UFHQC.
                            This is typically used for crosstalk suppression
                            and normalization. Requires optimal weights.
            - digitized  -> returns fraction of shots based on the threshold
                            defined in the UFHQC. Requires optimal weights.
        """
        super().__init__()

        self.UHFQC = UHFQC
        self.name = '{}_UHFQC_integration_logging_det'.format(
            result_logging_mode)
        self.channels = channels

        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = '{} w{}'.format(result_logging_mode,
                                                  channel)
        if result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.channels)
            self.scaling_factor = 1/(1.8e9*integration_length)
        else:
            self.value_units = ['']*len(self.channels)
            self.scaling_factor = 1

        self.AWG = AWG
        self.integration_length = integration_length
        self.nr_shots = nr_shots

        # 0/1/2 crosstalk supressed /digitized/raw
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        # mode 3 is statistics logging, this is implemented in a
        # different detector
        self.result_logging_mode_idx = res_logging_indices[result_logging_mode]
        self.result_logging_mode = result_logging_mode

    def get_values(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.quex_rl_readout(1)  # resets UHFQC internal readout counters
        self.UHFQC.acquisition_arm()
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(
            samples=self.nr_shots, arm=False, acquisition_time=0.01)
        data = np.array([data_raw[key]
                         for key in sorted(data_raw.keys())])*self.scaling_factor

        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'quex_trans_offset_weightfunction_{}'.format(channel))
        return data

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()

        # The averaging-count is used to specify how many times the AWG program
        # should run

        self.UHFQC.awgs_0_single(1)
        self.UHFQC.awgs_0_userregs_0(self.nr_shots)
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg (input avg)
        # The AWG program uses userregs/0 to define the number of iterations
        # in the loop

        self.UHFQC.quex_rl_length(self.nr_shots)
        self.UHFQC.quex_rl_avgcnt(0)  # log2(1) for single shot readout
        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))

        self.UHFQC.quex_rl_source(self.result_logging_mode_idx)
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class UHFQC_statistics_logging_det(Soft_Detector):
    """
    Detector used for the statistics logging mode of the UHFQC
    """

    def __init__(self, UHFQC, AWG, nr_shots: int,
                 integration_length: float,
                 channels: list,
                 statemap: dict,
                 channel_names: list=None,
                 normalize_counts: bool=True):
        """
        Detector for the statistics logger mode in the UHFQC.

            UHFQC (instrument) : data acquisition device
            AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller.
            integration_length (float): integration length in seconds
            nr_shots (int)     : nr of shots, if a number larger than the max
                of the UFHQC (4095) is chosen it will take chunks of 4095.
                The total number of chunks is rounded up.
            channels (list)    : index (channel) of UHFQC weight functions
                to use
            statemap (dict) : dictionary specifying the expected output state
                for each 2 bit input state.
                e.g.:
                    statemap ={'00': '00', '01':'10', '10':'10', '11':'00'}
            normalize_counts (bool) : if False, returns total counts if True
                return fraction of counts
        """
        super().__init__()
        self.UHFQC = UHFQC
        self.AWG = AWG
        self.nr_shots = nr_shots

        self.integration_length = integration_length
        self.normalize_counts = normalize_counts

        self.channels = channels
        if channel_names is None:
            channel_names = ['ch{}'.format(ch) for ch in channels]
        self.value_names = []
        for ch in channel_names:
            ch_value_names = ['{} counts'.format(ch), '{} flips'.format(ch),
                              '{} state errors'.format(ch)]
            self.value_names.extend(ch_value_names)
        self.value_names.append('Total state errors')
        if not self.normalize_counts:
            self.value_units = '#'*len(self.value_names)
        else:
            self.value_units = ['frac']*len(self.value_names)
        self.statemap = statemap

        self.max_shots = 4095  # hardware limit of UHFQC

    @staticmethod
    def statemap_to_array(statemap: dict):
        """
        Converts a statemap dictionary to a numpy array that the
        UHFQC can accept as input.

        Args:
            statemap (dict) : dictionary specifying the expected output state
                for each 2 bit input state.
                e.g.:
                    statemap ={'00': '00', '01':'10', '10':'10', '11':'00'}

        """

        if not statemap.keys() == {'00', '01', '10', '11'}:
            raise ValueError('Invalid statemap: {}'.format(statemap))
        fm = {'00': 0x0,
              '01': 0x1,
              '10': 0x2,
              '11': 0x3}
        arr = np.array([fm[statemap['00']], fm[statemap['01']],
                        fm[statemap['10']], fm[statemap['11']]],
                       dtype=np.uint32)
        return arr

    def prepare(self):
        if self.AWG is not None:
            self.AWG.stop()

        max_shots = 4095  # hardware limit of UHFQC
        self.nr_chunks = self.nr_shots//max_shots + 1
        self.shots_per_chunk = np.min([self.nr_shots, max_shots])

        # The averaging-count is used to specify how many times the AWG program
        # should run
        self.UHFQC.awgs_0_single(1)
        self.UHFQC.awgs_0_userregs_0(self.shots_per_chunk)
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg (input avg)
        # The AWG program uses userregs/0 to define the number of iterations
        # in the loop

        # Configure the results logger not to do any averaging
        # self.UHFQC.quex_rl_length(self.nr_shots)
        self.UHFQC.quex_rl_avgcnt(0)  # log2(1) for single shot readout
        # self.UHFQC.quex_rl_source(0)

        # configure the results logger (rl)
        self.UHFQC.quex_rl_length(4)  # 4 values per channel for stat mode
        self.UHFQC.quex_rl_source(3)  # statistics logging is rl mode "3"

        # Configure the statistics logger (sl)
        self.UHFQC.quex_sl_length(self.shots_per_chunk)
        self.UHFQC.quex_sl_statemap(self.statemap_to_array(self.statemap))

        # above should be all

        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))
        # I think this should not be there but let's leave it
        #
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.acquisition_finalize()

    def acquire_data_point(self, **kw):
        if self.AWG is not None:
            self.AWG.stop()
        data_concat = np.zeros(7)
        for i in range(self.nr_chunks):
            # resets UHFQC internal readout counters
            self.UHFQC.quex_rl_readout(1)
            # resets UHFQC internal sl counters ?
            self.UHFQC.quex_sl_readout(0)

            self.UHFQC.acquisition_arm()
            # starting AWG
            if self.AWG is not None and i == 0:
                self.AWG.start()

            data = self.UHFQC.acquisition_poll(samples=4,  # double check this
                                               arm=False,
                                               acquisition_time=0.01)

            data_concat += np.concatenate((data[0][:-1], data[1]))

        if self.normalize_counts:
            return data_concat/(self.nr_chunks*self.shots_per_chunk)
        return data_concat


class UHFQC_single_qubit_statistics_logging_det(UHFQC_statistics_logging_det):

    def __init__(self, UHFQC, AWG, nr_shots: int,
                 integration_length: float,
                 channel: int,
                 statemap: dict,
                 channel_name: str=None,
                 normalize_counts: bool=True):
        """
        Detector for the statistics logger mode in the UHFQC.

            UHFQC (instrument) : data acquisition device
            AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller.
            integration_length (float): integration length in seconds
            nr_shots (int)     : nr of shots (max is 4095)
            channel  (int)    : index (channel) of UHFQC weight function
            statemap (dict) : dictionary specifying the expected output state
                for each 1 bit input state.
                e.g.:
                    statemap ={'0': '0', '1':'1'}
            channel_name (str) : optional name of the channel


        """
        super(UHFQC_statistics_logging_det, self).__init__()
        self.UHFQC = UHFQC
        self.AWG = AWG
        self.nr_shots = nr_shots
        self.integration_length = integration_length
        self.normalize_counts = normalize_counts

        self.channels = [channel, int((channel+1) % 5)]

        if channel_name is None:
            channel_name = ['ch{}'.format(channel)]

        self.value_names = ['{} counts'.format(channel),
                            '{} flips'.format(channel),
                            '{} state errors'.format(channel)]
        if not self.normalize_counts:
            self.value_units = '#'*len(self.value_names)
        else:
            self.value_units = ['frac']*len(self.value_names)
        self.statemap = statemap

        self.max_shots = 4095  # hardware limit of UHFQC

        self.statemap = self.statemap_one2two_bit(statemap)

    @staticmethod
    def statemap_one2two_bit(one_bit_sm: dict):
        """
        Converts a one bit statemap to an appropriate dummy 2-bit statemap
        """
        sm = one_bit_sm
        two_bit_sm = {'00': '{}{}'.format(sm['0'], sm['0']),
                      '10': '{}{}'.format(sm['0'], sm['0']),
                      '01': '{}{}'.format(sm['1'], sm['1']),
                      '11': '{}{}'.format(sm['1'], sm['1'])}
        return two_bit_sm

    def acquire_data_point(self, **kw):
        # Returns only the data for the relevant channel
        return super().acquire_data_point()[0:3]


class UHFQC_single_qubit_statistics_logging_det(UHFQC_statistics_logging_det):

    def __init__(self, UHFQC, AWG, nr_shots: int,
                 integration_length: float,
                 channel: int,
                 statemap: dict,
                 channel_name: str=None,
                 normalize_counts: bool=True):
        """
        Detector for the statistics logger mode in the UHFQC.

            UHFQC (instrument) : data acquisition device
            AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller.
            integration_length (float): integration length in seconds
            nr_shots (int)     : nr of shots (max is 4095)
            channel  (int)    : index (channel) of UHFQC weight function
            statemap (dict) : dictionary specifying the expected output state
                for each 1 bit input state.
                e.g.:
                    statemap ={'0': '0', '1':'1'}
            channel_name (str) : optional name of the channel


        """
        super(UHFQC_statistics_logging_det, self).__init__()
        self.UHFQC = UHFQC
        self.AWG = AWG
        self.nr_shots = nr_shots
        self.integration_length = integration_length
        self.normalize_counts = normalize_counts

        self.channels = [channel, int((channel+1) % 5)]

        if channel_name is None:
            channel_name = ['ch{}'.format(channel)]

        self.value_names = ['{} counts'.format(channel),
                            '{} flips'.format(channel),
                            '{} state errors'.format(channel)]
        if not self.normalize_counts:
            self.value_units = '#'*len(self.value_names)
        else:
            self.value_units = ['frac']*len(self.value_names)
        self.statemap = statemap

        self.max_shots = 4095  # hardware limit of UHFQC

        self.statemap = self.statemap_one2two_bit(statemap)

    @staticmethod
    def statemap_one2two_bit(one_bit_sm: dict):
        """
        Converts a one bit statemap to an appropriate dummy 2-bit statemap
        """
        sm = one_bit_sm
        two_bit_sm = {'00': '{}{}'.format(sm['0'], sm['0']),
                      '10': '{}{}'.format(sm['0'], sm['0']),
                      '01': '{}{}'.format(sm['1'], sm['1']),
                      '11': '{}{}'.format(sm['1'], sm['1'])}
        return two_bit_sm

    def acquire_data_point(self, **kw):
        # Returns only the data for the relevant channel
        return super().acquire_data_point()[0:3]

# --------------------------------------------
# Fake detectors
# --------------------------------------------


class Chevron_sim(Hard_Detector):

    """
    Returns a simulated chevron as if it was measured.
    """

    def __init__(self, simulation_dict, **kw):
        super(Chevron_sim, self).__init__(**kw)
        self.simulation_dict = simulation_dict
        self.name = 'Simulated Chevron'
        self.value_names = []
        self.value_units = []

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points
        self.t_start = self.sweep_points[0]
        self.dt = self.sweep_points[1] - self.sweep_points[0]

    def get_values(self):
        return chev_lib.chevron_slice(self.simulation_dict['detuning'],
                                      self.simulation_dict['dist_step'],
                                      self.simulation_dict['g'],
                                      self.t_start,
                                      self.dt,
                                      self.simulation_dict['dist_step'])


class ATS_integrated_average_continuous_detector(Hard_Detector):
    # deprecated

    def __init__(self, ATS, ATS_acq, AWG, seg_per_point=1, normalize=False, rotate=False,
                 nr_averages=1024, integration_length=1e-6, **kw):
        '''
        Integration average detector.
        '''
        super().__init__(**kw)
        self.ATS_acq = ATS_acq
        self.ATS = ATS
        self.name = 'ATS_integrated_average_detector'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        self.AWG = AWG
        self.seg_per_point = seg_per_point
        self.rotate = rotate
        self.normalize = normalize
        self.cal_points = kw.get('cal_points', None)
        self.nr_averages = nr_averages
        self.integration_length = integration_length

    def get_values(self):
        self.AWG.stop()
        self.AWG.start()
        data = self.ATS_acq.acquisition()
        return data

    def rotate_and_normalize(self, data):
        """
        Rotates and normalizes
        """
        if self.cal_points is None:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=data,
                    cal_zero_points=list(range(-4, -2)),
                    cal_one_points=list(range(-2, 0)))
        else:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.measured_values[0:2],
                    cal_zero_points=self.cal_points[0],
                    cal_one_points=self.cal_points[1])
        return self.corr_data, self.corr_data

    def prepare(self, sweep_points):
        self.ATS.config(clock_source='INTERNAL_CLOCK',
                        sample_rate=100000000,
                        clock_edge='CLOCK_EDGE_RISING',
                        decimation=0,
                        coupling=['AC', 'AC'],
                        channel_range=[2., 2.],
                        impedance=[50, 50],
                        bwlimit=['DISABLED', 'DISABLED'],
                        trigger_operation='TRIG_ENGINE_OP_J',
                        trigger_engine1='TRIG_ENGINE_J',
                        trigger_source1='EXTERNAL',
                        trigger_slope1='TRIG_SLOPE_POSITIVE',
                        trigger_level1=128,
                        trigger_engine2='TRIG_ENGINE_K',
                        trigger_source2='DISABLE',
                        trigger_slope2='TRIG_SLOPE_POSITIVE',
                        trigger_level2=128,
                        external_trigger_coupling='AC',
                        external_trigger_range='ETR_5V',
                        trigger_delay=0,
                        timeout_ticks=0
                        )
        self.ATS.update_acquisitionkwargs(samples_per_record=1024,
                                          records_per_buffer=70,
                                          buffers_per_acquisition=self.nr_averages,
                                          channel_selection='AB',
                                          transfer_offset=0,
                                          external_startcapture='ENABLED',
                                          enable_record_headers='DISABLED',
                                          alloc_buffers='DISABLED',
                                          fifo_only_streaming='DISABLED',
                                          interleave_samples='DISABLED',
                                          get_processed_data='DISABLED',
                                          allocated_buffers=self.nr_averages,
                                          buffer_timeout=1000
                                          )

    def finish(self):
        pass


# DDM detector functions
class DDM_input_average_detector(Hard_Detector):

    '''
    Detector used for acquiring averaged input traces withe the DDM

    '''

    '''
    Detector used for acquiring single points of the DDM while externally
    triggered by the AWG.
    Soft version of the regular integrated avg detector.
    # not yet pair specific
    '''

    def __init__(self, DDM, AWG, channels=[1, 2], nr_averages=1024, nr_samples=1024, **kw):
        super(DDM_input_average_detector, self).__init__()

        self.DDM = DDM
        self.name = 'DDM_input_averaging_data'
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = 'ch{}'.format(channel)
            self.value_units[i] = 'V'
        self.AWG = AWG
        self.nr_samples = nr_samples
        self.nr_averages = nr_averages

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()
        self.DDM.ch_pair1_inavg_scansize.set(self.nr_samples)
        self.DDM.ch_pair1_inavg_Navg(self.nr_averages)
        self.nr_sweep_points = self.nr_samples

    def get_values(self):
        # arming DDM trigger
        self.DDM.ch_pair1_inavg_enable.set(1)
        self.DDM.ch_pair1_run.set(1)
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()
        # polling the data, function checks that measurement is finished
        data = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            data[i] = eval("self.DDM.ch{}_inavg_data()".format(channel))/127
        return data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class DDM_integrated_average_detector(Hard_Detector):

    '''
    Detector used for integrated average results with the DDM

    '''

    def __init__(self, DDM, AWG, integration_length=1e-6, nr_averages=1024, rotate=False,
                 channels=[1, 2, 3, 4, 5], crosstalk_suppression=False,
                 **kw):
        super(DDM_integrated_average_detector, self).__init__()
        self.DDM = DDM
        self.name = 'DDM_integrated_average'
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        self.cal_points = kw.get('cal_points', None)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = 'w{}'.format(channel)
            self.value_units[i] = 'V'
        self.rotate = rotate
        self.AWG = AWG
        self.nr_averages = nr_averages
        self.integration_length = integration_length
        self.rotate = rotate
        self.crosstalk_suppression = crosstalk_suppression
        self.scaling_factor = 1/(500e6*integration_length)/127

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()
        if sweep_points is None:
            self.nr_sweep_points = 1
        else:
            self.nr_sweep_points = len(sweep_points)
        # this sets the result to integration and rotation outcome
        for i, channel in enumerate(self.channels):
            eval("self.DDM.ch_pair1_weight{}_wint_intlength({})".format(
                channel, self.integration_length*500e6))
        self.DDM.ch_pair1_tvmode_naverages(self.nr_averages)
        self.DDM.ch_pair1_tvmode_nsegments(self.nr_sweep_points)

    def get_values(self):
        # arming DDM trigger
        self.DDM.ch_pair1_tvmode_enable.set(1)
        self.DDM.ch_pair1_run.set(1)
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()
        # polling the data, function checks that measurement is finished
        data = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            data[i] = eval("self.DDM.ch_pair1_weight{}_tvmode_data()".format(
                channel))*self.scaling_factor
        if self.rotate:
            return self.rotate_and_normalize(data)
        else:
            return data

    def acquire_data_point(self):
        return self.get_values()

    def rotate_and_normalize(self, data):
        """
        Rotates and normalizes
        """
        if self.cal_points is None:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=data,
                    cal_zero_points=list(range(-4, -2)),
                    cal_one_points=list(range(-2, 0)))
        else:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.measured_values[0:2],
                    cal_zero_points=self.cal_points[0],
                    cal_one_points=self.cal_points[1])
        return self.corr_data, self.corr_data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class DDM_integration_logging_det(Hard_Detector):

    '''
    Detector used for integrated average results with the UHFQC

    '''

    def __init__(self, DDM, AWG, integration_length=1e-6,
                 channels=[1, 2], nr_shots=4096, **kw):
        super(DDM_integration_logging_det, self).__init__()
        self.DDM = DDM
        self.name = 'DDM_integration_logging_det'
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = 'w{}'.format(channel)
            self.value_units[i] = 'V'
        if len(self.channels) == 2:
            self.value_names = ['I', 'Q']
            self.value_units = ['V', 'V']
        self.AWG = AWG
        self.integration_length = integration_length
        self.nr_shots = nr_shots
        self.scaling_factor = 1/(500e6*integration_length)/127

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()
        if sweep_points is None:
            self.nr_sweep_points = 1
        else:
            self.nr_sweep_points = len(sweep_points)
        # this sets the result to integration and rotation outcome
        for i, channel in enumerate(self.channels):
            eval("self.DDM.ch_pair1_weight{}_wint_intlength({})".format(
                channel, self.integration_length*500e6))
        self.DDM.ch_pair1_logging_nshots(self.nr_shots)

    def get_values(self):
        # arming DDM trigger
        self.DDM.ch_pair1_logging_enable.set(1)
        self.DDM.ch_pair1_run.set(1)
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()
        # polling the data, function checks that measurement is finished
        data = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            data[i] = eval("self.DDM.ch_pair1_weight{}_logging_int()".format(
                channel))*self.scaling_factor
        return data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
