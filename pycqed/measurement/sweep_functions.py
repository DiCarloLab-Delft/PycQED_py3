import logging
import time
import numpy as np
from pycqed.measurement import mc_parameter_wrapper
import qcodes


class Sweep_function(object):
    '''
    sweep_functions class for MeasurementControl(Instrument)
    '''

    def __init__(self, **kw):
        self.set_kw()

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s' % (key, kw[key]))

    def prepare(self, **kw):
        pass

    def finish(self, **kw):
        pass

    # note that set_paramter is only actively used in soft sweeps.
    # it is added here so that performing a "hard 2D" experiment
    # (see tests for MC) the missing set_parameter in the hard sweep does not
    # lead to unwanted errors
    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        pass


class Soft_Sweep(Sweep_function):
    def __init__(self, **kw):
        self.set_kw()
        self.sweep_control = 'soft'


##############################################################################


class Elapsed_Time_Sweep(Soft_Sweep):
    """
    A sweep function to do a measurement periodically.
    Set the sweep points to the times at which you want to probe the
    detector function.
    """

    def __init__(self,
                 sweep_control='soft',
                 as_fast_as_possible: bool = False,
                 **kw):
        super().__init__()
        self.sweep_control = sweep_control
        self.name = 'Elapsed_Time_Sweep'
        self.parameter_name = 'Time'
        self.unit = 's'
        self.as_fast_as_possible = as_fast_as_possible
        self.time_first_set = None

    def set_parameter(self, val):
        if self.time_first_set is None:
            self.time_first_set = time.time()
            return 0
        elapsed_time = time.time() - self.time_first_set
        if self.as_fast_as_possible:
            return elapsed_time

        if elapsed_time > val:
            logging.warning(
                'Elapsed time {:.2f}s larger than desired {:2f}s'.format(
                    elapsed_time, val))
            return elapsed_time

        while (time.time() - self.time_first_set) < val:
            pass  # wait
        elapsed_time = time.time() - self.time_first_set
        return elapsed_time


class Heterodyne_Frequency_Sweep(Soft_Sweep):
    def __init__(self,
                 RO_pulse_type,
                 LO_source,
                 IF,
                 RF_source=None,
                 sweep_control='soft',
                 sweep_points=None,
                 **kw):
        super(Heterodyne_Frequency_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'Heterodyne frequency'
        self.parameter_name = 'Frequency'
        self.unit = 'Hz'
        self.RO_pulse_type = RO_pulse_type
        self.sweep_points = sweep_points
        self.LO_source = LO_source
        self.IF = IF
        if 'gated' in self.RO_pulse_type.lower():
            self.RF_source = RF_source

    def set_parameter(self, val):
        # RF = LO + IF
        self.LO_source.frequency(val - self.IF)
        if 'gated' in self.RO_pulse_type.lower():
            self.RF_source.frequency(val)


class Heterodyne_Frequency_Sweep_simple(Soft_Sweep):
    # Same as above but less input arguments

    def __init__(self, MW_LO_source, IF, sweep_points=None, **kw):
        super().__init__()
        self.name = 'Heterodyne frequency'
        self.parameter_name = 'Frequency'
        self.unit = 'Hz'
        self.sweep_points = sweep_points
        self.MW_LO_source = MW_LO_source
        self.IF = IF

    def set_parameter(self, val):
        # RF = LO + IF
        self.MW_LO_source.frequency(val - self.IF)


class None_Sweep(Soft_Sweep):
    def __init__(self,
                 sweep_control='soft',
                 sweep_points=None,
                 name: str = 'None_Sweep',
                 parameter_name: str = 'pts',
                 unit: str = 'arb. unit',
                 **kw):
        super(None_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = name
        self.parameter_name = parameter_name
        self.unit = unit
        self.sweep_points = sweep_points

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be swept. Differs per sweep function
        '''
        pass


class None_Sweep_idx(None_Sweep):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.num_calls = 0

    def set_parameter(self, val):
        self.num_calls += 1


class QX_Sweep(Soft_Sweep):
    """
    QX Input Test
    """

    def __init__(self, qxc, sweep_control='soft', sweep_points=None, **kw):
        super(QX_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'QX_Sweep'
        self.parameter_name = 'Error Rate'
        self.unit = 'P'
        self.sweep_points = sweep_points
        self.__qxc = qxc
        self.__qxc.create_qubits(2)
        self.__cnt = 0

    def set_parameter(self, val):
        circuit_name = ("circuit%i" % self.__cnt)
        self.__qxc.create_circuit(circuit_name, [
            "prepz q0", "h q0", "x q0", "z q0", "y q0", "y q0", "z q0", "x q0",
            "h q0", "measure q0"
        ])
        self.__cnt = self.__cnt + 1
        # pass


class Delayed_None_Sweep(Soft_Sweep):
    def __init__(self, sweep_control='soft', delay=0, mode='cycle_delay',
                 **kw):
        super().__init__()
        self.sweep_control = sweep_control
        self.name = 'None_Sweep'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.delay = delay
        self.time_last_set = 0
        self.mode = mode
        if delay > 60:
            logging.warning(
                'setting a delay of {:.g}s are you sure?'.format(delay))

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        if self.mode != 'cycle_delay':
            self.time_last_set = time.time()
        while (time.time() - self.time_last_set) < self.delay:
            pass  # wait
        if self.mode == 'cycle_delay':
            self.time_last_set = time.time()


###################################


class AWG_amp(Soft_Sweep):
    def __init__(self, channel, AWG):
        super().__init__()
        self.name = 'AWG Channel Amplitude'
        self.channel = channel
        self.parameter_name = 'AWG_ch{}_amp'.format(channel)
        self.AWG = AWG
        self.unit = 'V'

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        self.AWG.stop()
        if type(self.channel) == int:
            exec('self.AWG.ch{}_amp({})'.format(self.channel, val))
        else:
            exec('self.AWG.{}_amp({})'.format(self.channel, val))
        self.AWG.start()


class AWG_multi_channel_amplitude(Soft_Sweep):
    '''
    Sweep function to sweep multiple AWG channels simultaneously
    '''

    def __init__(self, AWG, channels, delay=0, **kw):
        super().__init__()
        self.name = 'AWG channel amplitude chs %s' % channels
        self.parameter_name = 'AWG chs %s' % channels
        self.unit = 'V'
        self.AWG = AWG
        self.channels = channels
        self.delay = delay

    def set_parameter(self, val):
        for ch in self.channels:
            self.AWG.set('ch{}_amp'.format(ch), val)
        time.sleep(self.delay)


###############################################################################
####################          Hardware Sweeps      ############################
###############################################################################


class Hard_Sweep(Sweep_function):
    def __init__(self, **kw):
        super().__init__()
        self.sweep_control = 'hard'
        self.parameter_name = 'None'
        self.name = 'Hard_Sweep'
        self.unit = 'a.u.'

    def start_acquistion(self):
        pass

    def set_parameter(self, value):
        logging.warning('set_parameter called for hardware sweep.')


class Segment_Sweep(Hard_Sweep):
    def __init__(self, **kw):
        super().__init__()
        self.parameter_name = 'Segment index'
        self.name = 'Segment_Sweep'
        self.unit = ''


class OpenQL_Sweep(Hard_Sweep):
    def __init__(self,
                 openql_program,
                 CCL,
                 parameter_name: str = 'Points',
                 unit: str = 'a.u.',
                 upload: bool = True):
        super().__init__()
        self.name = 'OpenQL_Sweep'
        self.openql_program = openql_program
        self.CCL = CCL
        self.upload = upload
        self.parameter_name = parameter_name
        self.unit = unit

    def prepare(self, **kw):
        if self.upload:
            self.CCL.eqasm_program(self.openql_program.filename)


class OpenQL_File_Sweep(Hard_Sweep):
    def __init__(self,
                 filename: str,
                 CCL,
                 parameter_name: str = 'Points',
                 unit: str = 'a.u.',
                 upload: bool = True):
        super().__init__()
        self.name = 'OpenQL_Sweep'
        self.filename = filename
        self.CCL = CCL
        self.upload = upload
        self.parameter_name = parameter_name
        self.unit = unit

    def prepare(self, **kw):
        if self.upload:
            self.CCL.eqasm_program(self.filename)


class ZNB_VNA_sweep(Hard_Sweep):
    def __init__(self,
                 VNA,
                 start_freq=None,
                 stop_freq=None,
                 center_freq=None,
                 span=None,
                 segment_list=None,
                 npts=100,
                 force_reset=False):
        '''
        Frequencies are in Hz.
        Defines the frequency sweep using one of the following methods:
        1) start a and stop frequency
        2) center frequency and span
        3) segment sweep (this requires a list of elements. Each element fully
           defines a sweep)
           segment_list = [[start_frequency, stop_frequency, nbr_points,
                            power, segment_time, mesurement_delay, bandwidth],
                           [elements for segment #2],
                           ...,
                           [elements for segment #n]]

        If force_reset = True the VNA is reset to default settings
        '''
        super(ZNB_VNA_sweep, self).__init__()
        self.VNA = VNA
        self.name = 'ZNB_VNA_sweep'
        self.parameter_name = 'frequency'
        self.unit = 'Hz'
        self.filename = 'VNA_sweep'

        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.center_freq = center_freq
        self.segment_list = segment_list
        self.span = span
        self.npts = npts

        if force_reset == True:
            VNA.reset()

    def prepare(self):
        '''
        Prepare the VNA for measurements by defining basic settings.
        Set the frequency sweep and get the frequency points back from the insturment
        '''
        self.VNA.continuous_mode_all('off')  # measure only if required
        # optimize the sweep time for the fastest measurement
        self.VNA.min_sweep_time('on')
        # start a measurement once the trigger signal arrives
        self.VNA.trigger_source('immediate')
        # trigger signal is generated with the command:
        # VNA.start_sweep_all()

        if self.segment_list == None:
            self.VNA.sweep_type('linear')  # set a linear sweep
            if self.start_freq != None and self.stop_freq != None:
                self.VNA.start_frequency(self.start_freq)
                self.VNA.stop_frequency(self.stop_freq)
            elif self.center_freq != None and self.span != None:
                self.VNA.center_frequency(self.center_freq)
                self.VNA.span_frequency(self.span)

            self.VNA.npts(self.npts)
        elif self.segment_list != None:
            # delete all previous stored segments
            self.VNA.delete_all_segments()

            # Load segments in reverse order to have them executed properly
            for idx_segment in range(len(self.segment_list), 0, -1):
                current_segment = self.segment_list[idx_segment - 1]
                str_to_write = 'SENSE:SEGMENT:INSERT %s, %s, %s, %s, %s, %s, %s' % (
                    current_segment[0], current_segment[1], current_segment[2],
                    current_segment[3], current_segment[4], current_segment[5],
                    current_segment[6])
                self.VNA.write(str_to_write)

            self.VNA.sweep_type('segment')  # set a segment sweep

        # get the list of frequency used in the span from the VNA
        self.sweep_points = self.VNA.get_stimulus()


class QWG_lutman_par(Soft_Sweep):
    def __init__(self, LutMan, LutMan_parameter, **kw):
        self.set_kw()
        self.name = LutMan_parameter.name
        self.parameter_name = LutMan_parameter.label
        self.unit = LutMan_parameter.unit
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter

    def set_parameter(self, val):
        self.LutMan.AWG.get_instr().stop()
        self.LutMan_parameter.set(val)
        self.LutMan.load_waveforms_onto_AWG_lookuptable(
            regenerate_waveforms=True)
        self.LutMan.AWG.get_instr().start()
        self.LutMan.AWG.get_instr().getOperationComplete()


class QWG_flux_amp(Soft_Sweep):
    """
    Sweep function
    """

    def __init__(self, QWG, channel: int, frac_amp: float, **kw):
        self.set_kw()
        self.QWG = QWG
        self.qwg_channel_amp_par = QWG.parameters['ch{}_amp'.format(channel)]
        self.name = 'Flux_amp'
        self.parameter_name = 'Flux_amp'
        self.unit = 'V'
        self.sweep_control = 'soft'

        # Amp = frac * Vpp/2
        self.scale_factor = 2 / frac_amp

    def set_parameter(self, val):
        Vpp = val * self.scale_factor
        self.qwg_channel_amp_par(Vpp)
        # Ensure the amplitude was set correctly
        self.QWG.getOperationComplete()


class QWG_lutman_par_chunks(Soft_Sweep):
    '''
    Sweep function that divides sweep points into chunks. Every chunk is
    measured with a QASM sweep, and the operation dictionary can change between
    different chunks. Pulses are re-uploaded between chunks.
    '''

    def __init__(self,
                 LutMan,
                 LutMan_parameter,
                 sweep_points,
                 chunk_size,
                 codewords=np.arange(128),
                 flux_pulse_type='square',
                 **kw):
        super().__init__(**kw)
        self.sweep_points = sweep_points
        self.chunk_size = chunk_size
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter
        self.name = LutMan_parameter.name
        self.parameter_name = LutMan_parameter.label
        self.unit = LutMan_parameter.unit
        self.flux_pulse_type = flux_pulse_type
        self.codewords = codewords

    def set_parameter(self, val):
        # Find index of val in sweep_points
        ind = np.where(np.isclose(self.sweep_points, val, atol=1e-10))[0]
        if len(ind) == 0:
            # val was not found in the sweep points
            raise ValueError('Value {} is not in the sweep points'.format(val))
        ind = ind[0]  # set index to the first occurence of val in sweep points

        QWG = self.LutMan.QWG.get_instr()
        QWG.stop()

        for i, paramVal in enumerate(
                self.sweep_points[ind:ind + self.chunk_size]):
            pulseName = 'pulse_{}'.format(i)

            # Generate new pulse
            self.LutMan_parameter.set(paramVal)
            self.LutMan.regenerate_pulse(self.flux_pulse_type)

            if self.LutMan.wave_dict_unit() == 'V':
                scaleFac = QWG.get('ch{}_amp'.format(self.LutMan.F_ch())) / 2
            else:
                scaleFac = 1

            # Load onto QWG
            QWG.createWaveformReal(
                pulseName,
                self.LutMan._wave_dict[self.flux_pulse_type] / scaleFac)

            # Assign codeword
            QWG.set(
                'codeword_{}_ch{}_waveform'.format(self.codewords[i],
                                                   self.LutMan.F_ch()),
                pulseName)

        QWG.start()
        QWG.getOperationComplete()


class QWG_lutman_custom_wave_chunks(Soft_Sweep):
    '''
    Sweep function that divides sweep points into chunks. Every chunk is
    measured with a QASM sweep, and the operation dictionary can change between
    different chunks. Pulses are re-uploaded between chunks.
    The custom waveform is defined by a function (wave_func), which takes
    exactly one parameter, the current sweep point, and returns an array of the
    pulse samples (without compensation or delay; these are added by the
    QWG_lutman in the usual way).
     '''

    def __init__(self,
                 LutMan,
                 wave_func,
                 sweep_points,
                 chunk_size,
                 codewords=None,
                 param_name='flux pulse parameter',
                 param_unit='a.u.',
                 **kw):
        super().__init__(**kw)
        self.wave_func = wave_func
        self.chunk_size = chunk_size
        self.LutMan = LutMan
        if codewords is None:
            self.codewords = np.arange(chunk_size)
        else:
            self.codewords = codewords
        self.name = param_name
        self.parameter_name = param_name
        self.unit = param_unit
        # Setting self.custom_swp_pts because self.sweep_points is overwritten
        # by the MC.
        self.custom_swp_pts = sweep_points

    def set_parameter(self, val):
        # Find index of val in sweep_points
        ind = np.where(self.custom_swp_pts == val)[0]
        if len(ind) == 0:
            # val was not found in the sweep points
            raise ValueError('Value {} is not in the sweep points'.format(val))
        ind = ind[0]  # set index to the first occurence of val in sweep points

        for i, paramVal in enumerate(
                self.custom_swp_pts[ind:ind + self.chunk_size]):
            pulseName = 'pulse_{}'.format(i)

            self.LutMan.load_custom_pulse_onto_AWG_lookuptable(
                self.wave_func(paramVal),
                append_compensation=True,
                pulse_name=pulseName,
                codeword=self.codewords[i])


class lutman_par_dB_attenuation_QWG(Soft_Sweep):
    def __init__(self, LutMan, LutMan_parameter, **kw):
        self.set_kw()
        self.name = LutMan_parameter.name
        self.parameter_name = LutMan_parameter.label
        self.unit = 'dB'
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter

    def set_parameter(self, val):
        self.LutMan_parameter.set(10**(val / 20))
        self.LutMan.load_pulses_onto_AWG_lookuptable(regenerate_pulses=True)
        self.LutMan.QWG.get_instr().start()
        self.LutMan.QWG.get_instr().getOperationComplete()


class lutman_par_dB_attenuation_UHFQC(Soft_Sweep):
    def __init__(self, LutMan, LutMan_parameter, run=False, single=True, **kw):
        self.set_kw()
        self.name = LutMan_parameter.name
        self.parameter_name = LutMan_parameter.label
        self.unit = 'dB'
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter
        self.run = run
        self.single = single

    def set_parameter(self, val):
        self.LutMan_parameter.set(10**(val / 20))
        if self.run:
            self.LutMan.UHFQC.awgs_0_enable(False)
        self.LutMan.load_pulse_onto_AWG_lookuptable(
            'M_square', regenerate_pulses=True)
        if self.run:
            self.LutMan.UHFQC.acquisition_arm(single=self.single)


class lutman_par_UHFQC_dig_trig(Soft_Sweep):
    def __init__(self, LutMan, LutMan_parameter, single=True, run=False, **kw):
        self.set_kw()
        self.name = LutMan_parameter.name
        self.parameter_name = LutMan_parameter.label
        self.unit = LutMan_parameter.unit
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter
        self.run = run
        self.single = single

    def set_parameter(self, val):
        self.LutMan_parameter.set(val)
        if self.run:
            self.LutMan.AWG.get_instr().awgs_0_enable(False)
        self.LutMan.load_DIO_triggered_sequence_onto_UHFQC()
        if self.run:
            self.LutMan.AWG.get_instr().acquisition_arm(single=self.single)


class lutman_par_depletion_pulse_global_scaling(Soft_Sweep):
    def __init__(self,
                 LutMan,
                 resonator_numbers,
                 optimization_M_amps,
                 optimization_M_amp_down0s,
                 optimization_M_amp_down1s,
                 upload=True,
                 **kw):
        # sweeps the readout-and depletion pules of the listed resonators.
        # sets the remaining readout and depletion pulses to 0 amplitude.

        self.set_kw()
        self.name = 'depletion_pulse_sweeper'
        self.parameter_name = 'relative_depletion_pulse_scaling_amp'
        self.unit = 'a.u.'
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.optimization_M_amps = optimization_M_amps
        self.optimization_M_amp_down0s = optimization_M_amp_down0s
        self.optimization_M_amp_down1s = optimization_M_amp_down1s
        self.resonator_numbers = resonator_numbers
        self.upload = upload

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be swept. Differs per sweep function
        Sweeping the amplitudes of the readout-and-depletion-pulses in the list
        relative to the initially optimized amplitude.
        Sets the remaining depletion pulses to zero.
        '''
        for resonator_number in self.LutMan._resonator_codeword_bit_mapping:
            if resonator_number in self.resonator_numbers:
                i = self.resonator_numbers.index(resonator_number)
                self.LutMan.set('M_amp_R{}'.format(resonator_number),
                                val * self.optimization_M_amps[i])
                self.LutMan.set('M_down_amp0_R{}'.format(resonator_number),
                                val * self.optimization_M_amp_down0s[i])
                self.LutMan.set('M_down_amp1_R{}'.format(resonator_number),
                                val * self.optimization_M_amp_down1s[i])
            else:
                self.LutMan.set('M_amp_R{}'.format(resonator_number), 0)
                self.LutMan.set('M_down_amp0_R{}'.format(resonator_number), 0)
                self.LutMan.set('M_down_amp1_R{}'.format(resonator_number), 0)
        if self.upload:
            self.LutMan.load_DIO_triggered_sequence_onto_UHFQC(
                regenerate_waveforms=True)


class lutman_par_dB_attenuation_UHFQC_dig_trig(Soft_Sweep):
    def __init__(self, LutMan, LutMan_parameter, run=False, **kw):
        self.set_kw()
        self.name = LutMan_parameter.name
        self.parameter_name = LutMan_parameter.label
        self.unit = 'dB'
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter
        self.run = run

    def set_parameter(self, val):
        self.LutMan_parameter.set(10**(val / 20))
        if self.run:
            self.LutMan.AWG.get_instr().awgs_0_enable(False)
        self.LutMan.load_DIO_triggered_sequence_onto_UHFQC()
        if self.run:
            self.LutMan.AWG.get_instr().acquisition_arm(single=self.single)


class dB_attenuation_UHFQC_dig_trig(Soft_Sweep):
    def __init__(self, LutMan, LutMan_parameter, run=False, **kw):
        self.set_kw()
        self.name = LutMan_parameter.name
        self.parameter_name = LutMan_parameter.label
        self.unit = 'dB'
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter
        self.run = run

    def set_parameter(self, val):
        self.LutMan_parameter.set(10**(val / 20))
        if self.run:
            self.LutMan.AWG.get_instr().awgs_0_enable(False)
        self.LutMan.load_DIO_triggered_sequence_onto_UHFQC()
        if self.run:
            self.LutMan.AWG.get_instr().acquisition_arm(single=self.single)


class UHFQC_pulse_dB_attenuation(Soft_Sweep):
    def __init__(self, UHFQC, IF, dig_trigger=True, **kw):
        self.set_kw()
        self.name = 'UHFQC pulse attenuation'
        self.parameter_name = 'pulse attenuation'
        self.unit = 'dB'
        self.sweep_control = 'soft'
        self.UHFQC = UHFQC
        self.dig_trigger = dig_trigger
        self.IF = IF

    def set_parameter(self, val):
        self.UHFQC.awg_sequence_acquisition_and_pulse_SSB(
            f_RO_mod=self.IF,
            RO_amp=10**(val / 20),
            RO_pulse_length=2e-6,
            acquisition_delay=200e-9,
            dig_trigger=self.dig_trigger)
        time.sleep(1)
        #print('refreshed UHFQC')


class multi_sweep_function(Soft_Sweep):
    '''
    cascades several sweep functions into a single joint sweep functions.
    '''

    def __init__(self,
                 sweep_functions: list,
                 parameter_name=None,
                 name=None,
                 **kw):
        self.set_kw()
        self.sweep_functions = sweep_functions
        self.sweep_control = 'soft'
        self.name = name or 'multi_sweep'
        self.unit = sweep_functions[0].unit
        self.parameter_name = parameter_name or 'multiple_parameters'
        for i, sweep_function in enumerate(sweep_functions):
            if self.unit.lower() != sweep_function.unit.lower():
                raise ValueError('units of the sweepfunctions are not equal')

    def set_parameter(self, val):
        for sweep_function in self.sweep_functions:
            sweep_function.set_parameter(val)


class two_par_joint_sweep(Soft_Sweep):
    """
    Allows jointly sweeping two parameters while preserving their
    respective ratios.
    Allows par_A and par_B to be arrays of parameters
    """

    def __init__(self, par_A, par_B, preserve_ratio: bool = True, **kw):
        self.set_kw()
        self.unit = par_A.unit
        self.sweep_control = 'soft'
        self.par_A = par_A
        self.par_B = par_B
        self.name = par_A.name
        self.parameter_name = par_A.name
        if preserve_ratio:
            try:
                self.par_ratio = self.par_B.get() / self.par_A.get()
            except:
                self.par_ratio = (
                    self.par_B.get_latest() / self.par_A.get_latest())
        else:
            self.par_ratio = 1

    def set_parameter(self, val):
        self.par_A.set(val)
        self.par_B.set(val * self.par_ratio)


class Offset_Sweep(Soft_Sweep):
    """A sweep soft sweep function that calls an other sweep function with
    an offset."""

    def __init__(self,
                 sweep_function,
                 offset,
                 name=None,
                 parameter_name=None,
                 unit=None):
        super().__init__()
        if isinstance(sweep_function, qcodes.Parameter):
            sweep_function = mc_parameter_wrapper.wrap_par_to_swf(
                sweep_function)
        if sweep_function.sweep_control != 'soft':
            raise ValueError('Offset_Sweep: Only software sweeps supported')
        self.sweep_function = sweep_function
        self.offset = offset
        self.sweep_control = sweep_function.sweep_control
        if parameter_name is None:
            self.parameter_name = sweep_function.parameter_name + \
                ' {:+} {}'.format(-offset, sweep_function.unit)
        else:
            self.parameter_name = parameter_name
        if name is None:
            self.name = sweep_function.name
        else:
            self.name = name
        if unit is None:
            self.unit = sweep_function.unit
        else:
            self.unit = unit

    def prepare(self, *args, **kwargs):
        self.sweep_function.prepare(*args, **kwargs)

    def finish(self, *args, **kwargs):
        self.sweep_function.finish(*args, **kwargs)

    def set_parameter(self, val):
        self.sweep_function.set_parameter(val + self.offset)
