# FIXME: split-off QWG/UHFQA/etc sweeps into separate files
# FIXME: deprecate unused sweep functions

import logging
import time
import numpy as np

from pycqed.analysis_v2.tools import contours2d as c2d


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

    def __init__(self, sweep_control='soft',
                 as_fast_as_possible: bool=False, **kw):
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
            logging.warning('Elapsed time {:.2f}s larger than desired {:2f}s'
                            .format(elapsed_time, val))
            return elapsed_time

        while (time.time() - self.time_first_set) < val:
            pass  # wait
        elapsed_time = time.time() - self.time_first_set
        return elapsed_time


class Heterodyne_Frequency_Sweep(Soft_Sweep):
    """
    Performs a joint sweep of two microwave sources for the purpose of
    varying a heterodyne frequency.
    """

    def __init__(self, 
            RO_pulse_type:str,
            LO_source, 
            IF:float,
            RF_source=None,
            sweep_control:str='soft',
            sweep_points=None,
            **kw):
        """
        RO_pulse_type (str) : determines wether to only set the LO source
            (in case of a modulated RF pulse) or set both the LO and RF source
            to the required frequency. Can be:
                "gated"             Will set both the LO and RF source
                "pulse_modulated"   Will only set the LO source
                "CW"                Will set both the LO and RF source
        LO_source (instr) : instance of the LO instrument
        IF (float)        : intermodulation frequency in Hz
        RF_source (instr) : instance of the RF instrument, can be None
            if the pulse type is "pulse_modulated"
        """

        super(Heterodyne_Frequency_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'Heterodyne frequency'
        self.parameter_name = 'Frequency'
        self.unit = 'Hz'
        self.RO_pulse_type = RO_pulse_type
        self.sweep_points = sweep_points
        self.LO_source = LO_source
        self.IF = IF
        if (('gated' in self.RO_pulse_type.lower()) or
            ('cw' in self.RO_pulse_type.lower())):
            self.RF_source = RF_source

    def set_parameter(self, val):
        # RF = LO + IF
        self.LO_source.frequency(val-self.IF)
        if (('gated' in self.RO_pulse_type.lower()) or
            ('cw' in self.RO_pulse_type.lower())):
            self.RF_source.frequency(val)


class Heterodyne_Frequency_Sweep_simple(Soft_Sweep):
    # Same as above but less input arguments

    def __init__(self, MW_LO_source, IF,
                 sweep_points=None,
                 **kw):
        super().__init__()
        self.name = 'Heterodyne frequency'
        self.parameter_name = 'Frequency'
        self.unit = 'Hz'
        self.sweep_points = sweep_points
        self.MW_LO_source = MW_LO_source
        self.IF = IF

    def set_parameter(self, val):
        # RF = LO + IF
        self.MW_LO_source.frequency(val-self.IF)
        # retrieve paramter to ensure setting  is complete.
        self.MW_LO_source.frequency()


class None_Sweep(Soft_Sweep):

    def __init__(self, sweep_control='soft', sweep_points=None,
                 name: str='None_Sweep', parameter_name: str='pts',
                 unit: str='arb. unit',
                 **kw):
        super(None_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = name
        self.parameter_name = parameter_name
        self.unit = unit
        self.sweep_points = sweep_points

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        pass


class None_Sweep_With_Parameter_Returned(Soft_Sweep):

    def __init__(self, sweep_control='soft', sweep_points=None,
                 name: str='None_Sweep', parameter_name: str='pts',
                 unit: str='arb. unit',
                 **kw):
        super().__init__()
        self.sweep_control = sweep_control
        self.name = name
        self.parameter_name = parameter_name
        self.unit = unit
        self.sweep_points = sweep_points

    def set_parameter(self, val):
        '''
        returns something a bit different to simulate the set_parameter reading
        out the set parameter from the instrument
        '''
        return val+0.1


class None_Sweep_idx(None_Sweep):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.num_calls = 0

    def set_parameter(self, val):
        self.num_calls += 1


class Delayed_None_Sweep(Soft_Sweep):

    def __init__(self, sweep_control='soft', delay=0, **kw):
        super().__init__()
        self.sweep_control = sweep_control
        self.name = 'None_Sweep'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.delay = delay
        self.time_last_set = 0
        if delay > 60:
            logging.warning(
                'setting a delay of {:.g}s are you sure?'.format(delay))

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        while (time.time() - self.time_last_set) < self.delay:
            pass  # wait
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

    def set_parameter(self, val):
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

class mw_lutman_amp_sweep(Soft_Sweep):
    """
    """

    def __init__(self,qubits,device):
        super().__init__()
        self.device = device
        self.name = 'mw_lutman_amp_sweep'
        self.qubits = qubits
        self.parameter_name = 'mw_amp'
        self.unit = 'a.u.'

    def set_parameter(self, val):
        for q in self.qubits:
          qub  = self.device.find_instrument(q)
          mw_lutman = qub.instr_LutMan_MW.get_instr()
          mw_lutman.channel_amp(val)


class motzoi_lutman_amp_sweep(Soft_Sweep):
    """
    """

    def __init__(self,qubits,device):
        super().__init__()
        self.device = device
        self.name = 'motzoi_lutman_amp_sweep'
        self.qubits = qubits
        self.parameter_name = 'motzoi_amp'
        self.unit = 'a.u.'

    def set_parameter(self, val):
        for q in self.qubits:
          qub = self.device.find_instrument(q)
          mw_lutman = qub.instr_LutMan_MW.get_instr()
          mw_lutman.mw_motzoi(val)
          mw_lutman.load_waveforms_onto_AWG_lookuptable(
            regenerate_waveforms=True)

###############################################################################
####################          Hardware Sweeps      ############################
###############################################################################


class Hard_Sweep(Sweep_function):

    def __init__(self, **kw):
        super(Hard_Sweep, self).__init__()
        self.sweep_control = 'hard'
        self.parameter_name = 'None'
        self.name = 'Hard_Sweep'
        self.unit = 'a.u.'

    def start_acquistion(self):
        pass


class OpenQL_Sweep(Hard_Sweep):

    def __init__(self, openql_program, CCL,
                 parameter_name: str ='Points', unit: str='a.u.',
                 upload: bool=True):
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

    def __init__(self, filename: str, CCL,
                 parameter_name: str ='Points', unit: str='a.u.',
                 upload: bool=True):
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

    def __init__(self, VNA,
                 start_freq=None, stop_freq=None,
                 center_freq=None, span=None,
                 segment_list=None,
                 npts=100, force_reset=False):
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
        self.VNA.rf_on()
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
                current_segment = self.segment_list[idx_segment-1]
                str_to_write = 'SENSE:SEGMENT:INSERT %s, %s, %s, %s, %s, %s, %s' % (current_segment[0], current_segment[
                                                                                    1], current_segment[2], current_segment[3], current_segment[4], current_segment[5], current_segment[6])
                self.VNA.write(str_to_write)

            self.VNA.sweep_type('segment')  # set a segment sweep

        # get the list of frequency used in the span from the VNA
        self.sweep_points = self.VNA.get_stimulus()

    def finish(self, **kw):
        self.VNA.rf_off()

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
        self.LutMan.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
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
        self.scale_factor = 2/frac_amp

    def set_parameter(self, val):
        Vpp = val * self.scale_factor
        self.qwg_channel_amp_par(Vpp)
        # Ensure the amplitude was set correctly
        self.QWG.getOperationComplete()


class lutman_par(Soft_Sweep):
    """
    Sweeps a LutMan parameter and uploads the waveforms to AWG (in real-time if
    supported)
    """

    def __init__(self, LutMan, LutMan_parameter):
        self.set_kw()
        self.name = LutMan_parameter.name
        self.parameter_name = LutMan_parameter.label
        self.unit = LutMan_parameter.unit
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter

    def set_parameter(self, val):
        self.LutMan_parameter.set(val)
        self.LutMan.load_waveforms_onto_AWG_lookuptable(
            regenerate_waveforms=True)


class anharmonicity_sweep(Soft_Sweep):
    """
    Sweeps a LutMan parameter and uploads the waveforms to AWG (in real-time if
    supported)
    """

    def __init__(self, qubit, amps):
        self.set_kw()
        self.name = qubit.anharmonicity.name
        self.parameter_name = qubit.anharmonicity.label
        self.unit = qubit.anharmonicity.unit
        self.sweep_control = 'soft'
        self.qubit = qubit
        self.amps = amps

    def set_parameter(self, val):
        self.qubit.anharmonicity.set(val)
        # _prep_mw_pulses will upload anharmonicity val to LutMan
        self.qubit._prep_mw_pulses()
        # and we regenerate the waveform with that new modulation
        mw_lutman = self.qubit.instr_LutMan_MW.get_instr()
        mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable(amps=self.amps)


class joint_HDAWG_lutman_parameters(Soft_Sweep):
    """
    Sweeps two parameteres toghether, assigning the same value
    name is defined by user
    label and units are grabbed from parameter_1
    """

    def __init__(self, name, parameter_1, parameter_2,
                 AWG, lutman):
        self.set_kw()
        self.name = name
        self.parameter_name = parameter_1.label
        self.unit = parameter_1.unit
        self.lm = lutman
        self.AWG = AWG
        self.sweep_control = 'soft'
        self.parameter_1 = parameter_1
        self.parameter_2 = parameter_2

    def set_parameter(self, val):
        self.parameter_1.set(val)
        self.parameter_2.set(-val)
        self.AWG.stop()
        self.lm.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
        self.AWG.start()


class RO_freq_sweep(Soft_Sweep):
    """
    Sweeps two parameteres toghether, assigning the same value
    name is defined by user
    label and units are grabbed from parameter_1
    """

    def __init__(self, name, qubit, ro_lutman, idx, parameter):
        self.set_kw()
        self.name = name
        self.parameter_name = parameter.label
        self.unit = parameter.unit
        self.sweep_control = 'soft'
        self.qubit = qubit
        self.ro_lm = ro_lutman
        self.idx = idx

    def set_parameter(self, val):
        LO_freq = self.ro_lm.LO_freq()
        IF_freq = val - LO_freq
        # Parameter 1 will be qubit.ro_freq()
        # self.qubit.ro_freq.set(val)
        # Parameter 2 will be qubit.ro_freq_mod()
        self.qubit.ro_freq_mod.set(IF_freq)

        self.ro_lm.set('M_modulation_R{}'.format(self.idx), IF_freq)
        self.ro_lm.load_waveforms_onto_AWG_lookuptable()


class QWG_lutman_par_chunks(Soft_Sweep):
    '''
    Sweep function that divides sweep points into chunks. Every chunk is
    measured with a QASM sweep, and the operation dictionary can change between
    different chunks. Pulses are re-uploaded between chunks.
    '''

    def __init__(self, LutMan, LutMan_parameter,
                 sweep_points, chunk_size, codewords=np.arange(128),
                 flux_pulse_type='square', **kw):
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

        for i, paramVal in enumerate(self.sweep_points[ind:ind +
                                                       self.chunk_size]):
            pulseName = 'pulse_{}'.format(i)

            # Generate new pulse
            self.LutMan_parameter.set(paramVal)
            self.LutMan.regenerate_pulse(self.flux_pulse_type)

            if self.LutMan.wave_dict_unit() == 'V':
                scaleFac = QWG.get('ch{}_amp'.format(self.LutMan.F_ch())) / 2
            else:
                scaleFac = 1

            # Load onto QWG: FIXME: should be performed by LutMan, we shouldn't mess with its internals
            QWG.createWaveformReal(
                pulseName,
                self.LutMan._wave_dict[self.flux_pulse_type]/scaleFac)

            # Assign codeword
            QWG.set('codeword_{}_ch{}_waveform'
                    .format(self.codewords[i], self.LutMan.F_ch()), pulseName)

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

    def __init__(self, LutMan, wave_func, sweep_points, chunk_size,
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

        for i, paramVal in enumerate(self.custom_swp_pts[ind:ind +
                                                         self.chunk_size]):
            pulseName = 'pulse_{}'.format(i)

            self.LutMan.load_custom_pulse_onto_AWG_lookuptable(
                self.wave_func(paramVal), append_compensation=True,
                pulse_name=pulseName, codeword=self.codewords[i])


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
        self.LutMan_parameter.set(10**(val/20))
        self.LutMan.load_pulses_onto_AWG_lookuptable(regenerate_pulses=True)
        self.LutMan.QWG.get_instr().start()
        self.LutMan.QWG.get_instr().getOperationComplete()


class lutman_par_dB_attenuation_UHFQC(Soft_Sweep):

    def __init__(self, LutMan, LutMan_parameter, run=False, single=True,**kw):
        self.set_kw()
        self.name = LutMan_parameter.name
        self.parameter_name = LutMan_parameter.label
        self.unit = 'dB'
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter
        self.run=run
        self.single = single

    def set_parameter(self, val):
        self.LutMan_parameter.set(10**(val/20))
        if self.run:
            self.LutMan.UHFQC.awgs_0_enable(False)
        self.LutMan.load_pulse_onto_AWG_lookuptable('M_square',regenerate_pulses=True)
        if self.run:
            self.LutMan.UHFQC.acquisition_arm(single=self.single)


# FIXME: deprecate?
class par_dB_attenuation_UHFQC_AWG_direct(Soft_Sweep):
    def __init__(self, UHFQC, **kw):
        self.set_kw()
        self.name = "UHFQC attenuation"
        self.parameter_name = "UHFQC attenuation"
        self.unit = 'dB'
        self.sweep_control = 'soft'
        self.UHFQC = UHFQC

    # def set_parameter(self, val):
    #     UHFQC.awgs_0_outputs_1_amplitude(10**(val/20))  # FIXME: broken code
    #     UHFQC.awgs_0_outputs_0_amplitude(10**(val/20))


class lutman_par_UHFQC_dig_trig(Soft_Sweep):
    def __init__(self, LutMan, LutMan_parameter, single=True, run=False,**kw):
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
    def __init__(self, LutMan, resonator_numbers, optimization_M_amps,
                 optimization_M_amp_down0s, optimization_M_amp_down1s,
                 upload=True, **kw):
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
                                val*self.optimization_M_amps[i])
                self.LutMan.set('M_down_amp0_R{}'.format(resonator_number),
                                val*self.optimization_M_amp_down0s[i])
                self.LutMan.set('M_down_amp1_R{}'.format(resonator_number),
                                val*self.optimization_M_amp_down1s[i])
            else:
                self.LutMan.set('M_amp_R{}'.format(resonator_number), 0)
                self.LutMan.set('M_down_amp0_R{}'.format(resonator_number), 0)
                self.LutMan.set('M_down_amp1_R{}'.format(resonator_number), 0)
        if self.upload:
            self.LutMan.load_DIO_triggered_sequence_onto_UHFQC(regenerate_waveforms=True)


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
        self.LutMan_parameter.set(10**(val/20))
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
        self.LutMan_parameter.set(10**(val/20))
        if self.run:
            self.LutMan.AWG.get_instr().awgs_0_enable(False)
        self.LutMan.load_DIO_triggered_sequence_onto_UHFQC()
        if self.run:
            self.LutMan.AWG.get_instr().acquisition_arm(single=self.single)


class UHFQC_pulse_dB_attenuation(Soft_Sweep):

    def __init__(self, UHFQC, IF, dig_trigger=True,**kw):
        self.set_kw()
        self.name = 'UHFQC pulse attenuation'
        self.parameter_name = 'pulse attenuation'
        self.unit = 'dB'
        self.sweep_control = 'soft'
        self.UHFQC = UHFQC
        self.dig_trigger = dig_trigger
        self.IF = IF

    def set_parameter(self, val):
        self.UHFQC.awg_sequence_acquisition_and_pulse_SSB(f_RO_mod=self.IF,RO_amp=10**(val/20),RO_pulse_length=2e-6,acquisition_delay=200e-9,dig_trigger=self.dig_trigger)
        time.sleep(1)
        #print('refreshed UHFQC')


class multi_sweep_function(Soft_Sweep):
    '''
    cascades several sweep functions into a single joint sweep functions.
    '''
    def __init__(self, sweep_functions: list, sweep_point_ratios: list=None,
                 parameter_name=None, name=None, **kw):
        self.set_kw()
        self.sweep_functions = sweep_functions
        self.sweep_control = 'soft'
        self.name = name or 'multi_sweep'
        self.unit = sweep_functions[0].unit
        self.parameter_name = parameter_name or 'multiple_parameters'
        self.sweep_point_ratios = sweep_point_ratios
        for i, sweep_function in enumerate(sweep_functions):
            if self.unit.lower() != sweep_function.unit.lower():
                raise ValueError('units of the sweepfunctions are not equal')

    def set_parameter(self, val):
        if self.sweep_point_ratios is None:
            for sweep_function in self.sweep_functions:
                sweep_function.set_parameter(val)
        else:
            for i, sweep_function in enumerate(self.sweep_functions):
                v = (val-1)*self.sweep_point_ratios[i]+1
                sweep_function.set_parameter(v)

class multi_sweep_function_ranges(Soft_Sweep):
    '''
    cascades several sweep functions into a single joint sweep functions.
    '''
    def __init__(self, sweep_functions: list, sweep_ranges: list, n_points: int,
                 parameter_name=None, name=None,**kw):
        self.set_kw()
        self.sweep_functions = sweep_functions
        self.sweep_control = 'soft'
        self.name = name or 'multi_sweep'
        self.unit = sweep_functions[0].unit
        self.parameter_name = parameter_name or 'multiple_parameters'
        self.sweep_ranges = sweep_ranges
        self.n_points = n_points
        for i, sweep_function in enumerate(sweep_functions):
            if self.unit.lower() != sweep_function.unit.lower():
                raise ValueError('units of the sweepfunctions are not equal')

    def set_parameter(self, val):
        Sweep_points = [ np.linspace(self.sweep_ranges[i][0], 
                                     self.sweep_ranges[i][1],
                                     self.n_points) for i in range(len(self.sweep_ranges)) ]
        for i, sweep_function in enumerate(self.sweep_functions):
            sweep_function.set(Sweep_points[i][val])


class two_par_joint_sweep(Soft_Sweep):
    """
    Allows jointly sweeping two parameters while preserving their
    respective ratios.
    Allows par_A and par_B to be arrays of parameters
    """
    def __init__(self, par_A, par_B, preserve_ratio: bool=True,
                 retrieve_value=False, instr=None, **kw):
        self.set_kw()
        self.unit = par_A.unit
        self.sweep_control = 'soft'
        self.par_A = par_A
        self.par_B = par_B
        self.name = par_A.name
        self.parameter_name = par_A.name
        self.retrieve_value = retrieve_value
        self.instr=instr
        if preserve_ratio:
            try:
                self.par_ratio = self.par_B.get()/self.par_A.get()
            except:
                self.par_ratio = (self.par_B.get_latest()/
                                  self.par_A.get_latest())
        else:
            self.par_ratio = 1

    def set_parameter(self, val):
        self.par_A.set(val)
        self.par_B.set(val*self.par_ratio)
        if self.retrieve_value:
            if self.instr:
                self.instr.operationComplete()  # only get first one to prevent overhead


class FLsweep(Soft_Sweep):
    """
    Special sweep function for AWG8 and QWG flux pulses.
    """
    def __init__(self, 
            lm, 
            par, 
            waveform_name: str, 
            amp_for_generation: float = None, 
            upload_waveforms_always: bool=True,
            bypass_waveform_upload: bool=False
        ):
        super().__init__()
        self.lm = lm
        self.par = par
        self.waveform_name = waveform_name
        self.parameter_name = par.name
        self.unit = par.unit
        self.name = par.name
        self.amp_for_generation = amp_for_generation
        self.upload_waveforms_always = upload_waveforms_always
        self.bypass_waveform_upload = bypass_waveform_upload

        self.AWG = self.lm.AWG.get_instr()
        self.awg_model_QWG = self.AWG.IDN()['model'] == 'QWG'

    def set_parameter(self, val):
        # Just in case there is some resolution or number precision differences
        # when setting the value
        old_par_val = self.par()
        self.par(val)
        updated_par_val = self.par()
        if self.upload_waveforms_always \
                or (updated_par_val != old_par_val and not self.bypass_waveform_upload):
            if self.awg_model_QWG:
                self.set_parameter_QWG(val)
            else:
                self.set_parameter_HDAWG(val)

    def set_parameter_HDAWG(self, val):
        self.par(val)
        if self.amp_for_generation:
            old_val_amp = self.lm.cfg_awg_channel_amplitude()
            self.lm.cfg_awg_channel_amplitude(self.amp_for_generation)
        self.AWG.stop()
        self.lm.load_waveform_onto_AWG_lookuptable(self.waveform_name,
                                                   regenerate_waveforms=True)
        if self.amp_for_generation:
            self.lm.cfg_awg_channel_amplitude(abs(old_val_amp))

        self.AWG.start()
        return

    def set_parameter_QWG(self, val):
        self.AWG.stop()
        self.lm.load_waveform_onto_AWG_lookuptable(
            self.waveform_name, regenerate_waveforms=True,
            force_load_sequencer_program=True) # FIXME: parameter only exists for AWG8_MW_LutMan (and shouldn't)
        self.AWG.start()
        return

class flux_t_middle_sweep(Soft_Sweep):

    def __init__(self, 
            fl_lm_tm: list, 
            fl_lm_park: list,
            which_gate: list,
            t_pulse: list,
            duration: float = 40e-9
        ):
        super().__init__()
        self.name = 'time_middle'
        self.parameter_name = 'time_middle'
        self.unit = 's'
        self.fl_lm_tm = fl_lm_tm
        self.fl_lm_park = fl_lm_park
        self.which_gate = which_gate
        self.t_pulse = t_pulse
        self.duration = duration

    def set_parameter(self, val):
        which_gate = self.which_gate
        t_pulse = np.repeat(self.t_pulse, 2)
        sampling_rate = self.fl_lm_tm[0].sampling_rate()
        total_points = self.duration*sampling_rate
        
        # Calculate vcz times for each flux pulse
        time_mid = val / sampling_rate
        n_points = [ np.ceil(tp / 2 * sampling_rate) for tp in t_pulse ]
        time_sq  = [ n / sampling_rate for n in n_points ]
        time_park= np.max(time_sq)*2 + time_mid + 4/sampling_rate
        time_park_pad = np.ceil((self.duration-time_park)/2*sampling_rate)/sampling_rate
        time_pad = np.abs(np.array(time_sq)-np.max(time_sq))+time_park_pad

        # update parameters and upload waveforms
        Lutmans = self.fl_lm_tm + self.fl_lm_park
        AWGs = np.unique([lm.AWG() for lm in Lutmans])
        for AWG in AWGs:
            Lutmans[0].find_instrument(AWG).stop()
        # set flux lutman parameters of CZ qubits
        for i, fl_lm in enumerate(self.fl_lm_tm):
            fl_lm.set('vcz_time_single_sq_{}'.format(which_gate[i]), time_sq[i])
            fl_lm.set('vcz_time_middle_{}'.format(which_gate[i]), time_mid)
            fl_lm.set('vcz_time_pad_{}'.format(which_gate[i]), time_pad[i])
            fl_lm.set('vcz_amp_fine_{}'.format(which_gate[i]), .5)
            fl_lm.load_waveform_onto_AWG_lookuptable(
                wave_id=f'cz_{which_gate[i]}', regenerate_waveforms=True)
        # set flux lutman parameters of Park qubits
        for fl_lm in self.fl_lm_park:
            fl_lm.park_pad_length(time_park_pad)
            fl_lm.park_length(time_park)
            fl_lm.load_waveform_onto_AWG_lookuptable(
                wave_id='park', regenerate_waveforms=True)
        for AWG in AWGs:
            Lutmans[0].find_instrument(AWG).start()

        return val


class Nested_resonator_tracker(Soft_Sweep):
    """
    Sets a parameter and performs a "find_resonator_frequency" measurement
    after setting the parameter.
    """
    def __init__(self, qubit, nested_MC, par,
                 use_min=False, freqs=None, reload_sequence=False,
                 cc=None, sequence_file=None, **kw):
        super().__init__(**kw)
        self.qubit = qubit
        self.freqs = freqs
        self.par = par
        self.nested_MC = nested_MC
        self.parameter_name = par.name
        self.unit = par.unit
        self.name = par.name
        self.reload_marked_sequence = reload_sequence
        self.sequence_file = sequence_file
        self.cc = cc
        self.use_min = use_min

    def set_parameter(self, val):
        self.par(val)
        self.qubit.find_resonator_frequency(
            freqs=self.freqs,
            MC=self.nested_MC, use_min=self.use_min)
        self.qubit._prep_ro_sources()
        if self.reload_marked_sequence:
            # reload the meaningfull sequence
            self.cc.stop()
            self.cc.eqasm_program(self.sequence_file.filename)
            self.cc.start()
        spec_source = self.qubit.instr_spec_source.get_instr()
        spec_source.on()
        self.cc.start()

class Nested_spec_source_pow(Soft_Sweep):
    """
    Sets a parameter and performs a "find_resonator_frequency" measurement
    after setting the parameter.
    """
    def __init__(self, qubit, nested_MC, par, reload_sequence=False,
                 cc=None, sequence_file=None, **kw):
        super().__init__(**kw)
        self.qubit = qubit
        self.par = par
        self.nested_MC = nested_MC
        self.parameter_name = par.name
        self.unit = par.unit
        self.name = par.name
        self.reload_marked_sequence = reload_sequence
        self.sequence_file = sequence_file
        self.cc = cc

    def set_parameter(self, val):
        spec_source = self.qubit.instr_spec_source.get_instr()
        spec_source.power.set(val)
        if self.reload_marked_sequence:
            # reload the meaningfull sequence
            self.cc.eqasm_program(self.sequence_file.filename)
        spec_source.on()
        self.cc.start()

class Nested_amp_ro(Soft_Sweep):
    """
    Sets a parameter and performs a "find_resonator_frequency" measurement
    after setting the parameter.
    """
    def __init__(self, qubit, nested_MC, par, reload_sequence=False,
                 cc=None, sequence_file=None, **kw):
        super().__init__(**kw)
        self.qubit = qubit
        self.par = par
        self.nested_MC = nested_MC
        self.parameter_name = par.name
        self.unit = par.unit
        self.name = par.name
        self.reload_marked_sequence = reload_sequence
        self.sequence_file = sequence_file
        self.cc = cc

    def set_parameter(self, val):
        self.par(val)
        self.qubit._prep_ro_pulse(CW=True)
        if self.reload_marked_sequence:
            # reload the meaningfull sequence
            self.cc.eqasm_program(self.sequence_file.filename)
        self.cc.start()

class tim_flux_latency_sweep(Soft_Sweep):
    def __init__(self, device):
        super().__init__()
        self.dev = device
        self.name = 'Flux latency'
        self.parameter_name = 'Flux latency'
        self.unit = 's'

    def set_parameter(self, val):
        self.dev.tim_flux_latency_0(val)
        self.dev.tim_flux_latency_1(val)
        self.dev.tim_flux_latency_2(val)
        self.dev.prepare_timing()

        time.sleep(.5)
        return val


class tim_ro_latency_sweep(Soft_Sweep):
    def __init__(self, device):
        super().__init__()
        self.dev = device
        self.name = 'RO latency'
        self.parameter_name = 'RO latency'
        self.unit = 's'

    def set_parameter(self, val):
        self.dev.tim_ro_latency_0(val)
        self.dev.tim_ro_latency_1(val)
        self.dev.tim_ro_latency_2(val)
        self.dev.prepare_timing()
        time.sleep(.5)
        return val


class tim_mw_latency_sweep(Soft_Sweep):
    def __init__(self, device):
        super().__init__()
        self.dev = device
        self.name = 'MW latency'
        self.parameter_name = 'MW latency'
        self.unit = 's'

    def set_parameter(self, val):
        self.dev.tim_mw_latency_0(val)
        self.dev.tim_mw_latency_1(val)
        self.dev.tim_mw_latency_2(val)
        self.dev.tim_mw_latency_3(val)
        self.dev.tim_mw_latency_4(val)
        self.dev.prepare_timing()

        time.sleep(.5)
        return val


class tim_mw_latency_sweep_1D(Soft_Sweep):
    def __init__(self, device):
        super().__init__()
        self.dev = device
        self.name = 'MW latency'
        self.parameter_name = 'MW latency'
        self.unit = 's'

    def set_parameter(self, val):
        self.dev.tim_mw_latency_0(val)
        self.dev.tim_mw_latency_1(val)
        self.dev.prepare_timing()
        return val


class SweepAlong2DContour(Soft_Sweep):
    """
    Performs a sweep along a 2D contour by setting two parameters at the same
    time
    """
    def __init__(self, par_A, par_B, contour_pnts, interp_kw: dict = {}):
        super().__init__()
        self.par_A = par_A
        self.par_B = par_B
        self.name = 'Contour sweep'
        self.parameter_name = 'Contour sweep'
        self.unit = 'a.u.'
        self.interpolator = c2d.interp_2D_contour(contour_pnts, **interp_kw)

    def set_parameter(self, val):
        val_par_A, val_par_B = self.interpolator(val)
        self.par_A(val_par_A)
        self.par_B(val_par_B)

        return val
