import logging
import time
import os
import numpy as np
from pycqed.utilities.general import setInDict
from pycqed.measurement.waveform_control_CC import qasm_compiler as qcx
from pycqed.instrument_drivers.virtual_instruments.pyqx import qasm_loader as ql
from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta
from pycqed.measurement.waveform_control_CC import qasm_compiler as qcx


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


class Heterodyne_Frequency_Sweep(Soft_Sweep):

    def __init__(self, RO_pulse_type,
                 LO_source, IF,
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
        self.LO_source.frequency(val-self.IF)
        if 'gated' in self.RO_pulse_type.lower():
            self.RF_source.frequency(val)


class None_Sweep(Soft_Sweep):

    def __init__(self, sweep_control='soft', sweep_points=None,
                 **kw):
        super(None_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'None_Sweep'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.sweep_points = sweep_points

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
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
                                  "prepz q0", "h q0", "x q0", "z q0", "y q0", "y q0", "z q0", "x q0", "h q0", "measure q0"])
        self.__cnt = self.__cnt+1
        # pass


class QX_RB_Sweep(Soft_Sweep):

    """
       QX Randomized Benchmarking Test
    """

    def __init__(self, qxc, filename, num_circuits, sweep_control='soft',
                 sweep_points=None, **kw):
        super(QX_RB_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'QX_RB_Sweep'
        self.parameter_name = 'N_Clifford'
        self.unit = 'P'
        self.sweep_points = sweep_points
        self.__qxc = qxc
        self.__qxc.create_qubits(2)
        self.__cnt = 0
        self.filename = filename
        self.num_circuits = num_circuits
        qasm = ql.qasm_loader(filename)
        qasm.load_circuits()
        self.circuits = qasm.get_circuits()
        for c in self.circuits:
            self.__qxc.create_circuit(c[0], c[1])

    def set_parameter(self, val):
        assert(self.__cnt < self.num_circuits)
        self.__cnt = self.__cnt+1


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
        super(Hard_Sweep, self).__init__()
        self.sweep_control = 'hard'
        self.parameter_name = 'None'
        self.unit = 'a.u.'

    def start_acquistion(self):
        pass


class QASM_Sweep(Hard_Sweep):

    def __init__(self, filename, CBox, op_dict,
                 parameter_name='Points', unit='a.u.', upload=True):
        super().__init__()
        self.name = 'QASM_Sweep'
        self.filename = filename
        self.upload = upload
        self.CBox = CBox
        self.op_dict = op_dict
        self.parameter_name = parameter_name
        self.unit = unit

    def prepare(self, **kw):
        self.CBox.trigger_source('internal')
        if self.upload:
            qumis_file = qta.qasm_to_asm(self.filename, self.op_dict)
            self.CBox.load_instructions(qumis_file.name)


class QASM_Sweep_v2(Hard_Sweep):
    """
    Sweep function for a QASM file, using the XFu compiler to generate QuMis
    """

    def __init__(self, qasm_fn: str, config: dict, CBox,
                 parameter_name: str ='Points', unit: str='a.u.',
                 upload: bool=True, verbosity_level: int=0):
        super().__init__()
        self.name = 'QASM_Sweep_v2'

        self.qasm_fn = qasm_fn
        self.config = config
        self.CBox = CBox
        self.upload = upload

        self.parameter_name = parameter_name
        self.unit = unit
        self.verbosity_level = verbosity_level

    def prepare(self, **kw):
        self.compile_and_upload()

    def compile_and_upload(self):
        if self.upload:
            self.CBox.trigger_source('internal')
        qasm_folder, fn = os.path.split(self.qasm_fn)
        base_fn = fn.split('.')[0]
        qumis_fn = os.path.join(qasm_folder, base_fn + ".qumis")
        self.compiler = qcx.QASM_QuMIS_Compiler(
            verbosity_level=self.verbosity_level)
        self.compiler.compile(self.qasm_fn, qumis_fn=qumis_fn,
                              config=self.config)
        if self.upload:
            self.CBox.load_instructions(qumis_fn)
        return self.compiler


class QASM_config_sweep(QASM_Sweep_v2):
    """
    Sweep function for a QASM file, using the XFu compiler to generate QuMis
    """

    def __init__(self, qasm_fn: str, config: dict,
                 config_par_map: list, CBox,
                 parameter_name: str =None, unit: str='a.u.',
                 par_scale_factor=1, set_parser=None,
                 upload: bool=True, verbosity_level: int=0):
        self.name = 'QASM_config_sweep'
        self.sweep_control = 'soft'
        self.qasm_fn = qasm_fn
        self.config = config
        self.CBox = CBox
        self.set_parser = set_parser
        self.upload = upload
        self.config_par_map = config_par_map
        self.par_scale_factor = par_scale_factor

        if parameter_name is None:
            self.parameter_name = self.config_par_map[-1]
        else:
            self.parameter_name = parameter_name
        self.unit = unit
        self.verbosity_level = verbosity_level

    def set_parameter(self, val):
        val *= self.par_scale_factor
        if self.set_parser is not None:
            val = self.set_parser(val)
        setInDict(self.config, self.config_par_map, val)
        self.compile_and_upload()

    def prepare(self, **kw):
        pass


class QWG_flux_QASM_Sweep(QASM_Sweep_v2):

    def __init__(self, qasm_fn: str, config: dict,
                 CBox, QWG_flux_lutman,
                 parameter_name: str ='Points', unit: str='a.u.',
                 upload: bool=True, verbosity_level: int=0):
        super(QASM_Sweep_v2, self).__init__()
        self.name = 'QWG_flux_QASM_Sweep'

        self.qasm_fn = qasm_fn
        self.config = config
        self.CBox = CBox
        self.QWG_flux_lutman = QWG_flux_lutman
        self.upload = upload

        self.parameter_name = parameter_name
        self.unit = unit
        self.verbosity_level = verbosity_level

    def prepare(self, **kw):
        from pycqed.measurement.waveform_control_CC.qasm_compiler_helpers \
            import get_timetuples_since_event
        # assume this corresponds 1 to 1 with the QWG_trigger
        compiler = self.compile_and_upload()
        for i in range(len(self.sweep_points)):
            self.time_tuples, end_time_ns = get_timetuples_since_event(
                start_label='qwg_trigger_{}'.format(i),
                target_labels=['square', 'cz'],
                timing_grid=compiler.timing_grid, end_label='ro',
                convert_clk_to_ns=True)
            self.comp_fp = self.QWG_flux_lutman.generate_composite_flux_pulse(
                time_tuples=self.time_tuples, end_time_ns=end_time_ns)
        if self.upload:
            self.QWG_flux_lutman.load_custom_pulse_onto_AWG_lookuptable(
                waveform=self.comp_fp, pulse_name='custom_{}'.format(i),
                distort=True, append_compensation=True,
                codeword=i)


class QuMis_Sweep(Hard_Sweep):

    def __init__(self, filename, CBox,
                 parameter_name='Points', unit='a.u.', upload=True):
        super().__init__()
        self.name = 'QASM_Sweep'
        self.filename = filename
        self.upload = upload
        self.CBox = CBox
        self.parameter_name = parameter_name
        self.unit = unit

    def prepare(self, **kw):
        self.CBox.trigger_source('internal')
        if self.upload:
            self.CBox.load_instructions(self.filename)


class QX_Hard_Sweep(Hard_Sweep):

    def __init__(self, qxc, filename):  # , num_circuits):
        super().__init__()
        self.name = 'QX_Hard_Sweep'
        self.filename = filename
        self.__qxc = qxc
        # self.num_circuits = num_circuits
        qasm = ql.qasm_loader(filename, qxc.get_nr_qubits())
        qasm.load_circuits()
        self.circuits = qasm.get_circuits()

    def get_circuits_names(self):
        ids = []
        for c in self.circuits:
            ids.append(c[0])
        return ids

    def prepare(self, **kw):
        # self.CBox.trigger_source('internal')
        print("QX_Hard_Sweep.prepare() called...")
        # self.__qxc.create_qubits(2)
        # for c in self.circuits:
        #     self.__qxc.create_circuit(c[0], c[1])


class QX_RB_Hard_Sweep(Hard_Sweep):

    def __init__(self, qxc, qubits=2):
        super().__init__()
        self.name = 'QX_RB_Hard_Sweep'
        self.qubits = qubits
        self.__qxc = qxc
        self.__qxc.create_qubits(2)
        # qasm = ql.qasm_loader(filename)
        # qasm.load_circuits()
        # self.circuits = qasm.get_circuits()
        # print(self.circuits[0])

    def prepare(self, **kw):
        # self.CBox.trigger_source('internal')
        print("QX_Hard_Sweep.prepare() called...")
        # for c in self.circuits:
        # self.__qxc.create_circuit(c[0],c[1])


# NOTE: AWG_sweeps are located in AWG_sweep_functions


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
        self.LutMan.QWG.get_instr().stop()
        self.LutMan_parameter.set(val)
        self.LutMan.load_pulses_onto_AWG_lookuptable(regenerate_pulses=True)
        self.LutMan.QWG.get_instr().start()
        self.LutMan.QWG.get_instr().getOperationComplete()


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

            # Load onto QWG
            QWG.createWaveformReal(
                pulseName, self.LutMan._wave_dict[self.flux_pulse_type])

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
                 codewords=np.arange(128),
                 param_name='flux pulse parameter',
                 param_unit='a.u.',
                 **kw):
        super().__init__(**kw)
        self.wave_func = wave_func
        self.chunk_size = chunk_size
        self.LutMan = LutMan
        self.codewords = codewords
        self.name = param_name
        self.parameter_name = param_name
        self.unit = param_unit

    def set_parameter(self, val):
        # Find index of val in sweep_points
        ind = np.where(self.sweep_points == val)[0]
        if len(ind) == 0:
            # val was not found in the sweep points
            raise ValueError('Value {} is not in the sweep points'.format(val))
        ind = ind[0]  # set index to the first occurence of val in sweep points

        for i, paramVal in enumerate(self.sweep_points[ind:ind +
                                                       self.chunk_size]):
            pulseName = 'pulse_{}'.format(i)

            self.LutMan.load_custom_pulse_onto_AWG_lookuptable(
                self.wave_func(paramVal), append_compensation=True,
                pulse_name=pulseName, codeword=i)
