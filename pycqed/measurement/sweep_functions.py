import logging
import time

from pycqed.instrument_drivers.virtual_instruments.pyqx import qasm_loader as ql


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


class Soft_Sweep(Sweep_function):

    def __init__(self, **kw):
        self.set_kw()
        self.sweep_control = 'soft'

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        pass
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
        self.name = 'None_Sweep'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.RO_pulse_type = RO_pulse_type
        self.sweep_points = sweep_points
        self.LO_source = LO_source
        self.IF = IF
        if self.RO_pulse_type == 'Gated_MW_RO_pulse':
            self.RF_source = RF_source

    def set_parameter(self, val):
        # RF + IF = LO
        self.LO_source.frequency(val+self.IF)
        if self.RO_pulse_type == 'Gated_MW_RO_pulse':
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

    def __init__(self, qxc, filename, num_circuits, sweep_control='soft', sweep_points=None, **kw):
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


class QX_Hard_Sweep(Hard_Sweep):

    def __init__(self, qxc, filename):  # , num_circuits):
        super().__init__()
        self.name = 'QX_Hard_Sweep'
        self.filename = filename
        self.__qxc = qxc
        # self.num_circuits = num_circuits
        qasm = ql.qasm_loader(filename)
        qasm.load_circuits()
        self.circuits = qasm.get_circuits()
        # print(self.circuits[0])

    def get_circuits_names(self):
        ids = []
        for c in self.circuits:
            ids.append(c[0])
        return ids

    def prepare(self, **kw):
        # self.CBox.trigger_source('internal')
        print("QX_Hard_Sweep.prepare() called...")
        self.__qxc.create_qubits(2)
        for c in self.circuits:
            # print(c)
            self.__qxc.create_circuit(c[0], c[1])

'''
QX RB Sweep (Multi QASM Files)
'''


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
        3) segment sweep (this requires a list of elements. Each element fully defines a sweep)
           segment_list = [[start_frequency, stop_frequency, nbr_points, power, segment_time, mesurement_delay, bandwidth],
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
        self.unit = LutMan_parameter.units
        self.sweep_control = 'soft'
        self.LutMan = LutMan
        self.LutMan_parameter = LutMan_parameter

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''

        self.LutMan_parameter.set(val)
        self.LutMan.load_pulses_onto_AWG_lookuptable()
