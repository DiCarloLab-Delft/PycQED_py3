import qt
from instrument import Instrument
import logging
import types
import instruments


class HeterodyneSource(Instrument):
    '''
    This instrument combines the LO and RF source into an joint
    instrument for Heterodyne readout.
    No support yet for multiple RF's can be copied from existing TD Meas

    NOTE: this instrument differs from the HomodyneSource in that it does
    not concern itself with the ATS-CW
    '''
    def __init__(self, name,
                 RF='RF', LO='LO',
                 IF=0, **kw):
        logging.info(__name__ + ': Initializing HeterodyneSource instrument')
        Instrument.__init__(self, name, tags=['Meta-Instrument'])

        self.add_parameter('frequency', flag=Instrument.FLAG_GETSET,
                           type=float, units='GHz')
        self.add_parameter('IF', flag=Instrument.FLAG_GETSET,
                           type=float, units='GHz')
        self.add_parameter(
            'LO_source', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'RF_source', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)

        self.add_parameter('LO_power', flag=Instrument.FLAG_GETSET,
                           type=float, units='dBm')
        self.add_parameter('RF_power', flag=Instrument.FLAG_GETSET,
                           type=float, units='dBm')
        self.set_RF_source(RF)
        self.set_LO_source(LO)
        self.set_IF(IF)

    def do_set_IF(self, val):
        '''
        Sets the IF between the RF and LO frequencies.
        The following convention is used to define the sign of the IF:
            IF = LO - RF

        Changing this value only changes the value used when setting LO and RF
        it currently does not update the existing RF and LO frequencies.
        '''
        self.IF = val

    def do_get_IF(self):
        '''
        Returns the Inermediate Frequency between the LO and RF
            IF = LO - RF

        Returns the stored value in this virtual instrument does not querie the
        physical RF and LO and get's their difference.
        '''
        return self.IF

    def set_RF_and_LO_params(self):
        '''
        Uses the settings from this instrument to set the RF and LO.
        '''
        self.RF.set_frequency(self.frequency)
        self.RF.set_pulsemod_state('On')
        self.RF.set_power(self.RF_power)
        self.RF.on()

        self.LO.set_frequency((self.frequency+self.IF)*1e9)
        # conversion of GHz to Hz
        self.LO.on()

    def do_set_frequency(self, freq):
        self.frequency = freq
        self.LO.set_frequency((self.frequency+self.IF)*1e9)
        self.RF.set_frequency(self.frequency*1e9)
        # conversion of GHz to Hz

    def do_get_frequency(self):
        return self.frequency

    def do_set_RF_source(self, name):
        self.RF = qt.instruments[name]

    def do_get_RF_source(self):
        return self.RF.get_name()

    def get_RF_status(self):
        return self.RF.get_status()

    def do_set_RF_power(self, val):
        self.RF_power = val
        self.RF.set_power(val)

    def do_get_RF_power(self):
        return self.RF_power

    def do_set_LO_source(self, val):
        self.LO = qt.instruments[val]

    def do_get_LO_source(self):
        return self.LO.get_name()

    def get_LO_status(self):
        return self.LO.get_status()

    def do_set_LO_power(self, val):
        self.LO_power = val
        self.LO.set_power(val)

    def do_get_LO_power(self):
        return self.LO_power
