from qcodes import Instrument
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_PQSC as pqsc
import qcodes.utils.validators as vals


class PQSC_control(Instrument):
    """Meta instrument used as a wrapper to control the PQSC instrument via the
    ZI driver"""

    def __init__(self, name, pqsc_instr):

        super().__init__(name=name)
        self.pqsc_instr = pqsc_instr

        #FIXME: find the real boundary values
        self.add_parameter(
            'pulse_period',
            label='Pulse period',
            get_cmd=(lambda self=self: self.pqsc_instr.execution_holdoff()),
            set_cmd=(
                lambda val, self=self: self.pqsc_instr.execution_holdoff(val)),
            unit='s',
            vals=vals.Numbers(min_value=20e-9, max_value=2e3),
            initial_value=20e-9,
            docstring=('Command for setting the desired pulse\
                                period. Min value: 20 ns, Max value: 2000 s.'))
        # FIXME: get real boundary values
        self.add_parameter(
            'repetitions',
            label='Number of repetitions',
            get_cmd=(
                lambda self=self: self.pqsc_instr.execution_repetitions()),
            set_cmd=(lambda val, self=self: self.pqsc_instr.
                     execution_repetitions(int(val))),
            unit='',
            vals=vals.Ints(1, int(1e15)),
            docstring=('Command for setting the number of repetitions'))

    def start(self):
        """Start the playback of trigger pulses at the PQSC device"""

        self.pqsc_instr.start()

    def stop(self):
        """Start the playback of trigger pulses at the PQSC device"""

        self.pqsc_instr.stop()