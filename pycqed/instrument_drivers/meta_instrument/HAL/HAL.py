import logging

from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals

log = logging.getLogger(__name__)


"""
signal chain elements
- we define QCoDeS parameters at this level to define known instrument profiles
- derived classes can extend parameters with get_commands, validators, etc (FIXME: TBC)
- derived classes can also add parameters, which can then not be accessed through HAL
- derived classes can also choose to not act upon parameters if no meaningful support can be provided
"""

class Element(InstrumentBase):  # FIXME: or create HALInstrument?
    def __init__(self):
        pass



class Sequencer(Element):
    def __init__(self):
        super().__init__()

        # add parameters

    def start(self):
        raise RuntimeError("call overridden method")

    def stop(self):
        raise RuntimeError("call overridden method")


# Arbitraty Waveform Generator
class AWG(Element):
    def __init__(self):
        super().__init__()

# acquisition
class ACQ(Element):
    def __init__(self):
        super().__init__()


# subset of QCoDeS SGS100
class SignalGenerator(Element):
    def __init__(self):
        super().__init__()

        # add parameters
        self.add_parameter(
            name='frequency',
            label='Frequency',
            unit='Hz',
            get_parser=float)
        self.add_parameter(
            name='phase',
            label='Phase',
            unit='deg',
            get_parser=float,
            vals=vals.Numbers(0, 360))
        self.add_parameter(
            name='power',
            label='Power',
            unit='dBm',
            get_parser=float)

        # FIXME: add more

    def on(self):
        raise RuntimeError("call overridden method")

    def off(self):
        raise RuntimeError("call overridden method")


"""
elements to add:

PhaseModulator
TimeShift
AmplitudeModulator
VariableGain
Switch
FilterFIR
FilterIIR
Offset
CrosstalkMatrix
Threshold

composite:
UpConverter
DownConverter
Demodulator
Discriminator/Detector

exercise:
model everything currently used
"""


"""
the HAL constructs a single object representing all instruments used, based on a JSON configuration

FIXME: notes:
- factory pattern
- an instrument implements particular elements, e.g. tx.AWG, tx.filterIIR, tx.upconverter, rx.downconverter, rx.detector
- how to handle (e.g.) internal upconverter vs. external: attach instrument to root node, driver determines sub nodes.
  how to manage profiles then?
- define allowed node structure: tree and leaf nodes, check JSON against this?
- terminology: 
    -   elements vs. nodes vs tree
        -   the actual variables live inside the elements, can persist through serialization
    -   resources? targets? stuff that you can control in real time? (Virtual) place where you can generate/measure signal
        A waveform can be output on 'q0.flux'. But we can als set q0.flux.amplitude, maybe from the sequencer, or from sw.
        What makes a node eligible to be a real-time target: the presence of an AWG at that point (or ACQ for measurement)
        
        Rephrase:
        a resource is something you can control. If it is a rt capable resource, it can be controlled in rt. The waveform 
        compiler applies a waveform program to the rt resources.
        
    -   group (see OpenQL): several electrical ports can operate together (e.g. I&Q), but also, several signals can be
        multiplexed on single port/set of ports (e.g. UHF). Refers to a 'unit' that can be controlled, irrespective how
        that is wired to instrument ports. Maybe rename to unit/subunit?
        -   Would we like to be able to assign I&Q separately
        -   Also handle pulsar, which has 4 AWGs per output pair (numbers TBC) 
- Example:
    -   hw sideband modulator versus software. Maybe implement sw version in base class of composite instrument, and allow override
-   add note on detailed electronics knowledge of elements: none
- HAL-driver
    - init should not fail (e.g. because instrument is not responding)
        - add open() for that, which should be configurable from JSON
    - init and open should be 'fast'
    - parameter stucture according to profile
    - instruments down in list override prior definitions (TBC)
- qubits can share element, and thus clash. Variables should point to same element (use reference)
    
    element structure created
    q*.tx.AWG
    q*.tx.upconverter 
    q*.tx.upconverter.lo
    q*.tx.filterIIR
    
    q*.rx.downconverter
    q*.rx.
    
    also:
    .instrument
    .group/channel
    
# FIXME: use add_submodule?


"""


# this should also be usable for OpenQL (or other compilers)
# FIXME: allow instruments within instruments?

json_2 = """
    instruments [
        {
            "name": "ro_1"
            "class": "UHFQC"
            "open": {
                "name": "UHFQC_1",
                "device": "dev2216"
                "use_dio": true
            },
            "resources": [["q0.readout"], ["q2.readout"], ["q3.readout"], ["q4.readout"]],
        },
        {   "name": "ro_1_tx_lo",
            "class": "rs-sgs100a",
            "resources": [["q0.tx.upconverter", "q2.tx.upconverter", "q3.tx.upconverter", "q4.tx.upconverter"]]
        },
        {
            "name": "mw_1",
            "class":"ZI_HDAWG8",
            "open": {
                "name": "HDAWG_8070",
                "device": "dev8070"
            },
            "resources": [["q0.mw"], ["q2.mw"], ["q3.mw"], ["q4.mw"]],
        },
        {
            "name": "mw_1",
            "class":"ZI_HDAWG8",
            "open": {
                "name": "HDAWG_8070",
                "device": "dev8070"
            },
            "resources": [["q0.flux"], ["q2.flux"], ["q3.flux"], ["q4.flux"], [], [], [], []],
        },
        {
            "name": "cc",
            "class":"CC",
            "open": {
                "class": "IPTransport",
                "init": TBD
            },
            "resources": [],    // define resources per slot, how to link to connected instrument, link to name?
        },
        {
            "name": "",
            "class":"",
            "open": {
            },
            "resources": [],
        }
    ]


"""

"""
LO_ro = rs.RohdeSchwarz_SGS100A(name='LO_ro', address='TCPIP0::192.168.0.73')
UHFQC_1 = ZI_UHFQC.UHFQC('UHFQC_1', device='dev2216',use_dio=True)
HDAWG_8070 = ZI_HDAWG8.ZI_HDAWG8('HDAWG_8070', device='dev8070')
qwg_12 = QuTech_AWG_Module(
    'qwg_12', address=qwg_12_ip, port=5025, reset=True, dio_mode='SLAVE',
    run_mode='CODeword', codeword_protocol='MICROWAVE_NO_VSM')
fluxcurrent = s4g.QuTech_SPI_S4g_FluxCurrent(
    "fluxcurrent", address='COM9',
    channel_map={
       "FBL_D1": (15, 0),
       "FBL_D2": (15, 1),
       "FBL_D3": (15, 2),
       "FBL_D4": (15, 3),
       "FBL_X" : (4, 0),
       "FBL_Z1": (4, 1),
       "FBL_Z2": (4, 2),
       })
VSM = vsm.QuTechVSMModule(name='VSM', address='192.168.0.10', port=5025)
SH = sh.SignalHound_USB_SA124B('SH', dll_path='C:\Windows\System32\sa_api.dll')
    
"""

# or: use piece of Python, but: we want JSON for OpenQL and like to use one definition


class HAL:
    def __init__(self):
        pass    # FIXME: implement

    def register_element(self, element: Element):   # or just inherit from HAL
        pass    # FIXME: implement

    def from_JSON(self, json: str, simul: bool=False):
        pass    # FIXME: implement

    def open(self):
        pass    # FIXME: implement

    def close(self):
        pass    # FIXME: implement

    def has_hw_sideband_modulation(self): # or provide software implementation in base class
        pass    # FIXME: implement
