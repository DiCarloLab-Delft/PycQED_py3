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
    def __init__(self, name: str):
        super().__init__(name)

        self._meta_attrs = ['name'] # FIXME: not set by InstrumentBase, and required by snapshot
        pass



class Sequencer(Element):
    def __init__(self, name: str):
        super().__init__(name)

        # add parameters

    def start(self):
        raise RuntimeError("call overridden method")

    def stop(self):
        raise RuntimeError("call overridden method")


# Arbitrary Waveform Generator
class AWG(Element):
    def __init__(self, name: str):
        super().__init__(name)

    def start(self):
        raise RuntimeError("call overridden method")

    def stop(self):
        raise RuntimeError("call overridden method")

    def add_waveform(self):
        pass


# acquisition
class ACQ(Element):
    def __init__(self, name: str):
        super().__init__(name)

    def start(self):
        raise RuntimeError("call overridden method")

    def stop(self):
        raise RuntimeError("call overridden method")

    def add_waveform(self):
        pass


# subset of QCoDeS SGS100
class SignalGenerator(Element):
    def __init__(self, name: str):
        super().__init__(name)

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


class CrosstalkMatrix(Element):
    pass # FIXME


class MixerCorrection(Element):
    # FIXME: cleanup, inherit from crosstalk matrix?
    def __init__(self, name: str):
        super().__init__(name)

        # add parameters
        self.add_parameter(
            name='matrix',
            unit='', # FIXME: relative
            get_parser=float)  # FIXME: 2x2 matrix

    # get transfer matrix. use complex numbers. use Affine matrix?. allow to specify FIR filter. But how about IIR
    # (use transformation function on samples instead?)
    # Create base class for Elements that have transfer function (as opposed to sources): XferElement?
    def get_transfer(self):
        pass

    # from: mw_lutman.py
    def _add_mixer_corr_pars(self):
        self.add_parameter('mixer_alpha', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('mixer_phi', vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter(
            'mixer_apply_predistortion_matrix', vals=vals.Bool(), docstring=(
                'If True applies a mixer correction using mixer_phi and '
                'mixer_alpha to all microwave pulses using.'),
            parameter_class=ManualParameter, initial_value=True)


    # from: pycqed/measurement/waveform_control_CC/waveform.py
    def mixer_predistortion_matrix(alpha, phi):
        '''
        predistortion matrix correcting for a mixer with amplitude
        mismatch "mixer_alpha" and skewness "phi"
        M = [ 1            tan(phi) ]
            [ 0   1/mixer_alpha * sec(phi)]
        Notes on the procedure for acquiring this matrix can be found in
        PycQED/docs/notes/MixerSkewnessCalibration_LDC_150629.pdf
        '''
        predistortion_matrix = np.array(
            [[1,  np.tan(phi*2*np.pi/360)],
             [0, 1/alpha * 1/np.cos(phi*2*np.pi/360)]])
        return predistortion_matrix

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
Threshold
"""

"""

    Composite Elements
    
FIXME: how to associate a composite element with an instrument unit/group

"""

class FluxOutput(Element): # BasebandOutput?
    # awg+seq
    # IIR filter
    # DC offset
    pass

class MicrowaveOutput(Element):
    def __init__(self, name: str):
        super().__init__(name)

        awg = AWG("awg") # FIXME: should be final AWG, names must be unique
        self.add_submodule('awg', awg)
        seq = Sequencer('seq')
        self.add_submodule('seq', seq)
        # sideband modulator
        # AM
        # PM
        mixer_corr = MixerCorrection('mc')
        self.add_submodule('mixer_corr', mixer_corr)
        lo = SignalGenerator('lo')
        self.add_submodule('lo', lo)

    def on(self):
        pass # FIXME: switch relavant sub_modules

    def off(self):
        pass # FIXME: switch relavant sub_modules

    def get_transfer(self):
        """
        combined transfer_matrix for changing waveforms generated, based on built-in SP chain semantics
        :return:
        """
        pass

class MicrowaveInput(Element):
    pass

"""

UpConverter
DownConverter
Demodulator
Discriminator/Detector

exercise:
model 'everything' currently used, see how it fits
"""


"""
the HAL constructs a single object representing all instruments used, based on a JSON configuration

FIXME: notes:
- factory pattern
- an instrument implements particular elements, e.g. tx.AWG, tx.filterIIR, tx.upconverter, rx.downconverter, rx.detector
- how to handle (e.g.) internal upconverter vs. external: attach instrument to root node, driver determines sub nodes.
  how to manage profiles then?
- define allowed node structure: tree and leaf nodes, check JSON against this? Use JSON schema?
- terminology: 
    -   elements vs. nodes vs tree
        -   the actual variables live inside the elements, can persist through serialization
    -   locations (resources? targets?) stuff that you can control in real time? (Virtual) place where you can generate/measure signal
        A waveform can be output on 'q0.flux'. But we can als set q0.flux.amplitude, maybe from the sequencer, or from sw.
        What makes a node eligible to be a real-time target: the presence of an AWG at that point (or ACQ for measurement) (i.e. have certain parent class?)
        
        Rephrase:
        a resource is something you can control. If it is a rt capable resource, it can be controlled in rt. The waveform 
        compiler applies a waveform program to the rt locations.
        
    -   group (see OpenQL): several electrical ports can operate together (e.g. I&Q), but also, several signals can be
        multiplexed on single port/set of ports (e.g. UHF). Refers to a 'unit' that can be controlled, irrespective how
        that is wired to instrument ports. Maybe rename to unit/subunit?
        -   Would we like to be able to assign I&Q separately
        -   Also handle pulsar, which has 4 (TBC) AWGs per output pair
- Example:
    -   hw sideband modulator versus software. Maybe implement sw version in base class of composite instrument, and allow override
- add note on detailed electronics knowledge of elements: none
- instruments down in list override prior definitions (TBC)
- qubits can share element, and thus clash. Variables should point to same element (use reference)
- nothing stops you from also instantiating instruments outside of the HAL
- add hooks to allow user additions, or use inheritance
- where to know about sampling rate
- compare: https://qcodes.github.io/Qcodes/examples/Station.html
    -   uses YAML
    -   also sets parameters, which we dislike
    -   
    
- element structure created
    q*.tx.AWG
    q*.tx.upconverter 
    q*.tx.upconverter.lo
    q*.tx.filterIIR
    
    q*.rx.downconverter
    q*.rx.
    
    also:
    .instrument
    .group/channel
    
"""

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

    def start(self):    # add parameter for user intent to limit instruments that need to be controlled
        pass    # FIXME: implement

    def stop(self):
        pass    # FIXME: implement

    def has_hw_sideband_modulation(self): # or provide software implementation in base class
        pass    # FIXME: implement




# this should also be usable for OpenQL (or other compilers)
# FIXME: allow instruments within instruments?

hal_configuration = """
    instruments [
        {
            "name": "ro_1",
            "class": "UHFQC",
            "init": {
                "device": "dev2216",
                "use_dio": true
            },
            "locations": [["q0.readout"], ["q2.readout"], ["q3.readout"], ["q4.readout"]],
        },
        {   "name": "ro_1_tx_lo",
            "class": "RohdeSchwarz_SGS100A",
            "init": {
                "address": "TCPIP0::192.168.0.73"
            },
            "locations": [
                [   "q0.readout.tx.lo", 
                    "q2.readout.tx.lo", 
                    "q3.readout.tx.lo", 
                    "q4.readout.tx.lo"
                ]
            ]
        },
        {
            "name": "mw_1",
            "class":"ZI_HDAWG8",
            "init": {
                "device": "dev8070"
            },
            "locations": [["q0.mw"], ["q2.mw"], ["q3.mw"], ["q4.mw"]],
        },
        {   "name": "mw_1_lo",
            "class": "RohdeSchwarz_SGS100A",
            "init": {
                "address": "TCPIP0::192.168.0.74"
            },
            "locations": [
                [   "q0.mw.lo", 
                    "q2.mw.lo", 
                    "q3.mw.lo", 
                    "q4.mw.lo"
                ]
            ]
        },
        {
            "name": "flux_0",
            "class":"ZI_HDAWG8",
            "init": {
                "device": "dev8070"
            },
            "locations": [["q0.flux"], ["q2.flux"], ["q3.flux"], ["q4.flux"], [], [], [], []],
        },
        {
            "name": "cc",
            "class":"CC",
            "init": {
                "class": "IPTransport",
                "init": TBD
            },
            "locations": [],    // define locations per slot, how to link to connected instrument, link to name?
        },
        {
            "name": "qwg_12",
            "class":"QuTech_AWG_Module",
            "init": {
                "address": "192.168.0.50"
            },
            "locations": [],
        },
        {
            "name": "fluxcurrent",
            "class":"QuTech_SPI_S4g_FluxCurrent",
            "init": {
                "address": "COM9",
                "channel_map": {
                   "FBL_D1": (15, 0),
                   "FBL_D2": (15, 1),
                   "FBL_D3": (15, 2),
                   "FBL_D4": (15, 3),
                   "FBL_X" : (4, 0),
                   "FBL_Z1": (4, 1),
                   "FBL_Z2": (4, 2),
                }
            },
            "locations": [],
        },
        {
            "name": "",
            "class":"",
            "init": {
            },
            "locations": [],
        }
    ]
"""




"""
HAL-driver
-   must be child of Element
-   init should not fail (e.g. because instrument is not responding or has some trouble), because that stops us from
    creating HAL object (properly)
    -   add open() for that, which should be configurable from JSON. Or: keep parameters in init, but postpone
        connecting to instrument to open (factory should only call constructors)
-   init and open should be 'fast'. Maybe we can at some point parallalize
-   we should be able to apply snapshot before opening?
-   parameter structure according to profile (user defined?). But, instrument drivers will probably be composite 
    (.awg[], .lo[], .acq.[]) and thus contain part of structure (or allow configuration for that)
-   adaptation/shim driver to rename parameters (and functions). Renaming, so there is only one storage location. May
    need some glue around parameters/add_parameter to allow redefinition of parameters


VSM = vsm.QuTechVSMModule(name='VSM', address='192.168.0.10', port=5025)
SH = sh.SignalHound_USB_SA124B('SH', dll_path='C:\Windows\System32\sa_api.dll')
    
"""
