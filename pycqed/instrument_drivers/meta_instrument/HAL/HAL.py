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

    def program(self, src: str):
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
        """
        switches SG on. Blocking call, all parameters must have settled and SG must be ready after call.
        FIXME: must also be efficient, see HAL below for different idea
        :return:
        """
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
    
FIXME: how to associate a composite element with an instrument unit/group:
- instrument driver can inherit from CE
- should allow replacing submodules by other instruments 
- no, instrument driver creates CEs for resources that are present

"""

class FluxOutput(Element): # BasebandOutput?
    # awg+seq
    # IIR filter
    # DC offset
    pass

class MicrowaveOutput(Element):
    def __init__(self, name: str):
        super().__init__(name)

        self._awg = AWG("awg")
        self.add_submodule('awg', self._awg)
        self._seq = Sequencer('seq')
        self.add_submodule('seq', self._seq)
        # sideband modulator
        # AM
        # PM
        self._mixer_corr = MixerCorrection('mc')
        self.add_submodule('mixer_corr', self._mixer_corr)
        self._lo = SignalGenerator('lo')
        self.add_submodule('lo', self._lo)

    def arm(self):
        self._lo.on()
        self._awg.start()

    def on(self):
        self._seq.start()

    def off(self):
        self._lo.off()  # FIXME: or (optionally) keep on for optimal latency when quickly switching on and off
        self._seq.stop()
        self._awg.stop()

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
- an instrument implements particular elements, e.g. one or more of tx.AWG, tx.filterIIR, tx.upconverter, rx.downconverter, rx.detector
- how to handle (e.g.) internal upconverter vs. external: attach instrument to root node, driver determines sub nodes.
  how to manage profiles then?
    - define allowed node structure: tree and leaf nodes, check JSON against this? Use JSON schema?
- terminology: 
    -   elements vs. nodes vs tree
        -   the actual variables live inside the elements, can persist through serialization
    -   locations (resources? targets?) stuff that you can control in real time? (Virtual) place where you can generate/measure signal
        A waveform can be output on 'q0.flux'. But we can also set q0.flux.amplitude, maybe from the sequencer, or from sw.
        What makes a node eligible to be a real-time target: the presence of an AWG at that point (or ACQ for measurement) (i.e. have certain parent class?)
        NB: also see ZI nodes, that can partly be set from seqc
        
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
    -   also sets parameters, which we dislike (separate parameters from structure)
    -   
    
- element structure created
    q*.tx.AWG
    q*.tx.upconverter 
    q*.tx.upconverter.lo
    q*.tx.filterIIR
    
    q*.rx.downconverter
    q*.rx.
    
    also:
    q*.instrument
    q*.group/channel
    
    - what is the type of q*? Qubit? Type of choice?
    - do we also have access to the calibration routines from Qubit object from these nodes? Or do we redesign these,
        using some base class
    
"""

class HAL:
    def __init__(self):
        self._instruments = []

    def register_element(self, element: Element):   # or just inherit from HAL
        pass    # FIXME: implement

    def from_JSON(self, config, simul: bool=False):
        for instr in config["instruments"]:
            print(instr["name"], instr["class"])


    def open(self):
        pass    # FIXME: implement

    def close(self):
        pass    # FIXME: implement

    def start(self):    # add parameter for user intent to limit instruments that need to be controlled
        # optimize latency. So first start everything apart from sequencers, then do a get_operation_complete() on
        # all, then start sequencers
        self._arm()
        self._get_operation_complete()
        self._on()
        pass    # FIXME: implement

    def stop(self):
        pass    # FIXME: implement

    def has_hw_sideband_modulation(self): # or provide software implementation in base class
        pass    # FIXME: implement

    def _get_operation_complete(self):
        pass    # FIXME: implement

    def _arm(self):
        pass    # FIXME: implement

    def _on(self):
        pass    # FIXME: implement





"""
HAL-driver
-   must be child of Element. No, see above: creates elements
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
-   functions
    -   get_output_latency
    -


VSM = vsm.QuTechVSMModule(name='VSM', address='192.168.0.10', port=5025)
SH = sh.SignalHound_USB_SA124B('SH', dll_path='C:\Windows\System32\sa_api.dll')
    
"""
