import numpy as np
import time
import logging
import warnings
import adaptive
import networkx as nx
import datetime
from collections import OrderedDict
import multiprocessing
from importlib import reload

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import (
    ManualParameter,
    InstrumentRefParameter,
    Parameter,
)

from pycqed.analysis import multiplexed_RO_analysis as mra
from pycqed.measurement import detector_functions as det
reload(det)

from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import tomography as tomo
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.utilities.general import check_keyboard_interrupt, print_exception

from pycqed.instrument_drivers.physical_instruments.QuTech_AWG_Module import (
    QuTech_AWG_Module,
)
#from pycqed.instrument_drivers.physical_instruments.QuTech_CCL import CCL
from pycqed.instrument_drivers.physical_instruments.QuTech_QCC import QCC
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC
import pycqed.analysis_v2.tomography_2q_v2 as tomo_v2

from pycqed.utilities import learner1D_minimizer as l1dm

log = logging.getLogger(__name__)

try:
    from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
    import pycqed.measurement.openql_experiments.multi_qubit_oql as mqo
    from pycqed.measurement.openql_experiments import clifford_rb_oql as cl_oql
    from pycqed.measurement.openql_experiments import openql_helpers as oqh
    from pycqed.measurement import cz_cost_functions as czcf

    reload(sqo)
    reload(mqo)
    reload(cl_oql)
    reload(oqh)
    reload(czcf)
except ImportError:
    log.warning('Could not import OpenQL')
    mqo = None
    sqo = None
    cl_oql = None
    oqh = None
    czcf = None


class DeviceCCL(Instrument):
    """
    Device object for systems controlled using the
    CCLight (CCL), QuMa based CC (QCC) or Distributed CC (CC).
    FIXME: class name is outdated
    """
    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        self.msmt_suffix = '_' + name

        self.add_parameter('qubits',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Lists(elt_validator=vals.Strings()))

        self.add_parameter('qubit_edges',
                           parameter_class=ManualParameter,
                           docstring="Denotes edges that connect qubits. "
                           "Used to define the device topology.",
                           initial_value=[[]],
                           vals=vals.Lists(elt_validator=vals.Lists()))

        self.add_parameter(
            'ro_lo_freq', unit='Hz',
            docstring=('Frequency of the common LO for all RO pulses.'),
            parameter_class=ManualParameter)

        # actually, it should be possible to build the integration
        # weights obeying different settings for different
        # qubits, but for now we use a fixed common value.
        self.add_parameter(
            "ro_acq_integration_length",
            initial_value=500e-9,
            vals=vals.Numbers(min_value=0, max_value=20e6),
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "ro_pow_LO",
            label="RO power LO",
            unit="dBm",
            initial_value=20,
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "ro_acq_averages",
            initial_value=1024,
            vals=vals.Numbers(min_value=0, max_value=1e6),
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "ro_acq_delay",
            unit="s",
            label="Readout acquisition delay",
            vals=vals.Numbers(min_value=0),
            initial_value=0,
            parameter_class=ManualParameter,
            docstring=(
                "The time between the instruction that trigger the"
                " readout pulse and the instruction that triggers the "
                "acquisition. The positive number means that the "
                "acquisition is started after the pulse is send."
            ),
        )

        self.add_parameter(
            "instr_MC",
            label="MeasurementControl",
            parameter_class=InstrumentRefParameter,
        )
        self.add_parameter(
            "instr_VSM",
            label="Vector Switch Matrix",
            parameter_class=InstrumentRefParameter,
        )
        self.add_parameter(
            "instr_CC",
            label="Central Controller",
            docstring=(
                "Device responsible for controlling the experiment"
                " using eQASM generated using OpenQL, in the near"
                " future will be the CC_Light."
            ),
            parameter_class=InstrumentRefParameter,
        )

        for i in range(3):  # S17 has 3 feedlines
            self.add_parameter(
                "instr_acq_{}".format(i), parameter_class=InstrumentRefParameter
            )
        # Two microwave AWGs are used for S17
        self.add_parameter("instr_AWG_mw_0", parameter_class=InstrumentRefParameter)
        self.add_parameter("instr_AWG_mw_1", parameter_class=InstrumentRefParameter)
        self.add_parameter("instr_AWG_mw_2", parameter_class=InstrumentRefParameter)
        self.add_parameter("instr_AWG_mw_3", parameter_class=InstrumentRefParameter)
        self.add_parameter("instr_AWG_mw_4", parameter_class=InstrumentRefParameter)

        self.add_parameter("instr_AWG_flux_0", parameter_class=InstrumentRefParameter)
        self.add_parameter("instr_AWG_flux_1", parameter_class=InstrumentRefParameter)
        self.add_parameter("instr_AWG_flux_2", parameter_class=InstrumentRefParameter)

        ro_acq_docstr = (
            "Determines what type of integration weights to use: "
            "\n\t SSB: Single sideband demodulation\n\t"
            'optimal: waveforms specified in "RO_acq_weight_func_I" '
            '\n\tand "RO_acq_weight_func_Q"'
        )

        self.add_parameter(
            "ro_acq_weight_type",
            initial_value="SSB",
            vals=vals.Enum("SSB", "optimal","optimal IQ"),
            docstring=ro_acq_docstr,
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "ro_acq_digitized",
            vals=vals.Bool(),
            initial_value=False,
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "cfg_openql_platform_fn",
            label="OpenQL platform configuration filename",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
        )

        self.add_parameter(
            "ro_always_all",
            docstring="If true, configures the UHFQC to RO all qubits "
            "independent of codeword received.",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
        )

        # Timing related parameters
        self.add_parameter(
            "tim_ro_latency_0",
            unit="s",
            label="Readout latency 0",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_ro_latency_1",
            unit="s",
            label="Readout latency 1",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_ro_latency_2",
            unit="s",
            label="Readout latency 2",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_flux_latency_0",
            unit="s",
            label="Flux latency 0",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_flux_latency_1",
            unit="s",
            label="Flux latency 1",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_flux_latency_2",
            unit="s",
            label="Flux latency 2",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_mw_latency_0",
            unit="s",
            label="Microwave latency 0",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_mw_latency_1",
            unit="s",
            label="Microwave latency 1",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_mw_latency_2",
            unit="s",
            label="Microwave latency 2",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_mw_latency_3",
            unit="s",
            label="Microwave latency 3",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "tim_mw_latency_4",
            unit="s",
            label="Microwave latency 4",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "dio_map",
            docstring="The map between DIO"
            " channel number and functionality (ro_x, mw_x, flux_x). "
            "From 2020-03-19 on, Requires to be configured by the user in each set up. "
            "For convenience here are the mapping for the devices with fixed mappings:\n"
            "CCL:\n"
            "    {\n"
            "        'ro_0': 1,\n"
            "        'ro_1': 2,\n"
            "        'flux_0': 3,\n"
            "        'mw_0': 4,\n"
            "        'mw_1': 5\n"
            "    }\n"
            "QCC:\n"
            "    {\n"
            "        'ro_0': 1,\n"
            "        'ro_1': 2,\n"
            "        'ro_2': 3,\n"
            "        'mw_0': 4,\n"
            "        'mw_1': 5,\n"
            "        'flux_0': 6,\n"
            "        'flux_1': 7,\n"
            "        'flux_2': 8,\n"
            "        'flux_3': 9,\n"
            "        'mw_2': 10,\n"
            "        'mw_3': 11\n"
            "        'mw_4': 12\n"
            "    }\n"
            "Tip: run `device.dio_map?` to print the docstring of this parameter",
            initial_value=None,
            set_cmd=self._set_dio_map,
            vals=vals.Dict(),
        )

    def _set_dio_map(self, dio_map_dict):
        allowed_keys = {"ro_", "mw_", "flux_"}
        for key in dio_map_dict:
            assert np.any(
                [a_key in key and len(key) > len(a_key) for a_key in allowed_keys]
            ), "Key `{}` must start with:" " `{}`!".format(key, list(allowed_keys))
        return dio_map_dict

    def _grab_instruments_from_qb(self):
        """
        initialize instruments that should only exist once from the first
        qubit. Maybe must be done in a more elegant way (at least check
        uniqueness).
        """

        qb = self.find_instrument(self.qubits()[0])
        self.instr_MC(qb.instr_MC())
        self.instr_VSM(qb.instr_VSM())
        self.instr_CC(qb.instr_CC())
        self.cfg_openql_platform_fn(qb.cfg_openql_platform_fn())

    def prepare_timing(self):
        """
        Responsible for ensuring timing is configured correctly.

        Takes parameters starting with `tim_` and uses them to set the correct
        latencies on the DIO ports of the CCL or QCC.

        N.B. latencies are set in multiples of 20ns in the DIO.
        Latencies shorter than 20ns are set as channel delays in the AWGs.
        These are set globally. If individual (per channel) setting of latency
        is required in the future, we can add this.

        """
        # 2. Setting the latencies
        cc = self.instr_CC.get_instr()
        if cc.IDN()['model']=='CCL':
            latencies = OrderedDict(
                [
                    ("ro_0", self.tim_ro_latency_0()),
                    ("ro_1", self.tim_ro_latency_1()),
                    # ('ro_2', self.tim_ro_latency_2()),
                    ("mw_0", self.tim_mw_latency_0()),
                    ("mw_1", self.tim_mw_latency_1()),
                    ("flux_0", self.tim_flux_latency_0())
                    # ('flux_1', self.tim_flux_latency_1()),
                    # ('flux_2', self.tim_flux_latency_2()),
                    # ('mw_2', self.tim_mw_latency_2()),
                    # ('mw_3', self.tim_mw_latency_3()),
                    # ('mw_4', self.tim_mw_latency_4())]
                ]
            )
        else:
            latencies = OrderedDict(
                [
                    ("ro_0", self.tim_ro_latency_0()),
                    ("ro_1", self.tim_ro_latency_1()),
                    ("ro_2", self.tim_ro_latency_2()),
                    ("flux_0", self.tim_flux_latency_0()),
                    ("flux_1", self.tim_flux_latency_1()),
                    ("flux_2", self.tim_flux_latency_2()),
                    ("mw_0", self.tim_mw_latency_0()),
                    ("mw_1", self.tim_mw_latency_1()),
                    ("mw_2", self.tim_mw_latency_2()),
                    ("mw_3", self.tim_mw_latency_3()),
                    ("mw_4", self.tim_mw_latency_4()),
                ]
            )

        # NB: Mind that here number precision matters a lot!
        # Tripple check everything if any changes are to be made

        # Substract lowest value to ensure minimal latency is used.
        # note that this also supports negative delays (which is useful for
        # calibrating)
        lowest_value = min(latencies.values())
        for key, val in latencies.items():
            # Align to minimum and change to ns to avoid number precision problems
            # The individual multiplications are on purpose
            latencies[key] = val * 1e9 - lowest_value * 1e9

        # Only apply fine latencies above 1 ps (HDAWG8 minimum fine delay)
        ns_tol = 1e-3

        # ensuring that RO latency is a multiple of 20 ns as the UHFQC does
        # not have a fine timing control.
        ro_latency_modulo_20 = latencies["ro_0"] % 20
        # `% 20` is for the case ro_latency_modulo_20 == 20 ns
        correction_for_multiple = (20 - ro_latency_modulo_20) % 20
        if correction_for_multiple >= ns_tol:  # at least one 1 ps
            # Only apply corrections if they are significant
            for key, val in latencies.items():
                latencies[key] = val + correction_for_multiple

        # Setting the latencies in the CCL
        # Iterate over keys in dio_map as this ensures only relevant
        # timing setting are set.
        for lat_key, dio_ch in self.dio_map().items():
            lat = latencies[lat_key]
            lat_coarse = int(np.round(lat) // 20)  # Convert to CC dio value
            lat_fine = lat % 20
            lat_fine = lat_fine * 1e-9 if lat_fine <= 20 - ns_tol else 0
            log.debug(
                "Setting `dio{}_out_delay` for `{}` to `{}`. (lat_fine: {:4g})".format(
                    dio_ch, lat_key, lat_coarse, lat_fine
                )
            )
            cc.set("dio{}_out_delay".format(dio_ch), lat_coarse)

            # RO devices do not support fine delay setting.
            if "mw" in lat_key or "flux" in lat_key:
                # Check name to prevent crash when instrument not specified
                AWG_name = self.get("instr_AWG_{}".format(lat_key))
                if AWG_name is not None:
                    AWG = self.find_instrument(AWG_name)
                    using_QWG = AWG.__class__.__name__ == "QuTech_AWG_Module"
                    if not using_QWG:
                        # All channels are set globally from the device object.
                        AWG.stop()
                        for i in range(8):  # assumes the AWG is an HDAWG
                            log.debug(
                                "Setting `sigouts_{}_delay` to {:4g}"
                                " in {}".format(i, lat_fine, AWG.name)
                            )
                            AWG.set("sigouts_{}_delay".format(i), lat_fine)
                        AWG.start()
                        ch_not_ready = 8
                        while ch_not_ready > 0:
                            ch_not_ready = 0
                            for i in range(8):
                                ch_not_ready += AWG.geti("sigouts/{}/busy".format(i))
                            check_keyboard_interrupt()

    def prepare_fluxing(self, qubits):
        for qb_name in qubits:
            qb = self.find_instrument(qb_name)
            try:
                fl_lutman = qb.instr_LutMan_Flux.get_instr()
                fl_lutman.load_waveforms_onto_AWG_lookuptable()
            except Exception as e:
                warnings.warn("Could not load flux pulses for {}".format(qb))
                warnings.warn("Exception {}".format(e))

    def prepare_readout(self, qubits, reduced: bool = False):
        """
        Configures readout for specified qubits.

        Args:
            qubits (list of str):
                list of qubit names that have to be prepared
        """
        log.info('Configuring readout for {}'.format(qubits))
        if not reduced:
            self._prep_ro_sources(qubits=qubits)

        acq_ch_map = self._prep_ro_assign_weights(qubits=qubits)
        self._prep_ro_integration_weights(qubits=qubits)
        if not reduced:
            self._prep_ro_pulses(qubits=qubits) 
            self._prep_ro_instantiate_detectors(qubits=qubits, acq_ch_map=acq_ch_map) 
            
        # TODO:
        # - update global readout parameters (relating to mixer settings)
        #  the pulse mixer
        #       - ro_mixer_alpha, ro_mixer_phi
        #       - ro_mixer_offs_I, ro_mixer_offs_Q
        #       - ro_acq_delay
        #  the acquisition mixer
        # commented out because it conflicts with setting in the qubit object

        #     # These parameters affect all resonators.
        #     # Should not be part of individual qubits
        #     ro_lm.set('pulse_type', 'M_' + qb.ro_pulse_type())
        #     ro_lm.set('mixer_alpha',
        #               qb.ro_pulse_mixer_alpha())
        #     ro_lm.set('mixer_phi',
        #               qb.ro_pulse_mixer_phi())
        #     ro_lm.set('mixer_offs_I', qb.ro_pulse_mixer_offs_I())
        #     ro_lm.set('mixer_offs_Q', qb.ro_pulse_mixer_offs_Q())
        #     ro_lm.acquisition_delay(qb.ro_acq_delay())

        #     ro_lm.set_mixer_offsets()

    def _prep_ro_sources(self, qubits):
        """
        turn on and configure the RO LO's of all qubits to be measured and
        update the modulation frequency of all qubits.
        """
        # This device object works under the assumption that a single LO
        # is used to drive all readout lines.
        LO = self.find_instrument(qubits[0]).instr_LO_ro.get_instr()
        LO.frequency.set(self.ro_lo_freq())
        LO.power(self.ro_pow_LO())
        LO.on()

        for qb_name in qubits:
            qb = self.find_instrument(qb_name)
            # set RO modulation to use common LO frequency
            mod_freq = qb.ro_freq() - self.ro_lo_freq()
            log.info("Setting modulation freq of {} to {}".format(qb_name, mod_freq))
            qb.ro_freq_mod(mod_freq)

            LO_q = qb.instr_LO_ro.get_instr()
            if LO_q is not LO:
                raise ValueError("Expect a single LO to drive all feedlines")

    def _prep_ro_assign_weights(self, qubits):
        """
        Assign acquisition weight channels to the different qubits.

        Args:
            qubits (list of str):
                list of qubit names that have to be prepared

        Returns
            acq_ch_map (dict)
                a mapping of acquisition instruments and channels used
                for each qubit.

        The assignment is done based on  the acq_instr used for each qubit
        and the number of channels used per qubit. N.B. This method of mapping
        has no implicit feedline or UHFQC contraint built in.

        The mapping of acq_channels to qubits is stored in self._acq_ch_map
        for debugging purposes.
        """
        log.info('Setting up acquisition channels')
        if self.ro_acq_weight_type() == 'optimal':
            log.debug('ro_acq_weight_type = "optimal" using 1 ch per qubit')
            nr_of_acq_ch_per_qubit = 1
        else:
            log.debug('Using 2 ch per qubit')
            nr_of_acq_ch_per_qubit = 2

        acq_ch_map = {}
        for qb_name in qubits:
            qb = self.find_instrument(qb_name)
            acq_instr = qb.instr_acquisition()
            if not acq_instr in acq_ch_map.keys():
                acq_ch_map[acq_instr] = {}

            assigned_weight = len(acq_ch_map[acq_instr]) * nr_of_acq_ch_per_qubit
            log.info(
                "Assigning {} w{} to qubit {}".format(
                    acq_instr, assigned_weight, qb_name
                )
            )
            acq_ch_map[acq_instr][qb_name] = assigned_weight
            if assigned_weight > 9:
                # There are only 10 acq_weight_channels per UHF.
                # use optimal ro weights or read out less qubits.
                raise ValueError("Trying to assign too many acquisition weights")

            qb.ro_acq_weight_chI(assigned_weight)
            # even if the mode does not use Q weight, we still assign this
            # this is for when switching back to the qubit itself
            qb.ro_acq_weight_chQ(assigned_weight + 1)

        log.info("acq_channel_map: \n\t{}".format(acq_ch_map))

        log.info("Clearing UHF correlation settings")
        for acq_instr_name in acq_ch_map.keys():
            self.find_instrument(acq_instr).reset_correlation_params()
            self.find_instrument(acq_instr).reset_crosstalk_matrix()

        # Stored as a private attribute for debugging purposes.
        self._acq_ch_map = acq_ch_map

        return acq_ch_map

    def _prep_ro_integration_weights(self, qubits):
        """
        Set the acquisition integration weights on each channel.

        Args:
            qubits (list of str):
                list of qubit names that have to be prepared
        """
        log.info("Setting integration weights")

        if self.ro_acq_weight_type() == "SSB":
            log.info("using SSB weights")
            for qb_name in qubits:
                qb = self.find_instrument(qb_name)
                acq_instr = qb.instr_acquisition.get_instr()

                acq_instr.prepare_SSB_weight_and_rotation(
                    IF=qb.ro_freq_mod(),
                    weight_function_I=qb.ro_acq_weight_chI(),
                    weight_function_Q=qb.ro_acq_weight_chQ(),
                )

        elif 'optimal' in self.ro_acq_weight_type():
            log.info("using optimal weights")
            for qb_name in qubits:
                qb = self.find_instrument(qb_name)
                acq_instr = qb.instr_acquisition.get_instr()
                opt_WI = qb.ro_acq_weight_func_I()
                opt_WQ = qb.ro_acq_weight_func_Q()
                # N.B. no support for "delay samples" relating to #63
                if opt_WI is None or opt_WQ is None:
                    # do not raise an exception as it should be possible to
                    # run input avg experiments to calibrate the optimal weights.
                    log.warning("No optimal weights defined for"
                                " {}, not updating weights".format(qb_name))
                else:
                    acq_instr.set("qas_0_integration_weights_{}_real".format(
                                qb.ro_acq_weight_chI()), opt_WI,)
                    acq_instr.set("qas_0_integration_weights_{}_imag".format(
                                qb.ro_acq_weight_chI()), opt_WQ,)
                    acq_instr.set("qas_0_rotations_{}".format(
                                qb.ro_acq_weight_chI()), 1.0 - 1.0j)
                    if self.ro_acq_weight_type() == 'optimal IQ':
                        print('setting the optimal Q')
                        acq_instr.set('qas_0_integration_weights_{}_real'.format(
                            qb.ro_acq_weight_chQ()), opt_WQ)
                        acq_instr.set('qas_0_integration_weights_{}_imag'.format(
                            qb.ro_acq_weight_chQ()), opt_WI)
                        acq_instr.set('qas_0_rotations_{}'.format(
                            qb.ro_acq_weight_chQ()), 1.0 + 1.0j)

                if self.ro_acq_digitized():
                    # Update the RO theshold
                    if (qb.ro_acq_rotated_SSB_when_optimal() and 
                            abs(qb.ro_acq_threshold()) > 32):
                        threshold = 32
                        log.warning(
                            "Clipping ro_acq threshold of {} to 32".format(qb.name))
                        # working around the limitation of threshold in UHFQC
                        # which cannot be >abs(32).
                        # See also self._prep_ro_integration_weights scaling the weights
                    else:
                        threshold = qb.ro_acq_threshold()

                    qb.instr_acquisition.get_instr().set(
                        "qas_0_thresholds_{}_level".format(qb.ro_acq_weight_chI()),
                        threshold,
                    )
                    log.info("Setting threshold of {} to {}".format(qb.name, threshold))

            # Note, no support for optimal IQ in mux RO
            # Note, no support for ro_cq_rotated_SSB_when_optimal
        else:
            raise NotImplementedError('ro_acq_weight_type "{}" not supported'.format(
                self.ro_acq_weight_type()))

    def _prep_ro_pulses(self, qubits):
        """
        Configure the ro lutmans.

        The configuration includes
            - setting the right parameters for all readout pulses
            - uploading the waveforms to the UHFQC
            - setting the "resonator_combinations" that determine allowed pulses
                N.B. by convention we support all individual readouts and
                the readout all qubits instruction.
        """

        ro_lms = []

        resonators_in_lm = {}

        for qb_name in qubits:
            qb = self.find_instrument(qb_name)
            # qubit and resonator number are identical
            res_nr = qb.cfg_qubit_nr()
            ro_lm = qb.instr_LutMan_RO.get_instr()

            # Add resonator to list of resonators in lm
            if ro_lm not in ro_lms:
                ro_lms.append(ro_lm)
                resonators_in_lm[ro_lm.name] = []
            resonators_in_lm[ro_lm.name].append(res_nr)

            # update parameters of RO pulse in ro lutman

            # ro_freq_mod was updated in self._prep_ro_sources
            ro_lm.set("M_modulation_R{}".format(res_nr), qb.ro_freq_mod())

            ro_lm.set("M_length_R{}".format(res_nr), qb.ro_pulse_length())
            ro_lm.set("M_amp_R{}".format(res_nr), qb.ro_pulse_amp())
            ro_lm.set("M_delay_R{}".format(res_nr), qb.ro_pulse_delay())
            ro_lm.set("M_phi_R{}".format(res_nr), qb.ro_pulse_phi())
            ro_lm.set("M_down_length0_R{}".format(res_nr), qb.ro_pulse_down_length0())
            ro_lm.set("M_down_amp0_R{}".format(res_nr), qb.ro_pulse_down_amp0())
            ro_lm.set("M_down_phi0_R{}".format(res_nr), qb.ro_pulse_down_phi0())
            ro_lm.set("M_down_length1_R{}".format(res_nr), qb.ro_pulse_down_length1())
            ro_lm.set("M_down_amp1_R{}".format(res_nr), qb.ro_pulse_down_amp1())
            ro_lm.set("M_down_phi1_R{}".format(res_nr), qb.ro_pulse_down_phi1())

        for ro_lm in ro_lms:
            # list comprehension should result in a list with each
            # individual resonator + the combination of all simultaneously
            resonator_combs = [[r] for r in resonators_in_lm[ro_lm.name]] + \
                [resonators_in_lm[ro_lm.name]]
            log.info('Setting resonator combinations for {} to {}'.format(
                ro_lm.name, resonator_combs))

            # FIXME: temporary fix so device object doesnt mess with 
            #       the resonator combinations. Better strategy should be implemented
            #ro_lm.resonator_combinations(resonator_combs)
            ro_lm.load_DIO_triggered_sequence_onto_UHFQC()

    def get_correlation_detector(self, qubits: list,
                                 single_int_avg: bool = False,
                                 seg_per_point: int = 1,
                                 always_prepare: bool = False):
        if self.ro_acq_digitized():
            log.warning('Digitized mode gives bad results')
        if len(qubits) != 2:
            raise ValueError("Not possible to define correlation "
                             "detector for more than two qubits")
        if self.ro_acq_weight_type() != 'optimal':
            raise ValueError('Correlation detector only works '
                             'with optimal weights')
        q0 = self.find_instrument(qubits[0])
        q1 = self.find_instrument(qubits[1])

        w0 = q0.ro_acq_weight_chI()
        w1 = q1.ro_acq_weight_chI()

        if q0.instr_acquisition.get_instr() == q1.instr_acquisition.get_instr():
            d = det.UHFQC_correlation_detector(
                UHFQC=q0.instr_acquisition.get_instr(),  # <- hack line
                thresholding=self.ro_acq_digitized(),
                AWG=self.instr_CC.get_instr(),
                channels=[w0, w1], correlations=[(w0, w1)],
                nr_averages=self.ro_acq_averages(),
                integration_length=q0.ro_acq_integration_length(),
                single_int_avg=single_int_avg,
                seg_per_point=seg_per_point,
                always_prepare=always_prepare)
            d.value_names = ['{} w{}'.format(qubits[0], w0),
                             '{} w{}'.format(qubits[1], w1),
                             'Corr ({}, {})'.format(qubits[0], qubits[1])]
        else:
            # This should raise a ValueError but exists for legacy reasons.
            # WARNING DEBUG HACK
            d = self.get_int_avg_det(qubits=qubits,
                                     single_int_avg=single_int_avg,
                                     seg_per_point=seg_per_point,
                                     always_prepare=always_prepare)

        return d

    def get_int_logging_detector(self, qubits=None, result_logging_mode='raw'):

        if self.ro_acq_weight_type() == 'SSB':
            result_logging_mode = 'raw'
        elif 'optimal' in self.ro_acq_weight_type():
            # lin_trans includes
            result_logging_mode = 'lin_trans'
            if self.ro_acq_digitized():
                result_logging_mode = 'digitized'

        log.info('Setting result logging mode to {}'.format(result_logging_mode))

        if self.ro_acq_weight_type() != "optimal":
            acq_ch_map = _acq_ch_map_to_IQ_ch_map(self._acq_ch_map)
        else:
            acq_ch_map = self._acq_ch_map

        int_log_dets = []
        for i, acq_instr_name in enumerate(self._acq_ch_map.keys()):
            if i == 0:
                CC = self.instr_CC.get_instr()
            else:
                CC = None

            UHFQC = self.find_instrument(acq_instr_name)
            int_log_dets.append(
                det.UHFQC_integration_logging_det(
                    channels=list(acq_ch_map[acq_instr_name].values()),
                    value_names=list(acq_ch_map[acq_instr_name].keys()),
                    UHFQC=UHFQC, AWG=CC,
                    result_logging_mode=result_logging_mode,
                    integration_length=self.ro_acq_integration_length(),
                )
            )

        int_log_det = det.Multi_Detector_UHF(
            detectors=int_log_dets, detector_labels=list(self._acq_ch_map.keys())
        )

        return int_log_det

    def _prep_ro_instantiate_detectors(self, qubits, acq_ch_map):
        """
        Instantiate acquisition detectors.

        Args:
            qubits (list of str):
                list of qubit names that have to be prepared
            acq_ch_map (dict)
                dict specifying the mapping
        """
        log.info("Instantiating readout detectors")
        self.input_average_detector = self.get_input_avg_det()
        self.int_avg_det = self.get_int_avg_det()
        self.int_avg_det_single = self.get_int_avg_det(single_int_avg=True)
        self.int_log_det = self.get_int_logging_detector()

        if len(qubits) == 2 and self.ro_acq_weight_type() == 'optimal':
            self.corr_det = self.get_correlation_detector(qubits=qubits)
        else:
            self.corr_det = None

    def get_input_avg_det(self, **kw):
        """
        Create an input average multi detector based.

        The input average multi detector is based on the self._acq_ch_map
        that gets set when calling self.prepare_readout(qubits).
        """
        input_average_detectors = []

        for i, acq_instr_name in enumerate(self._acq_ch_map.keys()):
            if i == 0:
                CC = self.instr_CC.get_instr()
            else:
                CC = None
            UHFQC = self.find_instrument(acq_instr_name)

            input_average_detectors.append(
                det.UHFQC_input_average_detector(
                    UHFQC=UHFQC,
                    AWG=CC,
                    nr_averages=self.ro_acq_averages(),
                    nr_samples=int(self.ro_acq_integration_length() * 1.8e9),
                ),
                **kw
            )

        input_average_detector = det.Multi_Detector_UHF(
            detectors=input_average_detectors,
            detector_labels=list(self._acq_ch_map.keys()),
        )

        return input_average_detector

    def get_int_avg_det(self, qubits=None, **kw):
        """
        """
        if qubits is not None:
            log.warning("qubits is deprecated")

        if self.ro_acq_weight_type() == "SSB":
            result_logging_mode = "raw"
        elif 'optimal' in self.ro_acq_weight_type():
            # lin_trans includes
            result_logging_mode = "lin_trans"
            if self.ro_acq_digitized():
                result_logging_mode = "digitized"

        log.info("Setting result logging mode to {}".format(result_logging_mode))

        if self.ro_acq_weight_type() != "optimal":
            acq_ch_map = _acq_ch_map_to_IQ_ch_map(self._acq_ch_map)
        else:
            acq_ch_map = self._acq_ch_map

        int_avg_dets = []
        for i, acq_instr_name in enumerate(acq_ch_map.keys()):
            # The master detector is the one that holds the CC object
            if i == 0:
                CC = self.instr_CC.get_instr()
            else:
                CC = None
            int_avg_dets.append(
                det.UHFQC_integrated_average_detector(
                    channels=list(acq_ch_map[acq_instr_name].values()),
                    value_names=list(acq_ch_map[acq_instr_name].keys()),
                    UHFQC=self.find_instrument(acq_instr_name),
                    AWG=CC,
                    result_logging_mode=result_logging_mode,
                    nr_averages=self.ro_acq_averages(),
                    integration_length=self.ro_acq_integration_length(), **kw
                )
            )

        int_average_detector = det.Multi_Detector_UHF(
            detectors=int_avg_dets, detector_labels=list(self._acq_ch_map.keys())
        )
        return int_average_detector

    def _prep_td_configure_VSM(self):
        """
        turn off all VSM channels and then use qubit settings to
        turn on the required channels again.
        """

        # turn all channels on all VSMs off
        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)
            VSM = qb.instr_VSM.get_instr()
            # VSM.set_all_switches_to('OFF')  # FIXME: commented out

        # turn the desired channels on
        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)
            log

            # Configure VSM
            # N.B. This configure VSM block is geared specifically to the
            # Duplexer/BlueBox VSM
            # FIXME: code below commented out
            # VSM = qb.instr_VSM.get_instr()
            # Gin = qb.mw_vsm_ch_in()
            # Din = qb.mw_vsm_ch_in()
            # out = qb.mw_vsm_mod_out()

            # VSM.set('in{}_out{}_switch'.format(Gin, out), qb.mw_vsm_switch())
            # VSM.set('in{}_out{}_switch'.format(Din, out), qb.mw_vsm_switch())

            # VSM.set('in{}_out{}_att'.format(Gin, out), qb.mw_vsm_G_att())
            # VSM.set('in{}_out{}_att'.format(Din, out), qb.mw_vsm_D_att())
            # VSM.set('in{}_out{}_phase'.format(Gin, out), qb.mw_vsm_G_phase())
            # VSM.set('in{}_out{}_phase'.format(Din, out), qb.mw_vsm_D_phase())

            # self.instr_CC.get_instr().set(
            #     'vsm_channel_delay{}'.format(qb.cfg_qubit_nr()),
            #     qb.mw_vsm_delay())

    def prepare_for_timedomain(self, qubits: list, reduced: bool = False):
        """
        Prepare setup for a timedomain experiment:

        Args:
            qubits (list of str):
                list of qubit names that have to be prepared
        """
        self.prepare_readout(qubits=qubits, reduced=reduced)
        if reduced:
            return
        if self.find_instrument(qubits[0]).instr_LutMan_Flux() != None:
            self.prepare_fluxing(qubits=qubits)
        self.prepare_timing()

        for qb_name in qubits:
            qb = self.find_instrument(qb_name)
            qb._prep_td_sources()
            qb._prep_mw_pulses()

        # self._prep_td_configure_VSM()

    ########################################################
    # Measurement methods
    ########################################################

    def measure_conditional_oscillation(
        self,
        q0: str,
        q1: str,
        q2: str = None,
        q3: str = None,
        flux_codeword="cz",
        flux_codeword_park=None,
        parked_qubit_seq=None,
        downsample_swp_points=1,  # x2 and x3 available
        prepare_for_timedomain=True,
        MC=None,
        disable_cz: bool = False,
        disabled_cz_duration_ns: int = 60,
        cz_repetitions: int = 1,
        wait_time_before_flux_ns: int = 0,
        wait_time_after_flux_ns: int = 0,
        disable_parallel_single_q_gates: bool = False,
        label="",
        verbose=True,
        disable_metadata=False,
        extract_only=False,
    ):
        """
        Measures the "conventional cost function" for the CZ gate that
        is a conditional oscillation. In this experiment the conditional phase
        in the two-qubit Cphase gate is measured using Ramsey-lie sequence.
        Specifically qubit q0 is prepared in the superposition, while q1 is in 0 or 1 state.
        Next the flux pulse is applied. Finally pi/2 afterrotation around various axes
        is applied to q0, and q1 is flipped back (if neccessary) to 0 state.
        Plotting the probabilities of the zero state for each qubit as a function of
        the afterrotation axis angle, and comparing case of q1 in 0 or 1 state, enables to
        measure the conditional phase and estimale the leakage of the Cphase gate.

        Refs:
        Rol arXiv:1903.02492, Suppl. Sec. D

        Args:
            q0 (str):
                target qubit name (i.e. the qubit in the superposition state)

            q1 (str):
                control qubit name (i.e. the qubit remaining in 0 or 1 state)
            q2, q3 (str):
                names of optional extra qubit to either park or apply a CZ to.
            flux_codeword (str):
                the gate to be applied to the qubit pair q0, q1
            flux_codeword_park (str):
                optionally park qubits q2 (and q3) with either a 'park' pulse
                (single qubit operation on q2) or a 'cz' pulse on q2-q3.
                NB: depending on the CC configurations the parking can be
                implicit in the main `cz`
            prepare_for_timedomain (bool):
                should the insruments be reconfigured for time domain measurement
            disable_cz (bool):
                execute the experiment with no flux pulse applied
            disabled_cz_duration_ns (int):
                waiting time to emulate the flux pulse
            wait_time_after_flux_ns (int):
                additional waiting time (in ns) after the flux pulse, before
                the final afterrotations

        """
        if MC is None:
            MC = self.instr_MC.get_instr()
        assert q0 in self.qubits()
        assert q1 in self.qubits()
        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()
        list_qubits_used = [q0, q1]
        if q2 is None:
            q2idx = None
        else:
            q2idx = self.find_instrument(q2).cfg_qubit_nr()
            list_qubits_used.append(q2)
        if q3 is None:
            q3idx = None
        else:
            q3idx = self.find_instrument(q3).cfg_qubit_nr()
            list_qubits_used.append(q3)

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=list_qubits_used)
            for q in list_qubits_used:  #only on the CZ qubits we add the ef pulses
                mw_lutman = self.find_instrument(q).instr_LutMan_MW.get_instr()
                lm = mw_lutman.LutMap()
                # we hardcode the X on the ef transition to CW 31 here.
                lm[31] = {"name": "rX12", "theta": 180, "phi": 0, "type": "ef"}
                # load_phase_pulses will also upload other waveforms
                mw_lutman.load_phase_pulses_to_AWG_lookuptable()
                mw_lutman.load_waveforms_onto_AWG_lookuptable(
                    regenerate_waveforms=True)

        # These are hardcoded angles in the mw_lutman for the AWG8
        # only x2 and x3 downsample_swp_points available
        angles = np.arange(0, 341, 20 * downsample_swp_points)

        if parked_qubit_seq is None:
            parked_qubit_seq = "ramsey" if q2 is not None else "ground"

        p = mqo.conditional_oscillation_seq(
            q0idx,
            q1idx,
            q2idx,
            q3idx,
            platf_cfg=self.cfg_openql_platform_fn(),
            disable_cz=disable_cz,
            disabled_cz_duration=disabled_cz_duration_ns,
            angles=angles,
            wait_time_before_flux=wait_time_before_flux_ns,
            wait_time_after_flux=wait_time_after_flux_ns,
            flux_codeword=flux_codeword,
            flux_codeword_park=flux_codeword_park,
            cz_repetitions=cz_repetitions,
            parked_qubit_seq=parked_qubit_seq,
            disable_parallel_single_q_gates=disable_parallel_single_q_gates
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name="Phase",
            unit="deg",
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(p.sweep_points)

        measured_qubits = [q0, q1]
        if q2 is not None:
            measured_qubits.append(q2)
        if q3 is not None:
            measured_qubits.append(q3)

        MC.set_detector_function(self.get_int_avg_det(qubits=measured_qubits))

        MC.run(
            "conditional_oscillation_{}_{}_&_{}_{}_x{}_wb{}_wa{}{}{}".format(
                q0, q1, q2, q3, cz_repetitions,
                wait_time_before_flux_ns, wait_time_after_flux_ns,
                self.msmt_suffix, label,
            ),
            disable_snapshot_metadata=disable_metadata,
        )

        # [2020-06-24] parallel cz not supported (yet)
        # should be implemented by just running the analysis twice with
        # corresponding channels

        options_dict = {
            'ch_idx_osc': 0,
            'ch_idx_spec': 1
        }

        if q2 is not None:
            options_dict['ch_idx_park'] = 2

        a = ma2.Conditional_Oscillation_Analysis(
            options_dict=options_dict,
            extract_only=extract_only)

        return a

    def measure_two_qubit_grovers_repeated(
        self,
        qubits: list,
        nr_of_grover_iterations=40,
        prepare_for_timedomain=True,
        MC=None,
    ):

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        for q in qubits:
            assert q in self.qubits()

        q0idx = self.find_instrument(qubits[-1]).cfg_qubit_nr()
        q1idx = self.find_instrument(qubits[-2]).cfg_qubit_nr()

        p = mqo.grovers_two_qubits_repeated(
            qubits=[q1idx, q0idx],
            nr_of_grover_iterations=nr_of_grover_iterations,
            platf_cfg=self.cfg_openql_platform_fn(),
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())
        d = self.get_correlation_detector()
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_of_grover_iterations))
        MC.set_detector_function(d)
        MC.run(
            "Grovers_two_qubit_repeated_{}_{}{}".format(
                qubits[-2], qubits[-1], self.msmt_suffix
            )
        )

        a = ma.MeasurementAnalysis()
        return a

    def measure_two_qubit_tomo_bell(
        self,
        qubits: list,
        bell_state=0,
        wait_after_flux=None,
        analyze=True,
        close_fig=True,
        prepare_for_timedomain=True,
        MC=None,
        label="",
        shots_logging: bool = False,
        shots_per_meas=2 ** 16,
        flux_codeword="cz"
    ):
        """
        Prepares and performs a tomography of the one of the bell states, indicated
        by its index.

        Args:
            bell_state (int):
                index of prepared bell state
                0 -> |Phi_m>=|00>-|11>
                1 -> |Phi_p>=|00>+|11>
                2 -> |Psi_m>=|01>-|10>
                3 -> |Psi_p>=|01>+|10>

            qubits (list):
                list of names of the target qubits

            wait_after_flux (float):
                wait time (in seconds) after the flux pulse and
                after-rotation before tomographic rotations
            shots_logging (bool):
                if False uses correlation mode to acquire shots for tomography.
                if True uses single shot mode to acquire shots.
        """
        q0 = qubits[0]
        q1 = qubits[1]

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[q0, q1])
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        assert q1 in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()

        p = mqo.two_qubit_tomo_bell(
            bell_state,
            q0idx,
            q1idx,
            wait_after_flux=wait_after_flux,
            platf_cfg=self.cfg_openql_platform_fn(),
            flux_codeword=flux_codeword
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())
        MC.set_sweep_function(s)
        # 36 tomo rotations + 7*4 calibration points
        cases = np.arange(36 + 7 * 4)
        if not shots_logging:
            d = self.get_correlation_detector([q0, q1])
            MC.set_sweep_points(cases)
            MC.set_detector_function(d)
            MC.run("TwoQubitBellTomo_{}_{}{}".format(q0, q1, self.msmt_suffix) + label)
            if analyze:
                a = tomo.Tomo_Multiplexed(
                    label="Tomo",
                    MLE=True,
                    target_bell=bell_state,
                    single_shots=False,
                    q0_label=q0,
                    q1_label=q1,

                )
                return a

        else:
            nr_cases = len(cases)
            d = self.get_int_logging_detector(qubits)
            nr_shots = self.ro_acq_averages() * nr_cases
            shots_per_meas = int(
                np.floor(np.min([shots_per_meas, nr_shots]) / nr_cases) * nr_cases
            )
            d.set_child_attr("nr_shots", shots_per_meas)

            MC.set_sweep_points(np.tile(cases, self.ro_acq_averages()))
            MC.set_detector_function(d)
            MC.run(
                "TwoQubitBellTomo_{}_{}{}".format(q0, q1, self.msmt_suffix) + label,
                bins=cases,
            )

    def measure_two_qubit_allxy(
        self,
        q0: str,
        q1: str,
        sequence_type="sequential",
        replace_q1_pulses_with: str = None,
        repetitions: int = 2,
        analyze: bool = True,
        close_fig: bool = True,
        detector: str = "correl",
        prepare_for_timedomain: bool = True,
        MC=None
    ):
        """
        Perform AllXY measurement simultaneously of two qubits (c.f. measure_allxy
        method of the Qubit class). Order in which the mw pulses are executed
        can be varied.

        For detailed description of the (single qubit) AllXY measurement
        and symptomes of different errors see PhD thesis
        by Matthed Reed (2013, Schoelkopf lab), pp. 124.
        https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf

        Args:
            q0 (str):
                first quibit to perform allxy measurement on

            q1 (str):
                second quibit to perform allxy measurement on

            replace_q1_pulses_with (str):
                replaces all gates for q1 with the specified gate
                main use case: replace with "i" or "rx180" for crosstalks
                assessments

            sequence_type (str) : Describes the timing/order of the pulses.
                options are: sequential | interleaved | simultaneous | sandwiched
                           q0|q0|q1|q1   q0|q1|q0|q1     q01|q01       q1|q0|q0|q1
                describes the order of the AllXY pulses
        """
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[q0, q1])
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        assert q1 in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()

        p = mqo.two_qubit_AllXY(
            q0idx,
            q1idx,
            platf_cfg=self.cfg_openql_platform_fn(),
            sequence_type=sequence_type,
            replace_q1_pulses_with=replace_q1_pulses_with,
            repetitions=repetitions,
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())

        if detector == "correl":
            d = self.get_correlation_detector([q0, q1])
        elif detector == "int_avg":
            d = self.get_int_avg_det(qubits=[q0, q1])
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(21 * repetitions))
        MC.set_detector_function(d)
        MC.run("TwoQubitAllXY_{}_{}_{}_q1_repl={}{}".format(
            q0, q1, sequence_type, replace_q1_pulses_with,
            self.msmt_suffix))
        if analyze:
            a = ma.MeasurementAnalysis(close_main_fig=close_fig)
            a = ma2.Basic1DAnalysis()
        return a

    def measure_two_qubit_allXY_crosstalk(
        self, q0: str,
        q1: str,
        q1_replace_cases: list = [
            None, "i", "rx180", "rx180", "rx180"
        ],
        sequence_type_cases: list = [
            'sequential', 'sequential', 'sequential', 'simultaneous', 'sandwiched'
        ],
        repetitions: int = 1,
        **kw
    ):
        timestamps = []
        legend_labels = []

        for seq_type, q1_replace in zip(sequence_type_cases, q1_replace_cases):
            a = self.measure_two_qubit_allxy(
                q0=q0,
                q1=q1,
                replace_q1_pulses_with=q1_replace,
                sequence_type=seq_type,
                repetitions=repetitions,
                **kw)
            timestamps.append(a.timestamps[0])
            legend_labels.append("{}, {} replace: {}".format(seq_type, q1, q1_replace))

        a_full = ma2.Basic1DAnalysis(
            t_start=timestamps[0],
            t_stop=timestamps[-1],
            legend_labels=legend_labels,
            hide_pnts=True)

        # This one is to compare only the specific sequences we are after
        a_seq = ma2.Basic1DAnalysis(
            t_start=timestamps[-3],
            t_stop=timestamps[-1],
            legend_labels=legend_labels,
            hide_pnts=True)

        return a_full, a_seq

    def measure_single_qubit_parity(
        self,
        qD: str,
        qA: str,
        number_of_repetitions: int = 1,
        initialization_msmt: bool = False,
        initial_states=["0", "1"],
        nr_shots: int = 4088 * 4,
        flux_codeword: str = "cz",
        analyze: bool = True,
        close_fig: bool = True,
        prepare_for_timedomain: bool = True,
        MC=None,
        parity_axis="Z",
    ):
        assert qD in self.qubits()
        assert qA in self.qubits()
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[qD, qA])
        if MC is None:
            MC = self.instr_MC.get_instr()

        qDidx = self.find_instrument(qD).cfg_qubit_nr()
        qAidx = self.find_instrument(qA).cfg_qubit_nr()

        p = mqo.single_qubit_parity_check(
            qDidx,
            qAidx,
            self.cfg_openql_platform_fn(),
            number_of_repetitions=number_of_repetitions,
            initialization_msmt=initialization_msmt,
            initial_states=initial_states,
            flux_codeword=flux_codeword,
            parity_axis=parity_axis,
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())

        d = self.get_int_logging_detector(qubits=[qA], result_logging_mode="lin_trans")
        # d.nr_shots = 4088  # To ensure proper data binning
        # Because we are using a multi-detector
        d.set_child_attr("nr_shots", 4088)
        old_soft_avg = MC.soft_avg()
        old_live_plot_enabled = MC.live_plot_enabled()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)
        name = "Single_qubit_parity_{}_{}_{}".format(qD, qA, number_of_repetitions)
        MC.run(name)

        MC.soft_avg(old_soft_avg)
        MC.live_plot_enabled(old_live_plot_enabled)
        if analyze:
            a = ma2.Singleshot_Readout_Analysis(
                t_start=None,
                t_stop=None,
                label=name,
                options_dict={
                    "post_select": initialization_msmt,
                    "nr_samples": 2 + 2 * initialization_msmt,
                    "post_select_threshold": self.find_instrument(
                        qA
                    ).ro_acq_threshold(),
                },
                extract_only=False,
            )
        return a

    def measure_two_qubit_parity(
        self,
        qD0: str,
        qD1: str,
        qA: str,
        number_of_repetitions: int = 1,
        initialization_msmt: bool = False,
        initial_states=[
            ["0", "0"],
            ["0", "1"],
            ["1", "1",],
            ["1", "0"],
        ],  # nb: this groups even and odd
        # nr_shots: int=4088*4,
        flux_codeword: str = "cz",
        # flux_codeword1: str = "cz",
        analyze: bool = True,
        close_fig: bool = True,
        prepare_for_timedomain: bool = True,
        MC=None,
        echo: bool = True,
        post_select_threshold: float = None,
        parity_axes=["ZZ"],
        tomo=False,
        tomo_after=False,
        ro_time=600e-9,
        echo_during_ancilla_mmt: bool = True,
        idling_time=780e-9,
        idling_time_echo=480e-9,
        idling_rounds=0,
    ):
        assert qD0 in self.qubits()
        assert qD1 in self.qubits()
        assert qA in self.qubits()
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[qD1, qD0, qA])
        if MC is None:
            MC = self.instr_MC.get_instr()

        qD0idx = self.find_instrument(qD0).cfg_qubit_nr()
        qD1idx = self.find_instrument(qD1).cfg_qubit_nr()
        qAidx = self.find_instrument(qA).cfg_qubit_nr()

        p = mqo.two_qubit_parity_check(
            qD0idx,
            qD1idx,
            qAidx,
            self.cfg_openql_platform_fn(),
            number_of_repetitions=number_of_repetitions,
            initialization_msmt=initialization_msmt,
            initial_states=initial_states,
            flux_codeword=flux_codeword,
            # flux_codeword1=flux_codeword1,
            echo=echo,
            parity_axes=parity_axes,
            tomo=tomo,
            tomo_after=tomo_after,
            ro_time=ro_time,
            echo_during_ancilla_mmt=echo_during_ancilla_mmt,
            idling_time=idling_time,
            idling_time_echo=idling_time_echo,
            idling_rounds=idling_rounds,
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())

        d = self.get_int_logging_detector(
            qubits=[qD1, qD0, qA], result_logging_mode="lin_trans"
        )

        if tomo:
            mmts_per_round = (
                number_of_repetitions * len(parity_axes)
                + 1 * initialization_msmt
                + 1 * tomo_after
            )
            print("mmts_per_round", mmts_per_round)
            nr_shots = 4096 * 64 * mmts_per_round  # To ensure proper data binning
            if mmts_per_round < 4:
                nr_shots = 4096 * 64 * mmts_per_round  # To ensure proper data binning
            elif mmts_per_round < 10:
                nr_shots = 64 * 64 * mmts_per_round  # To ensure proper data binning
            elif mmts_per_round < 20:
                nr_shots = 16 * 64 * mmts_per_round  # To ensure proper data binning
            elif mmts_per_round < 40:
                nr_shots = 16 * 64 * mmts_per_round  # To ensure proper data binning
            else:
                nr_shots = 8 * 64 * mmts_per_round  # To ensure proper data binning
            d.set_child_attr("nr_shots", nr_shots)

        else:
            nr_shots = 4096 * 8  # To ensure proper data binning
            d.set_child_attr("nr_shots", nr_shots)

        old_soft_avg = MC.soft_avg()
        old_live_plot_enabled = MC.live_plot_enabled()
        self.msmt_suffix = "rounds{}".format(number_of_repetitions)
        MC.soft_avg(1)
        MC.live_plot_enabled(False)

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)
        name = "Two_qubit_parity_{}_{}_{}_{}_{}".format(
            parity_axes, qD1, qD0, qA, self.msmt_suffix
        )
        MC.run(name)
        MC.soft_avg(old_soft_avg)
        MC.live_plot_enabled(old_live_plot_enabled)
        if analyze:
            if not tomo:
                if not initialization_msmt:
                    a = mra.two_qubit_ssro_fidelity(name)
            a = ma2.Singleshot_Readout_Analysis(
                t_start=None,
                t_stop=None,
                label=name,
                options_dict={
                    "post_select": initialization_msmt,
                    "nr_samples": 2 + 2 * initialization_msmt,
                    "post_select_threshold": self.find_instrument(
                        qA
                    ).ro_acq_threshold(),
                    "preparation_labels": ["prep. 00, 11", "prep. 01, 10"],
                },
                extract_only=False,
            )
            return a

    def measure_residual_ZZ_coupling(
        self,
        q0: str,
        q_spectators: list,
        spectator_state="0",
        times=np.linspace(0, 10e-6, 26),
        analyze: bool = True,
        close_fig: bool = True,
        prepare_for_timedomain: bool = True,
        MC=None,
    ):

        assert q0 in self.qubits()
        for q_s in q_spectators:
            assert q_s in self.qubits()

        all_qubits = [q0] + q_spectators
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=all_qubits)
        if MC is None:
            MC = self.instr_MC.get_instr()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q_spec_idx_list = [
            self.find_instrument(q_s).cfg_qubit_nr() for q_s in q_spectators
        ]

        p = mqo.residual_coupling_sequence(
            times,
            q0idx,
            q_spec_idx_list,
            spectator_state,
            self.cfg_openql_platform_fn(),
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())
        d = self.get_int_avg_det(qubits=all_qubits)
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('Residual_ZZ_{}_{}_{}{}'.format(q0, q_spectators, spectator_state, self.msmt_suffix),
               exp_metadata={'target_qubit': q0,
                             'spectator_qubits': str(q_spectators),
                             'spectator_state': spectator_state})
        if analyze:
            a = ma.MeasurementAnalysis(close_main_fig=close_fig)
        return a

    def measure_two_qubit_ssro(
        self,
        qubits: list,
        nr_shots_per_case: int = 2 ** 13,  # 8192
        prepare_for_timedomain: bool = True,
        result_logging_mode="raw",
        initialize: bool = False,
        analyze=True,
        shots_per_meas: int = 2 ** 16,
        MC=None,
    ):
        """
        Perform a simultaneous ssro experiment on 2 qubits.

        Args:
            qubits (list of str)
                list of qubit names
            nr_shots_per_case (int):
                total number of measurements for each case under consideration
                    e.g., n*|00> , n*|01>, n*|10> , n*|11>

            shots_per_meas (int):
                number of single shot measurements per single
                acquisition with UHFQC


        FIXME: should be abstracted to measure multi qubit SSRO
        """

        # off and on, not including post selection init measurements yet
        nr_cases = 4  # 00, 01 ,10 and 11
        nr_shots = nr_shots_per_case * nr_cases

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits)
        if MC is None:
            MC = self.instr_MC.get_instr()

        # count from back because q0 is the least significant qubit
        q0 = qubits[-1]
        q1 = qubits[-2]

        assert q0 in self.qubits()
        assert q1 in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()
        p = mqo.multi_qubit_off_on(
            [q1idx, q0idx],
            initialize=initialize,
            second_excited_state=False,
            platf_cfg=self.cfg_openql_platform_fn(),
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())

        # right is LSQ
        d = self.get_int_logging_detector(
            qubits, result_logging_mode=result_logging_mode
        )

        shots_per_meas = int(
            np.floor(np.min([shots_per_meas, nr_shots]) / nr_cases) * nr_cases
        )

        d.set_child_attr("nr_shots", shots_per_meas)

        old_soft_avg = MC.soft_avg()
        old_live_plot_enabled = MC.live_plot_enabled()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)
        MC.run("SSRO_{}_{}_{}".format(q1, q0, self.msmt_suffix))

        MC.soft_avg(old_soft_avg)
        MC.live_plot_enabled(old_live_plot_enabled)
        if analyze:
            a = mra.two_qubit_ssro_fidelity("SSRO_{}_{}".format(q1, q0))
            a = ma2.Multiplexed_Readout_Analysis()
        return a

    def measure_state_tomography(self, qubits=['D2', 'X'],
                                 MC=None,
                                 bell_state: float=None,
                                 product_state: float=None,
                                 wait_after_flux: float=None,
                                 prepare_for_timedomain: bool =False,
                                 live_plot=False,
                                 nr_shots_per_case=2**14,
                                 shots_per_meas=2**16,
                                 disable_snapshot_metadata: bool = False,
                                 label='State_Tomography_',
                                 flux_codeword="cz"):
        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits)

        qubit_idxs = [self.find_instrument(qn).cfg_qubit_nr()
                      for qn in qubits]
        p = mqo.two_qubit_state_tomography(qubit_idxs, bell_state=bell_state,
                                           product_state=product_state,
                                           wait_after_flux=wait_after_flux,
                                           platf_cfg=self.cfg_openql_platform_fn(),
                                           flux_codeword=flux_codeword)
        # Special argument added to program
        combinations = p.combinations

        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        d = self.get_int_logging_detector(qubits)
        nr_cases = len(combinations)
        nr_shots = nr_shots_per_case*nr_cases
        shots_per_meas = int(np.floor(
            np.min([shots_per_meas, nr_shots])/nr_cases)*nr_cases)

        # Ensures shots per measurement is a multiple of the number of cases
        shots_per_meas -= shots_per_meas % nr_cases

        d.set_child_attr('nr_shots', shots_per_meas)

        MC.live_plot_enabled(live_plot)

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.tile(np.arange(nr_cases), nr_shots_per_case))
        MC.set_detector_function(d)
        MC.run('{}'.format(label),
               exp_metadata={'combinations': combinations},
               disable_snapshot_metadata=disable_snapshot_metadata)
        # mra.Multiplexed_Readout_Analysis(extract_combinations=True, options_dict={'skip_cross_fidelity': True})
        tomo_v2.Full_State_Tomography_2Q(label=label,
                                         qubit_ro_channels=qubits, # channels we will want to use for tomo
                                         correl_ro_channels=[qubits], # correlations we will want for the tomo
                                         tomo_qubits_idx=qubits)

    def measure_ssro_multi_qubit(
            self,
            qubits: list,
            nr_shots_per_case: int = 2**13,  # 8192
            prepare_for_timedomain: bool = True,
            result_logging_mode='raw',
            initialize: bool = False,
            analyze=True,
            shots_per_meas: int = 2**16,
            label='Mux_SSRO',
            MC=None):
        """
        Perform a simultaneous ssro experiment on multiple qubits.
        Args:
            qubits (list of str)
                list of qubit names
            nr_shots_per_case (int):
                total number of measurements for each case under consideration
                    e.g., n*|00> , n*|01>, n*|10> , n*|11> for two qubits

            shots_per_meas (int):
                number of single shot measurements per single
                acquisition with UHFQC

        """
        log.info("{}.measure_ssro_multi_qubit for qubits{}".format(self.name, qubits))

        # # off and on, not including post selection init measurements yet
        # nr_cases = 2**len(qubits)  # e.g., 00, 01 ,10 and 11 in the case of 2q
        # nr_shots = nr_shots_per_case*nr_cases

        # off and on, not including post selection init measurements yet
        nr_cases = 2 ** len(qubits)  # e.g., 00, 01 ,10 and 11 in the case of 2q

        if initialize:
            nr_shots = 2 * nr_shots_per_case * nr_cases
        else:
            nr_shots = nr_shots_per_case * nr_cases

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits)
        if MC is None:
            MC = self.instr_MC.get_instr()

        qubit_idxs = [self.find_instrument(qn).cfg_qubit_nr() for qn in qubits]
        p = mqo.multi_qubit_off_on(
            qubit_idxs,
            initialize=initialize,
            second_excited_state=False,
            platf_cfg=self.cfg_openql_platform_fn(),
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())

        # right is LSQ
        d = self.get_int_logging_detector(
            qubits, result_logging_mode=result_logging_mode
        )

        # This assumes qubit names do not contain spaces
        det_qubits = [v.split()[-1] for v in d.value_names]
        if (qubits != det_qubits) and (self.ro_acq_weight_type() == 'optimal'):
            # this occurs because the detector groups qubits per feedline.
            # If you do not pay attention, this will mess up the analysis of
            # this experiment.
            raise ValueError('Detector qubits do not match order specified.{} vs {}'.format(qubits, det_qubits))

        shots_per_meas = int(
            np.floor(np.min([shots_per_meas, nr_shots]) / nr_cases) * nr_cases
        )

        d.set_child_attr("nr_shots", shots_per_meas)

        old_soft_avg = MC.soft_avg()
        old_live_plot_enabled = MC.live_plot_enabled()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)
        MC.run("{}_{}_{}".format(label, qubits, self.msmt_suffix))
        MC.soft_avg(old_soft_avg)
        MC.live_plot_enabled(old_live_plot_enabled)

        if analyze:
            if initialize:
                thresholds = [
                    self.find_instrument(qubit).ro_acq_threshold()
                    for qubit in qubits]
                a = ma2.Multiplexed_Readout_Analysis(
                    label=label,
                    nr_qubits=len(qubits),
                    post_selection=True,
                    post_selec_thresholds=thresholds)
                # Print fraction of discarded shots
                # Dict = a.proc_data_dict['Post_selected_shots']
                # key = next(iter(Dict))
                # fraction=0
                # for comb in Dict[key].keys():
                #    fraction += len(Dict[key][comb])/(2**12 * 4)
                # print('Fraction of discarded results was {:.2f}'.format(1-fraction))
            else:
                a = ma2.Multiplexed_Readout_Analysis(
                    label=label,
                    nr_qubits=len(qubits))
            # Set thresholds
            for i, qubit in enumerate(qubits):
                label = a.Channels[i]
                threshold = a.qoi[label]['threshold_raw']
                self.find_instrument(qubit).ro_acq_threshold(threshold)
        return

    def measure_ssro_single_qubit(
            self,
            qubits: list,
            q_target: str,
            nr_shots: int = 2**13,  # 8192
            prepare_for_timedomain: bool = True,
            result_logging_mode='raw',
            initialize: bool = False,
            analyze=True,
            shots_per_meas: int = 2**16,
            label='Mux_SSRO',
            MC=None):
        '''
        Performs MUX single shot readout experiments of all possible
        combinations of prepared states of <qubits>. Outputs analysis
        of a single qubit <q_target>. This function is meant to
        assess a particular qubit readout in the multiplexed context.

        Args:
            qubits: List of qubits adressed in the mux readout.

            q_target: Qubit targeted in the analysis.

            nr_shots: number of shots for each prepared state of
            q_target. That is the experiment will include
            <nr_shots> shots of the qubit prepared in the ground state
            and <nr_shots> shots of the qubit prepared in the excited
            state. The remaining qubits will be prepared such that the
            experiment goes through all 2**n possible combinations of
            computational states.

            initialize: Include measurement post-selection by
            initialization.
        '''

        log.info('{}.measure_ssro_multi_qubit for qubits{}'.format(
            self.name, qubits))

        # off and on, not including post selection init measurements yet
        nr_cases = 2 ** len(qubits)  # e.g., 00, 01 ,10 and 11 in the case of 2q
        if initialize == True:
            nr_shots = 4 * nr_shots
        else:
            nr_shots = 2 * nr_shots

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits)
        if MC is None:
            MC = self.instr_MC.get_instr()

        qubit_idxs = [self.find_instrument(qn).cfg_qubit_nr()
                      for qn in qubits]

        p = mqo.multi_qubit_off_on(qubit_idxs,
                                   initialize=initialize,
                                   second_excited_state=False,
                                   platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())

        # right is LSQ
        d = self.get_int_logging_detector(qubits,
                                          result_logging_mode=result_logging_mode)

        # This assumes qubit names do not contain spaces
        det_qubits = [v.split()[-1] for v in d.value_names]
        if (qubits != det_qubits) and (self.ro_acq_weight_type() == 'optimal'):
            # this occurs because the detector groups qubits per feedline.
            # If you do not pay attention, this will mess up the analysis of
            # this experiment.
            raise ValueError('Detector qubits do not match order specified.{} vs {}'.format(qubits, det_qubits))

        shots_per_meas = int(np.floor(
            np.min([shots_per_meas, nr_shots])/nr_cases)*nr_cases)

        d.set_child_attr('nr_shots', shots_per_meas)

        old_soft_avg = MC.soft_avg()
        old_live_plot_enabled = MC.live_plot_enabled()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)
        MC.run('{}_{}_{}'.format(label, q_target, self.msmt_suffix))

        MC.soft_avg(old_soft_avg)
        MC.live_plot_enabled(old_live_plot_enabled)

        if analyze:
            if initialize == True:
                thresholds = [self.find_instrument(qubit).ro_acq_threshold() \
                    for qubit in qubits]
                a = ma2.Multiplexed_Readout_Analysis(label=label,
                                            nr_qubits = len(qubits),
                                            q_target = q_target,
                                            post_selection=True,
                                            post_selec_thresholds=thresholds)
                # Print fraction of discarded shots
                #Dict = a.proc_data_dict['Post_selected_shots']
                #key = next(iter(Dict))
                #fraction=0
                #for comb in Dict[key].keys():
                #    fraction += len(Dict[key][comb])/(2**12 * 4)
                #print('Fraction of discarded results was {:.2f}'.format(1-fraction))
            else:
                a = ma2.Multiplexed_Readout_Analysis(label=label,
                                                     nr_qubits=len(qubits),
                                                     q_target=q_target)
            q_ch = [ch for ch in a.Channels if q_target in ch.decode()][0]
            # Set thresholds
            for i, qubit in enumerate(qubits):
                label = a.raw_data_dict['value_names'][i]
                threshold = a.qoi[label]['threshold_raw']
                self.find_instrument(qubit).ro_acq_threshold(threshold)
            return a.qoi[q_ch]

    def measure_transients(self,
                           qubits: list,
                           q_target: str,
                           cases: list = ['off', 'on'],
                           MC=None,
                           prepare_for_timedomain: bool = True,
                           analyze: bool = True):
        '''
        Documentation.
        '''
        if q_target not in qubits:
            raise ValueError("q_target must be included in qubits.")
        # Ensure all qubits use same acquisition instrument
        instruments = [self.find_instrument(q).instr_acquisition() for q in qubits]
        if instruments[1:] != instruments[:-1]:
            raise ValueError("All qubits must have common acquisition instrument")

        qubits_nr = [self.find_instrument(q).cfg_qubit_nr() for q in qubits]
        q_target_nr = self.find_instrument(q_target).cfg_qubit_nr()

        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits)

        p = mqo.targeted_off_on(
                    qubits=qubits_nr,
                    q_target=q_target_nr,
                    pulse_comb='on',
                    platf_cfg=self.cfg_openql_platform_fn()
                )

        analysis = [None for case in cases]
        for i, pulse_comb in enumerate(cases):
            if 'off' in pulse_comb.lower():
                self.find_instrument(q_target).instr_LO_mw.get_instr().off()
            elif 'on' in pulse_comb.lower():
                self.find_instrument(q_target).instr_LO_mw.get_instr().on()
            else:
                raise ValueError(
                    "pulse_comb {} not understood: Only 'on' and 'off' allowed.".
                    format(pulse_comb))

            s = swf.OpenQL_Sweep(openql_program=p,
                                 parameter_name='Transient time', unit='s',
                                 CCL=self.instr_CC.get_instr())

            if 'UHFQC' in instruments[0]:
                sampling_rate = 1.8e9
            else:
                raise NotImplementedError()
            nr_samples = self.ro_acq_integration_length()*sampling_rate

            d = det.UHFQC_input_average_detector(
                            UHFQC=self.find_instrument(instruments[0]),
                            AWG=self.instr_CC.get_instr(),
                            nr_averages=self.ro_acq_averages(),
                            nr_samples=int(nr_samples))

            MC.set_sweep_function(s)
            MC.set_sweep_points(np.arange(nr_samples)/sampling_rate)
            MC.set_detector_function(d)
            MC.run('Mux_transients_{}_{}_{}'.format(q_target, pulse_comb,
                                                    self.msmt_suffix))
            if analyze:
                analysis[i] = ma2.Multiplexed_Transient_Analysis(
                    q_target='{}_{}'.format(q_target, pulse_comb))
        return analysis

    def calibrate_optimal_weights_mux(self,
                                      qubits: list,
                                      q_target: str,
                                      update=True,
                                      verify=True,
                                      averages=2**15
                                      ):

        """
        Measures the multiplexed readout transients of <qubits> for <q_target>
        in ground and excited state. After that, it calculates optimal
        integration weights that are used to weigh measuremet traces to maximize
        the SNR.

        Args:
            qubits (list):
                List of strings specifying qubits included in the multiplexed
                readout signal.
            q_target (str):
                ()
            verify (bool):
                indicates whether to run measure_ssro at the end of the routine
                to find the new SNR and readout fidelities with optimized weights
            update (bool):
                specifies whether to update the weights in the qubit object
        """
        if q_target not in qubits:
            raise ValueError("q_target must be included in qubits.")

        # Ensure that enough averages are used to get accurate weights
        old_avg = self.ro_acq_averages()
        self.ro_acq_averages(averages)

        Q_target = self.find_instrument(q_target)
        # Transient analysis
        A = self.measure_transients(qubits=qubits, q_target=q_target,
                                    cases=['on', 'off'])
        #return parameters
        self.ro_acq_averages(old_avg)

        # Optimal weights
        B = ma2.Multiplexed_Weights_Analysis(q_target=q_target,
                                             IF=Q_target.ro_freq_mod(),
                                             pulse_duration=Q_target.ro_pulse_length(),
                                             A_ground=A[1], A_excited=A[0])

        if update:
            Q_target.ro_acq_weight_func_I(B.qoi['W_I'])
            Q_target.ro_acq_weight_func_Q(B.qoi['W_Q'])
            Q_target.ro_acq_weight_type('optimal')

            if verify:
                Q_target._prep_ro_integration_weights()
                Q_target._prep_ro_instantiate_detectors()
                ssro_dict= self.measure_ssro_single_qubit(qubits=qubits,
                                                          q_target=q_target)
            return ssro_dict

    def measure_msmt_induced_dephasing_matrix(self, qubits: list,
                                              analyze=True, MC=None,
                                              prepare_for_timedomain=True,
                                              amps_rel=np.linspace(0, 1, 11),
                                              verbose=True,
                                              get_quantum_eff: bool = False,
                                              dephasing_sequence='ramsey',
                                              selected_target=None,
                                              selected_measured=None,
                                              target_qubit_excited=False,
                                              extra_echo=False,
                                              echo_delay=0e-9):
        """
        Measures the msmt induced dephasing for readout the readout of qubits
        i on qubit j. Additionally measures the SNR as a function of amplitude
        for the diagonal elements to obtain the quantum efficiency.
        In order to use this: make sure that
        - all readout_and_depletion pulses are of equal total length
        - the cc light to has the readout time configured equal to the
            measurement and depletion time + 60 ns buffer

        FIXME: not sure if the weight function assignment is working correctly.

        the qubit objects will use SSB for the dephasing measurements.
        """

        lpatt = "_trgt_{TQ}_measured_{RQ}"
        if prepare_for_timedomain:
            # for q in qubits:
            #    q.prepare_for_timedomain()
            self.prepare_for_timedomain(qubits=qubits)

        # Save old qubit suffixes
        old_suffixes = [self.find_instrument(q).msmt_suffix for q in qubits]
        old_suffix = self.msmt_suffix

        # Save the start-time of the experiment for analysis
        start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Loop over all target and measurement qubits
        target_qubits = [self.find_instrument(q) for q in qubits]
        measured_qubits = [self.find_instrument(q) for q in qubits]
        if selected_target != None:
            target_qubits = [target_qubits[selected_target]]
        if selected_measured != None:
            measured_qubits = [measured_qubits[selected_measured]]
        for target_qubit in target_qubits:
            for measured_qubit in measured_qubits:
                # Set measurement label suffix
                s = lpatt.replace("{TQ}", target_qubit.name)
                s = s.replace("{RQ}", measured_qubit.name)
                measured_qubit.msmt_suffix = s
                target_qubit.msmt_suffix = s

                # Print label
                if verbose:
                    print(s)

                # Slight differences if diagonal element
                if target_qubit == measured_qubit:
                    amps_rel = amps_rel
                    mqp = None
                    list_target_qubits = None
                else:
                    # t_amp_max = max(target_qubit.ro_pulse_down_amp0(),
                    #                target_qubit.ro_pulse_down_amp1(),
                    #                target_qubit.ro_pulse_amp())
                    # amp_max = max(t_amp_max, measured_qubit.ro_pulse_amp())
                    # amps_rel = np.linspace(0, 0.49/(amp_max), n_amps_rel)
                    amps_rel = amps_rel
                    mqp = self.cfg_openql_platform_fn()
                    list_target_qubits = [
                        target_qubit,
                    ]

                # If a diagonal element, consider doing the full quantum
                # efficiency matrix.
                if target_qubit == measured_qubit and get_quantum_eff:
                    res = measured_qubit.measure_quantum_efficiency(
                        verbose=verbose,
                        amps_rel=amps_rel,
                        dephasing_sequence=dephasing_sequence,
                    )
                else:
                    res = measured_qubit.measure_msmt_induced_dephasing_sweeping_amps(
                        verbose=verbose,
                        amps_rel=amps_rel,
                        cross_target_qubits=list_target_qubits,
                        multi_qubit_platf_cfg=mqp,
                        analyze=True,
                        sequence=dephasing_sequence,
                        target_qubit_excited=target_qubit_excited,
                        extra_echo=extra_echo,
                        # buffer_time=buffer_time
                    )
                # Print the result of the measurement
                if verbose:
                    print(res)

        # Save the end-time of the experiment
        stop = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # reset the msmt_suffix'es
        for qi, q in enumerate(qubits):
            self.find_instrument(q).msmt_suffix = old_suffixes[qi]
        self.msmt_suffix = old_suffix

        # Run the analysis for this experiment
        if analyze:
            options_dict = {
                "verbose": True,
            }
            qarr = qubits
            labelpatt = 'ro_amp_sweep_dephasing'+lpatt
            ca = ma2.CrossDephasingAnalysis(t_start=start, t_stop=stop,
                                            label_pattern=labelpatt,
                                            qubit_labels=qarr,
                                            options_dict=options_dict)

    def measure_chevron(
        self,
        q0: str,
        q_spec: str,
        q_park: str = None,
        amps=np.arange(0, 1, 0.05),
        lengths=np.arange(5e-9, 51e-9, 5e-9),
        adaptive_sampling=False,
        adaptive_sampling_pts=None,
        adaptive_pars: dict = None,
        prepare_for_timedomain=True,
        MC=None,
        freq_tone=6e9,
        pow_tone=-10,
        spec_tone=False,
        measure_parked_qubit=False,
        target_qubit_sequence: str = "ramsey",
        waveform_name="square",
        recover_q_spec: bool = False,
    ):
        """
        Measure a chevron patter of esulting from swapping of the excitations
        of the two qubits. Qubit q0 is prepared in 1 state and flux-pulsed
        close to the interaction zone using (usually) a rectangular pulse.
        Meanwhile q1 is prepared in 0, 1 or superposition state. If it is in 0
    state flipping between 01-10 can be observed. It if is in 1 state flipping
        between 11-20 as well as 11-02 show up. In superpostion everything is visible.

        Args:
            q0 (str):
                flux-pulsed qubit (prepared in 1 state at the beginning)
            q_spec (str):
                stationary qubit (in 0, 1 or superposition)
            q_park (str):
                qubit to move out of the interaction zone by applying a
                square flux pulse. Note that this is optional. Not specifying
                this means no extra pulses are applied.
                Note that this qubit is not read out.

            amps (array):
                amplitudes of the applied flux pulse controlled via the amplitude
                of the correspnding AWG channel

            lengths (array):
                durations of the applied flux pulses

            adaptive_sampling (bool):
                indicates whether to adaptivelly probe
                values of ampitude and duration, with points more dense where
                the data has more fine features

            adaptive_sampling_pts (int):
                number of points to measur in the adaptive_sampling mode

            prepare_for_timedomain (bool):
                should all instruments be reconfigured to
                time domain measurements

            target_qubit_sequence (str {"ground", "extited", "ramsey"}):
                specifies whether the spectator qubit should be
                prepared in the 0 state ('ground'), 1 state ('extited') or
                in superposition ('ramsey')

            spec_tone (bool):
                uses the spectroscopy source (in CW mode) of the qubit to produce
                a fake chevron.

            freq_tone (float):
                When spec_tone = True, controls the frequency of the spec source

            pow_tone (float):
                When spec_tone = True, controls the power of the spec source

            recover_q_spec (bool):
                applies the first gate of qspec at the end as well if `True`

        Circuit:
            q0    -x180-flux-x180-RO-
            qspec --x90-----(x90)-RO- (target_qubit_sequence='ramsey')

            q0    -x180-flux-x180-RO-
            qspec -x180----(x180)-RO- (target_qubit_sequence='excited')

            q0    -x180-flux-x180-RO-
            qspec ----------------RO- (target_qubit_sequence='ground')
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        assert q_spec in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q_specidx = self.find_instrument(q_spec).cfg_qubit_nr()
        if q_park is not None:
            q_park_idx = self.find_instrument(q_park).cfg_qubit_nr()
            fl_lutman_park = self.find_instrument(q_park).instr_LutMan_Flux.get_instr()
            if fl_lutman_park.sq_amp() < 0.1:
                # This can cause weird behaviour if not paid attention to.
                log.warning("Square amp for park pulse < 0.1")
            if fl_lutman_park.sq_length() < np.max(lengths):
                log.warning("Square length shorter than max Chevron length")
        else:
            q_park_idx = None

        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()
        fl_lutman_spec = self.find_instrument(q_spec).instr_LutMan_Flux.get_instr()

        if waveform_name == "square":
            length_par = fl_lutman.sq_length
            flux_cw = 6
        elif "cz" in waveform_name:
            length_par = fl_lutman.cz_length
            flux_cw = fl_lutman._get_cw_from_wf_name(waveform_name)
        else:
            raise ValueError("Waveform shape not understood")

        if prepare_for_timedomain:
            if measure_parked_qubit:
                self.prepare_for_timedomain(qubits=[q0, q_spec, q_park])
            else:
                self.prepare_for_timedomain(qubits=[q0, q_spec])

        awg = fl_lutman.AWG.get_instr()
        using_QWG = isinstance(awg, QuTech_AWG_Module)
        if using_QWG:
            awg_ch = fl_lutman.cfg_awg_channel()
            amp_par = awg.parameters["ch{}_amp".format(awg_ch)]
        else:
            awg_ch = (
                fl_lutman.cfg_awg_channel() - 1
            )  # -1 is to account for starting at 1
            ch_pair = awg_ch % 2
            awg_nr = awg_ch // 2

            amp_par = awg.parameters[
                "awgs_{}_outputs_{}_amplitude".format(awg_nr, ch_pair)
            ]

        sw = swf.FLsweep(fl_lutman, length_par, waveform_name=waveform_name)

        p = mqo.Chevron(
            q0idx,
            q_specidx,
            q_park_idx,
            buffer_time=40e-9,
            buffer_time2=max(lengths) + 40e-9,
            flux_cw=flux_cw,
            measure_parked_qubit=measure_parked_qubit,
            platf_cfg=self.cfg_openql_platform_fn(),
            target_qubit_sequence=target_qubit_sequence,
            cc=self.instr_CC.get_instr().name,
            recover_q_spec=recover_q_spec,
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)
        self.instr_CC.get_instr().start()

        if measure_parked_qubit:
            d = self.get_int_avg_det(
                qubits=[q0, q_spec, q_park],
                single_int_avg=True,
                seg_per_point=1,
                always_prepare=True,
            )
        else:
            d = self.get_correlation_detector(
                qubits=[q0, q_spec],
                single_int_avg=True,
                seg_per_point=1,
                always_prepare=True,
            )

        # if we want to add a spec tone
        if spec_tone:
            spec_source = self.find_instrument(q0).instr_spec_source.get_instr()
            spec_source.pulsemod_state(False)
            spec_source.power(pow_tone)
            spec_source.frequency(freq_tone)
            spec_source.on()

        MC.set_sweep_function(amp_par)
        MC.set_sweep_function_2D(sw)
        MC.set_detector_function(d)

        label = "Chevron {} {} {}".format(q0, q_spec, target_qubit_sequence)

        if not adaptive_sampling:
            MC.set_sweep_points(amps)
            MC.set_sweep_points_2D(lengths)
            MC.run(label, mode="2D")
            ma.TwoD_Analysis()
        else:
            if adaptive_pars is None:
                adaptive_pars = {
                    "adaptive_function": adaptive.Learner2D,
                    "goal": lambda l: l.npoints > adaptive_sampling_pts,
                    "bounds": (amps, lengths),
                }
            MC.set_adaptive_function_parameters(adaptive_pars)
            MC.run(label + " adaptive", mode="adaptive")
            ma2.Basic2DInterpolatedAnalysis()

    def measure_chevron_1D_bias_sweeps(
        self,
        q0: str,
        q_spec: str,
        q_park: str = None,
        amps=np.arange(0, 1, 0.05),
        prepare_for_timedomain=True,
        MC=None,
        freq_tone=6e9,
        pow_tone=-10,
        spec_tone=False,
        measure_parked_qubit=False,
        target_qubit_sequence: str = "excited",
        waveform_name="square",
        sq_duration=None,
        adaptive_sampling=False,
        adaptive_num_pts_max=None,
        adaptive_sample_for_alignment=True,
        max_pnts_beyond_threshold=10,
        adaptive_num_pnts_uniform=0,
        minimizer_threshold=0.5,
        par_idx=1,
        peak_is_inverted=True,
        mv_bias_by=[-150e-6, 150e-6],
        flux_buffer_time=40e-9,  # use multiples of 20 ns
    ):
        """
        Measure a chevron patter resulting from swapping of the excitations
        of the two qubits. Qubit q0 is prepared in 1 state and flux-pulsed
        close to the interaction zone using (usually) a rectangular pulse.
        Meanwhile q1 is prepared in 0, 1 or superposition state. If it is in 0
        state flipping between 10-01 can be observed. It if is in 1 state flipping
        between 11-20 as well as 11-02 show up. In superposition everything is visible.

        Args:
            q0 (str):
                flux-pulsed qubit (prepared in 1 state at the beginning)
            q_spec (str):
                stationary qubit (in 0, 1 or superposition)
            q_park (str):
                qubit to move out of the interaction zone by applying a
                square flux pulse. Note that this is optional. Not specifying
                this means no extra pulses are applied.
                Note that this qubit is not read out.

            amps (array):
                amplitudes of the applied flux pulse controlled via the amplitude
                of the corresponding AWG channel

            lengths (array):
                durations of the applied flux pulses

            adaptive_sampling (bool):
                indicates whether to adaptively probe
                values of amplitude and duration, with points more dense where
                the data has more fine features

            adaptive_num_pts_max (int):
                number of points to measure in the adaptive_sampling mode

            adaptive_num_pnts_uniform (bool):
                number of points to measure uniformly before giving control to
                adaptive sampler. Only relevant for `adaptive_sample_for_alignment`

            prepare_for_timedomain (bool):
                should all instruments be reconfigured to
                time domain measurements

            target_qubit_sequence (str {"ground", "excited", "ramsey"}):
                specifies whether the spectator qubit should be
                prepared in the 0 state ('ground'), 1 state ('excited') or
                in superposition ('ramsey')

            flux_buffer_time (float):
                buffer time added before and after the flux pulse

        Circuit:
            q0    -x180-flux-x180-RO-
            qspec --x90-----------RO- (target_qubit_sequence='ramsey')

            q0    -x180-flux-x180-RO-
            qspec -x180-----------RO- (target_qubit_sequence='excited')

            q0    -x180-flux-x180-RO-
            qspec ----------------RO- (target_qubit_sequence='ground')
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        assert q_spec in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q_specidx = self.find_instrument(q_spec).cfg_qubit_nr()
        if q_park is not None:
            q_park_idx = self.find_instrument(q_park).cfg_qubit_nr()
            fl_lutman_park = self.find_instrument(q_park).instr_LutMan_Flux.get_instr()
            if fl_lutman_park.sq_amp() < 0.1:
                # This can cause weird behavior if not paid attention to.
                log.warning("Square amp for park pulse < 0.1")
        else:
            q_park_idx = None

        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()

        if waveform_name == "square":
            length_par = fl_lutman.sq_length
            flux_cw = 6  # Hard-coded for now [2020-04-28]
            if sq_duration is None:
                raise ValueError("Square pulse duration must be specified.")
        else:
            raise ValueError("Waveform name not recognized.")

        awg = fl_lutman.AWG.get_instr()
        using_QWG = isinstance(awg, QuTech_AWG_Module)
        if using_QWG:
            awg_ch = fl_lutman.cfg_awg_channel()
            amp_par = awg.parameters["ch{}_amp".format(awg_ch)]
        else:
            # -1 is to account for starting at 1
            awg_ch = fl_lutman.cfg_awg_channel() - 1
            ch_pair = awg_ch % 2
            awg_nr = awg_ch // 2

            amp_par = awg.parameters[
                "awgs_{}_outputs_{}_amplitude".format(awg_nr, ch_pair)
            ]

        p = mqo.Chevron(
            q0idx,
            q_specidx,
            q_park_idx,
            buffer_time=flux_buffer_time,
            buffer_time2=length_par() + flux_buffer_time,
            flux_cw=flux_cw,
            measure_parked_qubit=measure_parked_qubit,
            platf_cfg=self.cfg_openql_platform_fn(),
            target_qubit_sequence=target_qubit_sequence,
            cc=self.instr_CC.get_instr().name,
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)

        qubits = [q0, q_spec]
        if measure_parked_qubit:
            # NB not tested in this function yet [2020-04-27]
            qubits.append(q_park)

        d = self.get_int_avg_det(qubits=qubits)

        # if we want to add a spec tone
        # NB: not tested [2020-04-27]
        if spec_tone:
            spec_source = self.find_instrument(q0).instr_spec_source.get_instr()
            spec_source.pulsemod_state(False)
            spec_source.power(pow_tone)
            spec_source.frequency(freq_tone)
            spec_source.on()

        MC.set_sweep_function(amp_par)
        MC.set_detector_function(d)

        old_sq_duration = length_par()
        # Assumes the waveforms will be generated below in the prepare_for_timedomain
        length_par(sq_duration)
        old_amp_par = amp_par()

        fluxcurrent_instr = self.find_instrument(q0).instr_FluxCtrl.get_instr()
        flux_bias_par_name = "FBL_" + q0
        flux_bias_par = fluxcurrent_instr[flux_bias_par_name]

        flux_bias_old_val = flux_bias_par()

        label = "Chevron {} {} [cut @ {:4g} ns]".format(q0, q_spec, length_par() / 1e-9)

        def restore_pars():
            length_par(old_sq_duration)
            amp_par(old_amp_par)
            flux_bias_par(flux_bias_old_val)

        # Keep below the length_par
        if prepare_for_timedomain:
            if measure_parked_qubit:
                self.prepare_for_timedomain(qubits=[q0, q_spec, q_park])
            else:
                self.prepare_for_timedomain(qubits=[q0, q_spec])
        else:
            log.warning("The flux waveform is not being uploaded!")

        if not adaptive_sampling:
            # Just single 1D sweep
            MC.set_sweep_points(amps)
            MC.run(label, mode="1D")

            restore_pars()

            ma2.Basic1DAnalysis()
        elif adaptive_sample_for_alignment:
            # Adaptive sampling intended for the calibration of the flux bias
            # (centering the chevron, and the qubit at the sweetspot)
            goal = l1dm.mk_min_threshold_goal_func(
                max_pnts_beyond_threshold=max_pnts_beyond_threshold
            )
            minimize = peak_is_inverted
            loss = l1dm.mk_minimization_loss_func(
                # Just in case it is ever changed to maximize
                threshold=(-1) ** (minimize + 1) * minimizer_threshold,
                interval_weight=200.0
            )
            bounds = (np.min(amps), np.max(amps))
            # q0 is the one leaking in the first CZ interaction point
            # because |2> amplitude is generally unpredictable, we use the
            # population in qspec to ensure there will be a peak for the
            # adaptive sampler
            # par_idx = 1 # Moved to method's arguments
            adaptive_pars_pos = {
                "adaptive_function": l1dm.Learner1D_Minimizer,
                "goal": lambda l: goal(l) or l.npoints > adaptive_num_pts_max,
                "bounds": bounds,
                "loss_per_interval": loss,
                "minimize": minimize,
                # A few uniform points to make more likely to find the peak
                "X0": np.linspace(
                    np.min(bounds),
                    np.max(bounds),
                    adaptive_num_pnts_uniform + 2)[1:-1]
            }
            bounds_neg = np.flip(-np.array(bounds), 0)
            adaptive_pars_neg = {
                "adaptive_function": l1dm.Learner1D_Minimizer,
                "goal": lambda l: goal(l) or l.npoints > adaptive_num_pts_max,
                # NB: order of the bounds matters, mind negative numbers ordering
                "bounds": bounds_neg,
                "loss_per_interval": loss,
                "minimize": minimize,
                # A few uniform points to make more likely to find the peak
                "X0": np.linspace(
                    np.min(bounds_neg),
                    np.max(bounds_neg),
                    adaptive_num_pnts_uniform + 2)[1:-1]
            }

            MC.set_sweep_functions([amp_par, flux_bias_par])
            adaptive_pars = {
                "multi_adaptive_single_dset": True,
                "adaptive_pars_list": [adaptive_pars_pos, adaptive_pars_neg],
                "extra_dims_sweep_pnts": flux_bias_par() + np.array(mv_bias_by),
                "par_idx": par_idx,
            }

            MC.set_adaptive_function_parameters(adaptive_pars)
            MC.run(label, mode="adaptive")

            restore_pars()

            a = ma2.Chevron_Alignment_Analysis(
                label=label,
                sq_pulse_duration=length_par(),
                fit_threshold=minimizer_threshold,
                fit_from=d.value_names[par_idx],
                peak_is_inverted=minimize,
            )

            return a

        else:
            # Default single 1D adaptive sampling
            adaptive_pars = {
                "adaptive_function": adaptive.Learner1D,
                "goal": lambda l: l.npoints > adaptive_num_pts_max,
                "bounds": (np.min(amps), np.max(amps)),
            }
            MC.set_adaptive_function_parameters(adaptive_pars)
            MC.run(label, mode="adaptive")

            restore_pars()

            ma2.Basic1DAnalysis()

    def measure_two_qubit_ramsey(
        self,
        q0: str,
        q_spec: str,
        times,
        prepare_for_timedomain=True,
        MC=None,
        target_qubit_sequence: str = "excited",
        chunk_size: int = None,
    ):
        """
        Measure a ramsey on q0 while setting the q_spec to excited state ('excited'),
        ground state ('ground') or superposition ('ramsey'). Suitable to measure
        large values of residual ZZ coupling.

        Args:
            q0 (str):
                qubit on which ramsey measurement is performed

            q1 (str):
                spectator qubit prepared in 0, 1 or superposition state

            times (array):
                durations of the ramsey sequence

            prepare_for_timedomain (bool):
                should all instruments be reconfigured to
                time domain measurements

            target_qubit_sequence (str {"ground", "extited", "ramsey"}):
                specifies whether the spectator qubit should be
                prepared in the 0 state ('ground'), 1 state ('extited') or
                in superposition ('ramsey')
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        assert q_spec in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q_specidx = self.find_instrument(q_spec).cfg_qubit_nr()

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[q0, q_spec])

        p = mqo.two_qubit_ramsey(
            times,
            q0idx,
            q_specidx,
            platf_cfg=self.cfg_openql_platform_fn(),
            target_qubit_sequence=target_qubit_sequence,
        )
        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name="Time",
            unit="s",
        )

        dt = times[1] - times[0]
        times = np.concatenate((times, [times[-1] + k * dt for k in range(1, 9)]))

        MC.set_sweep_function(s)
        MC.set_sweep_points(times)

        d = self.get_correlation_detector(qubits=[q0, q_spec])
        # d.chunk_size = chunk_size
        MC.set_detector_function(d)

        MC.run(
            "Two_qubit_ramsey_{}_{}_{}".format(q0, q_spec, target_qubit_sequence),
            mode="1D",
        )
        ma.MeasurementAnalysis()

    def measure_cryoscope(
        self,
        q0: str,
        times,
        MC=None,
        label="Cryoscope",
        double_projections: bool = True,
        waveform_name: str = "square",
        max_delay: float = "auto",
        twoq_pair=[2, 0],
        init_buffer=0,
        prepare_for_timedomain: bool = True,
    ):
        """
        Performs a cryoscope experiment to measure the shape of a flux pulse.

        Args:
            q0  (str)     :
                name of the target qubit

            times   (array):
                array of measurment times

            label (str):
                used to label the experiment

            waveform_name (str {"square", "custom_wf"}) :
                defines the name of the waveform used in the
                cryoscope. Valid values are either "square" or "custom_wf"

            max_delay {float, "auto"} :
                determines the delay in the pulse sequence
                if set to "auto" this is automatically set to the largest
                pulse duration for the cryoscope.

            prepare_for_timedomain (bool):
                calls self.prepare_for_timedomain on start
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        q0idx = self.find_instrument(q0).cfg_qubit_nr()

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[q0])

        if max_delay == "auto":
            max_delay = np.max(times) + 40e-9

        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()

        if waveform_name == "square":
            sw = swf.FLsweep(fl_lutman, fl_lutman.sq_length, waveform_name="square")
            flux_cw = "fl_cw_06"

        elif waveform_name == "custom_wf":
            sw = swf.FLsweep(
                fl_lutman, fl_lutman.custom_wf_length, waveform_name="custom_wf"
            )
            flux_cw = "fl_cw_05"

        else:
            raise ValueError(
                'waveform_name "{}" should be either '
                '"square" or "custom_wf"'.format(waveform_name)
            )

        p = mqo.Cryoscope(
            q0idx,
            buffer_time1=init_buffer,
            buffer_time2=max_delay,
            flux_cw=flux_cw,
            twoq_pair=twoq_pair,
            platf_cfg=self.cfg_openql_platform_fn(),
            cc=self.instr_CC.get_instr().name,
            double_projections=double_projections,
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)
        self.instr_CC.get_instr().start()

        MC.set_sweep_function(sw)
        MC.set_sweep_points(times)

        if double_projections:
            # Cryoscope v2
            values_per_point = 4
            values_per_point_suffex = ["cos", "sin", "mcos", "msin"]
        else:
            # Cryoscope v1
            values_per_point = 2
            values_per_point_suffex = ["cos", "sin"]

        d = self.get_int_avg_det(
            qubits=[q0],
            values_per_point=values_per_point,
            values_per_point_suffex=values_per_point_suffex,
            single_int_avg=True,
            always_prepare=True
        )
        MC.set_detector_function(d)
        MC.run(label)
        ma2.Basic1DAnalysis()

    def measure_cryoscope_vs_amp(
        self,
        q0: str,
        amps,
        duration: float = 100e-9,
        amp_parameter: str = "channel",
        MC=None,
        label="Cryoscope",
        max_delay: float = "auto",
        prepare_for_timedomain: bool = True,
    ):
        """
        Performs a cryoscope experiment to measure the shape of a flux pulse.


        Args:
            q0  (str)     :
                name of the target qubit

            amps   (array):
                array of square pulse amplitudes

            amps_paramater (str):
                The parameter through which the amplitude is changed either
                    {"channel",  "dac"}
                    channel : uses the AWG channel amplitude parameter
                    to rescale all waveforms
                    dac : uploads a new waveform with a different amlitude
                    for each data point.

            label (str):
                used to label the experiment

            waveform_name (str {"square", "custom_wf"}) :
                defines the name of the waveform used in the
                cryoscope. Valid values are either "square" or "custom_wf"

            max_delay {float, "auto"} :
                determines the delay in the pulse sequence
                if set to "auto" this is automatically set to the largest
                pulse duration for the cryoscope.

            prepare_for_timedomain (bool):
                calls self.prepare_for_timedomain on start
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        q0idx = self.find_instrument(q0).cfg_qubit_nr()

        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()
        fl_lutman.sq_length(duration)

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[q0])

        if max_delay == "auto":
            max_delay = duration + 40e-9

        if amp_parameter == "channel":
            sw = fl_lutman.cfg_awg_channel_amplitude
            flux_cw = "fl_cw_06"

        elif amp_parameter == "dac":
            sw = swf.FLsweep(fl_lutman, fl_lutman.sq_amp, waveform_name="square")
            flux_cw = "fl_cw_06"

        else:
            raise ValueError(
                'amp_parameter "{}" should be either '
                '"channel" or "dac"'.format(amp_parameter)
            )

        p = mqo.Cryoscope(
            q0idx,
            buffer_time1=0,
            buffer_time2=max_delay,
            flux_cw=flux_cw,
            platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)
        self.instr_CC.get_instr().start()

        MC.set_sweep_function(sw)
        MC.set_sweep_points(amps)
        d = self.get_int_avg_det(
            qubits=[q0],
            values_per_point=2,
            values_per_point_suffex=["cos", "sin"],
            single_int_avg=True,
            always_prepare=True,
        )
        MC.set_detector_function(d)
        MC.run(label)
        ma2.Basic1DAnalysis()

    def measure_timing_diagram(self, q0, flux_latencies, microwave_latencies,
                               MC=None,  label='timing_{}_{}',
                               qotheridx=2,
                               mw_gate="rx90",
                               pulse_length=40e-9, flux_cw='fl_cw_06',
                               extra_buffer=0e-9,
                               prepare_for_timedomain: bool = True):
        """
        Measure the ramsey-like sequence with the 40 ns flux pulses played between
        the two pi/2. While playing this sequence the delay of flux and microwave pulses
        is varied (relative to the readout pulse), looking for configuration in which
        the pulses arrive at the sample in the desired order.

        After measuting the pattern use ma2.Timing_Cal_Flux_Fine with manually
        chosen parameters to match the drawn line to the measured patern.

        Args:
            q0  (str)     :
                name of the target qubit
            flux_latencies   (array):
                array of flux latencies to set (in seconds)
            microwave_latencies (array):
                array of microwave latencies to set (in seconds)

            label (str):
                used to label the experiment

            prepare_for_timedomain (bool):
                calls self.prepare_for_timedomain on start
        """
        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain([q0])

        assert q0 in self.qubits()
        self.prepare_for_timedomain(qubits=[q0])
        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()
        fl_lutman.sq_length(pulse_length)

        CC = self.instr_CC.get_instr()

        # Wait 40 results in a mw separation of flux_pulse_duration+40ns = 80ns
        p = sqo.FluxTimingCalibration(q0idx,
                                      times=[extra_buffer],
                                      platf_cfg=self.cfg_openql_platform_fn(),
                                      flux_cw=flux_cw,
                                      mw_gate=mw_gate,
                                      # qubit_other_idx=qotheridx,
                                      cal_points=False)

        CC.eqasm_program(p.filename)

        d = self.get_int_avg_det(qubits=[q0], single_int_avg=True)
        MC.set_detector_function(d)

        s = swf.tim_flux_latency_sweep(self)
        s2 = swf.tim_mw_latency_sweep(self)
        MC.set_sweep_functions([s, s2])
        # MC.set_sweep_functions(s2)

        # MC.set_sweep_points(microwave_latencies)
        MC.set_sweep_points(flux_latencies)
        MC.set_sweep_points_2D(microwave_latencies)
        MC.run_2D(label.format(self.name, q0))

        # This is the analysis that should be run but with custom delays
        ma2.Timing_Cal_Flux_Fine(ch_idx=0, close_figs=False,
                                 ro_latency=-100e-9,
                                 flux_latency=0,
                                 flux_pulse_duration=10e-9,
                                 mw_pulse_separation=80e-9)

    def measure_timing_1d_trace(self, q0, latencies, latency_type='flux',
                                MC=None,  label='timing_{}_{}',
                                buffer_time=40e-9,
                                prepare_for_timedomain: bool = True,
                                mw_gate: str = "rx90", sq_length: float = 60e-9):
        mmt_label = label.format(self.name, q0)
        if MC is None:
            MC = self.instr_MC.get_instr()
        assert q0 in self.qubits()
        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        self.prepare_for_timedomain([q0])
        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()
        fl_lutman.sq_length(sq_length)
        CC = self.instr_CC.get_instr()

        # Wait 40 results in a mw separation of flux_pulse_duration+40ns = 120ns
        p = sqo.FluxTimingCalibration(q0idx,
                                      times=[buffer_time],
                                      platf_cfg=self.cfg_openql_platform_fn(),
                                      flux_cw='fl_cw_06',
                                      cal_points=False,
                                      mw_gate=mw_gate)
        CC.eqasm_program(p.filename)

        d = self.get_int_avg_det(qubits=[q0], single_int_avg=True)
        MC.set_detector_function(d)

        if latency_type == 'flux':
            s = swf.tim_flux_latency_sweep(self)
        elif latency_type == 'mw':
            s = swf.tim_mw_latency_sweep(self)
        else:
            raise ValueError('Latency type {} not understood.'.format(latency_type))
        MC.set_sweep_function(s)
        MC.set_sweep_points(latencies)
        MC.run(mmt_label)

        a_obj = ma2.Basic1DAnalysis(label=mmt_label)
        return a_obj

    def measure_ramsey_with_flux_pulse(self, q0: str, times,
                                       MC=None,
                                       label='Fluxed_ramsey',
                                       prepare_for_timedomain: bool = True,
                                       pulse_shape: str = 'square',
                                       sq_eps: float = None):
        """
        Performs a cryoscope experiment to measure the shape of a flux pulse.

        Args:
            q0  (str)     :
                name of the target qubit

            times   (array):
                array of measurment times

            label (str):
                used to label the experiment

            prepare_for_timedomain (bool):
                calls self.prepare_for_timedomain on start

        Note: the amplitude and (expected) detuning of the flux pulse is saved
         in experimental metadata.
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        q0idx = self.find_instrument(q0).cfg_qubit_nr()

        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()
        partner_lutman = self.find_instrument(fl_lutman.instr_partner_lutman())
        old_max_length = fl_lutman.cfg_max_wf_length()
        old_sq_length = fl_lutman.sq_length()
        fl_lutman.cfg_max_wf_length(max(times) + 200e-9)
        partner_lutman.cfg_max_wf_length(max(times) + 200e-9)
        fl_lutman.custom_wf_length(max(times) + 200e-9)
        partner_lutman.custom_wf_length(max(times) + 200e-9)
        fl_lutman.load_waveforms_onto_AWG_lookuptable(force_load_sequencer_program=True)

        def set_flux_pulse_time(value):
            if pulse_shape == "square":
                flux_cw = "fl_cw_02"
                fl_lutman.sq_length(value)
                fl_lutman.load_waveform_realtime("square", regenerate_waveforms=True)
            elif pulse_shape == "single_sided_square":
                flux_cw = "fl_cw_05"

                dac_scalefactor = fl_lutman.get_amp_to_dac_val_scalefactor()
                dacval = dac_scalefactor * fl_lutman.calc_eps_to_amp(
                    sq_eps, state_A="01", state_B=None, positive_branch=True
                )

                sq_pulse = dacval * np.ones(int(value * fl_lutman.sampling_rate()))

                fl_lutman.custom_wf(sq_pulse)
                fl_lutman.load_waveform_realtime("custom_wf", regenerate_waveforms=True)
            elif pulse_shape == "double_sided_square":
                flux_cw = "fl_cw_05"

                dac_scalefactor = fl_lutman.get_amp_to_dac_val_scalefactor()
                pos_dacval = dac_scalefactor * fl_lutman.calc_eps_to_amp(
                    sq_eps, state_A="01", state_B=None, positive_branch=True
                )

                neg_dacval = dac_scalefactor * fl_lutman.calc_eps_to_amp(
                    sq_eps, state_A="01", state_B=None, positive_branch=False
                )

                sq_pulse_half = np.ones(int(value / 2 * fl_lutman.sampling_rate()))

                sq_pulse = np.concatenate(
                    [pos_dacval * sq_pulse_half, neg_dacval * sq_pulse_half]
                )
                fl_lutman.custom_wf(sq_pulse)
                fl_lutman.load_waveform_realtime("custom_wf", regenerate_waveforms=True)

            p = mqo.fluxed_ramsey(
                q0idx,
                wait_time=value,
                flux_cw=flux_cw,
                platf_cfg=self.cfg_openql_platform_fn(),
            )
            self.instr_CC.get_instr().eqasm_program(p.filename)
            self.instr_CC.get_instr().start()

        flux_pulse_time = Parameter("flux_pulse_time", set_cmd=set_flux_pulse_time)

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[q0])

        MC.set_sweep_function(flux_pulse_time)
        MC.set_sweep_points(times)
        d = self.get_int_avg_det(
            qubits=[q0],
            values_per_point=2,
            values_per_point_suffex=["final x90", "final y90"],
            single_int_avg=True,
            always_prepare=True,
        )
        MC.set_detector_function(d)
        metadata_dict = {"sq_eps": sq_eps}
        MC.run(label, exp_metadata=metadata_dict)

        fl_lutman.cfg_max_wf_length(old_max_length)
        partner_lutman.cfg_max_wf_length(old_max_length)
        fl_lutman.sq_length(old_sq_length)
        fl_lutman.load_waveforms_onto_AWG_lookuptable(force_load_sequencer_program=True)

    def measure_sliding_flux_pulses(
        self,
        qubits: list,
        times: list,
        MC,
        nested_MC,
        prepare_for_timedomain: bool = True,
        flux_cw: str = "fl_cw_01",
        disable_initial_pulse: bool = False,
        label="",
    ):
        """
        Performs a sliding pulses experiment in order to determine how
        the phase picked up by a flux pulse depends on preceding flux
        pulses.

        Args:
            qubits (list):
                two-element list of qubits. Only the second of the qubits
                listed matters. First needs to be provided for compatibility
                with OpenQl.

            times (array):
                delays between the two flux pulses to sweep over

            flux_cw (str):
                codeword specifying which of the flux pulses to execute

            disable_initial_pulse (bool):
                allows to execute the reference measurement without
                the first of the flux pulses

            label (str):
                suffix to append to the measurement label
        """
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubits)

        q0_name = qubits[-1]

        counter_par = ManualParameter("counter", unit="#")
        counter_par(0)

        gate_separation_par = ManualParameter("gate separation", unit="s")
        gate_separation_par(20e-9)

        d = det.Function_Detector(
            get_function=self._measure_sliding_pulse_phase,
            value_names=["Phase", "stderr"],
            value_units=["deg", "deg"],
            msmt_kw={
                "disable_initial_pulse": disable_initial_pulse,
                "qubits": qubits,
                "counter_par": [counter_par],
                "gate_separation_par": [gate_separation_par],
                "nested_MC": nested_MC,
                "flux_cw": flux_cw,
            },
        )

        MC.set_sweep_function(gate_separation_par)
        MC.set_sweep_points(times)

        MC.set_detector_function(d)
        MC.run("Sliding flux pulses {}{}".format(q0_name, label))

    def _measure_sliding_pulse_phase(
        self,
        disable_initial_pulse,
        counter_par,
        gate_separation_par,
        qubits: list,
        nested_MC,
        flux_cw="fl_cw_01",
    ):
        """
        Method relates to "measure_sliding_flux_pulses", this performs one
        phase measurement for the sliding pulses experiment.
        It is defined as a private method as it should not be used
        independently.
        """
        # FIXME passing as a list is a hack to work around Function detector
        counter_par = counter_par[0]
        gate_separation_par = gate_separation_par[0]

        if disable_initial_pulse:
            flux_codeword_a = "fl_cw_00"
        else:
            flux_codeword_a = flux_cw
        flux_codeword_b = flux_cw

        counter_par(counter_par() + 1)
        # substract mw_pulse_dur to correct for mw_pulse before 2nd flux pulse
        mw_pulse_dur = 20e-9
        wait_time = int((gate_separation_par() - mw_pulse_dur) * 1e9)

        if wait_time < 0:
            raise ValueError()

        # angles = np.arange(0, 341, 20*1)
        # These are hardcoded angles in the mw_lutman for the AWG8
        angles = np.concatenate(
            [np.arange(0, 101, 20), np.arange(140, 341, 20)]
        )  # avoid CW15, issue
        # angles = np.arange(0, 341, 20))

        qubit_idxs = [self.find_instrument(q).cfg_qubit_nr() for q in qubits]
        p = mqo.sliding_flux_pulses_seq(
            qubits=qubit_idxs,
            platf_cfg=self.cfg_openql_platform_fn(),
            wait_time=wait_time,
            angles=angles,
            flux_codeword_a=flux_codeword_a,
            flux_codeword_b=flux_codeword_b,
            add_cal_points=False,
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name="Phase",
            unit="deg",
        )
        nested_MC.set_sweep_function(s)
        nested_MC.set_sweep_points(angles)
        nested_MC.set_detector_function(self.get_correlation_detector(qubits=qubits))
        nested_MC.run(
            "sliding_CZ_oscillation_{}".format(counter_par()),
            disable_snapshot_metadata=True,
        )

        # ch_idx = 1 because of the order of the correlation detector
        a = ma2.Oscillation_Analysis(ch_idx=1)
        phi = np.rad2deg(a.fit_res["cos_fit"].params["phase"].value) % 360

        phi_stderr = np.rad2deg(a.fit_res["cos_fit"].params["phase"].stderr)

        return (phi, phi_stderr)

    def measure_two_qubit_randomized_benchmarking(
        self,
        qubits,
        MC,
        nr_cliffords=np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 12.0, 15.0, 20.0, 25.0, 30.0, 50.0]
        ),
        nr_seeds=100,
        interleaving_cliffords=[None],
        label="TwoQubit_RB_{}seeds_recompile={}_icl{}_{}_{}_{}",
        recompile: bool = "as needed",
        cal_points=True,
        flux_codeword="cz",
        flux_allocated_duration_ns: int = None,
        sim_cz_qubits: list = None,
        compile_only: bool = False,
        pool=None,  # a multiprocessing.Pool()
        rb_tasks=None  # used after called with `compile_only=True`
    ):
        """
        Measures two qubit randomized benchmarking, including
        the leakage estimate.

        [2020-07-04 Victor] this method was updated to allow for parallel
        compilation using all the cores of the measurement computer

        Refs:
        Knill PRA 77, 012307 (2008)
        Wood PRA 97, 032306 (2018)

        Args:
            qubits (list):
                pair of the qubit names on which to perform RB

            nr_cliffords (array):
                lengths of the clifford sequences to perform

            nr_seeds (int):
                number of different clifford sequences of each length

            interleaving_cliffords (list):
                list of integers (or None) which specifies which cliffords
                to interleave the sequence with (for interleaved RB)
                For indices of Clifford group elements go to
                two_qubit_clifford_group.py

            label (str):
                string for formatting the measurement name

            recompile (bool, str {'as needed'}):
                indicate whether to regenerate the sequences of clifford gates.
                By default it checks whether the needed sequences were already
                generated since the most recent change of OpenQL file
                specified in self.cfg_openql_platform_fn

            cal_points (bool):
                should calibration point (qubits in 0 and 1 states)
                be included in the measurement

            flux_codeword (str):
                flux codeword corresponding to the Cphase gate
            sim_cz_qubits (list):
                A list of qubit names on which a simultaneous cz
                instruction must be applied. This is for characterizing
                CZ gates that are intended to be performed in parallel
                with other CZ gates.
            flux_allocated_duration_ns (list):
                Duration in ns of the flux pulse used when interleaved gate is
                [100_000], i.e. idle identity
            compilation_only (bool):
                Compile only the RB sequences without measuring, intended for
                parallelizing iRB sequences compilation with measurements
            pool (multiprocessing.Pool):
                Only relevant for `compilation_only=True`
                Pool to which the compilation tasks will be assigned
            rb_tasks (list):
                Only relevant when running `compilation_only=True` previously,
                saving the rb_tasks, waiting for them to finish then running
                this method again and providing the `rb_tasks`.
                See the interleaved RB for use case.
        """

        # Settings that have to be preserved, change is required for
        # 2-state readout and postprocessing
        old_weight_type = self.ro_acq_weight_type()
        old_digitized = self.ro_acq_digitized()
        self.ro_acq_weight_type("optimal IQ")
        self.ro_acq_digitized(False)

        self.prepare_for_timedomain(qubits=qubits)
        MC.soft_avg(1)
        # The detector needs to be defined before setting back parameters
        d = self.get_int_logging_detector(qubits=qubits)
        # set back the settings
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

        for q in qubits:
            q_instr = self.find_instrument(q)
            mw_lutman = q_instr.instr_LutMan_MW.get_instr()
            mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()

        MC.soft_avg(1)

        qubit_idxs = [self.find_instrument(q).cfg_qubit_nr() for q in qubits]
        if sim_cz_qubits is not None:
            sim_cz_qubits_idxs = [
                self.find_instrument(q).cfg_qubit_nr() for q in sim_cz_qubits
            ]
        else:
            sim_cz_qubits_idxs = None

        net_cliffords = [0, 3 * 24 + 3]

        def send_rb_tasks(pool_):
            tasks_inputs = []
            for i in range(nr_seeds):
                task_dict = dict(
                    qubits=qubit_idxs,
                    nr_cliffords=nr_cliffords,
                    nr_seeds=1,
                    flux_codeword=flux_codeword,
                    flux_allocated_duration_ns=flux_allocated_duration_ns,
                    platf_cfg=self.cfg_openql_platform_fn(),
                    program_name="TwoQ_RB_int_cl_s{}_ncl{}_icl{}_netcl{}_{}_{}".format(
                        int(i),
                        list(map(int, nr_cliffords)),
                        interleaving_cliffords,
                        list(map(int, net_cliffords)),
                        qubits[0],
                        qubits[1],
                    ),
                    interleaving_cliffords=interleaving_cliffords,
                    cal_points=cal_points,
                    net_cliffords=net_cliffords,  # measures with and without inverting
                    f_state_cal_pts=True,
                    recompile=recompile,
                    sim_cz_qubits=sim_cz_qubits_idxs,
                )
                tasks_inputs.append(task_dict)

            rb_tasks = pool_.map_async(cl_oql.parallel_friendly_rb, tasks_inputs)

            return rb_tasks

        if compile_only:
            assert pool is not None
            rb_tasks = send_rb_tasks(pool)
            return rb_tasks

        if rb_tasks is None:
            # Using `with ...:` makes sure the other processes will be terminated
            # avoid starting too mane processes,
            # nr_processes = None will start as many as the PC can handle
            nr_processes = None if recompile else 1
            with multiprocessing.Pool(
                nr_processes,
                maxtasksperchild=cl_oql.maxtasksperchild  # avoid RAM issues
            ) as pool:
                rb_tasks = send_rb_tasks(pool)
                cl_oql.wait_for_rb_tasks(rb_tasks)

        programs_filenames = rb_tasks.get()

        # to include calibration points
        if cal_points:
            sweep_points = np.append(
                np.repeat(nr_cliffords, 2),
                [nr_cliffords[-1] + 0.5] * 2
                + [nr_cliffords[-1] + 1.5] * 2
                + [nr_cliffords[-1] + 2.5] * 3,
            )
        else:
            sweep_points = np.repeat(nr_cliffords, 2)

        counter_param = ManualParameter("name_ctr", initial_value=0)
        prepare_function_kwargs = {
            "counter_param": counter_param,
            "programs_filenames": programs_filenames,
            "CC": self.instr_CC.get_instr(),
        }

        # Using the first detector of the multi-detector as this is
        # in charge of controlling the CC (see self.get_int_logging_detector)
        d.set_prepare_function(
            oqh.load_range_of_oql_programs_from_filenames,
            prepare_function_kwargs, detectors="first"
        )
        # d.nr_averages = 128

        reps_per_seed = 4094 // len(sweep_points)
        nr_shots = reps_per_seed * len(sweep_points)
        d.set_child_attr("nr_shots", nr_shots)

        s = swf.None_Sweep(parameter_name="Number of Cliffords", unit="#")

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.tile(sweep_points, reps_per_seed * nr_seeds))

        MC.set_detector_function(d)
        label = label.format(
            nr_seeds,
            recompile,
            interleaving_cliffords,
            qubits[0],
            qubits[1],
            flux_codeword)
        MC.run(label, exp_metadata={"bins": sweep_points})
        # N.B. if interleaving cliffords are used, this won't work
        ma2.RandomizedBenchmarking_TwoQubit_Analysis(label=label)

    def measure_interleaved_randomized_benchmarking_statistics(
        self, RB_type: str = "CZ", nr_iRB_runs: int = 30, **iRB_kw
    ):
        """
        This is an optimized way of measuring statistics of the iRB
        Main advantage: it recompiles the RB sequences for the next run in the
        loop while measuring the current run. This ensures that measurements
        are as close to back-to-back as possible and saves a significant
        amount of idle time on the experimental setup
        """
        if not iRB_kw["recompile"]:
            log.warning(
                "iRB statistics are intended to be measured while " +
                "recompiling the RB sequences!"
            )

        if RB_type == "CZ":
            measurement_func = self.measure_two_qubit_interleaved_randomized_benchmarking
        elif RB_type == "CZ_parked_qubit":
            measurement_func = self.measure_single_qubit_interleaved_randomized_benchmarking_parking
        else:
            raise ValueError(
                "RB type `{}` not recognized!".format(RB_type)
            )

        rounds_success = np.zeros(nr_iRB_runs)
        t0 = time.time()
        # `maxtasksperchild` avoid RAM issues
        with multiprocessing.Pool(maxtasksperchild=cl_oql.maxtasksperchild) as pool:
            rb_tasks_start = None
            last_run = nr_iRB_runs - 1
            for i in range(nr_iRB_runs):
                iRB_kw["rb_tasks_start"] = rb_tasks_start
                iRB_kw["pool"] = pool
                iRB_kw["start_next_round_compilation"] = (i < last_run)
                round_successful = False
                try:
                    rb_tasks_start = measurement_func(
                        **iRB_kw
                    )
                    round_successful = True
                except Exception:
                    print_exception()
                finally:
                    rounds_success[i] = 1 if round_successful else 0
        t1 = time.time()
        good_rounds = int(np.sum(rounds_success))
        print("Performed {}/{} successful iRB measurements in {:>7.1f} s ({:>7.1f} min.).".format(
            good_rounds, nr_iRB_runs, t1 - t0, (t1 - t0) / 60
        ))
        if good_rounds < nr_iRB_runs:
            log.error("Not all iRB measurements were successful!")

    def measure_two_qubit_interleaved_randomized_benchmarking(
        self,
        qubits: list,
        MC,
        nr_cliffords=np.array(
            [1., 3., 5., 7., 9., 11., 15., 20., 25., 30., 40., 50., 70., 90., 120.]
        ),
        nr_seeds=100,
        recompile: bool = "as needed",
        flux_codeword="cz",
        flux_allocated_duration_ns: int = None,
        sim_cz_qubits: list = None,
        measure_idle_flux: bool = True,
        rb_tasks_start: list = None,
        pool=None,
        start_next_round_compilation: bool = False
    ):
        """
        Perform two qubit interleaved randomized benchmarking with an
        interleaved CZ gate, and optionally an interleaved idle identity with
        the duration of the CZ.

        If recompile is `True` or `as needed` it will parallelize RB sequence
        compilation with measurement (beside the parallelization of the RB
        sequences which will always happen in parallel).
        """
        def run_parallel_iRB(
            recompile, pool, rb_tasks_start: list = None,
            start_next_round_compilation: bool = False
        ):
            """
            We define the full parallel iRB procedure here as function such
            that we can control the flow of the parallel RB sequences
            compilations from the outside of this method, and allow for
            chaining RB compilations for sequential measurements intended for
            taking statistics of the RB performance
            """
            rb_tasks_next = None

            # 1. Start (non-blocking) compilation for [None]
            # We make it non-blocking such that the non-blocking feature
            # is used for the interleaved cases
            if rb_tasks_start is None:
                rb_tasks_start = self.measure_two_qubit_randomized_benchmarking(
                    qubits=qubits,
                    MC=MC,
                    nr_cliffords=nr_cliffords,
                    interleaving_cliffords=[None],
                    recompile=recompile,
                    flux_codeword=flux_codeword,
                    nr_seeds=nr_seeds,
                    sim_cz_qubits=sim_cz_qubits,
                    compile_only=True,
                    pool=pool
                )

            # 2. Wait for [None] compilation to finish
            cl_oql.wait_for_rb_tasks(rb_tasks_start)

            # 3. Start (non-blocking) compilation for [104368]
            rb_tasks_CZ = self.measure_two_qubit_randomized_benchmarking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[104368],
                recompile=recompile,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                sim_cz_qubits=sim_cz_qubits,
                compile_only=True,
                pool=pool
            )
            # 4. Start the measurement and run the analysis for [None]
            self.measure_two_qubit_randomized_benchmarking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[None],
                recompile=False,  # This of course needs to be False
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                sim_cz_qubits=sim_cz_qubits,
                rb_tasks=rb_tasks_start,
            )

            # 5. Wait for [104368] compilation to finish
            cl_oql.wait_for_rb_tasks(rb_tasks_CZ)

            # 6. Start (non-blocking) compilation for [100_000]
            if measure_idle_flux:
                rb_tasks_I = self.measure_two_qubit_randomized_benchmarking(
                    qubits=qubits,
                    MC=MC,
                    nr_cliffords=nr_cliffords,
                    interleaving_cliffords=[100_000],
                    recompile=recompile,
                    flux_codeword=flux_codeword,
                    flux_allocated_duration_ns=flux_allocated_duration_ns,
                    nr_seeds=nr_seeds,
                    sim_cz_qubits=sim_cz_qubits,
                    compile_only=True,
                    pool=pool,
                )
            elif start_next_round_compilation:
                # Optionally send to the `pool` the tasks of RB compilation to be
                # used on the next round of calling the iRB method
                rb_tasks_next = self.measure_two_qubit_randomized_benchmarking(
                    qubits=qubits,
                    MC=MC,
                    nr_cliffords=nr_cliffords,
                    interleaving_cliffords=[None],
                    recompile=recompile,
                    flux_codeword=flux_codeword,
                    nr_seeds=nr_seeds,
                    sim_cz_qubits=sim_cz_qubits,
                    compile_only=True,
                    pool=pool
                )
            # 7. Start the measurement and run the analysis for [104368]
            self.measure_two_qubit_randomized_benchmarking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[104368],
                recompile=False,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                sim_cz_qubits=sim_cz_qubits,
                rb_tasks=rb_tasks_CZ,
            )
            ma2.InterleavedRandomizedBenchmarkingAnalysis(
                label_base="icl[None]",
                label_int="icl[104368]"
            )

            if measure_idle_flux:
                # 8. Wait for [100_000] compilation to finish
                cl_oql.wait_for_rb_tasks(rb_tasks_I)

                # 8.a. Optionally send to the `pool` the tasks of RB compilation to be
                # used on the next round of calling the iRB method
                if start_next_round_compilation:
                    rb_tasks_next = self.measure_two_qubit_randomized_benchmarking(
                        qubits=qubits,
                        MC=MC,
                        nr_cliffords=nr_cliffords,
                        interleaving_cliffords=[None],
                        recompile=recompile,
                        flux_codeword=flux_codeword,
                        nr_seeds=nr_seeds,
                        sim_cz_qubits=sim_cz_qubits,
                        compile_only=True,
                        pool=pool
                    )

                # 9. Start the measurement and run the analysis for [100_000]
                self.measure_two_qubit_randomized_benchmarking(
                    qubits=qubits,
                    MC=MC,
                    nr_cliffords=nr_cliffords,
                    interleaving_cliffords=[100_000],
                    recompile=False,
                    flux_codeword=flux_codeword,
                    flux_allocated_duration_ns=flux_allocated_duration_ns,
                    nr_seeds=nr_seeds,
                    sim_cz_qubits=sim_cz_qubits,
                    rb_tasks=rb_tasks_I
                )
                ma2.InterleavedRandomizedBenchmarkingAnalysis(
                    label_base="icl[None]",
                    label_int="icl[104368]",
                    label_int_idle="icl[100000]"
                )

            return rb_tasks_next

        if recompile or recompile == "as needed":
            # This is an optimization that compiles the interleaved RB
            # sequences for the next measurement while measuring the previous
            # one
            if pool is None:
                # Using `with ...:` makes sure the other processes will be terminated
                # `maxtasksperchild` avoid RAM issues
                with multiprocessing.Pool(maxtasksperchild=cl_oql.maxtasksperchild) as pool:
                    run_parallel_iRB(
                        recompile=recompile,
                        pool=pool,
                        rb_tasks_start=rb_tasks_start)
            else:
                # In this case the `pool` to execute the RB compilation tasks
                # is provided, `rb_tasks_start` is expected to be as well
                rb_tasks_next = run_parallel_iRB(
                    recompile=recompile,
                    pool=pool,
                    rb_tasks_start=rb_tasks_start,
                    start_next_round_compilation=start_next_round_compilation)
                return rb_tasks_next
        else:
            # recompile=False no need to parallelize compilation with measurement
            # Perform two-qubit RB (no interleaved gate)
            self.measure_two_qubit_randomized_benchmarking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[None],
                recompile=recompile,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                sim_cz_qubits=sim_cz_qubits,
            )

            # Perform two-qubit RB with CZ interleaved
            self.measure_two_qubit_randomized_benchmarking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[104368],
                recompile=recompile,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                sim_cz_qubits=sim_cz_qubits,
            )

            ma2.InterleavedRandomizedBenchmarkingAnalysis(
                label_base="icl[None]",
                label_int="icl[104368]",
            )

            if measure_idle_flux:
                # Perform two-qubit iRB with idle identity of same duration as CZ
                self.measure_two_qubit_randomized_benchmarking(
                    qubits=qubits,
                    MC=MC,
                    nr_cliffords=nr_cliffords,
                    interleaving_cliffords=[100_000],
                    recompile=recompile,
                    flux_codeword=flux_codeword,
                    flux_allocated_duration_ns=flux_allocated_duration_ns,
                    nr_seeds=nr_seeds,
                    sim_cz_qubits=sim_cz_qubits,
                )
                ma2.InterleavedRandomizedBenchmarkingAnalysis(
                    label_base="icl[None]",
                    label_int="icl[104368]",
                    label_int_idle="icl[100000]"

                )

    def measure_single_qubit_interleaved_randomized_benchmarking_parking(
        self,
        qubits: list,
        MC,
        nr_cliffords=2**np.arange(12),
        nr_seeds: int = 100,
        recompile: bool = 'as needed',
        flux_codeword: str = "cz",
        rb_on_parked_qubit_only: bool = False,
        rb_tasks_start: list = None,
        pool=None,
        start_next_round_compilation: bool = False
    ):
        """
        This function uses the same parallelization approaches as the
        `measure_two_qubit_interleaved_randomized_benchmarking`. See it
        for details and useful comments
        """

        def run_parallel_iRB(
            recompile, pool, rb_tasks_start: list = None,
            start_next_round_compilation: bool = False
        ):

            rb_tasks_next = None

            # 1. Start (non-blocking) compilation for [None]
            if rb_tasks_start is None:
                rb_tasks_start = self.measure_single_qubit_randomized_benchmarking_parking(
                    qubits=qubits,
                    MC=MC,
                    nr_cliffords=nr_cliffords,
                    interleaving_cliffords=[None],
                    recompile=recompile,
                    flux_codeword=flux_codeword,
                    nr_seeds=nr_seeds,
                    rb_on_parked_qubit_only=rb_on_parked_qubit_only,
                    compile_only=True,
                    pool=pool
                )

            # 2. Wait for [None] compilation to finish
            cl_oql.wait_for_rb_tasks(rb_tasks_start)

            # 200_000 by convention is a CZ on the first two qubits with
            # implicit parking on the 3rd qubit
            # 3. Start (non-blocking) compilation for [200_000]
            rb_tasks_CZ_park = self.measure_single_qubit_randomized_benchmarking_parking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[200_000],
                recompile=recompile,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                rb_on_parked_qubit_only=rb_on_parked_qubit_only,
                compile_only=True,
                pool=pool
            )
            # 4. Start the measurement and run the analysis for [None]
            self.measure_single_qubit_randomized_benchmarking_parking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[None],
                recompile=False,  # This of course needs to be False
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                rb_on_parked_qubit_only=rb_on_parked_qubit_only,
                rb_tasks=rb_tasks_start,
            )

            # 5. Wait for [200_000] compilation to finish
            cl_oql.wait_for_rb_tasks(rb_tasks_CZ_park)

            if start_next_round_compilation:
                # Optionally send to the `pool` the tasks of RB compilation to be
                # used on the next round of calling the iRB method
                rb_tasks_next = self.measure_single_qubit_randomized_benchmarking_parking(
                    qubits=qubits,
                    MC=MC,
                    nr_cliffords=nr_cliffords,
                    interleaving_cliffords=[None],
                    recompile=recompile,
                    flux_codeword=flux_codeword,
                    nr_seeds=nr_seeds,
                    rb_on_parked_qubit_only=rb_on_parked_qubit_only,
                    compile_only=True,
                    pool=pool
                )
            # 7. Start the measurement and run the analysis for [200_000]
            self.measure_single_qubit_randomized_benchmarking_parking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[200_000],
                recompile=False,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                rb_on_parked_qubit_only=rb_on_parked_qubit_only,
                rb_tasks=rb_tasks_CZ_park,
            )

            ma2.InterleavedRandomizedBenchmarkingParkingAnalysis(
                label_base="icl[None]",
                label_int="icl[200000]"
            )

            return rb_tasks_next

        if recompile or recompile == "as needed":
            # This is an optimization that compiles the interleaved RB
            # sequences for the next measurement while measuring the previous
            # one
            if pool is None:
                # Using `with ...:` makes sure the other processes will be terminated
                with multiprocessing.Pool(maxtasksperchild=cl_oql.maxtasksperchild) as pool:
                    run_parallel_iRB(
                        recompile=recompile,
                        pool=pool,
                        rb_tasks_start=rb_tasks_start)
            else:
                # In this case the `pool` to execute the RB compilation tasks
                # is provided, `rb_tasks_start` is expected to be as well
                rb_tasks_next = run_parallel_iRB(
                    recompile=recompile,
                    pool=pool,
                    rb_tasks_start=rb_tasks_start,
                    start_next_round_compilation=start_next_round_compilation)
                return rb_tasks_next
        else:
            # recompile=False no need to parallelize compilation with measurement
            # Perform two-qubit RB (no interleaved gate)
            self.measure_single_qubit_randomized_benchmarking_parking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[None],
                recompile=recompile,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                rb_on_parked_qubit_only=rb_on_parked_qubit_only,
            )

            # Perform two-qubit RB with CZ interleaved
            self.measure_single_qubit_randomized_benchmarking_parking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[200_000],
                recompile=recompile,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                rb_on_parked_qubit_only=rb_on_parked_qubit_only,
            )

            ma2.InterleavedRandomizedBenchmarkingParkingAnalysis(
                label_base="icl[None]",
                label_int="icl[200000]"
            )

    def measure_single_qubit_randomized_benchmarking_parking(
        self,
        qubits: list,
        nr_cliffords=2**np.arange(10),
        nr_seeds: int = 100,
        MC=None,
        recompile: bool = 'as needed',
        prepare_for_timedomain: bool = True,
        cal_points: bool = True,
        ro_acq_weight_type: str = "optimal IQ",
        flux_codeword: str = "cz",
        rb_on_parked_qubit_only: bool = False,
        interleaving_cliffords: list = [None],
        compile_only: bool = False,
        pool=None,  # a multiprocessing.Pool()
        rb_tasks=None  # used after called with `compile_only=True`
    ):
        """
        [2020-07-06 Victor] This is a modified copy of the same method from CCL_Transmon.
        The modification is intended for measuring a single qubit RB on a qubit
        that is parked during an interleaving CZ. There is a single qubit RB
        going on in parallel on all 3 qubits. This should cover the most realistic
        case for benchmarking the parking flux pulse.

        Measures randomized benchmarking decay including second excited state
        population.

        For this it:
            - stores single shots using `ro_acq_weight_type` weights (int. logging)
            - uploads a pulse driving the ef/12 transition (should be calibr.)
            - performs RB both with and without an extra pi-pulse
            - Includes calibration poitns for 0, 1, and 2 (g,e, and f)
            - analysis extracts fidelity and leakage/seepage

        Refs:
        Knill PRA 77, 012307 (2008)
        Wood PRA 97, 032306 (2018)

        Args:
            nr_cliffords (array):
                list of lengths of the clifford gate sequences

            nr_seeds (int):
                number of random sequences for each sequence length

            recompile (bool, str {'as needed'}):
                indicate whether to regenerate the sequences of clifford gates.
                By default it checks whether the needed sequences were already
                generated since the most recent change of OpenQL file
                specified in self.cfg_openql_platform_fn

            rb_on_parked_qubit_only (bool):
                `True`: there is a single qubit RB being applied only on the
                3rd qubit (parked qubit)
                `False`: there will be a single qubit RB applied to all 3
                qubits
            other args: behave same way as for 1Q RB r 2Q RB
        """

        # because only 1 seed is uploaded each time
        if MC is None:
            MC = self.instr_MC.get_instr()

        # Settings that have to be preserved, change is required for
        # 2-state readout and postprocessing
        old_weight_type = self.ro_acq_weight_type()
        old_digitized = self.ro_acq_digitized()
        self.ro_acq_weight_type(ro_acq_weight_type)
        self.ro_acq_digitized(False)

        self.prepare_for_timedomain(qubits=qubits)
        MC.soft_avg(1)
        # The detector needs to be defined before setting back parameters
        d = self.get_int_logging_detector(qubits=qubits)
        # set back the settings
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

        for q in qubits:
            q_instr = self.find_instrument(q)
            mw_lutman = q_instr.instr_LutMan_MW.get_instr()
            mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()
        MC.soft_avg(1)  # Not sure this is necessary here...

        net_cliffords = [0, 3]  # always measure double sided
        qubit_idxs = [self.find_instrument(q).cfg_qubit_nr() for q in qubits]

        def send_rb_tasks(pool_):
            tasks_inputs = []
            for i in range(nr_seeds):
                task_dict = dict(
                    qubits=qubit_idxs,
                    nr_cliffords=nr_cliffords,
                    net_cliffords=net_cliffords,  # always measure double sided
                    nr_seeds=1,
                    platf_cfg=self.cfg_openql_platform_fn(),
                    program_name='RB_s{}_ncl{}_net{}_icl{}_{}_{}_park_{}_rb_on_parkonly{}'.format(
                        i, nr_cliffords, net_cliffords, interleaving_cliffords, *qubits,
                        rb_on_parked_qubit_only),
                    recompile=recompile,
                    simultaneous_single_qubit_parking_RB=True,
                    rb_on_parked_qubit_only=rb_on_parked_qubit_only,
                    cal_points=cal_points,
                    flux_codeword=flux_codeword,
                    interleaving_cliffords=interleaving_cliffords
                )
                tasks_inputs.append(task_dict)
            # pool.starmap_async can be used for positional arguments
            # but we are using a wrapper
            rb_tasks = pool_.map_async(cl_oql.parallel_friendly_rb, tasks_inputs)

            return rb_tasks

        if compile_only:
            assert pool is not None
            rb_tasks = send_rb_tasks(pool)
            return rb_tasks

        if rb_tasks is None:
            # Using `with ...:` makes sure the other processes will be terminated
            # avoid starting too mane processes,
            # nr_processes = None will start as many as the PC can handle
            nr_processes = None if recompile else 1
            with multiprocessing.Pool(
                nr_processes,
                maxtasksperchild=cl_oql.maxtasksperchild  # avoid RAM issues
            ) as pool:
                rb_tasks = send_rb_tasks(pool)
                cl_oql.wait_for_rb_tasks(rb_tasks)

        programs_filenames = rb_tasks.get()

        # to include calibration points
        if cal_points:
            sweep_points = np.append(
                # repeat twice because of net clifford being 0 and 3
                np.repeat(nr_cliffords, 2),
                [nr_cliffords[-1] + 0.5] * 2 +
                [nr_cliffords[-1] + 1.5] * 2 +
                [nr_cliffords[-1] + 2.5] * 2,
            )
        else:
            sweep_points = np.repeat(nr_cliffords, 2)

        counter_param = ManualParameter('name_ctr', initial_value=0)
        prepare_function_kwargs = {
            'counter_param': counter_param,
            'programs_filenames': programs_filenames,
            'CC': self.instr_CC.get_instr()}

        # Using the first detector of the multi-detector as this is
        # in charge of controlling the CC (see self.get_int_logging_detector)
        d.set_prepare_function(
            oqh.load_range_of_oql_programs_from_filenames,
            prepare_function_kwargs, detectors="first"
        )

        reps_per_seed = 4094 // len(sweep_points)
        d.set_child_attr("nr_shots", reps_per_seed * len(sweep_points))

        s = swf.None_Sweep(parameter_name='Number of Cliffords', unit='#')

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.tile(sweep_points, reps_per_seed * nr_seeds))

        MC.set_detector_function(d)
        label = 'RB_{}_{}_park_{}_{}seeds_recompile={}_rb_park_only={}_icl{}'.format(
            *qubits, nr_seeds, recompile, rb_on_parked_qubit_only, interleaving_cliffords)
        label += self.msmt_suffix
        # FIXME should include the indices in the exp_metadata and
        # use that in the analysis instead of being dependent on the
        # measurement for those parameters
        rates_I_quad_ch_idx = -2
        cal_pnts_in_dset = np.repeat(["0", "1", "2"], 2)
        MC.run(label, exp_metadata={
            'bins': sweep_points,
            "rates_I_quad_ch_idx": rates_I_quad_ch_idx,
            "cal_pnts_in_dset": list(cal_pnts_in_dset)  # needs to be list to save
        })

        a_q2 = ma2.RandomizedBenchmarking_SingleQubit_Analysis(
            label=label,
            rates_I_quad_ch_idx=rates_I_quad_ch_idx,
            cal_pnts_in_dset=cal_pnts_in_dset
        )
        return a_q2

    def measure_two_qubit_purity_benchmarking(
        self,
        qubits,
        MC,
        nr_cliffords=np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 12.0, 15.0, 20.0, 25.0]
        ),
        nr_seeds=100,
        interleaving_cliffords=[None],
        label="TwoQubit_purityB_{}seeds_{}_{}",
        recompile: bool = "as needed",
        cal_points: bool = True,
        flux_codeword: str = "cz",
    ):
        """
        Measures two qubit purity (aka unitarity) benchmarking.
        It is a modified RB routine which measures the length of
        the Bloch vector at the end of the sequence of cliffords
        to verify the putity of the final state. In this way it is
        not sensitive to systematic errors in the gates allowing
        to estimate whether the RB gate fidelity is limited by
        incoherent errors or inaccurate tuning.

        Refs:
        Joel Wallman, New J. Phys. 17, 113020 (2015)

        Args:
            qubits (list):
                pair of the qubit names on which to perform RB

            nr_cliffords (array):
                lengths of the clifford sequences to perform

            nr_seeds (int):
                number of different clifford sequences of each length

            interleaving_cliffords (list):
                list of integers (or None) which specifies which cliffords
                to interleave the sequence with (for interleaved RB)
                For indices of Clifford group elements go to
                two_qubit_clifford_group.py

            label (str):
                string for formatting the measurement name

            recompile (bool, str {'as needed'}):
                indicate whether to regenerate the sequences of clifford gates.
                By default it checks whether the needed sequences were already
                generated since the most recent change of OpenQL file
                specified in self.cfg_openql_platform_fn

            cal_points (bool):
                should aclibration point (qubits in 0 and 1 states)
                be included in the measurement
        """

        # Settings that have to be preserved, change is required for
        # 2-state readout and postprocessing
        old_weight_type = self.ro_acq_weight_type()
        old_digitized = self.ro_acq_digitized()
        # [2020-07-02] 'optimal IQ' mode is the standard now,
        self.ro_acq_weight_type("optimal IQ")
        self.ro_acq_digitized(False)

        self.prepare_for_timedomain(qubits=qubits)

        # Need to be created before setting back the ro mode
        d = self.get_int_logging_detector(qubits=qubits)

        MC.soft_avg(1)
        # set back the settings
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

        for q in qubits:
            q_instr = self.find_instrument(q)
            mw_lutman = q_instr.instr_LutMan_MW.get_instr()
            mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()

        MC.soft_avg(1)

        programs = []
        t0 = time.time()
        print("Generating {} PB programs".format(nr_seeds))
        qubit_idxs = [self.find_instrument(q).cfg_qubit_nr() for q in qubits]
        for i in range(nr_seeds):
            # check for keyboard interrupt q because generating can be slow
            check_keyboard_interrupt()
            sweep_points = np.concatenate([nr_cliffords, [nr_cliffords[-1] + 0.5] * 4])

            p = cl_oql.randomized_benchmarking(
                qubits=qubit_idxs,
                nr_cliffords=nr_cliffords,
                nr_seeds=1,
                platf_cfg=self.cfg_openql_platform_fn(),
                program_name="TwoQ_PB_int_cl{}_s{}_ncl{}_{}_{}_double".format(
                    i,
                    list(map(int, nr_cliffords)),
                    interleaving_cliffords,
                    qubits[0],
                    qubits[1],
                ),
                interleaving_cliffords=interleaving_cliffords,
                cal_points=cal_points,
                net_cliffords=[
                    0 * 24 + 0,
                    0 * 24 + 21,
                    0 * 24 + 16,
                    21 * 24 + 0,
                    21 * 24 + 21,
                    21 * 24 + 16,
                    16 * 24 + 0,
                    16 * 24 + 21,
                    16 * 24 + 16,
                    3 * 24 + 3,
                ],
                # ZZ, XZ, YZ,
                # ZX, XX, YX
                # ZY, XY, YY
                # (-Z)(-Z) (for f state calibration)
                f_state_cal_pts=True,
                recompile=recompile,
                flux_codeword=flux_codeword,
            )
            p.sweep_points = sweep_points
            programs.append(p)
            print(
                "Generated {} PB programs in {:>7.1f}s".format(i + 1, time.time() - t0),
                end="\r",
            )
        print(
            "Succesfully generated {} PB programs in {:>7.1f}s".format(
                nr_seeds, time.time() - t0
            )
        )

        # to include calibration points
        if cal_points:
            sweep_points = np.append(
                np.repeat(nr_cliffords, 10),
                [nr_cliffords[-1] + 0.5] * 2
                + [nr_cliffords[-1] + 1.5] * 2
                + [nr_cliffords[-1] + 2.5] * 3,
            )
        else:
            sweep_points = np.repeat(nr_cliffords, 10)

        counter_param = ManualParameter("name_ctr", initial_value=0)
        prepare_function_kwargs = {
            "counter_param": counter_param,
            "programs": programs,
            "CC": self.instr_CC.get_instr(),
        }

        # Using the first detector of the multi-detector as this is
        # in charge of controlling the CC (see self.get_int_logging_detector)
        d.set_prepare_function(
            oqh.load_range_of_oql_programs, prepare_function_kwargs,
            detectors="first"
        )
        # d.nr_averages = 128

        reps_per_seed = 4094 // len(sweep_points)
        nr_shots = reps_per_seed * len(sweep_points)
        d.set_child_attr("nr_shots", nr_shots)

        s = swf.None_Sweep(parameter_name="Number of Cliffords", unit="#")

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.tile(sweep_points, reps_per_seed * nr_seeds))

        MC.set_detector_function(d)
        MC.run(
            label.format(nr_seeds, qubits[0], qubits[1]),
            exp_metadata={"bins": sweep_points},
        )
        # N.B. if measurement was interrupted this wont work
        ma2.UnitarityBenchmarking_TwoQubit_Analysis(nseeds=nr_seeds)

    def measure_two_qubit_character_benchmarking(
        self,
        qubits,
        MC,
        nr_cliffords=np.array(
            [
                1.0,
                2.0,
                3.0,
                5.0,
                6.0,
                7.0,
                9.0,
                12.0,
                15.0,
                19.0,
                25.0,
                31.0,
                39.0,
                49,
                62,
                79,
            ]
        ),
        nr_seeds=100,
        interleaving_cliffords=[None, -4368],
        label="TwoQubit_CharBench_{}seeds_icl{}_{}_{}",
        flux_codeword="fl_cw_01",
        recompile: bool = "as needed",
        ch_idxs=np.array([1, 2]),
    ):
        # Refs:
        # Helsen arXiv:1806.02048v1
        # Xue PRX 9, 021011 (2019)

        # Settings that have to be preserved, change is required for
        # 2-state readout and postprocessing
        old_weight_type = self.ro_acq_weight_type()
        old_digitized = self.ro_acq_digitized()
        self.ro_acq_weight_type("SSB")
        self.ro_acq_digitized(False)

        self.prepare_for_timedomain(qubits=qubits)

        MC.soft_avg(1)
        # set back the settings
        d = self.get_int_logging_detector(qubits=qubits)
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

        for q in qubits:
            q_instr = self.find_instrument(q)
            mw_lutman = q_instr.instr_LutMan_MW.get_instr()
            mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()

        MC.soft_avg(1)

        programs = []
        t0 = time.time()
        print("Generating {} Character benchmarking programs".format(nr_seeds))
        qubit_idxs = [self.find_instrument(q).cfg_qubit_nr() for q in qubits]
        for i in range(nr_seeds):
            # check for keyboard interrupt q because generating can be slow
            check_keyboard_interrupt()
            sweep_points = np.concatenate(
                [
                    np.repeat(nr_cliffords, 4 * len(interleaving_cliffords)),
                    nr_cliffords[-1] + np.arange(7) * 0.05 + 0.5,
                ]
            )  # cal pts

            p = cl_oql.character_benchmarking(
                qubits=qubit_idxs,
                nr_cliffords=nr_cliffords,
                nr_seeds=1,
                program_name="Char_RB_s{}_ncl{}_icl{}_{}_{}".format(
                    i,
                    list(map(int, nr_cliffords)),
                    interleaving_cliffords,
                    qubits[0],
                    qubits[1],
                ),
                flux_codeword=flux_codeword,
                platf_cfg=self.cfg_openql_platform_fn(),
                interleaving_cliffords=interleaving_cliffords,
                recompile=recompile,
            )

            p.sweep_points = sweep_points
            programs.append(p)
            print(
                "Generated {} Character benchmarking programs in {:>7.1f}s".format(
                    i + 1, time.time() - t0
                ),
                end="\r",
            )
        print(
            "Succesfully generated {} Character benchmarking programs in {:>7.1f}s".format(
                nr_seeds, time.time() - t0
            )
        )

        counter_param = ManualParameter("name_ctr", initial_value=0)
        prepare_function_kwargs = {
            "counter_param": counter_param,
            "programs": programs,
            "CC": self.instr_CC.get_instr(),
        }

        # Using the first detector of the multi-detector as this is
        # in charge of controlling the CC (see self.get_int_logging_detector)
        d.set_prepare_function(
            oqh.load_range_of_oql_programs, prepare_function_kwargs, detectors="first"
        )
        # d.nr_averages = 128

        reps_per_seed = 4094 // len(sweep_points)
        nr_shots = reps_per_seed * len(sweep_points)
        d.set_child_attr("nr_shots", nr_shots)

        s = swf.None_Sweep(parameter_name="Number of Cliffords", unit="#")

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.tile(sweep_points, reps_per_seed * nr_seeds))

        MC.set_detector_function(d)
        MC.run(
            label.format(nr_seeds, interleaving_cliffords, qubits[0], qubits[1]),
            exp_metadata={"bins": sweep_points},
        )
        # N.B. if measurement was interrupted this wont work
        ma2.CharacterBenchmarking_TwoQubit_Analysis(ch_idxs=ch_idxs)

    def measure_two_qubit_simultaneous_randomized_benchmarking(
        self,
        qubits,
        MC=None,
        nr_cliffords=2 ** np.arange(11),
        nr_seeds=100,
        interleaving_cliffords=[None],
        label="TwoQubit_sim_RB_{}seeds_recompile={}_{}_{}",
        recompile: bool = "as needed",
        cal_points: bool = True,
        ro_acq_weight_type: str = "optimal IQ",
        compile_only: bool = False,
        pool=None,  # a multiprocessing.Pool()
        rb_tasks=None  # used after called with `compile_only=True`
    ):
        """
        Performs simultaneous single qubit RB on two qubits.
        The data of this experiment should be compared to the results of single
        qubit RB to reveal differences due to crosstalk and residual coupling

        Args:
            qubits (list):
                pair of the qubit names on which to perform RB

            nr_cliffords (array):
                lengths of the clifford sequences to perform

            nr_seeds (int):
                number of different clifford sequences of each length

            interleaving_cliffords (list):
                list of integers (or None) which specifies which cliffords
                to interleave the sequence with (for interleaved RB)
                For indices of Clifford group elements go to
                two_qubit_clifford_group.py

            label (str):
                string for formatting the measurement name

            recompile (bool, str {'as needed'}):
                indicate whether to regenerate the sequences of clifford gates.
                By default it checks whether the needed sequences were already
                generated since the most recent change of OpenQL file
                specified in self.cfg_openql_platform_fn

            cal_points (bool):
                should calibration point (qubits in 0, 1 and 2 states)
                be included in the measurement
        """

        # Settings that have to be preserved, change is required for
        # 2-state readout and postprocessing
        old_weight_type = self.ro_acq_weight_type()
        old_digitized = self.ro_acq_digitized()
        self.ro_acq_weight_type(ro_acq_weight_type)
        self.ro_acq_digitized(False)

        self.prepare_for_timedomain(qubits=qubits)
        if MC is None:
            MC = self.instr_MC.get_instr()
        MC.soft_avg(1)

        # The detector needs to be defined before setting back parameters
        d = self.get_int_logging_detector(qubits=qubits)
        # set back the settings
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

        for q in qubits:
            q_instr = self.find_instrument(q)
            mw_lutman = q_instr.instr_LutMan_MW.get_instr()
            mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()

        MC.soft_avg(1)

        def send_rb_tasks(pool_):
            tasks_inputs = []
            for i in range(nr_seeds):
                task_dict = dict(
                    qubits=[self.find_instrument(q).cfg_qubit_nr() for q in qubits],
                    nr_cliffords=nr_cliffords,
                    nr_seeds=1,
                    platf_cfg=self.cfg_openql_platform_fn(),
                    program_name="TwoQ_Sim_RB_int_cl{}_s{}_ncl{}_{}_{}_double".format(
                        i,
                        list(map(int, nr_cliffords)),
                        interleaving_cliffords,
                        qubits[0],
                        qubits[1],
                    ),
                    interleaving_cliffords=interleaving_cliffords,
                    simultaneous_single_qubit_RB=True,
                    cal_points=cal_points,
                    net_cliffords=[0, 3],  # measures with and without inverting
                    f_state_cal_pts=True,
                    recompile=recompile,
                )
                tasks_inputs.append(task_dict)
            # pool.starmap_async can be used for positional arguments
            # but we are using a wrapper
            rb_tasks = pool_.map_async(cl_oql.parallel_friendly_rb, tasks_inputs)

            return rb_tasks

        if compile_only:
            assert pool is not None
            rb_tasks = send_rb_tasks(pool)
            return rb_tasks

        if rb_tasks is None:
            # Using `with ...:` makes sure the other processes will be terminated
            # avoid starting too mane processes,
            # nr_processes = None will start as many as the PC can handle
            nr_processes = None if recompile else 1
            with multiprocessing.Pool(
                nr_processes,
                maxtasksperchild=cl_oql.maxtasksperchild  # avoid RAM issues
            ) as pool:
                rb_tasks = send_rb_tasks(pool)
                cl_oql.wait_for_rb_tasks(rb_tasks)

        programs_filenames = rb_tasks.get()

        # to include calibration points
        if cal_points:
            sweep_points = np.append(
                np.repeat(nr_cliffords, 2),
                [nr_cliffords[-1] + 0.5] * 2
                + [nr_cliffords[-1] + 1.5] * 2
                + [nr_cliffords[-1] + 2.5] * 3,
            )
        else:
            sweep_points = np.repeat(nr_cliffords, 2)

        counter_param = ManualParameter("name_ctr", initial_value=0)
        prepare_function_kwargs = {
            "counter_param": counter_param,
            "programs_filenames": programs_filenames,
            "CC": self.instr_CC.get_instr(),
        }

        # Using the first detector of the multi-detector as this is
        # in charge of controlling the CC (see self.get_int_logging_detector)
        d.set_prepare_function(
            oqh.load_range_of_oql_programs_from_filenames,
            prepare_function_kwargs, detectors="first"
        )
        # d.nr_averages = 128

        reps_per_seed = 4094 // len(sweep_points)
        d.set_child_attr("nr_shots", reps_per_seed * len(sweep_points))

        s = swf.None_Sweep(parameter_name="Number of Cliffords", unit="#")

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.tile(sweep_points, reps_per_seed * nr_seeds))

        MC.set_detector_function(d)
        label = label.format(nr_seeds, recompile, qubits[0], qubits[1])
        MC.run(label, exp_metadata={"bins": sweep_points})

        # N.B. if interleaving cliffords are used, this won't work
        # [2020-07-11 Victor] not sure if NB still holds

        cal_2Q = ["00", "01", "10", "11", "02", "20", "22"]

        rates_I_quad_ch_idx = 0
        cal_1Q = [state[rates_I_quad_ch_idx // 2] for state in cal_2Q]
        a_q0 = ma2.RandomizedBenchmarking_SingleQubit_Analysis(
            label=label,
            rates_I_quad_ch_idx=rates_I_quad_ch_idx,
            cal_pnts_in_dset=cal_1Q
        )
        rates_I_quad_ch_idx = 2
        cal_1Q = [state[rates_I_quad_ch_idx // 2] for state in cal_2Q]
        a_q1 = ma2.RandomizedBenchmarking_SingleQubit_Analysis(
            label=label,
            rates_I_quad_ch_idx=rates_I_quad_ch_idx,
            cal_pnts_in_dset=cal_1Q
        )

        return a_q0, a_q1

    ########################################################
    # Calibration methods
    ########################################################

    def calibrate_mux_ro(
        self,
        qubits,
        calibrate_optimal_weights=True,
        calibrate_threshold=True,
        # option should be here but is currently not implementd
        # update_threshold: bool=True,
        mux_ro_label="Mux_SSRO",
        update_cross_talk_matrix: bool = False,
    ) -> bool:
        """
        Calibrates multiplexed Readout.

        Multiplexed readout is calibrated by
            - iterating over all qubits and calibrating optimal weights.
                This steps involves measuring the transients
                Measuring single qubit SSRO to determine the threshold and
                updating it.
            - Measuring multi qubit SSRO using the optimal weights.

        N.B. Currently only works for 2 qubits
        """

        q0 = self.find_instrument(qubits[0])
        q1 = self.find_instrument(qubits[1])

        q0idx = q0.cfg_qubit_nr()
        q1idx = q1.cfg_qubit_nr()

        UHFQC = q0.instr_acquisition.get_instr()
        self.ro_acq_weight_type("optimal")
        log.info("Setting ro acq weight type to Optimal")
        self.prepare_for_timedomain(qubits)

        if calibrate_optimal_weights:
            # Important that this happens before calibrating the weights
            # 10 is the number of channels in the UHFQC
            for i in range(9):
                UHFQC.set("qas_0_trans_offset_weightfunction_{}".format(i), 0)

            # This resets the crosstalk correction matrix
            UHFQC.upload_crosstalk_matrix(np.eye(10))

            for q_name in qubits:
                q = self.find_instrument(q_name)
                # The optimal weights are calibrated for each individual qubit
                # verify = True -> measure SSRO aftewards to determin the
                # acquisition threshold.
                if calibrate_optimal_weights:
                    q.calibrate_optimal_weights(analyze=True, verify=False, update=True)
                if calibrate_optimal_weights and not calibrate_threshold:
                    log.warning("Updated acq weights but not updating threshold")
                if calibrate_threshold:
                    q.measure_ssro(update=True, nr_shots_per_case=2 ** 13)

        self.measure_ssro_multi_qubit(
            qubits, label=mux_ro_label, result_logging_mode="lin_trans"
        )

        # if len (qubits)> 2:
        #     raise NotImplementedError

        # res_dict = mra.two_qubit_ssro_fidelity(
        #     label='{}_{}'.format(q0.name, q1.name),
        #     qubit_labels=[q0.name, q1.name])
        # V_offset_cor = res_dict['V_offset_cor']

        # N.B. no crosstalk parameters are assigned
        # # weights 0 and 1 are the correct indices because I set the numbering
        # # at the start of this calibration script.
        # UHFQC.qas_trans_offset_weightfunction_0(V_offset_cor[0])
        # UHFQC.qas_trans_offset_weightfunction_1(V_offset_cor[1])

        # # Does not work because axes are not normalized
        # matrix_normalized = res_dict['mu_matrix_inv']
        # matrix_rescaled = matrix_normalized/abs(matrix_normalized).max()
        # UHFQC.upload_transformation_matrix(matrix_rescaled)

        # a = self.check_mux_RO(update=update, update_threshold=update_threshold)
        return True

    def calibrate_cz_single_q_phase(
        self,
        q_osc: str,
        q_spec: str,
        amps,
        q2=None,
        q3=None,
        waveform="cz_NE",
        flux_codeword_park=None,
        update: bool = True,
        prepare_for_timedomain: bool = True,
        MC=None,
    ):
        """
        Calibrate single qubit phase corrections of CZ pulse.

        Parameters
        ----------
        q_osc : str
            Name of the "oscillating" qubit. The phase correction of this
            qubit will be calibrated.
        q_spec: str
            Name of the "spectator" qubit. This qubit is used as the control.
        amps: array_like
            Amplitudes of the phase correction to measure.
        waveform: str
            Name of the waveform used on the "oscillating" qubit. This waveform
            will be reuploaded for each datapoint. Common names are "cz_z" and
            "idle_z"

        Returns
        -------
        succes: bool
            True if calibration succeeded, False if it failed.

        procedure works by performing a conditional oscillation experiment at
        a phase of 90 degrees. If the phase correction is correct, the "off" and
        "on" curves (control qubit in 0 and 1) should interesect. The
        analysis looks for the intersect.
        """

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[q_osc, q_spec])
        if MC is None:
            MC = self.instr_MC.get_instr()

        which_gate = waveform[-2:]
        flux_codeword = waveform[:-3]

        q0idx = self.find_instrument(q_osc).cfg_qubit_nr()
        q1idx = self.find_instrument(q_spec).cfg_qubit_nr()
        if q2 is not None:
            q2idx = self.find_instrument(q2).cfg_qubit_nr()
            q3idx = self.find_instrument(q3).cfg_qubit_nr()
        else:
            q2idx = None
            q3idx = None
        fl_lutman_q0 = self.find_instrument(q_osc).instr_LutMan_Flux.get_instr()

        phase_par = fl_lutman_q0.parameters["cz_phase_corr_amp_{}".format(which_gate)]

        p = mqo.conditional_oscillation_seq(
            q0idx,
            q1idx,
            q2idx,
            q3idx,
            flux_codeword=flux_codeword,
            flux_codeword_park=flux_codeword_park,
            platf_cfg=self.cfg_openql_platform_fn(),
            CZ_disabled=False,
            add_cal_points=False,
            angles=[90],
        )

        CC = self.instr_CC.get_instr()
        CC.eqasm_program(p.filename)
        CC.start()

        s = swf.FLsweep(fl_lutman_q0, phase_par, waveform)
        d = self.get_correlation_detector(
            qubits=[q_osc, q_spec], single_int_avg=True, seg_per_point=2
        )
        d.detector_control = "hard"

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.repeat(amps, 2))
        MC.set_detector_function(d)
        MC.run("{}_CZphase".format(q_osc))

        # The correlation detector has q_osc on channel 0
        a = ma2.Intersect_Analysis(options_dict={"ch_idx_A": 0, "ch_idx_B": 0})

        phase_corr_amp = a.get_intersect()[0]
        if phase_corr_amp > np.max(amps) or phase_corr_amp < np.min(amps):
            print("Calibration failed, intersect outside of initial range")
            return False
        else:
            if update:
                phase_par(phase_corr_amp)
            return True

    def create_dep_graph(self):
        dags = []
        for qi in self.qubits():
            q_obj = self.find_instrument(qi)
            if hasattr(q_obj, "_dag"):
                dag = q_obj._dag
            else:
                dag = q_obj.create_dep_graph()
            dags.append(dag)

        dag = nx.compose_all(dags)

        dag.add_node(self.name + " multiplexed readout")
        dag.add_node(self.name + " resonator frequencies coarse")
        dag.add_node("AWG8 MW-staircase")
        dag.add_node("AWG8 Flux-staircase")

        # Timing of channels can be done independent of the qubits
        # it is on a per frequency per feedline basis so not qubit specific
        dag.add_node(self.name + " mw-ro timing")
        dag.add_edge(self.name + " mw-ro timing", "AWG8 MW-staircase")

        dag.add_node(self.name + " mw-vsm timing")
        dag.add_edge(self.name + " mw-vsm timing", self.name + " mw-ro timing")

        for edge_L, edge_R in self.qubit_edges():
            dag.add_node("Chevron {}-{}".format(edge_L, edge_R))
            dag.add_node("CZ {}-{}".format(edge_L, edge_R))

            dag.add_edge(
                "CZ {}-{}".format(edge_L, edge_R),
                "Chevron {}-{}".format(edge_L, edge_R),
            )
            dag.add_edge(
                "CZ {}-{}".format(edge_L, edge_R), "{} cryo dist. corr.".format(edge_L)
            )
            dag.add_edge(
                "CZ {}-{}".format(edge_L, edge_R), "{} cryo dist. corr.".format(edge_R)
            )

            dag.add_edge(
                "Chevron {}-{}".format(edge_L, edge_R),
                "{} single qubit gates fine".format(edge_L),
            )
            dag.add_edge(
                "Chevron {}-{}".format(edge_L, edge_R),
                "{} single qubit gates fine".format(edge_R),
            )
            dag.add_edge("Chevron {}-{}".format(edge_L, edge_R), "AWG8 Flux-staircase")
            dag.add_edge(
                "Chevron {}-{}".format(edge_L, edge_R),
                self.name + " multiplexed readout",
            )

            dag.add_node("{}-{} mw-flux timing".format(edge_L, edge_R))

            dag.add_edge(
                edge_L + " cryo dist. corr.",
                "{}-{} mw-flux timing".format(edge_L, edge_R),
            )
            dag.add_edge(
                edge_R + " cryo dist. corr.",
                "{}-{} mw-flux timing".format(edge_L, edge_R),
            )

            dag.add_edge(
                "Chevron {}-{}".format(edge_L, edge_R),
                "{}-{} mw-flux timing".format(edge_L, edge_R),
            )
            dag.add_edge(
                "{}-{} mw-flux timing".format(edge_L, edge_R), "AWG8 Flux-staircase"
            )

            dag.add_edge(
                "{}-{} mw-flux timing".format(edge_L, edge_R),
                self.name + " mw-ro timing",
            )

        for qubit in self.qubits():
            dag.add_edge(qubit + " ro pulse-acq window timing", "AWG8 MW-staircase")

            dag.add_edge(qubit + " room temp. dist. corr.", "AWG8 Flux-staircase")
            dag.add_edge(self.name + " multiplexed readout", qubit + " optimal weights")

            dag.add_edge(
                qubit + " resonator frequency",
                self.name + " resonator frequencies coarse",
            )
            dag.add_edge(qubit + " pulse amplitude coarse", "AWG8 MW-staircase")

        for qi in self.qubits():
            q_obj = self.find_instrument(qi)
            # ensures all references are to the main dag
            q_obj._dag = dag

        self._dag = dag
        return dag

    def measure_performance(self, number_of_repetitions: int = 1,
                            post_selection: bool = False,
                            qubit_pairs: list = [['QNW','QC'], ['QNE','QC'],
                                                ['QC','QSW','QSE'], ['QC','QSE','QSW']],
                            do_cond_osc: bool = True,
                            do_1q: bool = True, do_2q: bool = True,
                            do_ro: bool = True):

        """
        Routine runs readout, single-qubit and two-qubit metrics.

        Parameters
        ----------
        number_of_repetitions : int
            defines number of times the routine is repeated.
        post_selection: bool
            defines whether readout fidelities are measured with post-selection.
        qubit_pairs: list
            list of the qubit pairs for which 2-qubit metrics should be measured.
            Each pair should be a list of 2 strings (3 strings, if a parking operation
            is needed) of the respective qubit object names.

        Returns
        -------
        succes: bool
            True if performance metrics were run successfully, False if it failed.

        """

        for _ in range(0, number_of_repetitions):
            try:
                if do_ro: 
                    self.measure_ssro_multi_qubit(self.qubits(), initialize=post_selection)

                if do_1q:
                    for qubit in self.qubits():
                        qubit_obj = self.find_instrument(qubit)
                        qubit_obj.ro_acq_averages(4096)
                        qubit_obj.measure_T1()
                        qubit_obj.measure_ramsey()
                        qubit_obj.measure_echo()
                        qubit_obj.ro_acq_weight_type('SSB')
                        qubit_obj.ro_soft_avg(3)
                        qubit_obj.measure_allxy()
                        qubit_obj.ro_soft_avg(1)
                        qubit_obj.measure_single_qubit_randomized_benchmarking()
                        qubit_obj.ro_acq_weight_type('optimal')
                
                self.ro_acq_weight_type('optimal')
                if do_2q:
                    for pair in qubit_pairs:
                        self.measure_two_qubit_randomized_benchmarking(qubits=pair[:2],
                                                                        MC=self.instr_MC.get_instr())
                        self.measure_state_tomography(qubits=pair[:2], bell_state=0,
                                                        prepare_for_timedomain=True, live_plot=False,
                                                        nr_shots_per_case=2**10, shots_per_meas=2**14,
                                                        label='State_Tomography_Bell_0')
                        
                        if do_cond_osc:
                            self.measure_conditional_oscillation(q0=pair[0], q1=pair[1])
                            self.measure_conditional_oscillation(q0=pair[1], q1=pair[0])
                            # in case of parked qubit, assess its parked phase as well
                            if len(pair) == 3:
                                self.measure_conditional_oscillation( q0=pair[0], q1=pair[1], q2=pair[2], 
                                                                    parked_qubit_seq='ramsey')
            except KeyboardInterrupt:
                print('Keyboard Interrupt')
                break
            except:
                print("Exception encountered during measure_device_performance")


    def calibrate_phases(self, phase_offset_park: float = 0.003,
            phase_offset_sq: float = 0.05, do_park_cal: bool = True, do_sq_cal: bool = True,
            operation_pairs: list = [(['QNW','QC'],'SE'), (['QNE','QC'],'SW'),
                                    (['QC','QSW','QSE'],'SW'), (['QC','QSE','QSW'],'SE')]):    
        
        # First, fix parking phases
        # Set 'qubits': [q0.name, q1.name, q2.name] and 'parked_qubit_seq': 'ramsey'
        if do_park_cal:
            for operation_tuple in operation_pairs:
                pair, gate = operation_tuple
                if len(pair) != 3: continue
    
                q0 = self.find_instrument(pair[0]) # ramsey qubit (we make this be the fluxed one)
                q1 = self.find_instrument(pair[1]) # control qubit
                q2 = self.find_instrument(pair[2]) # parked qubit
    
                # cf.counter_param(0)
                flux_lm = q0.instr_LutMan_Flux.get_instr() # flux_lm of fluxed_qubit
                nested_mc = q0.instr_nested_MC.get_instr() # device object has no nested MC object, get from qubit object
                mc = self.instr_MC.get_instr()
    
                parked_seq = 'ramsey'
                conv_cost_det = det.Function_Detector( get_function=czcf.conventional_CZ_cost_func,
                        msmt_kw={'device': self, 'FL_LutMan_QR': flux_lm,
                            'MC': mc, 'waveform_name': 'cz_{}'.format(gate),
                            'qubits': [q0.name, q1.name, q2.name],
                            'parked_qubit_seq': parked_seq},
                        value_names=['Cost function value',
                            'Conditional phase', 'offset difference', 'missing fraction',
                            'Q0 phase', 'Park Phase OFF', 'Park Phase ON'],
                        result_keys=['cost_function_val',
                            'delta_phi', 'offset_difference', 'missing_fraction',
                            'single_qubit_phase_0', 'park_phase_off', 'park_phase_on'],
                        value_units=['a.u.', 'deg', '%', '%', 'deg', 'deg', 'deg'])
    
                park_flux_lm = q2.instr_LutMan_Flux.get_instr() # flux_lm of fluxed_qubit
    
                # 1D Scan of phase corrections after flux pulse 
                value_min = park_flux_lm.park_amp() - phase_offset_park
                value_max = park_flux_lm.park_amp() + phase_offset_park
                sw = swf.joint_HDAWG_lutman_parameters(name='park_amp',
                                                    parameter_1=park_flux_lm.park_amp,
                                                    parameter_2=park_flux_lm.park_amp_minus,
                                                    AWG=park_flux_lm.AWG.get_instr(),
                                                    lutman=park_flux_lm)
                
                nested_mc.set_sweep_function(sw)
                nested_mc.set_sweep_points(np.linspace(value_min, value_max, 10))
                label = '1D_park_phase_corr_{}_{}_{}'.format(q0.name,q1.name,q2.name)
                nested_mc.set_detector_function(conv_cost_det)
                result = nested_mc.run(label)
    
                # Use ch_to_analyze as 5 for parking phase
                a_obj = ma2.Crossing_Analysis(label=label,
                                            options_dict={'ch_idx': 5, 'target_crossing': 0})
                crossed_value = a_obj.proc_data_dict['root']
                park_flux_lm.park_amp(crossed_value)
                park_flux_lm.park_amp_minus(-crossed_value)

        # Then, fix single-qubit phases
        # Set 'qubits': [q0.name, q1.name] and 'parked_qubit_seq': 'ground'
        if do_sq_cal:
            for operation_tuple in operation_pairs:
                # For each qubit pair, calibrate both individually (requires inversion of arguments)
                for reverse in [False, True]:
                    pair, gate = operation_tuple
                    parked_seq = 'ground'
    
                    if reverse:
                        q0 = self.find_instrument(pair[1]) # ramsey qubit (we make this be the fluxed one)
                        q1 = self.find_instrument(pair[0]) # control qubit
                        if gate=='NE': gate='SW' 
                        elif gate=='NW': gate = 'SE'
                        elif gate=='SW': gate = 'NE'
                        elif gate=='SE': gate = 'NW'
                    else:
                        q0 = self.find_instrument(pair[0]) # ramsey qubit (we make this be the fluxed one)
                        q1 = self.find_instrument(pair[1]) # control qubit
                        gate = gate
            
                    q2 = None
                    # cf.counter_param(0)
                    flux_lm = q0.instr_LutMan_Flux.get_instr() # flux_lm of fluxed_qubit
                    nested_mc = q0.instr_nested_MC.get_instr() # device object has no nested MC object, get from qubit object
                    mc = self.instr_MC.get_instr()
            
                    conv_cost_det = det.Function_Detector( get_function=czcf.conventional_CZ_cost_func,
                                    msmt_kw={'device': self, 'FL_LutMan_QR': flux_lm,
                                        'MC': mc,'waveform_name': 'cz_{}'.format(gate),
                                        'qubits': [q0.name, q1.name], 'parked_qubit_seq': parked_seq},
                                    value_names=['Cost function value',
                                        'Conditional phase', 'offset difference', 'missing fraction',
                                        'Q0 phase', 'Park Phase OFF', 'Park Phase ON'],
                                    result_keys=['cost_function_val',
                                        'delta_phi', 'offset_difference', 'missing_fraction',
                                        'single_qubit_phase_0', 'park_phase_off', 'park_phase_on'],
                                    value_units=['a.u.', 'deg', '%', '%', 'deg', 'deg', 'deg'])
            
                    # 1D Scan of phase corrections after flux pulse 
                    #value_min = flux_lm.cz_phase_corr_amp_SW()-phase_offset
                    value_min = getattr(flux_lm, 'cz_phase_corr_amp_' + gate )()-phase_offset_sq
                    #value_max = flux_lm.cz_phase_corr_amp_SW()+phase_offset
                    value_max = getattr(flux_lm, 'cz_phase_corr_amp_' + gate )()+phase_offset_sq
            
                    label = 'CZ_1D_sweep_phase_corr_{}'.format(gate)
                    nested_mc.set_sweep_function(getattr(flux_lm, 'cz_phase_corr_amp_' + gate ))
                    nested_mc.set_sweep_points(np.linspace(value_min, value_max, 10))
                    nested_mc.set_detector_function(conv_cost_det)
                    result = nested_mc.run(label)
            
                    # Use ch_to_analyze as 4 for single qubit phases ('Q0 phase')
                    a_obj = ma2.Crossing_Analysis(label=label,
                                                  options_dict={'ch_idx': 4, 'target_crossing': 0})
                    crossed_value = a_obj.proc_data_dict['root']
                    getattr(flux_lm, 'cz_phase_corr_amp_' + gate )(crossed_value)


    def calibrate_cz_thetas(self, phase_offset: float = 1,
            operation_pairs: list = [(['QNW','QC'],'SE'), (['QNE','QC'],'SW'),
                                    (['QC','QSW','QSE'],'SW'), (['QC','QSE','QSW'],'SE')]):    

        # Set 'qubits': [q0.name, q1.name] and 'parked_qubit_seq': 'ground'
        for operation_tuple in operation_pairs:
            pair, gate = operation_tuple
            parked_seq = 'ground'
    
            q0 = self.find_instrument(pair[0]) # ramsey qubit (we make this be the fluxed one)
            q1 = self.find_instrument(pair[1]) # control qubit
            q2 = None
            gate = gate
        
            # cf.counter_param(0)
            flux_lm = q0.instr_LutMan_Flux.get_instr() # flux_lm of fluxed_qubit
            nested_mc = q0.instr_nested_MC.get_instr() # device object has no nested MC object, get from qubit object
            mc = self.instr_MC.get_instr()
        
            conv_cost_det = det.Function_Detector( get_function=czcf.conventional_CZ_cost_func,
                            msmt_kw={'device': self, 'FL_LutMan_QR': flux_lm,
                                'MC': mc,'waveform_name': 'cz_{}'.format(gate),
                                'qubits': [q0.name, q1.name], 'parked_qubit_seq': parked_seq},
                            value_names=['Cost function value',
                                'Conditional phase', 'offset difference', 'missing fraction',
                                'Q0 phase', 'Park Phase OFF', 'Park Phase ON'],
                            result_keys=['cost_function_val',
                                'delta_phi', 'offset_difference', 'missing_fraction',
                                'single_qubit_phase_0', 'park_phase_off', 'park_phase_on'],
                            value_units=['a.u.', 'deg', '%', '%', 'deg', 'deg', 'deg'])
        
            # 1D Scan of phase corrections after flux pulse 
            value_min = getattr(flux_lm, 'cz_theta_f_' + gate )()-phase_offset
            #value_max = flux_lm.cz_phase_corr_amp_SW()+phase_offset
            value_max = getattr(flux_lm, 'cz_theta_f_' + gate )()+phase_offset
        
            label = 'CZ_1D_sweep_theta_{}'.format(gate)
            nested_mc.set_sweep_function(getattr(flux_lm, 'cz_theta_f_' + gate ))
            nested_mc.set_sweep_points(np.linspace(value_min, value_max, 10))
            nested_mc.set_detector_function(conv_cost_det)
            result = nested_mc.run(label)
        
            # Use ch_to_analyze as 4 for single qubit phases ('Q0 phase')
            a_obj = ma2.Crossing_Analysis(label=label,
                                          options_dict={'ch_idx': 1, 'target_crossing': 180})
            crossed_value = a_obj.proc_data_dict['root']
            getattr(flux_lm, 'cz_theta_f_' + gate )(crossed_value)

    def prepare_for_inspire(self):
        for lutman in ['mw_lutman_QNW','mw_lutman_QNE','mw_lutman_QC','mw_lutman_QSW','mw_lutman_QSE']:
            self.find_instrument(lutman).set_inspire_lutmap()
        self.prepare_for_timedomain(qubits=self.qubits())
        self.find_instrument(self.instr_MC()).soft_avg(1)

def _acq_ch_map_to_IQ_ch_map(acq_ch_map):
    acq_ch_map_IQ = {}
    for acq_instr, ch_map in acq_ch_map.items():
        acq_ch_map_IQ[acq_instr] = {}
        for qubit, ch in ch_map.items():
            acq_ch_map_IQ[acq_instr]["{} I".format(qubit)] = ch
            acq_ch_map_IQ[acq_instr]["{} Q".format(qubit)] = ch + 1
    return acq_ch_map_IQ
