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
from typing import List, Union, Tuple, Optional
import itertools as itt
from math import ceil

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import (
    ManualParameter,
    InstrumentRefParameter,
    Parameter,
)
from qce_circuit.library.repetition_code_circuit import (
    InitialStateContainer,
    InitialStateEnum,
)

from pycqed.analysis import multiplexed_RO_analysis as mra
from pycqed.measurement import detector_functions as det
reload(det)
from pycqed.measurement import cz_cost_functions as cf
reload(cf)

from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import tomography as tomo
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.analysis_v2.repeated_stabilizer_analysis import RepeatedStabilizerAnalysis
from pycqed.utilities.general import (
    check_keyboard_interrupt,
    print_exception,
    get_gate_directions,
    get_frequency_waveform,
    get_DAC_amp_frequency,
    get_Ch_amp_frequency,
    get_parking_qubits,
)

from pycqed.instrument_drivers.physical_instruments.QuTech_AWG_Module import (
    QuTech_AWG_Module,
)
# from pycqed.instrument_drivers.physical_instruments.QuTech_CCL import CCL
# from pycqed.instrument_drivers.physical_instruments.QuTech_QCC import QCC
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
import pycqed.analysis_v2.tomography_2q_v2 as tomo_v2

from pycqed.utilities import learner1D_minimizer as l1dm
from pycqed.utilities import learnerND_minimizer as lndm

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


def _acq_ch_map_to_IQ_ch_map(acq_ch_map):
    acq_ch_map_IQ = {}
    for acq_instr, ch_map in acq_ch_map.items():
        acq_ch_map_IQ[acq_instr] = {}
        for qubit, ch in ch_map.items():
            acq_ch_map_IQ[acq_instr]["{} I".format(qubit)] = ch
            acq_ch_map_IQ[acq_instr]["{} Q".format(qubit)] = ch + 1
    return acq_ch_map_IQ


class DeviceCCL(Instrument):
    """
    Device object for systems controlled using the
    CCLight (CCL), QuMa based CC (QCC) or Distributed CC (CC).
    FIXME: class name is outdated
    """
    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        self.msmt_suffix = '_' + name

        self.add_parameter(
            'qubits',
            parameter_class=ManualParameter,
            initial_value=[],
            vals=vals.Lists(elt_validator=vals.Strings())
        )

        self.add_parameter(
            'qubit_edges',
            parameter_class=ManualParameter,
            docstring="Denotes edges that connect qubits. "
                        "Used to define the device topology, needed for two qubit gates.",
            initial_value=[[]],
            vals=vals.Lists(elt_validator=vals.Lists(elt_validator=vals.Strings()))
        )

        self.add_parameter(
            'qubits_by_feedline',
            parameter_class=ManualParameter,
            docstring="Nested list of qubits as divided by feedline."
                        "Used to sort qubits for readout preparation.",
            initial_value=[[]],
            vals=vals.Lists(elt_validator=vals.Lists(elt_validator=vals.Strings()))
        )

        self.add_parameter(
            'ro_lo_freq',
            unit='Hz',
            docstring='Frequency of the common LO for all RO pulses.',
            parameter_class=ManualParameter
        )

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
            parameter_class=InstrumentRefParameter,)
        self.add_parameter('instr_nested_MC',
                           label='Nested MeasurementControl',
                           parameter_class=InstrumentRefParameter)

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
            vals=vals.Enum("SSB", "optimal", "optimal IQ", "custom"),
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
            "tim_ro_latency_3",
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
                    ("ro_3", self.tim_ro_latency_3()),
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
            if "mw" in lat_key:
                # Check name to prevent crash when instrument not specified
                AWG_name = self.get("instr_AWG_{}".format(lat_key))

                if AWG_name is not None:
                    AWG = self.find_instrument(AWG_name)
                    using_QWG = AWG.__class__.__name__ == "QuTech_AWG_Module"
                    if not using_QWG:
                        AWG.stop()
                        for qubit in self.qubits():
                            q_obj = self.find_instrument(qubit)
                            MW_lm = self.find_instrument(q_obj.instr_LutMan_MW())
                            if AWG_name == MW_lm.AWG():
                                extra_delay = q_obj.mw_fine_delay()
                                awg_chs     = MW_lm.channel_I(), MW_lm.channel_Q()
                                log.debug("Setting `sigouts_{}_delay` to {:4g}"
                                          " in {}".format(awg_chs[0], lat_fine, AWG.name))
                                AWG.set("sigouts_{}_delay".format(awg_chs[0]-1), lat_fine+extra_delay)
                                AWG.set("sigouts_{}_delay".format(awg_chs[1]-1), lat_fine+extra_delay)
                            if self.find_instrument(qubit).instr_LutMan_LRU():
                                 q_obj = self.find_instrument(qubit)
                                 MW_lm_LRU = self.find_instrument(q_obj.instr_LutMan_LRU())
                                 if AWG_name == MW_lm_LRU.AWG():
                                    awg_chs     = MW_lm_LRU.channel_I(), MW_lm_LRU.channel_Q()
                                    log.debug("Setting `sigouts_{}_delay` to {:4g}"
                                              " in {}".format(awg_chs[0], lat_fine, AWG.name))
                                    AWG.set("sigouts_{}_delay".format(awg_chs[0]-1), lat_fine)
                                    AWG.set("sigouts_{}_delay".format(awg_chs[1]-1), lat_fine)
                        AWG.start()
                        # All channels are set globally from the device object.
                        # for i in range(8):  # assumes the AWG is an HDAWG
                        #     log.debug(
                        #         "Setting `sigouts_{}_delay` to {:4g}"
                        #         " in {}".format(i, lat_fine, AWG.name)
                        #     )
                        #     AWG.set("sigouts_{}_delay".format(i), lat_fine)
                        # ch_not_ready = 8
                        # while ch_not_ready > 0:
                        #     ch_not_ready = 0
                        #     for i in range(8):
                        #         ch_not_ready += AWG.geti("sigouts/{}/busy".format(i))
                        #     check_keyboard_interrupt()

    def prepare_fluxing(self, qubits):
        for qb_name in qubits:
            qb = self.find_instrument(qb_name)
            try:
                fl_lutman = qb.instr_LutMan_Flux.get_instr()
                fl_lutman.load_waveforms_onto_AWG_lookuptable()
            except Exception as e:
                warnings.warn("Could not load flux pulses for {}".format(qb))
                warnings.warn("Exception {}".format(e))

    def prepare_readout(self, qubits,
        reduced: bool = False,
        qubit_int_weight_type_dict: dict=None):
        """
        Configures readout for specified qubits.

        Args:
            qubits (list of str):
                list of qubit names that have to be prepared
            qubit_int_weight_type_dict (dict of str):
                dictionary specifying individual acquisition weight types
                for qubits. example:
                    qubit_int_weight_type_dict={'X3':'optimal IQ',
                                                'D4':'optimal',
                                                'D5':'optimal IQ'}
                Note: Make sure to use ro_acq_weight_type('custom') for
                this to work!
                Warning: Only allows for 'optimal' and 'optimal IQ'.
                Also, order of dictionary should be same as 'qubits'.
                Only works for the int_log_det and int_avg_det.
        """
        log.info('Configuring readout for {}'.format(qubits))

        if not reduced:
            self._prep_ro_sources(qubits=qubits)

        acq_ch_map = self._prep_ro_assign_weights(qubits=qubits,
            qubit_int_weight_type_dict = qubit_int_weight_type_dict)
        self._prep_ro_integration_weights(qubits=qubits,
            qubit_int_weight_type_dict = qubit_int_weight_type_dict)
        if not reduced:
            self._prep_ro_pulses(qubits=qubits)
        self._prep_ro_instantiate_detectors(qubits=qubits,
                                            acq_ch_map=acq_ch_map)

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
        LO_lutman = self.find_instrument(qubits[0]).instr_LutMan_RO.get_instr()
        LO.frequency.set(LO_lutman.LO_freq())
        LO.power(self.ro_pow_LO())
        LO.on()

        for qb_name in qubits:
            qb = self.find_instrument(qb_name)
            ro_lutman = qb.instr_LutMan_RO.get_instr()
            # set RO modulation to use common LO frequency
            mod_freq = qb.ro_freq() - ro_lutman.LO_freq()
            log.info("Setting modulation freq of {} to {}".format(qb_name, mod_freq))
            qb.ro_freq_mod(mod_freq)

            LO_q = qb.instr_LO_ro.get_instr()
            if LO_q is not LO:
                LO_q.frequency.set(ro_lutman.LO_freq())
                #LO_q.power(self.ro_pow_LO())
                LO_q.on()
                #raise ValueError("Expect a single LO to drive all feedlines")

    def _prep_ro_assign_weights(self, qubits, qubit_int_weight_type_dict=None):
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

        if not qubit_int_weight_type_dict:
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

        else:
            log.info('Using custom acq_channel_map')
            for q in qubits:
                assert q in qubit_int_weight_type_dict.keys(), f"Qubit {q} not present in qubit_int_weight_type_dict"
            acq_ch_map = {}
            for qb_name, w_type in qubit_int_weight_type_dict.items():
                qb = self.find_instrument(qb_name)
                acq_instr = qb.instr_acquisition()
                if not acq_instr in acq_ch_map.keys():
                    if w_type == 'optimal IQ':
                        acq_ch_map[acq_instr] = {f'{qb_name} I': 0,
                                                 f'{qb_name} Q': 1}
                        log.info("Assigning {} w0 and w1 to qubit {}".format(
                                 acq_instr, qb_name))
                    else:
                        acq_ch_map[acq_instr] = {f'{qb_name}': 0}
                        log.info("Assigning {} w0 to qubit {}".format(
                                 acq_instr, qb_name))
                    # even if the mode does not use Q weight, we still assign this
                    # this is for when switching back to the qubit itself
                    qb.ro_acq_weight_chI(0)
                    qb.ro_acq_weight_chQ(1)

                else:
                    _nr_channels_taken = len(acq_ch_map[acq_instr])
                    if w_type == 'optimal IQ':
                        acq_ch_map[acq_instr][f'{qb_name} I'] = _nr_channels_taken
                        acq_ch_map[acq_instr][f'{qb_name} Q'] = _nr_channels_taken+1
                        log.info("Assigning {} w{} and w{} to qubit {}".format(
                                 acq_instr, _nr_channels_taken, _nr_channels_taken+1, qb_name))
                        if _nr_channels_taken+1 > 10:
                            # There are only 10 acq_weight_channels per UHF.
                            # use optimal ro weights or read out less qubits.
                            raise ValueError("Trying to assign too many acquisition weights")
                        qb.ro_acq_weight_chI(_nr_channels_taken)
                        qb.ro_acq_weight_chQ(_nr_channels_taken+1)
                    else:
                        acq_ch_map[acq_instr][f'{qb_name}'] = _nr_channels_taken
                        log.info("Assigning {} w{} to qubit {}".format(
                                 acq_instr, _nr_channels_taken, qb_name))
                        if _nr_channels_taken > 10:
                            # There are only 10 acq_weight_channels per UHF.
                            # use optimal ro weights or read out less qubits.
                            raise ValueError("Trying to assign too many acquisition weights")
                        qb.ro_acq_weight_chI(_nr_channels_taken)

        log.info("Clearing UHF correlation settings")
        for acq_instr_name in acq_ch_map.keys():
            self.find_instrument(acq_instr).reset_correlation_params()
            self.find_instrument(acq_instr).reset_crosstalk_matrix()

        # Stored as a private attribute for debugging purposes.
        self._acq_ch_map = acq_ch_map

        return acq_ch_map

    def _prep_ro_integration_weights(self, qubits, qubit_int_weight_type_dict=None):
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

        elif 'custom'  in self.ro_acq_weight_type():
            assert qubit_int_weight_type_dict != None
            for q in qubits:
                assert q in qubit_int_weight_type_dict.keys(), f"Qubit {q} not present in qubit_int_weight_type_dict"
            log.info("using optimal custom mapping")
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
                    if qubit_int_weight_type_dict[qb_name] == 'optimal IQ':
                        print('setting the optimal Q')
                        acq_instr.set('qas_0_integration_weights_{}_real'.format(
                            qb.ro_acq_weight_chQ()), opt_WQ)
                        acq_instr.set('qas_0_integration_weights_{}_imag'.format(
                            qb.ro_acq_weight_chQ()), opt_WI)
                        acq_instr.set('qas_0_rotations_{}'.format(
                            qb.ro_acq_weight_chQ()), 1.0 + 1.0j)

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
            # resonator_combs = [[r] for r in resonators_in_lm[ro_lm.name]] + \
            #     [resonators_in_lm[ro_lm.name]]
            resonator_combs = [resonators_in_lm[ro_lm.name]]
            log.info('Setting resonator combinations for {} to {}'.format(
                ro_lm.name, resonator_combs))
            ro_lm.acquisition_delay(self.ro_acq_delay())
            ro_lm.resonator_combinations(resonator_combs)
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

        # qubits passed to but not used in function?

        if self.ro_acq_weight_type() == 'SSB':
            result_logging_mode = 'raw'
        elif 'optimal' in self.ro_acq_weight_type():
            # lin_trans includes
            result_logging_mode = 'lin_trans'
            if self.ro_acq_digitized():
                result_logging_mode = 'digitized'
        elif 'custom' in self.ro_acq_weight_type():
            # lin_trans includes
            result_logging_mode = "lin_trans"

        log.info("Setting result logging mode to {}".format(result_logging_mode))

        if (self.ro_acq_weight_type() != "optimal") and\
           (self.ro_acq_weight_type() != "custom"):
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
        elif 'custom' in self.ro_acq_weight_type():
            # lin_trans includes
            result_logging_mode = "lin_trans"

        log.info("Setting result logging mode to {}".format(result_logging_mode))

        if (self.ro_acq_weight_type() != "optimal") and\
           (self.ro_acq_weight_type() != "custom"):
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

    def prepare_for_timedomain(self, qubits: list, reduced: bool = False,
                               bypass_flux: bool = False, 
                               prepare_for_readout: bool = True):
        """
        Prepare setup for a timedomain experiment:

        Args:
            qubits (list of str):
                list of qubit names that have to be prepared
        """
        if prepare_for_readout:
            self.prepare_readout(qubits=qubits, reduced=reduced)
        if reduced:
            return
        if bypass_flux is False:
            self.prepare_fluxing(qubits=qubits)
        self.prepare_timing()

        for qb_name in qubits:
            qb = self.find_instrument(qb_name)
            qb._prep_td_sources()
            qb._prep_mw_pulses()
            # qb._set_mw_fine_delay(qb.mw_fine_delay())

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
            self.prepare_readout(qubits=list_qubits_used)
            self.prepare_fluxing(qubits=list_qubits_used)
            for q in list_qubits_used:  #only on the CZ qubits we add the ef pulses
                self.find_instrument(q)._prep_td_sources()
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

        measured_qubits = [q0,q1]
        if q2 is not None:
            measured_qubits.append(q2)
        if q3 is not None:
            measured_qubits.append(q3)

        MC.set_detector_function(self.get_int_avg_det(qubits=measured_qubits))

        MC.run(
            "conditional_oscillation_{}_{}_&_{}_{}_x{}_wb{}_wa{}{}{}{}".format(
                q0, q1, q2, q3, cz_repetitions,
                wait_time_before_flux_ns, wait_time_after_flux_ns,parked_qubit_seq,
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

    def measure_conditional_oscillation_multi(
            self,
            pairs: list, 
            parked_qbs: list,
            flux_codeword="cz",
            phase_offsets:list = None,
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
        Specifically qubit q0 of each pair is prepared in the superposition, while q1 is in 0 or 1 state.
        Next the flux pulse is applied. Finally pi/2 afterrotation around various axes
        is applied to q0, and q1 is flipped back (if neccessary) to 0 state.
        Plotting the probabilities of the zero state for each qubit as a function of
        the afterrotation axis angle, and comparing case of q1 in 0 or 1 state, enables to
        measure the conditional phase and estimale the leakage of the Cphase gate.

        Refs:
        Rol arXiv:1903.02492, Suppl. Sec. D
        IARPA M6 for the flux-dance, not publicly available 

        Args:
            pairs (lst(lst)):
                Contains all pairs with the order (q0,q1) where q0 in 'str' is the target and q1 in 
                'str' is the control. This is based on qubits that are parked in the flux-dance.  

            parked_qbs(lst):
                Contains a list of all qubits that are required to be parked.  
                This is based on qubits that are parked in the flux-dance.   

            flux_codeword (str):
                the gate to be applied to the qubit pair [q0, q1]

            flux_codeword_park (str):
                optionally park qubits. This is designed according to the flux-dance. if 
                one has to measure a single pair, has to provide more qubits for parking. 
                Problem here is parked qubits are hardcoded in cc config, thus one has to include the extra 
                parked qubits in this file.  
                (single qubit operation on q2) or a 'cz' pulse on q2-q3.
                NB: depending on the CC configurations the parking can be
                implicit in the main `cz`

            prepare_for_timedomain (bool):
                should the insruments be reconfigured for time domain measurement

            disable_cz (bool):
                execute the experiment with no flux pulse applied

            disabled_cz_duration_ns (int):
                waiting time to emulate the flux pulse

            wait_time_before_flux_ns (int):
                additional waiting time (in ns) before the flux pulse. 

            wait_time_after_flux_ns (int):
                additional waiting time (in ns) after the flux pulse, before
                the final afterrotations

        """

        if self.ro_acq_weight_type() != 'optimal':
            # this occurs because the detector groups qubits per feedline.
            # If you do not pay attention, this will mess up the analysis of
            # this experiment.
            raise ValueError('Current conditional analysis is not working with {}'.format(self.ro_acq_weight_type()))

        if MC is None:
            MC = self.instr_MC.get_instr()

        Q_idxs_target = []
        Q_idxs_control = []
        Q_idxs_parked = []
        list_qubits_used = []
        ramsey_qubits = []

        for i,pair in enumerate(pairs):
            # print ( 'Pair (target,control) {} : ({},{})'. format(i+1,pair[0],pair[1]))
            assert pair[0] in self.qubits()
            assert pair[1] in self.qubits()
            Q_idxs_target += [self.find_instrument(pair[0]).cfg_qubit_nr()]
            Q_idxs_control += [self.find_instrument(pair[1]).cfg_qubit_nr()]
            list_qubits_used += [pair[0], pair[1]]
            ramsey_qubits += [pair[0]]

        # print('Q_idxs_target : {}'.format(Q_idxs_target))
        # print('Q_idxs_control : {}'.format(Q_idxs_control))
        # print('list_qubits_used : {}'.format(list_qubits_used))

        if parked_qbs is not None:
            Q_idxs_parked = [self.find_instrument(Q).cfg_qubit_nr() for Q in parked_qbs]

        if prepare_for_timedomain:
            for i, q in enumerate(list_qubits_used): 
                mw_lutman = self.find_instrument(q).instr_LutMan_MW.get_instr()
                mw_lutman.set_default_lutmap()
                lm = mw_lutman.LutMap()
                lm[27] = {'name': 'rXm180', 'phi': 0, 'theta': -180, 'type': 'ge'}
                # This is awkward because cw9 should have rx12 by default
                lm[31] = {"name": "rX12", "theta": 180, "phi": 0, "type": "ef"}
                mw_lutman.LutMap(lm)

            self.prepare_for_timedomain(qubits=list_qubits_used)

            for i, q in enumerate(np.concatenate([ramsey_qubits])): 
                mw_lutman = self.find_instrument(q).instr_LutMan_MW.get_instr()
                # load_phase_pulses will also upload other waveforms
                if phase_offsets == None:
                    mw_lutman.load_phase_pulses_to_AWG_lookuptable()
                else:
                    mw_lutman.load_phase_pulses_to_AWG_lookuptable(
                        phases=np.arange(0,360,20)+phase_offsets[i])
                mw_lutman.load_waveforms_onto_AWG_lookuptable(
                    regenerate_waveforms=True)
            # prepare_parked qubits
            for q in parked_qbs:
                fl_lm = self.find_instrument(q).instr_LutMan_Flux.get_instr()
                fl_lm.load_waveform_onto_AWG_lookuptable(
                    wave_id='park', regenerate_waveforms=True)

        # These are hardcoded angles in the mw_lutman for the AWG8
        # only x2 and x3 downsample_swp_points available
        angles = np.arange(0, 341, 20 * downsample_swp_points)

        p = mqo.conditional_oscillation_seq_multi(
            Q_idxs_target,
            Q_idxs_control,
            Q_idxs_parked,
            platf_cfg=self.cfg_openql_platform_fn(),
            disable_cz=disable_cz,
            disabled_cz_duration=disabled_cz_duration_ns,
            angles=angles,
            wait_time_before_flux=wait_time_before_flux_ns,
            wait_time_after_flux=wait_time_after_flux_ns,
            flux_codeword=flux_codeword,
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
        d = self.get_int_avg_det(qubits=list_qubits_used)
        MC.set_detector_function(d)

        MC.run(
            "conditional_oscillation_{}_x{}_{}{}".format(
                list_qubits_used, cz_repetitions,
                self.msmt_suffix, label,
            ),
            disable_snapshot_metadata=disable_metadata,
        )

        if len(pairs) > 1:
            # qb_ro_order = np.sum([ list(self._acq_ch_map[key].keys()) for key in self._acq_ch_map.keys()])
            # qubits_by_feedline = [['D1','X1'],  
            #                         ['D2','Z1','D3','D4','D5','D7','X2','X3','Z3'],
            #                         ['D6','D8','D9','X4','Z2','Z4']]
            # qb_ro_order = sorted(np.array(pairs).flatten().tolist(), 
            #                     key=lambda x: [i for i,qubits in enumerate(qubits_by_feedline) if x in qubits])
            qb_ro_order = [qb for qb_dict in self._acq_ch_map.values() for qb in qb_dict.keys()]
        else:
            # qb_ro_order = [ list(self._acq_ch_map[key].keys()) for key in self._acq_ch_map.keys()][0]
            qb_ro_order = [pairs[0][0], pairs[0][1]]
        
        result_dict = {}
        for i, pair in enumerate(pairs):
            ch_osc = qb_ro_order.index(pair[0])
            ch_spec= qb_ro_order.index(pair[1])

            options_dict = {
                'ch_idx_osc': ch_osc,
                'ch_idx_spec': ch_spec
            }
            a = ma2.Conditional_Oscillation_Analysis(
                options_dict=options_dict,
                extract_only=extract_only)

            result_dict['pair_{}_delta_phi_a'.format(i+1)] = \
            a.proc_data_dict['quantities_of_interest']['phi_cond'].n % 360

            result_dict['pair_{}_missing_frac_a'.format(i+1)]  = \
            a.proc_data_dict['quantities_of_interest']['missing_fraction'].n

            result_dict['pair_{}_offset_difference_a'.format(i+1)]  = \
            a.proc_data_dict['quantities_of_interest']['offs_diff'].n

            result_dict['pair_{}_phi_0_a'.format(i+1)]  = \
            (a.proc_data_dict['quantities_of_interest']['phi_0'].n+180) % 360 - 180

            result_dict['pair_{}_phi_1_a'.format(i+1)]  = \
            (a.proc_data_dict['quantities_of_interest']['phi_1'].n+180) % 360 - 180

        return result_dict

    def measure_flux_arc_dc_conditional_oscillation(
            self,
            qubit_high: str,
            qubit_low: str,
            flux_array: Optional[np.ndarray] = None,
            flux_sample_points: int = 21,
            disable_metadata: bool = False,
            prepare_for_timedomain: bool = True,
            analyze: bool = True,
            ):
        assert self.ro_acq_weight_type() == 'optimal', "Expects device acquisition weight type to be 'optimal'"

        # Get instruments
        nested_MC = self.instr_nested_MC.get_instr()
        qubit_high_instrument = self.find_instrument(qubit_high)
        _flux_instrument = qubit_high_instrument.instr_FluxCtrl.get_instr()
        _flux_parameter = _flux_instrument[f'FBL_{qubit_high}']
        qubits_awaiting_prepare = [qubit_high, qubit_low]
        self.prepare_readout(qubits_awaiting_prepare)
        parked_qubits = get_parking_qubits(qubit_high, qubit_low)

        original_current: float = qubit_high_instrument.fl_dc_I0()
        if flux_array is None:
            flux_array = np.linspace(-40e-6, 40e-6, flux_sample_points) + original_current

        local_prepare = ManualParameter('local_prepare', initial_value=prepare_for_timedomain)
        def wrapper():
            a = self.measure_conditional_oscillation_multi(
                pairs=[[qubit_high, qubit_low]],
                parked_qbs=parked_qubits,
                disable_metadata=disable_metadata,
                prepare_for_timedomain=local_prepare(),
                extract_only=True,
            )
            local_prepare(False)  # Turn off prepare for followup measurements
            return {
                'pair_1_delta_phi_a': a['pair_1_delta_phi_a'],
                'pair_1_missing_frac_a': a['pair_1_missing_frac_a'],
                'pair_1_offset_difference_a': a['pair_1_offset_difference_a'],
                'pair_1_phi_0_a': a['pair_1_phi_0_a'],
                'pair_1_phi_1_a': a['pair_1_phi_1_a'],
            }

        d = det.Function_Detector(
            wrapper,
            result_keys=['pair_1_missing_frac_a', 'pair_1_delta_phi_a', 'pair_1_phi_0_a'],
            value_names=['missing_fraction', 'phi_cond', 'phi_0'],
            value_units=['a.u.', 'degree', 'degree'],
        )

        nested_MC.set_detector_function(d)
        nested_MC.set_sweep_function(_flux_parameter)
        nested_MC.set_sweep_points(np.atleast_1d(flux_array))

        response = None
        label = f'conditional_oscillation_dc_flux_arc_{qubit_high}'
        try:
            response = nested_MC.run(label, disable_snapshot_metadata=disable_metadata)
        except Exception as e:
            log.warn(e)
        finally:
            _flux_parameter(original_current)
        if analyze:
            a = ma2.FineBiasAnalysis(
                initial_bias=original_current,
                label=label,
            )
            a.run_analysis()
            return a
        return response


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

    def measure_ssro_multi_qubit(
            self,
            qubits: list,
            f_state: bool = False,
            nr_shots_per_case: int = 2**13,  # 8192
            prepare_for_timedomain: bool = True,
            result_logging_mode='raw',
            initialize: bool = False,
            analyze=True,
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


        """
        log.info("{}.measure_ssro_multi_qubit for qubits{}".format(self.name, qubits))

        # # off and on, not including post selection init measurements yet
        # nr_cases = 2**len(qubits)  # e.g., 00, 01 ,10 and 11 in the case of 2q
        # nr_shots = nr_shots_per_case*nr_cases

        # off and on, not including post selection init measurements yet
        if f_state:
            nr_cases = 3 ** len(qubits)
        else:
            nr_cases = 2 ** len(qubits)

        if initialize:
            nr_shots = 2 * nr_shots_per_case * nr_cases
        else:
            nr_shots = nr_shots_per_case * nr_cases

        self.ro_acq_digitized(False)

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits, bypass_flux=True)
        if MC is None:
            MC = self.instr_MC.get_instr()

        d = self.get_int_logging_detector(
            qubits, result_logging_mode=result_logging_mode
        )
        # Check detector order
        det_qubits, _idxs = np.unique([ name.split(' ')[2]
            for name in d.value_names], return_index=True)
        det_qubits = det_qubits[_idxs]
        if not all(qubits != det_qubits):
            # this occurs because the detector groups qubits per feedline.
            # If you do not pay attention, this will mess up the analysis of
            # this experiment.
            print('Detector qubits do not match order specified.{} vs {}'.format(qubits, det_qubits))
            qubits = det_qubits

        qubit_idxs = [self.find_instrument(qn).cfg_qubit_nr() for qn in qubits]
        p = mqo.multi_qubit_off_on(
            qubit_idxs,
            initialize=initialize,
            second_excited_state=f_state,
            platf_cfg=self.cfg_openql_platform_fn(),
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())


        shots_per_meas = int(
            np.floor(np.min([2**20, nr_shots]) / nr_cases) * nr_cases
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
            second_excited_state: bool = False,
            result_logging_mode='raw',
            initialize: bool = False,
            analyze=True,
            shots_per_meas: int = 2**16,
            nr_flux_dance:int=None,
            wait_time :float=None,
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
        if second_excited_state:
            nr_cases = 3 ** len(qubits)

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
                                   nr_flux_dance=nr_flux_dance,
                                   wait_time = wait_time,
                                   second_excited_state=second_excited_state,
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

    def measure_MUX_SSRO(self,
            qubits: list,
            f_state: bool = False,
            nr_shots_per_case: int = 2**13,
            heralded_init: bool = False,
            prepare_for_timedomain: bool = True,
            analyze: bool = True,
            disable_metadata: bool = False):
        '''
        Measures single shot readout multiplexed assignment fidelity matrix.
        Supports second excited state as well!
        '''
        assert self.ro_acq_digitized() == False, 'Analog readout required'
        assert 'IQ' in self.ro_acq_weight_type(), 'IQ readout is required!'
        MC = self.instr_MC.get_instr()
        # Configure lutmap
        for qubit in qubits:
            qb = self.find_instrument(qubit)
            mwl = qb.instr_LutMan_MW.get_instr()
            mwl.set_default_lutmap()
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits = qubits)

        # get qubit idx 
        Q_idxs = [self.find_instrument(q).cfg_qubit_nr() for q in qubits]
        # Set UHF number of shots
        _cycle = (2**len(qubits))
        _states = ['0', '1']
        if f_state:
            _cycle = (3**len(qubits))
            _states = ['0', '1', '2']
        nr_shots = _cycle*nr_shots_per_case
        if heralded_init:
            nr_shots *= 2
        uhfqc_max_shots = 2**20
        if nr_shots < uhfqc_max_shots:
            # all shots can be acquired in a single UHF run
            shots_per_run = nr_shots
        else:
            # Number of UHF acquisition runs
            nr_runs = ceil(nr_shots/uhfqc_max_shots) 
            shots_per_run = int((nr_shots/nr_runs)/_cycle)*_cycle
            nr_shots = nr_runs*shots_per_run
        # Compile sequence
        p = mqo.MUX_RO_sequence(
            qubit_idxs = Q_idxs,
            heralded_init = heralded_init,
            states = _states,
            platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='Shot', unit='#',
                             upload=True)
        MC.soft_avg(1)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        d = self.get_int_logging_detector(qubits=qubits)
        for det in d.detectors:
            det.nr_shots = shots_per_run
        MC.set_detector_function(d)
        MC.live_plot_enabled(False)
        label = f'MUX_SSRO_{"_".join(qubits)}'
        MC.run(label+self.msmt_suffix, disable_snapshot_metadata=disable_metadata)
        # MC.live_plot_enabled(True)
        # Analysis
        if analyze:
            ma2.ra.Multiplexed_Readout_Analysis(
                qubits=qubits,
                f_state=f_state,
                heralded_init=heralded_init,
                label=label)

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
                                      averages=2**15,
                                      return_analysis=True
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
            if return_analysis:
                return ssro_dict
            else:
                return True

    def measure_msmt_induced_dephasing(
            self,
            meas_qubits: list, 
            target_qubits: list,
            measurement_time_ns: int,
            echo_times: list = None,
            echo_phases: list = None,
            disable_metadata=False,
            prepare_for_timedomain: bool=True):

        assert self.ro_acq_digitized() == False
        assert self.ro_acq_weight_type() == 'optimal'
        ###################
        # setup qubit idxs
        ###################
        all_qubits = meas_qubits+target_qubits
        meas_idxs = [ self.find_instrument(q).cfg_qubit_nr() for q in meas_qubits ]
        target_idxs = [ self.find_instrument(q).cfg_qubit_nr() for q in target_qubits ]
        ###########################################
        # RO preparation (assign res_combinations)
        ###########################################
        RO_lms = np.unique([self.find_instrument(q).instr_LutMan_RO() for q in all_qubits])
        qubit_RO_lm = { self.find_instrument(q).cfg_qubit_nr() : 
                       (self.find_instrument(q).name, 
                        self.find_instrument(q).instr_LutMan_RO()) for q in all_qubits }
        main_qubits = [] # qubits that belong to RO lm where there is an ancilla
        exception_qubits = [] # qubits that belong to RO lm  without ancilla
        res_combs = {}
        for lm in RO_lms:
            res_combs[lm] = []
            comb1 = [] # used for target+meas_qubits
            comb2 = [] # used for only target qubits 
            comb3 = [] # used for only meas qubits 
            targ_q_in_lm = []
            meas_q_in_lm = []
            # Sort resontator combinations
            for idx in meas_idxs+target_idxs:
                if qubit_RO_lm[idx][1] == lm:
                    comb1 += [idx]
                    comb2 += [idx]
            res_combs[lm] += [comb1]
            for idx in meas_idxs:
                if qubit_RO_lm[idx][1] == lm:
                    comb2.remove(idx)
                    comb3 += [idx]
            if comb2 != comb1:
                res_combs[lm] += [comb2]
            if len(comb3) != 0:
                res_combs[lm] += [comb3]
            # Sort main and exception qubits
            for idx in meas_idxs+target_idxs:
                if qubit_RO_lm[idx][1] == lm:
                    if qubit_RO_lm[idx][0] in target_qubits:
                        targ_q_in_lm.append(qubit_RO_lm[idx][0])
                    if qubit_RO_lm[idx][0] in meas_qubits:
                        meas_q_in_lm.append(qubit_RO_lm[idx][0])
            if len(meas_q_in_lm) == 0:
                exception_qubits += targ_q_in_lm
            else:
                main_qubits += meas_q_in_lm
                main_qubits += targ_q_in_lm
        # Time-domain preparation
        ordered_qubits = main_qubits+exception_qubits
        if prepare_for_timedomain:
            self.prepare_for_timedomain(ordered_qubits, bypass_flux=True)
            for lm in RO_lms:
                ro_lm = self.find_instrument(lm)
                ro_lm.resonator_combinations(res_combs[lm])
                ro_lm.load_DIO_triggered_sequence_onto_UHFQC()
            for i, q in enumerate(target_qubits):
                mw_lm = self.find_instrument(f'MW_lutman_{q}')
                if echo_times != None:
                    assert echo_phases != None
                    print(f'Echo phase upload on {mw_lm.name}')    
                    mw_lm.LutMap()[30] = {'name': 'rEcho', 'theta': 180, 
                                          'phi': echo_phases[i], 'type': 'ge'}
                    mw_lm.LutMap()[27] = {"name": "rXm180", "theta": -180,
                                          "phi": 0, "type": "ge"}
                    mw_lm.LutMap()[28] = {"name": "rYm180", "theta": -180,
                                          "phi": 90, "type": "ge"}
                mw_lm.load_phase_pulses_to_AWG_lookuptable()
        if exception_qubits != []:
            exception_idxs = [self.find_instrument(q).cfg_qubit_nr()
                             for q in exception_qubits]
        else:
            exception_idxs = None

        p = mqo.Msmt_induced_dephasing_ramsey(
            q_rams = target_idxs,
            q_meas = meas_idxs,
            echo_times = echo_times,
            meas_time = measurement_time_ns,
            exception_qubits=exception_idxs,
            platf_cfg=self.cfg_openql_platform_fn())

        d = self.get_int_avg_det(qubits=ordered_qubits)
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        MC = self.instr_MC.get_instr()
        MC.soft_avg(1)
        # MC.live_plot_enabled(True)
        MC.set_sweep_function(s)
        sw_pts = np.concatenate((np.repeat(np.arange(0, 360, 20), 6), 
                                 np.array([360, 361, 362, 364])))
        MC.set_sweep_points(sw_pts)
        MC.set_detector_function(d)
        if isinstance(echo_times, str):
            echo_seq = echo_times
        elif isinstance(echo_times, type(None)):
            echo_seq = 'None'
        elif isinstance(echo_times, list):
            echo_seq = 'single_echo'
        MC.run(f'Msmt_induced_dephasing_echo_seq_{echo_seq}',
                disable_snapshot_metadata=disable_metadata)
        a = ma2.mra.measurement_dephasing_analysis(
                meas_time=measurement_time_ns*1e-9,
                target_qubits= target_qubits,
                exception_qubits=exception_qubits)

    def measure_chevron(
        self,
        q0: str,
        q_spec: str,
        q_parks=None,
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
        buffer_time=0,
        target_qubit_sequence: str = "ramsey",
        waveform_name="square",
        recover_q_spec: bool = False,
        second_excited_state: bool = False,
        disable_metadata: bool = False,
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
            q_parks (list):
                qubits to move out of the interaction zone by applying a
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

            second_excited_state (bool):
                Applies f12 transition pulse before flux pulse.

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
        if q_parks is not None:
            q_park_idxs = [self.find_instrument(q_park).cfg_qubit_nr() for q_park in q_parks]
            for q_park in q_parks:
                q_park_idx = self.find_instrument(q_park).cfg_qubit_nr()
                fl_lutman_park = self.find_instrument(q_park).instr_LutMan_Flux.get_instr()
                if fl_lutman_park.park_amp() < 0.1:
                    # This can cause weird behaviour if not paid attention to.
                    log.warning("Square amp for park pulse < 0.1")
                if fl_lutman_park.park_length() < np.max(lengths):
                    log.warning("Square length shorter than max Chevron length")
        else:
            q_park_idxs = None

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
            q_park_idxs,
            buffer_time=buffer_time,
            buffer_time2=buffer_time,
            flux_cw=flux_cw,
            platf_cfg=self.cfg_openql_platform_fn(),
            target_qubit_sequence=target_qubit_sequence,
            cc=self.instr_CC.get_instr().name,
            recover_q_spec=recover_q_spec,
            second_excited_state=second_excited_state,
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)
        self.instr_CC.get_instr().start()
        
        d = self.get_correlation_detector(
            qubits=[q0, q_spec],
            single_int_avg=True,
            seg_per_point=1,
            always_prepare=True,
        )
        # d = self.get_int_avg_det(qubits=[q0, q_spec])
        # d = self.int_log_det
        MC.set_sweep_function(amp_par)
        MC.set_sweep_function_2D(sw)
        MC.set_detector_function(d)

        prepared_state_label: str = "_prep_state_f" if second_excited_state else "_prep_state_e"
        label = f"Chevron{prepared_state_label} {q0} {q_spec} {target_qubit_sequence}"

        if not adaptive_sampling:
            MC.set_sweep_points(amps)
            MC.set_sweep_points_2D(lengths)
            MC.run(label, mode="2D",
                   disable_snapshot_metadata=disable_metadata)
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

    def measure_cryoscope(
        self,
        qubits,
        times,
        MC=None,
        nested_MC=None,
        double_projections: bool = False,
        wait_time_flux: int = 0,
        update_FIRs: bool=False,
        update_IIRs: bool=False,
        waveform_name: str = "square",
        max_delay=None,
        twoq_pair=[2, 0],
        disable_metadata: bool = False,
        init_buffer=0,
        analyze: bool = True,
        prepare_for_timedomain: bool = True,
        ):
        """
        Performs a cryoscope experiment to measure the shape of a flux pulse.

        Args:
            qubits  (list):
                a list of two target qubits

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
        assert self.ro_acq_weight_type() == 'optimal'
        assert not (update_FIRs and update_IIRs), 'Can only either update IIRs or FIRs' 
        if update_FIRs or update_IIRs:
            assert analyze==True, 'Analsis has to run for filter update'
        if MC is None:
            MC = self.instr_MC.get_instr()
        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()
        for q in qubits:
            assert q in self.qubits()
        Q_idxs = [self.find_instrument(q).cfg_qubit_nr() for q in qubits]
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubits)
        if max_delay is None:
            max_delay = 0 
        else:
            max_delay = np.max(times) + 40e-9
        Fl_lutmans = [self.find_instrument(q).instr_LutMan_Flux.get_instr() \
                      for q in qubits]
        if waveform_name == "square":
            Sw_functions = [swf.FLsweep(lutman, lutman.sq_length,
                            waveform_name="square") for lutman in Fl_lutmans]
            swfs = swf.multi_sweep_function(Sw_functions)
            flux_cw = "sf_square"
        elif waveform_name == "custom_wf":
            Sw_functions = [swf.FLsweep(lutman, lutman.custom_wf_length, 
                            waveform_name="custom_wf") for lutman in Fl_lutmans]
            swfs = swf.multi_sweep_function(Sw_functions)
            flux_cw = "sf_custom_wf"
        else:
            raise ValueError(
                'waveform_name "{}" should be either '
                '"square" or "custom_wf"'.format(waveform_name)
            )

        p = mqo.Cryoscope(
            qubit_idxs=Q_idxs,
            flux_cw=flux_cw,
            twoq_pair=twoq_pair,
            wait_time_flux=wait_time_flux,
            platf_cfg=self.cfg_openql_platform_fn(),
            cc=self.instr_CC.get_instr().name,
            double_projections=double_projections,
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)
        self.instr_CC.get_instr().start()

        MC.set_sweep_function(swfs)
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
            qubits=qubits,
            values_per_point=values_per_point,
            values_per_point_suffex=values_per_point_suffex,
            single_int_avg=True,
            always_prepare=True
        )
        MC.set_detector_function(d)
        label = 'Cryoscope_{}_amps'.format('_'.join(qubits))
        MC.run(label,disable_snapshot_metadata=disable_metadata)
        # Run analysis
        if analyze:
            a = ma2.cv2.multi_qubit_cryoscope_analysis(
                label='Cryoscope',
                update_IIRs=update_IIRs,
                update_FIRs=update_FIRs)
        if update_FIRs:
            for qubit, fltr in a.proc_data_dict['conv_filters'].items():
                lin_dist_kern = self.find_instrument(f'lin_dist_kern_{qubit}')
                filter_dict = {'params': {'weights': fltr},
                               'model': 'FIR', 'real-time': True }
                lin_dist_kern.filter_model_04(filter_dict)
        elif update_IIRs:
            for qubit, fltr in a.proc_data_dict['exponential_filter'].items():
                lin_dist_kern = self.find_instrument(f'lin_dist_kern_{qubit}')
                filter_dict = {'params': fltr,
                               'model': 'exponential', 'real-time': True }
                if fltr['amp'] > 0:
                    print('Amplitude of filter is positive (overfitting).')
                    print('Filter not updated.')
                    return True
                else:
                    # Check wich is the first empty exponential filter
                    for i in range(4):
                        _fltr = lin_dist_kern.get(f'filter_model_0{i}')
                        if _fltr == {}:
                            lin_dist_kern.set(f'filter_model_0{i}', filter_dict)
                            return True
                        else:
                            print(f'filter_model_0{i} used.')
                    print('All exponential filter tabs are full. Filter not updated.')
        return True

    def measure_cryoscope_long(
        self,
        qubit: str,
        times: list,
        frequencies: list,
        MC = None,
        nested_MC = None,
        analyze: bool = True,
        update_IIRs: bool = False,
        prepare_for_timedomain: bool = True,
        ):
        """
        Performs a cryoscope experiment to measure the shape of a flux pulse.
        This long version of cryoscope, uses a spectroscopy type experiment to
        probe the frequency of the qubit for pulses longer than those allowed
        by conventional ramsey measurements:
                   t     __                 _____________
        MW  :  |<----->_/  \_              |             |
                ________________________   | Measurement |    
        Flux: _|                        |_ |_____________|

        Args:
            qubit  (list):
                target qubit

            times   (array):
                array of measurment times

            prepare_for_timedomain (bool):
                calls self.prepare_for_timedomain on start
        """
        assert self.ro_acq_weight_type() == 'optimal'
        assert self.ro_acq_digitized() == True
        if update_IIRs:
            assert analyze
        if MC is None:
            MC = self.instr_MC.get_instr()
        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()
        # Setup sweep times of experiment
        max_length = np.max(times) + 200e-9
        Times_ns = times*1e9
        # Get instruments
        Q_inst = self.find_instrument(qubit)
        flux_lm = Q_inst.instr_LutMan_Flux.get_instr()
        HDAWG_inst = flux_lm.AWG.get_instr()
        MW_LO_inst = Q_inst.instr_LO_mw.get_instr()
        # Save previous operating parameters
        LO_frequency = MW_LO_inst.frequency()
        mw_gauss_width = Q_inst.mw_gauss_width() 
        mw_channel_amp = Q_inst.mw_channel_amp()
        cfg_max_length = flux_lm.cfg_max_wf_length()
        # For spectroscopy, we'll use a 80 ns qubit pi pulse.
        # (we increase the duration of the pulse to probe 
        # a narrower frequency spectrum)
        Q_inst.mw_gauss_width(20e-9)
        Q_inst.mw_channel_amp(mw_channel_amp/2*1.3)
        # Prepare for the experiment:
        # Additional to the normal qubit preparation, this requires
        # changing the driver of the HDAWG to allow for longer flux
        # pulses without running out of memory. To do this, we hack
        # the HDAWG by changing a method of the ZIHDAWG class at 
        # runtime. THEREFORE THIS ROUTINE MUST BE USED CAUTIOUSLY. 
        # MAKING SURE THAT THE METHOD OF CLASS IS RESET AT THE END 
        # OF THE ROUTINE IS NECESSARY FOR THE HDAWG TO WORK NORMALLY
        # AFTER COMPLETION!!!
        if prepare_for_timedomain:
            # HACK HDAWG to upload long square pulses
            from functools import partial
            import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase
            # Define new waveform table method
            def _get_waveform_table_new(self, awg_nr: int):
                '''
                Replaces the "_get_waveform_table" method of the HDAWG
                '''
                ch = awg_nr*2
                wf_table = []
                if 'flux' in self.cfg_codeword_protocol():
                    for cw_r in range(1):
                        for cw_l in range(1):
                            wf_table.append((zibase.gen_waveform_name(ch, cw_l),
                                             zibase.gen_waveform_name(ch+1, cw_r)))
                    
                    is_odd_channel = flux_lm.cfg_awg_channel()%2
                    if is_odd_channel: 
                        for cw_r in range(1):
                            for cw_l in range(1,2):
                                wf_table.append((zibase.gen_waveform_name(ch, cw_l),
                                                 zibase.gen_waveform_name(ch+1, cw_r)))
                    else:
                        for cw_r in range(1,2):
                            for cw_l in range(1):
                                wf_table.append((zibase.gen_waveform_name(ch, cw_l),
                                                 zibase.gen_waveform_name(ch+1, cw_r)))
                print('WARNING THIS HDAWG IS HACKED!!!!')
                print(wf_table)
                return wf_table
            # Store old method
            HDAWG_inst._original_method = HDAWG_inst._get_waveform_table
            # Replace to new method
            HDAWG_inst._get_waveform_table = partial(_get_waveform_table_new,
                                                     HDAWG_inst)
            # Define new codeword table method
            def _codeword_table_preamble_new(self, awg_nr):
                """
                Defines a snippet of code to use in the beginning of an AWG program in order to define the waveforms.
                The generated code depends on the instrument type. For the HDAWG instruments, we use the seWaveDIO
                function.
                """
                program = ''

                wf_table = self._get_waveform_table(awg_nr=awg_nr)
                is_odd_channel = flux_lm.cfg_awg_channel()%2 #this only works if wf_table has a length of 2 
                if is_odd_channel:
                    dio_cws = [0, 1]
                else:
                    dio_cws = [0, 8]
                # for dio_cw, (wf_l, wf_r) in enumerate(wf_table):
                # Assuming wf_table looks like this: [('wave_ch7_cw000', 'wave_ch8_cw000'), ('wave_ch7_cw000', 'wave_ch8_cw001')]
                for dio_cw, (wf_l, wf_r) in zip(dio_cws, wf_table):# hardcoded for long cryoscope on even awg channels
                    csvname_l = self.devname + '_' + wf_l
                    csvname_r = self.devname + '_' + wf_r

                    # FIXME: Unfortunately, 'static' here also refers to configuration required for flux HDAWG8
                    if self.cfg_sideband_mode() == 'static' or self.cfg_codeword_protocol() == 'flux':
                        # program += 'assignWaveIndex(\"{}\", \"{}\", {});\n'.format(
                        #     csvname_l, csvname_r, dio_cw)
                        program += 'setWaveDIO({}, \"{}\", \"{}\");\n'.format(
                            dio_cw, csvname_l, csvname_r)
                    elif self.cfg_sideband_mode() == 'real-time' and self.cfg_codeword_protocol() == 'novsm_microwave':
                        # program += 'setWaveDIO({}, 1, 2, \"{}\", 1, 2, \"{}\");\n'.format(
                        #     dio_cw, csvname_l, csvname_r)
                        program += 'assignWaveIndex(1, 2, \"{}\", 1, 2, \"{}\", {});\n'.format(
                            csvname_l, csvname_r, dio_cw)
                    else:
                        raise Exception("Unknown modulation type '{}' and codeword protocol '{}'" \
                                            .format(self.cfg_sideband_mode(), self.cfg_codeword_protocol()))

                if self.cfg_sideband_mode() == 'real-time':
                    program += '// Initialize the phase of the oscillators\n'
                    program += 'executeTableEntry(1023);\n'
                return program
            # Store old method
            HDAWG_inst._original_codeword_method = HDAWG_inst._codeword_table_preamble
            # Replace to new method
            HDAWG_inst._codeword_table_preamble = partial(_codeword_table_preamble_new,
                                                     HDAWG_inst)
            # Prepare flux pulse
            flux_lm.sq_length(max_length)
            flux_lm.cfg_max_wf_length(max_length)
            # Change LutMap accordingly to only upload one waveform
            flux_lm.LutMap({0: {'name': 'i', 'type': 'idle'}, # idle always required
                            1: {'name': 'square', 'type': 'square'}})
            try:
                # Load flux waveform
                flux_lm.AWG.get_instr().stop()
                flux_lm.load_waveform_onto_AWG_lookuptable(regenerate_waveforms=True, wave_id='square')
                flux_lm.cfg_awg_channel_amplitude()
                flux_lm.cfg_awg_channel_range()
                flux_lm.AWG.get_instr().start()
                self.prepare_for_timedomain(qubits=[qubit])
                Q_inst.prepare_readout()
            except:
                print_exception()
                print('Execution failed. Reseting HDAWG and flux lutman...')
                # Reset old method in HDAWG
                if prepare_for_timedomain:
                    # HDAWG_inst._get_waveform_table = partial(HDAWG_inst._original_method, HDAWG_inst)
                    HDAWG_inst._get_waveform_table = HDAWG_inst._original_method
                    HDAWG_inst._codeword_table_preamble = HDAWG_inst._original_codeword_method
                    del HDAWG_inst._original_method
                    del HDAWG_inst._original_codeword_method
                # Reset mw settings
                MW_LO_inst.frequency(LO_frequency)
                Q_inst.mw_gauss_width(mw_gauss_width)
                Q_inst.mw_channel_amp(mw_channel_amp)
                # Reset flux settings
                flux_lm.sq_length(20e-9)
                flux_lm.cfg_max_wf_length(cfg_max_length)
                HDAWG_inst.reset_waveforms_zeros()
                for i in range(4):
                    HDAWG_inst._clear_dirty_waveforms(i)
                flux_lm.set_default_lutmap()
                flux_lm.load_waveforms_onto_AWG_lookuptable()
                # Raise error
                raise RuntimeError('Preparation failed.')
        # Compile experiment sequence
        p = mqo.Cryoscope_long(
            qubit = Q_inst.cfg_qubit_nr(),
            times_ns = Times_ns,
            t_total_ns=max_length*1e9,
            platf_cfg = Q_inst.cfg_openql_platform_fn())
        # Sweep functions
        d = Q_inst.int_avg_det
        # d = self.get_int_avg_det(qubits=[qubit]) # this should be the right detector
        swf1 = swf.OpenQL_Sweep(openql_program=p,
                     CCL=self.instr_CC.get_instr())
        swf2 = MW_LO_inst.frequency
        sweep_freqs = frequencies-Q_inst.mw_freq_mod()
        # Setup measurement control
        MC.soft_avg(1)
        MC.live_plot_enabled(True)
        MC.set_sweep_function(swf1)
        MC.set_sweep_function_2D(swf2)
        MC.set_sweep_points(Times_ns)
        MC.set_sweep_points_2D(sweep_freqs)
        MC.set_detector_function(d)
        try:
            label = f"Cryoscope_long_{Q_inst.name}"
            _max_length = max_length - 200e-9
            if _max_length > 1e-6:
                label += f"_{_max_length*1e6:.0f}us"
            else:
                label += f"_{_max_length*1e9:.0f}ns"
            # Analysis relies on snapshot
            MC.run(label, mode='2D',
                    disable_snapshot_metadata=False)
        except:
            analyze = False
            print_exception()
        print('Reseting HDAWG and flux lutman...')
        # Reset operating parameters
        MW_LO_inst.frequency(LO_frequency)
        Q_inst.mw_gauss_width(mw_gauss_width)
        Q_inst.mw_channel_amp(mw_channel_amp)
        # Reset old method in HDAWG
        if prepare_for_timedomain:
            # HDAWG_inst._get_waveform_table = partial(HDAWG_inst._original_method, HDAWG_inst)
            HDAWG_inst._get_waveform_table = HDAWG_inst._original_method
            HDAWG_inst._codeword_table_preamble = HDAWG_inst._original_codeword_method
            del HDAWG_inst._original_method
            del HDAWG_inst._original_codeword_method
        # Reset flux settings
        flux_lm.sq_length(20e-9)
        flux_lm.cfg_max_wf_length(cfg_max_length)
        HDAWG_inst.reset_waveforms_zeros()
        for i in range(4):
            HDAWG_inst._clear_dirty_waveforms(i)
        flux_lm.set_default_lutmap()
        flux_lm.load_waveforms_onto_AWG_lookuptable()
        # Run analysis
        if analyze:
            a = ma2.cv2.Cryoscope_long_analysis(update_IIR=update_IIRs)
            if update_IIRs:
                lin_dist_kern = flux_lm.instr_distortion_kernel.get_instr()
                filtr = {'params': a.proc_data_dict['exponential_filter'],
                         'model': 'exponential', 'real-time': True }
                # Check wich is the first empty exponential filter
                for i in [0,1,2,3,5,6,7,8]:
                    _fltr = lin_dist_kern.get(f'filter_model_0{i}')
                    if _fltr == {}:
                        lin_dist_kern.set(f'filter_model_0{i}', filtr)
                        return True
                    else:
                        print(f'filter_model_0{i} used.')
                print('All exponential filter tabs are full. Filter not updated.')
        return True

    def measure_cryoscope_vs_amp(
        self,
        q0: str,
        amps,
        flux_cw: str = 'fl_cw_06',
        duration: float = 100e-9,
        amp_parameter: str = "channel",
        MC=None,
        twoq_pair=[2, 0],
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
        elif amp_parameter == "dac":
            sw = swf.FLsweep(fl_lutman, fl_lutman.sq_amp, waveform_name="square")
        else:
            raise ValueError(
                'amp_parameter "{}" should be either '
                '"channel" or "dac"'.format(amp_parameter)
            )

        p = mqo.Cryoscope(
            q0idx,
            buffer_time1=0,
            buffer_time2=max_delay,
            twoq_pair=twoq_pair,
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

    def measure_timing_diagram(self, qubits: list, 
                               flux_latencies, microwave_latencies,
                               MC=None,
                               pulse_length=40e-9, flux_cw='fl_cw_06',
                               prepare_for_timedomain: bool = True):
        """
        Measure the ramsey-like sequence with the 40 ns flux pulses played between
        the two pi/2. While playing this sequence the delay of flux and microwave pulses
        is varied (relative to the readout pulse), looking for configuration in which
        the pulses arrive at the sample in the desired order.

        After measuting the pattern use ma2.Timing_Cal_Flux_Fine with manually
        chosen parameters to match the drawn line to the measured patern.

        Args:
            qubits  (str)     :
                list of the target qubits
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
            self.prepare_for_timedomain(qubits)

        for q in qubits:
            assert q in self.qubits()
        
        Q_idxs = [self.find_instrument(q).cfg_qubit_nr() for q in qubits]

        Fl_lutmans = [self.find_instrument(q).instr_LutMan_Flux.get_instr() \
                      for q in qubits]
        for lutman in Fl_lutmans:
            lutman.sq_length(pulse_length)

        CC = self.instr_CC.get_instr()

        p = mqo.FluxTimingCalibration(qubit_idxs=Q_idxs,
                                      platf_cfg=self.cfg_openql_platform_fn(),
                                      flux_cw=flux_cw,
                                      cal_points=False)

        CC.eqasm_program(p.filename)

        d = self.get_int_avg_det(qubits=qubits, single_int_avg=True)
        MC.set_detector_function(d)
        
        s = swf.tim_flux_latency_sweep(self)
        s2 = swf.tim_mw_latency_sweep(self)
        MC.set_sweep_functions([s, s2])
        # MC.set_sweep_functions(s2)

        # MC.set_sweep_points(microwave_latencies)
        MC.set_sweep_points(flux_latencies)
        MC.set_sweep_points_2D(microwave_latencies)
        label = 'Timing_diag_{}'.format('_'.join(qubits))
        MC.run_2D(label)

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

    def measure_two_qubit_randomized_benchmarking(
        self,
        qubits,
        nr_cliffords=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0,
                               12.0, 15.0, 20.0, 25.0, 30.0, 50.0]),
        nr_seeds=100,
        interleaving_cliffords=[None],
        label="TwoQubit_RB_{}seeds_recompile={}_icl{}_{}_{}_{}",
        recompile: bool = "as needed",
        cal_points=True,
        flux_codeword="cz",
        flux_allocated_duration_ns: int = None,
        sim_cz_qubits: list = None,
        compile_only: bool = False,
        prepare_for_timedomain: bool = True,
        pool=None,  # a multiprocessing.Pool()
        rb_tasks=None,  # used after called with `compile_only=True`
        MC=None
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
        if MC is None:
            MC = self.instr_MC.get_instr()

        old_weight_type = self.ro_acq_weight_type()
        old_digitized = self.ro_acq_digitized()
        old_avg = self.ro_acq_averages()
        # Settings that have to be preserved, change is required for
        if prepare_for_timedomain:
            # 2-state readout and postprocessing
            self.ro_acq_weight_type("optimal IQ")
            self.ro_acq_digitized(False)          
            for q in qubits:
                q_instr = self.find_instrument(q)
                mw_lutman = q_instr.instr_LutMan_MW.get_instr()
                # mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()
                mw_lutman.set_default_lutmap()
            self.prepare_for_timedomain(qubits=qubits)
        MC.soft_avg(1)  # FIXME: changes state
        # The detector needs to be defined before setting back parameters
        d = self.get_int_logging_detector(qubits=qubits)
        # set back the settings
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

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
                    program_name="TwoQ_RB_int_cl_s{}_ncl{}_icl{}_{}_{}".format(
                        int(i),
                        list(map(int, nr_cliffords)),
                        interleaving_cliffords,
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
        MC.run(label, exp_metadata={"bins": sweep_points},
               disable_snapshot_metadata=True)
        # N.B. if interleaving cliffords are used, this won't work
        ma2.RandomizedBenchmarking_TwoQubit_Analysis(label=label)

    def measure_two_qubit_interleaved_randomized_benchmarking(
            self,
            qubits: list,
            nr_cliffords=np.array([1., 3., 5., 7., 9., 11., 15.,
                                   20., 25., 30., 40., 50.]),
            nr_seeds=100,
            recompile: bool = "as needed",
            flux_codeword="cz",
            flux_allocated_duration_ns: int = None,
            sim_cz_qubits: list = None,
            measure_idle_flux: bool = False,
            prepare_for_timedomain: bool = True,
            rb_tasks_start: list = None,
            pool=None,
            cardinal: dict = None,
            start_next_round_compilation: bool = False,
            maxtasksperchild=None,
            MC = None,
        ):
        # USED_BY: inspire_dependency_graph.py,
        """
        Perform two-qubit interleaved randomized benchmarking with an
        interleaved CZ gate, and optionally an interleaved idle identity with
        the duration of the CZ.

        If recompile is `True` or `as needed` it will parallelize RB sequence
        compilation with measurement (beside the parallelization of the RB
        sequences which will always happen in parallel).
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        def run_parallel_iRB(
                recompile, pool, rb_tasks_start: list = None,
                start_next_round_compilation: bool = False,
                cardinal=cardinal
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
                    prepare_for_timedomain = prepare_for_timedomain,
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
                prepare_for_timedomain = prepare_for_timedomain,
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
                recompile=recompile,  # This of course needs to be False
                prepare_for_timedomain = prepare_for_timedomain,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                sim_cz_qubits=sim_cz_qubits,
                rb_tasks=rb_tasks_start,
            )

            # 5. Wait for [104368] compilation to finish
            cl_oql.wait_for_rb_tasks(rb_tasks_CZ)

            # # 6. Start (non-blocking) compilation for [100_000]
            # if measure_idle_flux:
            #     rb_tasks_I = self.measure_two_qubit_randomized_benchmarking(
            #         qubits=qubits,
            #         MC=MC,
            #         nr_cliffords=nr_cliffords,
            #         interleaving_cliffords=[100_000],
            #         recompile=recompile,
            #         flux_codeword=flux_codeword,
            #         flux_allocated_duration_ns=flux_allocated_duration_ns,
            #         nr_seeds=nr_seeds,
            #         sim_cz_qubits=sim_cz_qubits,
            #         compile_only=True,
            #         pool=pool,
            #     )
            # elif start_next_round_compilation:
            #     # Optionally send to the `pool` the tasks of RB compilation to be
            #     # used on the next round of calling the iRB method
            #     rb_tasks_next = self.measure_two_qubit_randomized_benchmarking(
            #         qubits=qubits,
            #         MC=MC,
            #         nr_cliffords=nr_cliffords,
            #         interleaving_cliffords=[None],
            #         recompile=recompile,
            #         flux_codeword=flux_codeword,
            #         nr_seeds=nr_seeds,
            #         sim_cz_qubits=sim_cz_qubits,
            #         compile_only=True,
            #         pool=pool
            #     )

            # 7. Start the measurement and run the analysis for [104368]
            self.measure_two_qubit_randomized_benchmarking(
                qubits=qubits,
                MC=MC,
                nr_cliffords=nr_cliffords,
                interleaving_cliffords=[104368],
                recompile=recompile,
                prepare_for_timedomain = prepare_for_timedomain,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                sim_cz_qubits=sim_cz_qubits,
                rb_tasks=rb_tasks_CZ,
            )
            a = ma2.InterleavedRandomizedBenchmarkingAnalysis(
                label_base="icl[None]",
                label_int="icl[104368]"
            )
            # update qubit objects to record the attained CZ fidelity
            if cardinal:
                opposite_cardinal = {'NW':'SE', 'NE':'SW', 'SW':'NE', 'SE':'NW'}
                self.find_instrument(qubits[0]).parameters[f'F_2QRB_{cardinal}'].set(1-a.proc_data_dict['quantities_of_interest']['eps_CZ_simple'].n)
                self.find_instrument(qubits[1]).parameters[f'F_2QRB_{opposite_cardinal[cardinal]}'].set(1-a.proc_data_dict['quantities_of_interest']['eps_CZ_simple'].n)


            # if measure_idle_flux:
            #     # 8. Wait for [100_000] compilation to finish
            #     cl_oql.wait_for_rb_tasks(rb_tasks_I)

            #     # 8.a. Optionally send to the `pool` the tasks of RB compilation to be
            #     # used on the next round of calling the iRB method
            #     if start_next_round_compilation:
            #         rb_tasks_next = self.measure_two_qubit_randomized_benchmarking(
            #             qubits=qubits,
            #             MC=MC,
            #             nr_cliffords=nr_cliffords,
            #             interleaving_cliffords=[None],
            #             recompile=recompile,
            #             flux_codeword=flux_codeword,
            #             nr_seeds=nr_seeds,
            #             sim_cz_qubits=sim_cz_qubits,
            #             compile_only=True,
            #             pool=pool
            #         )

            #     # 9. Start the measurement and run the analysis for [100_000]
            #     self.measure_two_qubit_randomized_benchmarking(
            #         qubits=qubits,
            #         MC=MC,
            #         nr_cliffords=nr_cliffords,
            #         interleaving_cliffords=[100_000],
            #         recompile=False,
            #         flux_codeword=flux_codeword,
            #         flux_allocated_duration_ns=flux_allocated_duration_ns,
            #         nr_seeds=nr_seeds,
            #         sim_cz_qubits=sim_cz_qubits,
            #         rb_tasks=rb_tasks_I
            #     )
            #     ma2.InterleavedRandomizedBenchmarkingAnalysis(
            #         label_base="icl[None]",
            #         label_int="icl[104368]",
            #         label_int_idle="icl[100000]"
            #     )

            return rb_tasks_next

        if recompile or recompile == "as needed":
            # This is an optimization that compiles the interleaved RB
            # sequences for the next measurement while measuring the previous
            # one
            if pool is None:
                # Using `with ...:` makes sure the other processes will be terminated
                # `maxtasksperchild` avoid RAM issues
                if not maxtasksperchild:
                    maxtasksperchild = cl_oql.maxtasksperchild
                with multiprocessing.Pool(maxtasksperchild=maxtasksperchild) as pool:
                    run_parallel_iRB(recompile=recompile,
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
                prepare_for_timedomain = prepare_for_timedomain,
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
                prepare_for_timedomain = prepare_for_timedomain,
                flux_codeword=flux_codeword,
                nr_seeds=nr_seeds,
                sim_cz_qubits=sim_cz_qubits,
            )

            a = ma2.InterleavedRandomizedBenchmarkingAnalysis(
                label_base="icl[None]",
                label_int="icl[104368]",
            )

            # update qubit objects to record the attained CZ fidelity
            if cardinal:
                opposite_cardinal = {'NW':'SE', 'NE':'SW', 'SW':'NE', 'SE':'NW'}
                self.find_instrument(qubits[0]).parameters[f'F_2QRB_{cardinal}'].set(1-a.proc_data_dict['quantities_of_interest']['eps_CZ_simple'].n)
                self.find_instrument(qubits[1]).parameters[f'F_2QRB_{opposite_cardinal[cardinal]}'].set(1-a.proc_data_dict['quantities_of_interest']['eps_CZ_simple'].n)

            if measure_idle_flux:
                # Perform two-qubit iRB with idle identity of same duration as CZ
                self.measure_two_qubit_randomized_benchmarking(
                    qubits=qubits,
                    MC=MC,
                    nr_cliffords=nr_cliffords,
                    interleaving_cliffords=[100_000],
                    recompile=recompile,
                    prepare_for_timedomain = prepare_for_timedomain,
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
        return True

    def measure_multi_qubit_simultaneous_randomized_benchmarking(
            self,
            qubits,
            MC = None,
            nr_cliffords=2 ** np.arange(11),
            nr_seeds=100,
            recompile: bool = "as needed",
            cal_points: bool = True,
            ro_acq_weight_type: str = "optimal IQ",
            compile_only: bool = False,
            pool=None,  # a multiprocessing.Pool()
            rb_tasks=None,  # used after called with `compile_only=True
            label_name=None,
            prepare_for_timedomain=True
        ):
        """
        Performs simultaneous single qubit RB on multiple qubits.
        The data of this experiment should be compared to the results of single
        qubit RB to reveal differences due to MW crosstalk and residual coupling

        Args:
            qubits (list):
                list of the qubit names on which to perform RB

            nr_cliffords (array):
                lengths of the clifford sequences to perform

            nr_seeds (int):
                number of different clifford sequences of each length

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

        if MC is None:
            MC = self.instr_MC.get_instr()
        MC.soft_avg(1)

        # The detector needs to be defined before setting back parameters
        d = self.get_int_logging_detector(qubits=qubits)
        # set back the settings
        # self.ro_acq_weight_type(old_weight_type)
        # self.ro_acq_digitized(old_digitized)

        for q in qubits:
            q_instr = self.find_instrument(q)
            mw_lutman = q_instr.instr_LutMan_MW.get_instr()
            # mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()
            mw_lutman.set_default_lutmap()


        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubits, bypass_flux=True)

        MC.soft_avg(1)

        def send_rb_tasks(pool_):
            tasks_inputs = []
            for i in range(nr_seeds):
                task_dict = dict(
                    qubits=[self.find_instrument(q).cfg_qubit_nr() for q in qubits],
                    nr_cliffords=nr_cliffords,
                    nr_seeds=1,
                    platf_cfg=self.cfg_openql_platform_fn(),
                    program_name="MultiQ_RB_s{}_ncl{}_{}".format(
                        i,
                        list(map(int, nr_cliffords)),
                        '_'.join(qubits)
                    ),
                    interleaving_cliffords=[None],
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
                [nr_cliffords[-1] + 0.5]
                + [nr_cliffords[-1] + 1.5]
                + [nr_cliffords[-1] + 2.5],
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

        label="Multi_Qubit_sim_RB_{}seeds_recompile={}_".format(nr_seeds, recompile)
        if label_name is None:
            label += '_'.join(qubits)
        else:
            label += label_name
        MC.run(label, exp_metadata={"bins": sweep_points})

        cal_2Q = ["0"*len(qubits), "1"*len(qubits), "2"*len(qubits)]
        Analysis = []
        for i in range(len(qubits)):
            rates_I_quad_ch_idx = 2*i
            cal_1Q = [state[rates_I_quad_ch_idx // 2] for state in cal_2Q]
            a = ma2.RandomizedBenchmarking_SingleQubit_Analysis(
                label=label,
                rates_I_quad_ch_idx=rates_I_quad_ch_idx,
                cal_pnts_in_dset=cal_1Q
            )
            Analysis.append(a)

        return Analysis

    def measure_two_qubit_simultaneous_randomized_benchmarking(
        self,
        qubits,
        MC= None,
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

        for q in qubits:
            q_instr = self.find_instrument(q)
            mw_lutman = q_instr.instr_LutMan_MW.get_instr()
            mw_lutman.set_default_lutmap()

        self.prepare_for_timedomain(qubits=qubits)
        if MC is None:
            MC = self.instr_MC.get_instr()
        MC.soft_avg(1)

        # The detector needs to be defined before setting back parameters
        d = self.get_int_logging_detector(qubits=qubits)
        # set back the settings
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

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

    def measure_gate_process_tomography(
        self,
        meas_qubit: str,
        gate_qubits: list,
        gate_name: str,
        gate_duration_ns: int,
        wait_after_gate_ns: int = 0,
        nr_shots_per_case: int = 2**14,
        prepare_for_timedomain: bool= True,
        disable_metadata: bool = False,
        ):
        assert self.ro_acq_weight_type() != 'optimal', 'IQ readout required!'
        q_meas = self.find_instrument(meas_qubit)
        # q_gate = self.find_instrument(gate_qubit)
        q_gate_idx = [self.find_instrument(q).cfg_qubit_nr() for q in gate_qubits]

        MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            if not all([q==meas_qubit for q in gate_qubits]): 
                self.prepare_for_timedomain(qubits=gate_qubits, prepare_for_readout=False)
            self.prepare_for_timedomain(qubits=[meas_qubit])
        # Experiment
        p = mqo.gate_process_tomograhpy(
                meas_qubit_idx=q_meas.cfg_qubit_nr(),
                gate_qubit_idx=q_gate_idx,
                gate_name=gate_name,
                gate_duration_ns=gate_duration_ns,
                wait_after_gate_ns=wait_after_gate_ns,
                platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='Shot', unit='#',
                             upload=True)
        d = self.get_int_logging_detector()
        nr_shots = (2*18+3)*nr_shots_per_case
        if nr_shots < 2**20:
            d.detectors[0].nr_shots = nr_shots
        else:
            _shots_per_run = ((2**20)//(2*18+3))*(2*18+3)
            nr_shots = np.ceil(nr_shots/_shots_per_run)*_shots_per_run
            print(f'Number of shots per case increased to {nr_shots/(2*18+3)}.')
            d.detectors[0].nr_shots = _shots_per_run
        MC.soft_avg(1)
        MC.set_detector_function(d)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.live_plot_enabled(False)
        try:
            label = f'Gate_process_tomograhpy_gate_{gate_name}_{meas_qubit}'
            MC.run(label+self.msmt_suffix, disable_snapshot_metadata=disable_metadata)
        except:
            print_exception()
            # MC.live_plot_enabled(True)
        # Analysis
        ma2.tomoa.Gate_process_tomo_Analysis(qubit=q_meas.name, label='Gate_process')

    ########################################################
    # Calibration methods
    ########################################################
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

    def measure_multi_AllXY(self, qubits: list = None ,MC=None,
                            prepare_for_timedomain: bool = True,
                            disable_metadata: bool = False,
                            double_points =True,termination_opt=0.08):

        if qubits is None:
            qubits = self.qubits()
        if prepare_for_timedomain:
            self.ro_acq_weight_type('optimal')
            self.prepare_for_timedomain(qubits=qubits, bypass_flux=True)

        qubits_idx = []
        for q in qubits:
            q_ob = self.find_instrument(q)
            q_nr = q_ob.cfg_qubit_nr()
            qubits_idx.append(q_nr)

        p = mqo.multi_qubit_AllXY(qubits_idx=qubits_idx,
                                  platf_cfg=self.cfg_openql_platform_fn(),
                                  double_points = double_points)

        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        d = self.get_int_avg_det(qubits=qubits)
        if MC is None:
            MC = self.instr_MC.get_instr()
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(42))
        MC.set_detector_function(d)
        MC.run('Multi_AllXY_'+'_'.join(qubits),
               disable_snapshot_metadata = disable_metadata)
        a = ma2.Multi_AllXY_Analysis()

        dev = 0
        for Q in qubits:
            dev += a.proc_data_dict['deviation_{}'.format(Q)]
            if dev > len(qubits)*termination_opt:
                return False
            else:
                return True

    def measure_multi_rabi(self, qubits: list = None, prepare_for_timedomain=True ,MC=None, 
                           amps=np.linspace(0,1,31),calibrate=True):
        if qubits is None:
            qubits = self.qubits()
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubits)

        qubits_idx = []
        for q in qubits:
            qub = self.find_instrument(q)
            qubits_idx.append(qub.cfg_qubit_nr())

       
        p = mqo.multi_qubit_rabi(qubits_idx = qubits_idx,platf_cfg = self.cfg_openql_platform_fn())
        
        self.instr_CC.get_instr().eqasm_program(p.filename)

        s = swf.mw_lutman_amp_sweep(qubits = qubits,device=self)

        d = self.int_avg_det_single

        if MC is None:
            MC = self.instr_MC.get_instr()

        MC.set_sweep_function(s)    
        MC.set_sweep_points(amps)
        MC.set_detector_function(d)
        label = 'Multi_qubit_rabi_'+'_'.join(qubits)
        MC.run(name = label)
        a = ma2.Multi_Rabi_Analysis(qubits = qubits, label = label)
        if calibrate:
          b = a.proc_data_dict
          for q in qubits:
            pi_amp = b['quantities_of_interest'][q]['pi_amp']
            qub = self.find_instrument(q)
            qub.mw_channel_amp(pi_amp)
        return True

    def measure_multi_ramsey(self, qubits: list = None, times = None, GBT = True,
                         artificial_periods: float = None, label=None,
                         MC=None, prepare_for_timedomain=True,
                         update_T2=True,update_frequency = False):
      if MC is None:
          MC = self.instr_MC.get_instr()

      if qubits is None:
        qubits = self.qubits()

      if prepare_for_timedomain:
        self.prepare_for_timedomain(qubits=qubits, bypass_flux=True)

      if artificial_periods is None:
        artificial_periods = 5

      if times is None:
        t = True
        times = []
      else:
        t = False


      qubits_idx = []
      for i,q in enumerate(qubits):
        qub = self.find_instrument(q)
        qubits_idx.append(qub.cfg_qubit_nr())
        stepsize = max((4*qub.T2_star()/61)//(abs(qub.cfg_cycle_time()))
                          *abs(qub.cfg_cycle_time()),40e-9)
        if t is True:
            set_time = np.arange(0,stepsize*64,stepsize)
            times.append(set_time)

        artificial_detuning = artificial_periods/times[i][-1]
        freq_qubit = qub.freq_qubit()
        mw_mod = qub.mw_freq_mod.get()
        freq_det = freq_qubit - mw_mod + artificial_detuning
        qub.instr_LO_mw.get_instr().set('frequency', freq_det)

      points = len(times[0])

      p = mqo.multi_qubit_ramsey(times = times,qubits_idx=qubits_idx,
                                 platf_cfg=self.cfg_openql_platform_fn())

      s = swf.OpenQL_Sweep(openql_program=p,
                         CCL=self.instr_CC.get_instr())

      d = self.get_int_avg_det(qubits=qubits)

      MC.set_sweep_function(s)
      MC.set_sweep_points(np.arange(points))
      MC.set_detector_function(d)
      if label is None:
        label = 'Multi_Ramsey_'+'_'.join(qubits)
      MC.run(label)

      a = ma2.Multi_Ramsey_Analysis(qubits = qubits, times = times, artificial_detuning=artificial_detuning,label=label)
      qoi = a.proc_data_dict['quantities_of_interest']
      for q in qubits:
        qub = self.find_instrument(q)
        if update_T2:
            T2_star = qoi[q]['tau']
            qub.T2_star(T2_star)
        if update_frequency:
            new_freq = qoi[q]['freq_new']
            qub.freq_qubit(new_freq)
      if GBT:
        return True
      else:
        return a

    def calibrate_multi_frequency_fine(self,qubits: list = None,times = None,
                                   artificial_periods: float = None,
                                   MC=None, prepare_for_timedomain=True,
                                   update_T2=False,update_frequency = True,
                                   stepsize:float = None,termination_opt = 0,
                                   steps=[1, 1, 3, 10, 30, 100, 300, 1000]):
        if qubits is None:
            qubits = self.qubits()
        if artificial_periods is None:
            artificial_periods = 2.5
        if stepsize is None:
            stepsize = 20e-9
        for n in steps:
            times = []
            for q in qubits:
                qub = self.find_instrument(q)
                time = np.arange(0,50*n*stepsize,n*stepsize)
                times.append(time)

            label = 'Multi_Ramsey_{}_pulse_sep_'.format(n)+ '_'.join(qubits)

            a = self.measure_multi_ramsey(qubits = qubits, times =times, MC=MC, GBT=False,
                                     artificial_periods = artificial_periods, label = label,
                                     prepare_for_timedomain =prepare_for_timedomain,
                                     update_frequency=False,update_T2 = update_T2)
            for q in qubits:

                qub = self.find_instrument(q)
                freq = a.proc_data_dict['quantities_of_interest'][q]['freq_new']
                T2 = a.proc_data_dict['quantities_of_interest'][q]['tau']
                fit_error = a.proc_data_dict['{}_fit_res'.format(q)].chisqr

                if (times[0][-1] < 2.*T2) and (update_frequency is True):
                    # If the last step is > T2* then the next will be for sure
                    qub.freq_qubit(freq)



            T2_max = max(a.proc_data_dict['quantities_of_interest'][q]['tau'] for q in qubits)
            if times[0][-1] > 2.*T2_max:
                    # If the last step is > T2* then the next will be for sure
            
                    print('Breaking of measurement because of T2*')
                    break
        return True

    def measure_multi_T1(self,qubits: list = None, times = None, MC=None,
                         prepare_for_timedomain=True, analyze=True,
                         update=True):

      if MC is None:
          MC = self.instr_MC.get_instr()

      if qubits is None:
        qubits = self.qubits()

      if prepare_for_timedomain:
        self.prepare_for_timedomain(qubits=qubits)


      qubits_idx = []
      set_times = []
      for q in qubits:
        qub = self.find_instrument(q)
        qubits_idx.append(qub.cfg_qubit_nr())
        stepsize = max((4*qub.T1()/31)//(abs(qub.cfg_cycle_time()))
                          *abs(qub.cfg_cycle_time()),40e-9)
        set_time = np.arange(0,stepsize*34,stepsize)
        set_times.append(set_time)

      if times is None:
        times = set_times

      points = len(times[0])
      


      p = mqo.multi_qubit_T1(times = times,qubits_idx=qubits_idx,
                                 platf_cfg=self.cfg_openql_platform_fn())

      s = swf.OpenQL_Sweep(openql_program=p,
                         CCL=self.instr_CC.get_instr())

      d = self.get_int_avg_det(qubits=qubits)

      MC.set_sweep_function(s)
      MC.set_sweep_points(np.arange(points))
      MC.set_detector_function(d)
      label = 'Multi_T1_'+'_'.join(qubits)
      MC.run(label)

      if analyze:
        a = ma2.Multi_T1_Analysis(qubits=qubits,times = times)
      if update:
        for q in qubits:
          qub = self.find_instrument(q)
          T1 = a.proc_data_dict['quantities_of_interest'][q]['tau']
          qub.T1(T1) 

      return a

    def measure_multi_Echo(self,qubits: list=None, times = None, MC=None,
                           prepare_for_timedomain=True, analyze=True,
                           update=True):
      if MC is None:
          MC = self.instr_MC.get_instr()

      if qubits is None:
        qubits = self.qubits()

      if prepare_for_timedomain:
        self.prepare_for_timedomain(qubits=qubits)


      qubits_idx = []
      set_times = []
      for q in qubits:
        qub = self.find_instrument(q)
        qubits_idx.append(qub.cfg_qubit_nr())
        stepsize = max((2*qub.T2_echo()/61)//(abs(qub.cfg_cycle_time()))
                          *abs(qub.cfg_cycle_time()),20e-9)
        set_time = np.arange(0,stepsize*64,stepsize)
        set_times.append(set_time)

      if times is None:
        times = set_times

      points = len(times[0])


      p = mqo.multi_qubit_Echo(times = times,qubits_idx=qubits_idx,
                                 platf_cfg=self.cfg_openql_platform_fn())

      s = swf.OpenQL_Sweep(openql_program=p,
                         CCL=self.instr_CC.get_instr())

      d = self.get_int_avg_det(qubits=qubits)

      MC.set_sweep_function(s)
      MC.set_sweep_points(np.arange(points))
      MC.set_detector_function(d)
      label = 'Multi_Echo_'+'_'.join(qubits)
      MC.run(label)
      if analyze:
        a = ma2.Multi_Echo_Analysis(label = label, qubits = qubits,times = times)
      if update:
        qoi = a.proc_data_dict['quantities_of_interest']
        for q in qubits:
            qub = self.find_instrument(q)
            T2_echo = qoi[q]['tau']
            qub.T2_echo(T2_echo)

      return True 

    def measure_multi_flipping(self,
            qubits: list=None, 
            number_of_flips: int=None,
            equator=True, 
            ax='x', 
            angle='180', 
            MC=None,
            prepare_for_timedomain=True,
            update=False,
            scale_factor_based_on_line: bool = False
        ):
        # allow flipping only with pi/2 or pi, and x or y pulses
        assert angle in ['90','180']
        assert ax.lower() in ['x', 'y']

        if MC is None:
            MC = self.instr_MC.get_instr()

        if qubits is None:
            qubits = self.qubits()

        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubits, bypass_flux=True)

        if number_of_flips is None:
            number_of_flips = 30
        nf = np.arange(0,(number_of_flips+4)*2,2)

        qubits_idx = []
        for q in qubits:
            qub = self.find_instrument(q)
            qubits_idx.append(qub.cfg_qubit_nr())
        
        p = mqo.multi_qubit_flipping(number_of_flips = nf,qubits_idx=qubits_idx,
                                   platf_cfg=self.cfg_openql_platform_fn(),
                                   equator=equator,ax=ax, angle=angle)

        s = swf.OpenQL_Sweep(openql_program=p,unit = '#',
                               CCL=self.instr_CC.get_instr())

        d = self.get_int_avg_det(qubits=qubits)

        MC.set_sweep_function(s)
        MC.set_sweep_points(nf)
        MC.set_detector_function(d)
        label = 'Multi_flipping_'+'_'.join(qubits)
        MC.run(label)

        a = ma2.Multi_Flipping_Analysis(qubits=qubits, label=label)

        if update:
            for q in qubits:
                # Same as in single-qubit flipping:
                # Choose scale factor based on simple goodness-of-fit comparison,
                # unless it is forced by `scale_factor_based_on_line`
                # This method gives priority to the line fit: 
                # the cos fit will only be chosen if its chi^2 relative to the 
                # chi^2 of the line fit is at least 10% smaller
                # cos_chisqr = a.proc_data_dict['quantities_of_interest'][q]['cos_fit'].chisqr
                # line_chisqr = a.proc_data_dict['quantities_of_interest'][q]['line_fit'].chisqr

                # if scale_factor_based_on_line:
                #     scale_factor = a.proc_data_dict['quantities_of_interest'][q]['line_fit']['sf']
                # elif (line_chisqr - cos_chisqr)/line_chisqr > 0.1:
                #     scale_factor = a.proc_data_dict['quantities_of_interest'][q]['cos_fit']['sf']
                # else:
                #     scale_factor = a.proc_data_dict['quantities_of_interest'][q]['line_fit']['sf']

                if scale_factor_based_on_line:
                    scale_factor = a.proc_data_dict['quantities_of_interest'][q]['line_fit']['sf']
                else:
                    # choose scale factor preferred by analysis (currently based on BIC measure)                                                   
                    scale_factor = a.proc_data_dict['{}_scale_factor'.format(q)]
                
                if abs(scale_factor-1) < 1e-3:
                    print(f'Qubit {q}: Pulse amplitude accurate within 0.1%. Amplitude not updated.')
                    return a

                qb = self.find_instrument(q)
                if angle == '180':
                    if qb.cfg_with_vsm():
                        amp_old = qb.mw_vsm_G_amp()
                        qb.mw_vsm_G_amp(scale_factor*amp_old)
                    else:
                        amp_old = qb.mw_channel_amp()
                        qb.mw_channel_amp(scale_factor*amp_old)
                elif angle == '90':
                    amp_old = qb.mw_amp90_scale()
                    qb.mw_amp90_scale(scale_factor*amp_old)

                print('Qubit {}: Pulse amplitude for {}-{} pulse changed from {:.3f} to {:.3f}'.format(
                    q, ax, angle, amp_old, scale_factor*amp_old))

    def measure_multi_motzoi(self,qubits: list = None, prepare_for_timedomain=True ,MC=None, 
                             amps=None,calibrate=True):
        if qubits is None:
            qubits = self.qubits()
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubits)
        if amps is None:
          amps = np.linspace(-0.3,0.3,31)

        qubits_idx = []
        for q in qubits:
            qub = self.find_instrument(q)
            qubits_idx.append(qub.cfg_qubit_nr())

        p = mqo.multi_qubit_motzoi(qubits_idx = qubits_idx,platf_cfg = self.cfg_openql_platform_fn())
       
        self.instr_CC.get_instr().eqasm_program(p.filename)

        s = swf.motzoi_lutman_amp_sweep(qubits = qubits,device=self)

        d = self.get_int_avg_det(qubits = qubits,single_int_avg=True, 
                                   values_per_point=2,
                                   values_per_point_suffex=['yX', 'xY'],
                                   always_prepare=True)

        if MC is None:
            MC = self.instr_MC.get_instr()

        MC.set_sweep_function(s)    
        MC.set_sweep_points(amps)
        MC.set_detector_function(d)
        label = 'Multi_Motzoi_'+'_'.join(qubits)
        MC.run(name = label)

        a = ma2.Multi_Motzoi_Analysis(qubits=qubits, label = label)
        if calibrate:
            for q in qubits:
                qub = self.find_instrument(q)
                opt_motzoi = a.proc_data_dict['{}_intersect'.format(q)][0]
                qub.mw_motzoi(opt_motzoi)
            return True

    #######################################
    # Two qubit gate calibration functions
    #######################################
    def measure_ramsey_tomo(self, 
                            qubit_ramsey: list,
                            qubit_control: list,
                            excited_spectators: list = [],
                            nr_shots_per_case: int = 2**10,
                            flux_codeword: str = 'cz',
                            prepare_for_timedomain: bool = True,
                            MC=None):
        '''
        Doc string

        '''

        qubitR = [self.find_instrument(qr) for qr in qubit_ramsey]
        qubitR_idxs = [qr.cfg_qubit_nr() for qr in qubitR]
        
        qubitC = [self.find_instrument(qc) for qc in qubit_control]
        qubitC_idxs = [qc.cfg_qubit_nr() for qc in qubitC]
        
        # Get indices for spectator qubits
        qubitS = [self.find_instrument(q) for q in excited_spectators]
        qubitS_idxs = [q.cfg_qubit_nr() for q in qubitS]

        # Assert we have IQ readout
        assert self.ro_acq_weight_type() == 'optimal IQ', 'device not in "optimal IQ" mode'
        assert self.ro_acq_digitized() == False, 'RO should not be digitized'

        for qr in qubitR:
            mw_lutman = qr.instr_LutMan_MW.get_instr()
            mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[*excited_spectators], prepare_for_readout=False)
            self.prepare_for_timedomain(qubits=[*qubit_ramsey, *qubit_control])
            
        
        p = mqo.Ramsey_tomo(qR= qubitR_idxs,
                            qC= qubitC_idxs,
                            exc_specs= qubitS_idxs,
                            flux_codeword=flux_codeword,
                            platf_cfg=self.cfg_openql_platform_fn())

        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        
        # d = self.get_int_log_det(qubits=[qubit_ramsey, qubit_control])
        d = self.get_int_logging_detector(qubits=[*qubit_ramsey, *qubit_control], 
                                          result_logging_mode='raw')
        d.detectors[0].nr_shots = 4096
        try:
            d.detectors[1].nr_shots = 4096
        except:
            pass
        try:
            d.detectors[2].nr_shots = 4096
        except:
            pass

        nr_shots = int(16*256*2**4)
        if MC is None:
            MC = self.instr_MC.get_instr()
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)
        MC.run('Ramsey_tomo_R_{}_C_{}_S_{}'.format(qubit_ramsey, qubit_control, excited_spectators))
        # Analysis
        a = ma2.tqg.Two_qubit_gate_tomo_Analysis(label='Ramsey', n_pairs=len(qubit_ramsey))
        
        return a.qoi

    def measure_repeated_CZ_experiment(self,
            qubit_pair: list,
            rounds: int = 50,
            nr_shots_per_case: int = 2**13,
            flux_codeword: str = 'cz',
            gate_time_ns: int = 60,
            prepare_for_timedomain: bool = True,
            analyze: bool = True,
            disable_metadata: bool = False):
        '''
        Function used to measure CZ leakage using a repeated measurment scheme:
               xrounds
        Q0 ---H---o---Meas
                  |
        Q1 ---H---o---Meas
        
        Requires Qutrit readout.
        Also measures 2 qutrit readout assignment to accurately estimate
        leakage rates.
        '''
        assert self.ro_acq_digitized() == False, 'Analog readout required'
        assert 'IQ' in self.ro_acq_weight_type(), 'IQ readout is required!'
        MC = self.instr_MC.get_instr()
        # Configure lutmap
        for qubit in qubit_pair:
            qb = self.find_instrument(qubit)
            mwl = qb.instr_LutMan_MW.get_instr()
            mwl.set_default_lutmap()
        self.prepare_for_timedomain(qubits = qubit_pair)

        # get qubit idx 
        Q_idx = [self.find_instrument(q).cfg_qubit_nr() for q in qubit_pair]
        # Set UHF number of shots
        _cycle = (2*rounds+9)
        nr_shots = _cycle*nr_shots_per_case
        # if heralded_init:
        #     nr_shots *= 2 
        uhfqc_max_shots = 2**20
        if nr_shots < uhfqc_max_shots:
            # all shots can be acquired in a single UHF run
            shots_per_run = nr_shots
        else:
            # Number of UHF acquisition runs
            nr_runs = ceil(nr_shots/uhfqc_max_shots) 
            shots_per_run = int((nr_shots/nr_runs)/_cycle)*_cycle
            nr_shots = nr_runs*shots_per_run
        # Compile sequence
        p = mqo.repeated_CZ_experiment(
            qubit_idxs=Q_idx,
            rounds=rounds,
            flux_codeword=flux_codeword,
            gate_time_ns=gate_time_ns,
            heralded_init=False,
            platf_cfg=self.cfg_openql_platform_fn())
        
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='Shot', unit='#',
                             upload=True)
        MC.soft_avg(1)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        d = self.get_int_logging_detector(qubits=qubit_pair)
        for det in d.detectors:
            det.nr_shots = shots_per_run
        MC.set_detector_function(d)
        MC.live_plot_enabled(False)
        label = f'Repeated_CZ_experiment_qubit_pair_{"_".join(qubit_pair)}'
        MC.run(label+self.msmt_suffix, disable_snapshot_metadata=disable_metadata)
        MC.live_plot_enabled(True)
        # Analysis
        if analyze:
            a = ma2.tqg.Repeated_CZ_experiment_Analysis(rounds=rounds, label=label)

    def measure_vcz_A_tmid_landscape(
        self, 
        Q0,
        Q1,
        T_mids,
        A_ranges,
        A_points: int,
        Q_parks: list = None,
        Tp : float = None,
        flux_codeword: str = 'cz',
        flux_pulse_duration: float = 60e-9,
        prepare_for_timedomain: bool = True,
        disable_metadata: bool = False):
        """
        Perform 2D sweep of amplitude and wave parameter while measuring 
        conditional phase and missing fraction via the "conditional 
        oscillation" experiment.

        Q0 : High frequency qubit(s). Can be given as single qubit or list.
        Q1 : Low frequency qubit(s). Can be given as single qubit or list.
        T_mids : list of vcz "T_mid" values to sweep.
        A_ranges : list of tuples containing ranges of amplitude sweep.
        A_points : Number of points to sweep for amplitude range.
        Q_parks : list of qubits parked during operation.
        """
        if isinstance(Q0, str):
            Q0 = [Q0]
        if isinstance(Q1, str):
            Q1 = [Q1]
        assert len(Q0) == len(Q1)

        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()
        # get gate directions
        directions = [get_gate_directions(q0, q1) for q0, q1 in zip(Q0, Q1)]
        Flux_lm_0 = [self.find_instrument(q0).instr_LutMan_Flux.get_instr() for q0 in Q0]
        Flux_lm_1 = [self.find_instrument(q1).instr_LutMan_Flux.get_instr() for q1 in Q1]
        Flux_lms_park = [self.find_instrument(q).instr_LutMan_Flux.get_instr() for q in Q_parks]
        # Prepare for time domain
        if prepare_for_timedomain:
            self.prepare_for_timedomain(
                qubits=np.array([[Q0[i],Q1[i]] for i in range(len(Q0))]).flatten(),
                bypass_flux=True)
            for i, lm in enumerate(Flux_lm_0):
                print(f'Setting {Q0[i]} vcz_amp_sq_{directions[i][0]} to 1')
                print(f'Setting {Q0[i]} vcz_amp_fine_{directions[i][0]} to 0.5')
                print(f'Setting {Q0[i]} vcz_amp_dac_at_11_02_{directions[i][0]} to 0.5')
                lm.set(f'vcz_amp_sq_{directions[i][0]}', 1)
                lm.set(f'vcz_amp_fine_{directions[i][0]}', .5)
                lm.set(f'vcz_amp_dac_at_11_02_{directions[i][0]}', .5)
            for i, lm in enumerate(Flux_lm_1):
                print(f'Setting {Q1[i]} vcz_amp_dac_at_11_02_{directions[i][1]} to 0')
                lm.set(f'vcz_amp_dac_at_11_02_{directions[i][1]}',  0)
        # Look for Tp values
        if Tp:
            if isinstance(Tp, str):
                Tp = [Tp]
        else:
            Tp = [lm.get(f'vcz_time_single_sq_{directions[i][0]}')*2 for i, lm in enumerate(Flux_lm_0)]
        assert len(Q0) == len(Tp)
        #######################
        # Load phase pulses
        #######################
        for i, q in enumerate(Q0):
            # only on the CZ qubits we add the ef pulses 
            mw_lutman = self.find_instrument(q).instr_LutMan_MW.get_instr()
            lm = mw_lutman.LutMap()
            # we hardcode the X on the ef transition to CW 31 here.
            lm[27] = {'name': 'rXm180', 'phi': 0, 'theta': -180, 'type': 'ge'}
            lm[31] = {"name": "rX12", "theta": 180, "phi": 0, "type": "ef"}
            # load_phase_pulses will also upload other waveforms
            mw_lutman.load_phase_pulses_to_AWG_lookuptable()
        # Wrapper function for conditional oscillation detector function.
        def wrapper(Q0, Q1,
                    prepare_for_timedomain,
                    downsample_swp_points,
                    extract_only,
                    disable_metadata):
            a = self.measure_conditional_oscillation_multi(
                    pairs=[[Q0[i], Q1[i]] for i in range(len(Q0))], 
                    parked_qbs=Q_parks,
                    flux_codeword=flux_codeword,
                    prepare_for_timedomain=prepare_for_timedomain,
                    downsample_swp_points=downsample_swp_points,
                    extract_only=extract_only,
                    disable_metadata=disable_metadata,
                    verbose=False)
            cp = { f'phi_cond_{i+1}' : a[f'pair_{i+1}_delta_phi_a']\
                  for i in range(len(Q0)) }
            mf = { f'missing_fraction_{i+1}' : a[f'pair_{i+1}_missing_frac_a']\
                  for i in range(len(Q0)) }
            return { **cp, **mf} 

        d = det.Function_Detector(
            wrapper,
            msmt_kw={'Q0' : Q0, 'Q1' : Q1,
                     'prepare_for_timedomain' : False,
                     'downsample_swp_points': 3,
                     'extract_only': True,
                     'disable_metadata': True},
            result_keys=list(np.array([[f'phi_cond_{i+1}', f'missing_fraction_{i+1}']\
                                   for i in range(len(Q0))]).flatten()),
            value_names=list(np.array([[f'conditional_phase_{i+1}', f'missing_fraction_{i+1}']\
                                   for i in range(len(Q0))]).flatten()),
            value_units=list(np.array([['deg', '%']\
                                   for i in range(len(Q0))]).flatten()))
        nested_MC.set_detector_function(d)

        swf1 = swf.multi_sweep_function_ranges(
            sweep_functions=[Flux_lm_0[i].cfg_awg_channel_amplitude\
                             for i in range(len(Q0))],
            sweep_ranges= A_ranges,
            n_points=A_points)
        swf2 = swf.flux_t_middle_sweep(
            fl_lm_tm =  list(np.array([[Flux_lm_0[i], Flux_lm_1[i] ]\
                             for i in range(len(Q0))]).flatten()), 
            fl_lm_park = Flux_lms_park,
            which_gate = list(np.array(directions).flatten()),
            t_pulse = Tp,
            duration = flux_pulse_duration)
        nested_MC.set_sweep_function(swf1)
        nested_MC.set_sweep_points(np.arange(A_points))
        nested_MC.set_sweep_function_2D(swf2)
        nested_MC.set_sweep_points_2D(T_mids)
        MC.live_plot_enabled(False)
        nested_MC.run(f'VCZ_Amp_vs_Tmid_{Q0}_{Q1}_{Q_parks}',
                      mode='2D', disable_snapshot_metadata=disable_metadata)
        # MC.live_plot_enabled(True)
        ma2.tqg.VCZ_tmid_Analysis(Q0=Q0, Q1=Q1,
                                  A_ranges=A_ranges,
                                  label='VCZ_Amp_vs_Tmid')

    def measure_vcz_A_B_landscape(
        self, 
        Q0, Q1,
        A_ranges,
        A_points: int,
        B_amps: list,
        Q_parks: list = None,
        update_flux_params: bool = False,
        flux_codeword: str = 'cz',
        prepare_for_timedomain: bool = True,
        disable_metadata: bool = False):
        """
        Perform 2D sweep of amplitude and wave parameter while measuring 
        conditional phase and missing fraction via the "conditional 
        oscillation" experiment.

        Q0 : High frequency qubit(s). Can be given as single qubit or list.
        Q1 : Low frequency qubit(s). Can be given as single qubit or list.
        T_mids : list of vcz "T_mid" values to sweep.
        A_ranges : list of tuples containing ranges of amplitude sweep.
        A_points : Number of points to sweep for amplitude range.
        Q_parks : list of qubits parked during operation.
        """
        if isinstance(Q0, str):
            Q0 = [Q0]
        if isinstance(Q1, str):
            Q1 = [Q1]
        assert len(Q0) == len(Q1)
        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()
        # get gate directions
        directions = [get_gate_directions(q0, q1) for q0, q1 in zip(Q0, Q1)]
        Flux_lm_0 = [self.find_instrument(q0).instr_LutMan_Flux.get_instr() for q0 in Q0]
        Flux_lm_1 = [self.find_instrument(q1).instr_LutMan_Flux.get_instr() for q1 in Q1]
        Flux_lms_park = [self.find_instrument(q).instr_LutMan_Flux.get_instr() for q in Q_parks]
        # Prepare for time domain
        if prepare_for_timedomain:
            # Time-domain preparation
            self.prepare_for_timedomain(
                qubits=np.array([[Q0[i],Q1[i]] for i in range(len(Q0))]).flatten(),
                bypass_flux=True)
            for i, lm in enumerate(Flux_lm_0):
                print(f'Setting {Q0[i]} vcz_amp_sq_{directions[i][0]} to 1')
                print(f'Setting {Q0[i]} vcz_amp_dac_at_11_02_{directions[i][0]} to 0.5')
                lm.set(f'vcz_amp_sq_{directions[i][0]}', 1)
                lm.set(f'vcz_amp_dac_at_11_02_{directions[i][0]}', .5)
            for i, lm in enumerate(Flux_lm_1):
                print(f'Setting {Q1[i]} vcz_amp_dac_at_11_02_{directions[i][1]} to 0')
                lm.set(f'vcz_amp_dac_at_11_02_{directions[i][1]}',  0)
        # Update two qubit gate parameters
        if update_flux_params:
            # List of current flux lutman amplitudes
            Amps_11_02 = [{ d: lm.get(f'vcz_amp_dac_at_11_02_{d}')\
                         for d in ['NW', 'NE', 'SW', 'SE']} for lm in Flux_lm_0]
            # List of parking amplitudes
            Amps_park = [ lm.get('park_amp') for lm in Flux_lm_0 ]
            # List of current flux lutman channel gains
            Old_gains = [ lm.get('cfg_awg_channel_amplitude') for lm in Flux_lm_0]
        ###########################
        # Load phase pulses
        ###########################
        for i, q in enumerate(Q0):
            # only on the CZ qubits we add the ef pulses 
            mw_lutman = self.find_instrument(q).instr_LutMan_MW.get_instr()
            lm = mw_lutman.LutMap()
            # we hardcode the X on the ef transition to CW 31 here.
            lm[27] = {'name': 'rXm180', 'phi': 0, 'theta': -180, 'type': 'ge'}
            lm[31] = {"name": "rX12", "theta": 180, "phi": 0, "type": "ef"}
            # load_phase_pulses will also upload other waveforms
            mw_lutman.load_phase_pulses_to_AWG_lookuptable()
        # Wrapper function for conditional oscillation detector function.
        def wrapper(Q0, Q1,
                    prepare_for_timedomain,
                    downsample_swp_points,
                    extract_only,
                    disable_metadata):
            a = self.measure_conditional_oscillation_multi(
                    pairs=[[Q0[i], Q1[i]] for i in range(len(Q0))], 
                    parked_qbs=Q_parks,
                    flux_codeword=flux_codeword,
                    prepare_for_timedomain=prepare_for_timedomain,
                    downsample_swp_points=downsample_swp_points,
                    extract_only=extract_only,
                    disable_metadata=disable_metadata,
                    verbose=False)
            cp = { f'phi_cond_{i+1}' : a[f'pair_{i+1}_delta_phi_a']\
                  for i in range(len(Q0)) }
            mf = { f'missing_fraction_{i+1}' : a[f'pair_{i+1}_missing_frac_a']\
                  for i in range(len(Q0)) }
            return { **cp, **mf} 
            
        d = det.Function_Detector(
            wrapper,
            msmt_kw={'Q0' : Q0, 'Q1' : Q1,
                     'prepare_for_timedomain' : False,
                     'downsample_swp_points': 3,
                     'extract_only': True,
                     'disable_metadata': True},
            result_keys=list(np.array([[f'phi_cond_{i+1}', f'missing_fraction_{i+1}']\
                                   for i in range(len(Q0))]).flatten()),
            value_names=list(np.array([[f'conditional_phase_{i+1}', f'missing_fraction_{i+1}']\
                                   for i in range(len(Q0))]).flatten()),
            value_units=list(np.array([['deg', '%']\
                                   for i in range(len(Q0))]).flatten()))
        nested_MC.set_detector_function(d)

        swf1 = swf.multi_sweep_function_ranges(
            sweep_functions=[Flux_lm_0[i].cfg_awg_channel_amplitude
                             for i in range(len(Q0))],
            sweep_ranges= A_ranges,
            n_points=A_points)
        swfs = [swf.FLsweep(lm = lm,
                            par = lm.parameters[f'vcz_amp_fine_{directions[i][0]}'],
                            waveform_name = f'cz_{directions[i][0]}')
                for i, lm in enumerate(Flux_lm_0) ]
        swf2 = swf.multi_sweep_function(sweep_functions=swfs)
        nested_MC.set_sweep_function(swf1)
        nested_MC.set_sweep_points(np.arange(A_points))
        nested_MC.set_sweep_function_2D(swf2)
        nested_MC.set_sweep_points_2D(B_amps)

        MC.live_plot_enabled(False)
        nested_MC.run(f'VCZ_Amp_vs_B_{Q0}_{Q1}_{Q_parks}',
                      mode='2D', disable_snapshot_metadata=disable_metadata)
        # MC.live_plot_enabled(True)
        a = ma2.tqg.VCZ_B_Analysis(Q0=Q0, Q1=Q1,
                                   A_ranges=A_ranges,
                                   directions=directions,
                                   label='VCZ_Amp_vs_B')
        ###################################
        # Update flux parameters
        ###################################
        if update_flux_params:
            print('Updating flux lutman parameters:')
            def _set_amps_11_02(amps, lm, verbose=True):
                '''
                Helper function to set amplitudes in Flux_lutman
                '''
                for d in amps.keys():
                    lm.set(f'vcz_amp_dac_at_11_02_{d}', amps[d])
                    if verbose:
                        print(f'Set {lm.name}.vcz_amp_dac_at_11_02_{d} to {amps[d]}')
            # Update channel gains for each gate
            Opt_gains = [ a.qoi[f'Optimal_amps_{q}'][0] for q in Q0 ]
            Opt_Bvals = [ a.qoi[f'Optimal_amps_{q}'][1] for q in Q0 ]
            
            for i in range(len(Q0)):
                # If new channel gain is higher than old gain then scale dac
                # values accordingly: new_dac = old_dac*(old_gain/new_gain)
                if Opt_gains[i] > Old_gains[i]:
                    Flux_lm_0[i].set('cfg_awg_channel_amplitude', Opt_gains[i])
                    print(f'Set {Flux_lm_0[i].name}.cfg_awg_channel_amplitude to {Opt_gains[i]}')
                    for d in ['NW', 'NE', 'SW', 'SE']:
                        Amps_11_02[i][d] *= Old_gains[i]/Opt_gains[i]
                    Amps_11_02[i][directions[i][0]] = 0.5
                    Amps_park[i] *= Old_gains[i]/Opt_gains[i]
                # If new channel gain is lower than old gain, then choose
                # dac value for measured gate based on old gain
                else:
                    Flux_lm_0[i].set('cfg_awg_channel_amplitude', Old_gains[i])
                    print(f'Set {Flux_lm_0[i].name}.cfg_awg_channel_amplitude to {Old_gains[i]}')
                    Amps_11_02[i][directions[i][0]] = 0.5*Opt_gains[i]/Old_gains[i]
                # Set flux_lutman amplitudes
                _set_amps_11_02(Amps_11_02[i], Flux_lm_0[i])
                Flux_lm_0[i].set(f'vcz_amp_fine_{directions[i][0]}', Opt_Bvals[i])
                Flux_lm_0[i].set(f'park_amp', Amps_park[i])
        return a.qoi

    def measure_unipolar_A_t_landscape(
        self, 
        Q0, Q1,
        A_ranges,
        A_points: int,
        times: list,
        Q_parks: list = None,
        update_flux_params: bool = False,
        flux_codeword: str = 'sf_square',
        prepare_for_timedomain: bool = True,
        disable_metadata: bool = False):
        """
        Perform 2D sweep of amplitude and wave parameter while measuring 
        conditional phase and missing fraction via the "conditional 
        oscillation" experiment.

        Q0 : High frequency qubit(s). Can be given as single qubit or list.
        Q1 : Low frequency qubit(s). Can be given as single qubit or list.
        T_mids : list of vcz "T_mid" values to sweep.
        A_ranges : list of tuples containing ranges of amplitude sweep.
        A_points : Number of points to sweep for amplitude range.
        Q_parks : list of qubits parked during operation.
        """
        if isinstance(Q0, str):
            Q0 = [Q0]
        if isinstance(Q1, str):
            Q1 = [Q1]
        assert len(Q0) == len(Q1)
        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()
        # get gate directions
        directions = [get_gate_directions(q0, q1) for q0, q1 in zip(Q0, Q1)]
        Flux_lm_0 = [self.find_instrument(q0).instr_LutMan_Flux.get_instr() for q0 in Q0]
        Flux_lm_1 = [self.find_instrument(q1).instr_LutMan_Flux.get_instr() for q1 in Q1]
        Flux_lms_park = [self.find_instrument(q).instr_LutMan_Flux.get_instr() for q in Q_parks]
        # Prepare for time domain
        if prepare_for_timedomain:
            # Time-domain preparation
            self.prepare_for_timedomain(
                qubits=np.array([[Q0[i],Q1[i]] for i in range(len(Q0))]).flatten(),
                bypass_flux=True)
            for i, lm in enumerate(Flux_lm_0):
                print(f'Setting {Q0[i]} sq_amp to -.5')
                lm.set(f'sq_amp', -0.5)
        ###########################
        # Load phase pulses
        ###########################
        for i, q in enumerate(Q0):
            # only on the CZ qubits we add the ef pulses 
            mw_lutman = self.find_instrument(q).instr_LutMan_MW.get_instr()
            lm = mw_lutman.LutMap()
            # we hardcode the X on the ef transition to CW 31 here.
            lm[27] = {'name': 'rXm180', 'phi': 0, 'theta': -180, 'type': 'ge'}
            lm[31] = {"name": "rX12", "theta": 180, "phi": 0, "type": "ef"}
            # load_phase_pulses will also upload other waveforms
            mw_lutman.load_phase_pulses_to_AWG_lookuptable()
        # Wrapper function for conditional oscillation detector function.
        def wrapper(Q0, Q1,
                    prepare_for_timedomain,
                    downsample_swp_points,
                    extract_only,
                    disable_metadata):
            a = self.measure_conditional_oscillation_multi(
                    pairs=[[Q0[i], Q1[i]] for i in range(len(Q0))], 
                    parked_qbs=Q_parks,
                    flux_codeword=flux_codeword,
                    prepare_for_timedomain=prepare_for_timedomain,
                    downsample_swp_points=downsample_swp_points,
                    extract_only=extract_only,
                    disable_metadata=disable_metadata,
                    verbose=False)
            cp = { f'phi_cond_{i+1}' : a[f'pair_{i+1}_delta_phi_a']\
                  for i in range(len(Q0)) }
            mf = { f'missing_fraction_{i+1}' : a[f'pair_{i+1}_missing_frac_a']\
                  for i in range(len(Q0)) }
            return { **cp, **mf} 
            
        d = det.Function_Detector(
            wrapper,
            msmt_kw={'Q0' : Q0, 'Q1' : Q1,
                     'prepare_for_timedomain' : False,
                     'downsample_swp_points': 3,
                     'extract_only': True,
                     'disable_metadata': True},
            result_keys=list(np.array([[f'phi_cond_{i+1}', f'missing_fraction_{i+1}']\
                                   for i in range(len(Q0))]).flatten()),
            value_names=list(np.array([[f'conditional_phase_{i+1}', f'missing_fraction_{i+1}']\
                                   for i in range(len(Q0))]).flatten()),
            value_units=list(np.array([['deg', '%']\
                                   for i in range(len(Q0))]).flatten()))
        nested_MC.set_detector_function(d)

        swf1 = swf.multi_sweep_function_ranges(
            sweep_functions=[Flux_lm_0[i].cfg_awg_channel_amplitude
                             for i in range(len(Q0))],
            sweep_ranges= A_ranges,
            n_points=A_points)
        swfs = [swf.FLsweep(lm = lm,
                            par = lm.parameters['sq_length'],
                            waveform_name = 'square')
                for i, lm in enumerate(Flux_lm_0) ]
        swf2 = swf.multi_sweep_function(sweep_functions=swfs)
        nested_MC.set_sweep_function(swf1)
        nested_MC.set_sweep_points(np.arange(A_points))
        nested_MC.set_sweep_function_2D(swf2)
        nested_MC.set_sweep_points_2D(times)

        MC.live_plot_enabled(False)
        nested_MC.run(f'Unipolar_Amp_vs_t_{Q0}_{Q1}_{Q_parks}',
                      mode='2D', disable_snapshot_metadata=disable_metadata)
        # MC.live_plot_enabled(True)
        a = ma2.tqg.VCZ_B_Analysis(Q0=Q0, Q1=Q1,
                                   A_ranges=A_ranges,
                                   directions=directions,
                                   label='Unipolar_Amp_vs_t')
        ###################################
        # Update flux parameters
        ###################################
        if update_flux_params:
            pass
        return a.qoi

    def measure_parity_check_ramsey(
        self,
        Q_target: list,
        Q_control: list,
        flux_cw_list: list,
        control_cases: list = None,
        Q_spectator: list = None,
        pc_repetitions: int = 1,
        downsample_angle_points: int = 1,
        prepare_for_timedomain: bool = True,
        disable_metadata: bool = False,
        extract_only: bool = False,
        analyze: bool = True,
        solve_for_phase_gate_model: bool = False,
        update_mw_phase: bool = False,
        mw_phase_param: str = 'vcz_virtual_q_ph_corr_step_1',
        wait_time_before_flux: int = 0,
        wait_time_after_flux: int = 0):
        """
        Perform conditional oscillation like experiment in the context of a
        parity check.

        Q_target : Ancilla qubit where parity is projected.
        Q_control : List of control qubits in parity check.
        Q_spectator : Similar to control qubit, but will be treated as 
                      spectator in analysis.
        flux_cw_list : list of flux codewords to be played during the parity
                       check.
        Control_cases : list of different control qubit states. Defaults to all
                        possible combinations of states.
        """
        # assert len(Q_target) == 1
        assert self.ro_acq_weight_type().lower() == 'optimal'
        MC = self.instr_MC.get_instr()
        if Q_spectator:
            Q_control += Q_spectator
        if control_cases == None:
            control_cases = ['{:0{}b}'.format(i, len(Q_control))\
                              for i in range(2**len(Q_control))]
            solve_for_phase_gate_model = True
        else:
            for case in control_cases:
                assert len(case) == len(Q_control)

        qubit_list = Q_target + Q_control
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubit_list)
            for q in Q_target:
                mw_lm = self.find_instrument(q).instr_LutMan_MW.get_instr()
                mw_lm.set_default_lutmap()
                mw_lm.load_phase_pulses_to_AWG_lookuptable()
        Q_target_idx = [self.find_instrument(q).cfg_qubit_nr() for q in Q_target]
        Q_control_idx = [self.find_instrument(q).cfg_qubit_nr() for q in Q_control]
        # These are hardcoded angles in the mw_lutman for the AWG8
        # only x2 and x3 downsample_swp_points available
        angles = np.arange(0, 341, 20 * downsample_angle_points)
        p = mqo.parity_check_ramsey(
            Q_idxs_target = Q_target_idx,
            Q_idxs_control = Q_control_idx,
            control_cases = control_cases,
            flux_cw_list = flux_cw_list,
            platf_cfg = self.cfg_openql_platform_fn(),
            angles = angles,
            nr_spectators = len(Q_spectator) if Q_spectator else 0,
            pc_repetitions=pc_repetitions,
            wait_time_before_flux = wait_time_before_flux,
            wait_time_after_flux = wait_time_after_flux
            )
        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name="Cases",
            unit="a.u."
            )
        d = self.get_int_avg_det(qubits=qubit_list)
        MC.set_sweep_function(s)
        MC.set_sweep_points(p.sweep_points)
        MC.set_detector_function(d)
        label = f'Parity_check_ramsey_{"_".join(qubit_list)}'
        if pc_repetitions != 1:
            label += f'_x{pc_repetitions}'
        label += self.msmt_suffix
        MC.run(label, disable_snapshot_metadata=disable_metadata)
        if analyze:
            a = ma2.tqg.Parity_check_ramsey_analysis(
                label=label,
                Q_target = Q_target,
                Q_control = Q_control,
                Q_spectator = Q_spectator,
                control_cases = control_cases,
                angles = angles,
                solve_for_phase_gate_model = solve_for_phase_gate_model,
                extract_only = extract_only)
            if update_mw_phase:
                if type(mw_phase_param) is str:
                    mw_phase_param = [mw_phase_param for q in Q_target]
                for q, param in zip(Q_target, mw_phase_param):
                    # update single qubit phase
                    Q = self.find_instrument(q)
                    mw_lm = Q.instr_LutMan_MW.get_instr()
                    # Make sure mw phase parameter is valid
                    assert param in mw_lm.parameters.keys()
                    # Calculate new virtual phase
                    phi0 = mw_lm.get(param)
                    phi_new = list(a.qoi['Phase_model'][Q.name].values())[0]
                    phi_new = phi_new / pc_repetitions  # Divide by number of CZ repetitions
                    phi = np.mod(phi0+phi_new, 360)
                    mw_lm.set(param, phi)
                    print(f'{Q.name}.{param} changed to {phi} deg.')
            return a.qoi

    def calibrate_parity_check_phase(
        self,
        Q_ancilla: list,
        Q_control: list,
        Q_pair_target: list,
        flux_cw_list: list,
        B_amps: list = None,
        control_cases: list = None,
        pc_repetitions: int = 1,
        downsample_angle_points: int = 1,
        prepare_for_timedomain: bool = True,
        disable_metadata: bool = False,
        extract_only: bool = True,
        update_flux_param: bool = True,
        update_mw_phase: bool = True,
        mw_phase_param: str = 'vcz_virtual_q_ph_corr_step_1'):
        """
        Calibrate the phase of a gate in a parity-check by performing a sweep 
        of the SNZ B parameter while measuring the parity check phase gate
        coefficients.

        Q_ancilla : Ancilla qubit of the parity check.
        Q_control : List of control qubits in parity check.
        Q_pair_target : list of two qubits involved in the two qubit gate. Must
                        be given in the order [<high_freq_q>, <low_freq_q>]
        flux_cw_list : list of flux codewords to be played during the parity
                       check.
        B_amps : List of B parameters to sweep through.
        Control_cases : list of different control qubit states. Defaults to all
                        possible combinations of states.
        """
        assert self.ro_acq_weight_type().lower() == 'optimal'
        assert len(Q_ancilla) == 1
        qubit_list = Q_ancilla + Q_control
        assert Q_pair_target[0] in qubit_list
        assert Q_pair_target[1] in qubit_list

        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()

        # get gate directions of two-qubit gate codewords
        directions = get_gate_directions(Q_pair_target[0],
                                         Q_pair_target[1])
        fl_lm = self.find_instrument(Q_pair_target[0]).instr_LutMan_Flux.get_instr()
        fl_par = f'vcz_amp_fine_{directions[0]}'
        B0 = fl_lm.get(fl_par)
        if B_amps is None:
            B_amps = np.linspace(-.1, .1, 3)+B0
        if np.min(B_amps) < 0:
            B_amps -= np.min(B_amps)
        if np.max(B_amps) > 1:
            B_amps -= np.max(B_amps)-1

        # Prepare for timedomain
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubit_list)
            for q in Q_ancilla:
                mw_lm = self.find_instrument(q).instr_LutMan_MW.get_instr()
                mw_lm.set_default_lutmap()
                mw_lm.load_phase_pulses_to_AWG_lookuptable()
        # Wrapper function for parity check ramsey detector function.
        def wrapper(Q_target, Q_control,
                    flux_cw_list,
                    downsample_angle_points,
                    extract_only):
            a = self.measure_parity_check_ramsey(
                Q_target = Q_target,
                Q_control = Q_control,
                flux_cw_list = flux_cw_list,
                control_cases = None,
                downsample_angle_points = downsample_angle_points,
                prepare_for_timedomain = False,
                pc_repetitions=pc_repetitions,
                solve_for_phase_gate_model = True,
                disable_metadata = True,
                extract_only = extract_only)
            pm = { f'Phase_model_{op}' : a['Phase_model'][Q_ancilla[0]][op]\
                   for op in a['Phase_model'][Q_ancilla[0]].keys()}
            mf = { f'missing_fraction_{q}' : a['Missing_fraction'][q]\
                  for q in Q_control }
            return { **pm, **mf} 
        n = len(Q_control)
        Operators = ['{:0{}b}'.format(i, n).replace('0','I').replace('1','Z')\
                     for i in range(2**n)]
        d = det.Function_Detector(
            wrapper,
            msmt_kw={'Q_target' : Q_ancilla,
                     'Q_control' : Q_control,
                     'flux_cw_list': flux_cw_list,
                     'downsample_angle_points': downsample_angle_points,
                     'extract_only': extract_only},
            result_keys=[f'Phase_model_{op}' for op in Operators]+\
                        [f'missing_fraction_{q}' for q in Q_control],
            value_names=[f'Phase_model_{op}' for op in Operators]+\
                        [f'missing_fraction_{q}' for q in Q_control],
            value_units=['deg' for op in Operators]+\
                        ['fraction' for q in Q_control])
        nested_MC.set_detector_function(d)
        # Set sweep function
        swf1 = swf.FLsweep(
            lm = fl_lm,
            par = fl_lm.parameters[fl_par],
            waveform_name = f'cz_{directions[0]}')
        nested_MC.set_sweep_function(swf1)
        nested_MC.set_sweep_points(B_amps)

        MC.live_plot_enabled(False)
        label = f'Parity_check_calibration_gate_{"_".join(Q_pair_target)}'
        nested_MC.run(label, disable_snapshot_metadata=disable_metadata)
        # MC.live_plot_enabled(True)

        a = ma2.tqg.Parity_check_calibration_analysis(
            Q_ancilla = Q_ancilla,
            Q_control = Q_control,
            Q_pair_target = Q_pair_target,
            B_amps = B_amps,
            label = label)
        if update_flux_param:
            try :
                if (a.qoi['Optimal_B']>0) and (a.qoi['Optimal_B']<1):
                    # update flux parameter
                    fl_lm.set(fl_par, a.qoi['Optimal_B'])
                elif a.qoi['Optimal_B']<0:
                    fl_lm.set(fl_par, 0)
                elif a.qoi['Optimal_B']>1:
                    fl_lm.set(fl_par, 1)
            except:
                fl_lm.set(fl_par, B0)
                raise ValueError(f'B amplitude {a.qoi["Optimal_B"]:.3f} not valid. '+\
                                 f'Resetting {fl_par} to {B0:.3f}.')
        else:
            fl_lm.set(fl_par, B0)
            print(f'Resetting {fl_par} to {B0:.3f}.')

        if update_mw_phase:
            # update single qubit phase
            Qa = self.find_instrument(Q_ancilla[0])
            mw_lm = Qa.instr_LutMan_MW.get_instr()
            # Make sure mw phase parameter is valid
            assert mw_phase_param in mw_lm.parameters.keys()
            # Calculate new virtual phase
            phi0 = mw_lm.get(mw_phase_param)
            phi = np.mod(phi0+a.qoi['Phase_offset'], 360)
            mw_lm.set(mw_phase_param, phi)

        return a.qoi

    def calibrate_park_frequency(
        self,
        qH: str,
        qL: str,
        qP: str,
        Park_distances: list = np.arange(300e6, 1000e6, 5e6),
        flux_cw: str = 'cz',
        extract_only: bool = False,
        prepare_for_timedomain: bool = True,
        disable_metadata: bool = False):
        """
        Calibrate the parking amplitude of a spectator for a given two-qubit 
        gate. Does this by sweeping the parking frequency while measuring the 
        conditional phases and missing fraction of the three qubits involved. 

        qH : High frequency qubit in two-qubit gate.
        qL : Low frequency qubit in two-qubit gate.
        qP : Parked qubit on which we'll sweep park frequency.
        flux_cw : flux codeword of two-qubit gate.
        Park_distances : List of Park sweep (frequency) distances to low 
                         frequency qubit during the two-qubit gate.
        """
        assert self.ro_acq_weight_type() == 'optimal'
        # Get measurement control instances
        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()
        # setup up measurement
        dircts = get_gate_directions(qH, qL)
        Q_H = self.find_instrument(qH)
        Q_L = self.find_instrument(qL)
        Q_P = self.find_instrument(qP)
        flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
        flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
        flux_lm_P = Q_P.instr_LutMan_Flux.get_instr()
        # Prepare for timedomain
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[qH, qL, qP])
            # Upload phase pulses on qH
            mw_lm_H = Q_H.instr_LutMan_MW.get_instr()
            mw_lm_H.set_default_lutmap()
            mw_lm_H.load_phase_pulses_to_AWG_lookuptable()
        # Wrapper function for parity check ramsey detector function.
        def wrapper():
            # downsampling factor (makes sweep faster!)
            downsample = 3
            self.measure_parity_check_ramsey(
                Q_target = [qH],
                Q_control = [qL],
                Q_spectator = [qP],
                flux_cw_list = [flux_cw],
                prepare_for_timedomain = False,
                downsample_angle_points = downsample,
                update_mw_phase=False,
                analyze=False,
                disable_metadata=True)
            # Analyze
            a = ma2.tqg.Parity_check_ramsey_analysis(
                Q_target = [qH],
                Q_control = [qL, qP],
                Q_spectator = [qP],
                control_cases = ['{:0{}b}'.format(i, 2) for i in range(4)],
                angles = np.arange(0, 341, 20*downsample),
                solve_for_phase_gate_model = True,
                extract_only = True)
            # Get residual ZZ phase
            phi   = a.proc_data_dict['Fit_res'][qH]['00'][0]
            phi_s = a.proc_data_dict['Fit_res'][qH]['01'][0]
            delta_phi = phi_s-phi
            phi = np.mod(phi+180, 360)-180
            phi_s = np.mod(phi_s+180, 360)-180
            delta_phi = np.mod(delta_phi+180, 360)-180
            # Conditional phase difference
            phi_cond   = a.proc_data_dict['Fit_res'][qH]['00'][0]-a.proc_data_dict['Fit_res'][qH]['10'][0]
            phi_cond_s = a.proc_data_dict['Fit_res'][qH]['01'][0]-a.proc_data_dict['Fit_res'][qH]['11'][0]
            delta_phi_cond = phi_cond_s-phi_cond
            phi_cond = np.mod(phi_cond, 360)
            phi_cond_s = np.mod(phi_cond_s, 360)
            delta_phi_cond = np.mod(delta_phi_cond+180, 360)-180
            # Missing fraction
            miss_frac   = a.proc_data_dict['P_excited'][qL]['10']-a.proc_data_dict['P_excited'][qL]['00']
            miss_frac_s = a.proc_data_dict['P_excited'][qL]['11']-a.proc_data_dict['P_excited'][qL]['01']
            delta_miss_frac = miss_frac_s-miss_frac
            # result dictionary
            _r = {'phi': phi, 'phi_s': phi_s, 'delta_phi': delta_phi,
                  'phi_cond': phi_cond, 'phi_cond_s': phi_cond_s, 'delta_phi_cond': delta_phi_cond,
                  'miss_frac': miss_frac, 'miss_frac_s': miss_frac_s, 'delta_miss_frac': delta_miss_frac} 
            return _r 
        d = det.Function_Detector(
            wrapper,
            msmt_kw={},
            result_keys=['phi', 'phi_s', 'delta_phi', 
                         'phi_cond', 'phi_cond_s', 'delta_phi_cond', 
                         'miss_frac', 'miss_frac_s', 'delta_miss_frac'],
            value_names=['phi', 'phi_s', 'delta_phi', 
                         'phi_cond', 'phi_cond_s', 'delta_phi_cond', 
                         'miss_frac', 'miss_frac_s', 'delta_miss_frac'],
            value_units=['deg', 'deg', 'deg', 'deg', 'deg', 'deg',
                         'fraction', 'fraction', 'fraction'])
        nested_MC.set_detector_function(d)
        # Set sweep function
        swf1 = swf.FLsweep(
            lm = flux_lm_P,
            par = flux_lm_P.parameters['park_amp'],
            waveform_name = 'park')
        nested_MC.set_sweep_function(swf1)
        # Get parking amplitudes based on parking distances
        Park_detunings = []
        Park_amps = []
        for park_dist in Park_distances:
            # calculate detuning of qH during 2Q-gate
            det_qH = get_frequency_waveform(f'vcz_amp_dac_at_11_02_{dircts[0]}',
                                            flux_lm_H)
            det_qL = get_frequency_waveform(f'vcz_amp_dac_at_11_02_{dircts[1]}',
                                            flux_lm_L)
            # calculate required detuning of qP during 2Q-gate
            park_freq = Q_H.freq_qubit()-det_qH-park_dist
            park_det = Q_P.freq_qubit()-park_freq
            Park_detunings.append(park_det)
            # Only park if the qubit is closer than then 350 MHz
            amp_park = get_DAC_amp_frequency(park_det, flux_lm_P)
            Park_amps.append(amp_park)
        # If parking distance results in negative detuning, clip those values
        idx = np.where(np.array(Park_detunings)>0)[0]
        Park_amps = np.array(Park_amps)[idx]
        Park_distances = np.array(Park_distances)[idx]
        # set sweeping park amps
        nested_MC.set_sweep_points(Park_amps)
        # Measure!
        MC.live_plot_enabled(False)
        label = f'Park_frequency_calibration_gate_{qH}_{qL}_park_{qP}'
        try:
            nested_MC.run(label, disable_snapshot_metadata=disable_metadata)
        except:
            print_exception()
        self.msmt_suffix = '_device'
        # MC.live_plot_enabled(True)
        # Run analysis
        # if False:
        a = ma2.tqg.Park_frequency_sweep_analysis(
            label=label,
            qH=qH, qL=qL, qP=qP,
            Parking_distances=Park_distances,
            alpha_qH=Q_H.anharmonicity())
        
    def calibrate_parity_check_park_new(
        self,
        Q_neighbors: list,
        Q_ancilla: str,
        Q_park: str,
        flux_cw_list: list,
        Park_amps: list,
        downsample_angle_points: int = 1,
        prepare_for_timedomain: bool = True,
        extract_only: bool = False,
        update_park_amp: bool = True):
        """
        """
        qubit_list = Q_neighbors + [Q_ancilla] + [Q_park]

        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()

        fl_lm = self.find_instrument(Q_park).instr_LutMan_Flux.get_instr()
        fl_par = 'park_amp'
        P0 = fl_lm.get(fl_par)

        # Prepare for timedomain
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubit_list)
            for q in Q_neighbors:
                mw_lm = self.find_instrument(q).instr_LutMan_MW.get_instr()
                mw_lm.set_default_lutmap()
                mw_lm.load_phase_pulses_to_AWG_lookuptable()
        # Wrapper function for parity check ramsey detector function.
        def wrapper(Q_neighbors,
                    Q_park,
                    Q_ancilla,
                    flux_cw_list,
                    downsample_angle_points,
                    extract_only):
            a = self.measure_parity_check_ramsey(
                Q_target = Q_neighbors,
                Q_control = [Q_ancilla],
                Q_spectator = [Q_park],
                flux_cw_list = flux_cw_list,
                downsample_angle_points = downsample_angle_points,
                prepare_for_timedomain = False,
                disable_metadata = True,
                extract_only = extract_only)
            Phase_coeff = {}
            for q in Q_neighbors:
                Phase_coeff['_'.join([q]+[Q_park])] = a['Phase_model'][q]['IZ']
                Phase_coeff['_'.join([q]+[Q_ancilla]+[Q_park])] = \
                                        a['Phase_model'][q]['ZZ']
            missing_fraction = a['Missing_fraction']
            return {**Phase_coeff, **missing_fraction}

        keys = []
        for q in Q_neighbors:
            keys.append(f'{q}_{Q_park}')
            keys.append(f'{q}_{Q_ancilla}_{Q_park}')
        d = det.Function_Detector(
            wrapper,
            msmt_kw={'Q_neighbors' : Q_neighbors,
                     'Q_park' : Q_park,
                     'Q_ancilla' : Q_ancilla,
                     'flux_cw_list': flux_cw_list,
                     'downsample_angle_points': downsample_angle_points,
                     'extract_only': extract_only},
            result_keys=keys+[Q_ancilla]+[Q_park],
            value_names=[f'Phase_coeff_{k}' for k in keys]+\
                        [f'missing_fraction_{q}' for q in [Q_ancilla, Q_park]],
            value_units=['deg' for k in keys]+\
                        ['missing_fraction' for i in range(2)])
        nested_MC.set_detector_function(d)
        # Set sweep function
        swf1 = swf.FLsweep(
            lm = fl_lm,
            par = fl_lm.parameters[fl_par],
            waveform_name = 'park')
        nested_MC.set_sweep_function(swf1)
        nested_MC.set_sweep_points(Park_amps)

        # MC.live_plot_enabled(False)
        label = f'Parity_check_calibration_park_{Q_park}_{"_".join(Q_neighbors)}_{Q_ancilla}'
        nested_MC.run(label)
        # MC.live_plot_enabled(True)

        if update_park_amp:
            pass
            # P_opt = a.qoi['Amp_opt']
            # fl_lm.set(fl_par, P_opt)
            # print(f'Park amplitude of {Q_park_target} set to {P_opt}.')
        else:
            fl_lm.set(fl_par, P0)
            print(f'Park amplitude of {Q_park} reset to {P0}.')

    def calibrate_vcz_flux_offset(
        self, 
        Q0: str, Q1: str,
        Offsets: list = None,
        Q_parks: list = None,
        update_params: bool = True,
        flux_codeword: str = 'cz',
        disable_metadata: bool = False):
        """
        Perform a sweep of flux offset of high freq. qubit while measuring 
        conditional phase and missing fraction via the "conditional 
        oscillation" experiment.

        Q0 : High frequency qubit. Can be given as single qubit or list.
        Q1 : Low frequency qubit. Can be given as single qubit or list.
        Offsets : Offsets of pulse asymmetry.
        Q_parks : list of qubits parked during operation.
        """
        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()
        if Offsets == None:
            Q_inst = self.find_instrument(Q0)
            dc_off = Q_inst.fl_dc_I0()
            if 'D' in Q0:
                # When the Q0 is a data qubit, it can only be a high - mid 
                # CZ gate type. When this happens we want higher range
                Offsets = np.linspace(-40e-6, 40e-6, 7)+dc_off
            else:
                # When the Q0 is not a data qubit, it can only be a mid - low 
                # CZ gate type. When this happens we want lower range
                Offsets = np.linspace(-20e-6, 20e-6, 7)+dc_off
        # Time-domain preparation
        self.prepare_for_timedomain(
            qubits=[Q0, Q1],
            bypass_flux=False)
        ###########################
        # Load phase pulses
        ###########################
        # only on the CZ qubits we add the ef pulses 
        mw_lutman = self.find_instrument(Q0).instr_LutMan_MW.get_instr()
        lm = mw_lutman.LutMap()
        # we hardcode the X on the ef transition to CW 31 here.
        lm[27] = {'name': 'rXm180', 'phi': 0, 'theta': -180, 'type': 'ge'}
        lm[31] = {"name": "rX12", "theta": 180, "phi": 0, "type": "ef"}
        # load_phase_pulses will also upload other waveforms
        mw_lutman.load_phase_pulses_to_AWG_lookuptable()
        # Wrapper function for conditional oscillation detector function.
        def wrapper(Q0, Q1,
                    prepare_for_timedomain,
                    downsample_swp_points,
                    extract_only,
                    disable_metadata):
            a = self.measure_conditional_oscillation_multi(
                    pairs=[[Q0, Q1]], 
                    parked_qbs=Q_parks,
                    flux_codeword=flux_codeword,
                    prepare_for_timedomain=prepare_for_timedomain,
                    downsample_swp_points=downsample_swp_points,
                    extract_only=extract_only,
                    disable_metadata=disable_metadata,
                    verbose=False)
            cp = { f'phi_cond_{1}' : a[f'pair_{1}_delta_phi_a']}
            mf = { f'missing_fraction_{1}' : a[f'pair_{1}_missing_frac_a']}
            return { **cp, **mf} 
        d = det.Function_Detector(
            wrapper,
            msmt_kw={'Q0' : Q0, 'Q1' : Q1,
                     'prepare_for_timedomain' : False,
                     'downsample_swp_points': 3,
                     'extract_only': True,
                     'disable_metadata': True},
            result_keys=[f'phi_cond_{1}', f'missing_fraction_{1}'],
            value_names=[f'conditional_phase_{1}', f'missing_fraction_{1}'],
            value_units=['deg', '%'])
        nested_MC.set_detector_function(d)
        Q_inst = self.find_instrument(Q0)
        swf1 = Q_inst.instr_FluxCtrl.get_instr()[Q_inst.fl_dc_ch()]
        nested_MC.set_sweep_function(swf1)
        nested_MC.set_sweep_points(Offsets)
        try:
            MC.live_plot_enabled(False)
            nested_MC.run(f'VCZ_flux_offset_sweep_{Q0}_{Q1}_{Q_parks}', mode='1D', 
                          disable_snapshot_metadata=disable_metadata)
            MC.live_plot_enabled(True)
            a = ma2.tqg.VCZ_flux_offset_sweep_Analysis(label='VCZ_flux_offset_sweep')
        except:
            print_exception()
            print(f'Resetting flux offset of {Q0}...')
            swf1.set(Q_inst.fl_dc_I0())
        ################################
        # Update (or reset) params
        ################################
        if update_params:
            swf1(a.qoi[f'offset_opt'])
            Q_inst.fl_dc_I0(a.qoi[f'offset_opt'])
            print(f'Updated {swf1.name} to {a.qoi[f"offset_opt"]*1e3:.3f}mA')

    def calibrate_vcz_asymmetry(
        self, 
        Q0, Q1,
        Asymmetries: list = np.linspace(-.005, .005, 7),
        Q_parks: list = None,
        prepare_for_timedomain = True,
        update_params: bool = True,
        flux_codeword: str = 'cz',
        disable_metadata: bool = False):
        """
        Perform a sweep of vcz pulse asymmetry while measuring 
        conditional phase and missing fraction via the "conditional 
        oscillation" experiment.

        Q0 : High frequency qubit(s). Can be given as single qubit or list.
        Q1 : Low frequency qubit(s). Can be given as single qubit or list.
        Offsets : Offsets of pulse asymmetry.
        Q_parks : list of qubits parked during operation.
        """
        if isinstance(Q0, str):
            Q0 = [Q0]
        if isinstance(Q1, str):
            Q1 = [Q1]
        assert len(Q0) == len(Q1)
        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()
        # get gate directions
        directions = [get_gate_directions(q0, q1) for q0, q1 in zip(Q0, Q1)]
        Flux_lm_0 = [self.find_instrument(q0).instr_LutMan_Flux.get_instr() for q0 in Q0]
        Flux_lm_1 = [self.find_instrument(q1).instr_LutMan_Flux.get_instr() for q1 in Q1]
        Flux_lms_park = [self.find_instrument(q).instr_LutMan_Flux.get_instr() for q in Q_parks]
        # Make sure asymmetric pulses are enabled
        for i, flux_lm in enumerate(Flux_lm_0):
            param = flux_lm.parameters[f'vcz_use_asymmetric_amp_{directions[i][0]}']
            assert param() == True , 'Asymmetric pulses must be enabled.'
        if prepare_for_timedomain:    
            # Time-domain preparation
            self.prepare_for_timedomain(
                qubits=np.array([[Q0[i],Q1[i]] for i in range(len(Q0))]).flatten(),
                bypass_flux=True)
            ###########################
            # Load phase pulses
            ###########################
            for i, q in enumerate(Q0):
                # only on the CZ qubits we add the ef pulses 
                mw_lutman = self.find_instrument(q).instr_LutMan_MW.get_instr()
                lm = mw_lutman.LutMap()
                # we hardcode the X on the ef transition to CW 31 here.
                lm[27] = {'name': 'rXm180', 'phi': 0, 'theta': -180, 'type': 'ge'}
                lm[31] = {"name": "rX12", "theta": 180, "phi": 0, "type": "ef"}
                # load_phase_pulses will also upload other waveforms
                mw_lutman.load_phase_pulses_to_AWG_lookuptable()
        # Wrapper function for conditional oscillation detector function.
        def wrapper(Q0, Q1,
                    prepare_for_timedomain,
                    downsample_swp_points,
                    extract_only,
                    disable_metadata):
            a = self.measure_conditional_oscillation_multi(
                    pairs=[[Q0[i], Q1[i]] for i in range(len(Q0))], 
                    parked_qbs=Q_parks,
                    flux_codeword=flux_codeword,
                    prepare_for_timedomain=prepare_for_timedomain,
                    downsample_swp_points=downsample_swp_points,
                    extract_only=extract_only,
                    disable_metadata=disable_metadata,
                    verbose=False)
            cp = { f'phi_cond_{i+1}' : a[f'pair_{i+1}_delta_phi_a']\
                  for i in range(len(Q0)) }
            mf = { f'missing_fraction_{i+1}' : a[f'pair_{i+1}_missing_frac_a']\
                  for i in range(len(Q0)) }
            return { **cp, **mf} 
            
        d = det.Function_Detector(
            wrapper,
            msmt_kw={'Q0' : Q0, 'Q1' : Q1,
                     'prepare_for_timedomain' : False,
                     'downsample_swp_points': 3,
                     'extract_only': True,
                     'disable_metadata': True},
            result_keys=list(np.array([[f'phi_cond_{i+1}', f'missing_fraction_{i+1}']\
                                   for i in range(len(Q0))]).flatten()),
            value_names=list(np.array([[f'conditional_phase_{i+1}', f'missing_fraction_{i+1}']\
                                   for i in range(len(Q0))]).flatten()),
            value_units=list(np.array([['deg', '%']\
                                   for i in range(len(Q0))]).flatten()))
        nested_MC.set_detector_function(d)
        swfs = [swf.FLsweep(lm = lm,
                            par = lm.parameters[f'vcz_asymmetry_{directions[i][0]}'],
                            waveform_name = f'cz_{directions[i][0]}')
                for i, lm in enumerate(Flux_lm_0) ]
        swf1 = swf.multi_sweep_function(sweep_functions=swfs)
        nested_MC.set_sweep_function(swf1)
        nested_MC.set_sweep_points(Asymmetries)

        MC.live_plot_enabled(False)
        nested_MC.run(f'VCZ_asymmetry_sweep_{Q0}_{Q1}_{Q_parks}', mode='1D', 
                      disable_snapshot_metadata=disable_metadata)
        MC.live_plot_enabled(True)
        a = ma2.tqg.VCZ_asymmetry_sweep_Analysis(label='VCZ_asymmetry_sweep')
        ################################
        # Update (or reset) flux params
        ################################
        for i, flux_lm in enumerate(Flux_lm_0):
            param = flux_lm.parameters[f'vcz_asymmetry_{directions[i][0]}']
            if update_params:
                param(a.qoi[f'asymmetry_opt_{i}'])
                print(f'Updated {param.name} to {a.qoi[f"asymmetry_opt_{i}"]*100:.3f}%')
            else:
                param(0)
                print(f'Reset {param.name} to 0%')

    def calibrate_cz_pad_samples(
        self,
        Q_ramsey: str,
        Q_control: str,
        flux_cw: str = 'cz',
        Sample_points: list = None,
        downsample_angle_points: int = 1,
        update: bool = True,
        prepare_for_timedomain: bool = True,
        disable_metadata: bool = False,
        extract_only: bool = True):
        """
        Dont forget to write a description for this function.
        """
        assert self.ro_acq_weight_type().lower() == 'optimal'
        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()
        # get gate directions of two-qubit gate codewords
        directions = get_gate_directions(Q_ramsey, Q_control)
        fl_lm = self.find_instrument(Q_control).instr_LutMan_Flux.get_instr()
        fl_par = f'vcz_amp_pad_samples_{directions[1]}'
        # Calculate sweep points
        _sample_rate = fl_lm.sampling_rate()
        _time_pad = fl_lm.get(f'vcz_time_pad_{directions[1]}')
        max_samples = int(_time_pad*_sample_rate)
        if Sample_points is None:
            Sample_points = np.arange(1, max_samples)
        # Prepare for timedomain
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=[Q_ramsey, Q_control])
            for q in [Q_ramsey]:
                mw_lm = self.find_instrument(q).instr_LutMan_MW.get_instr()
                mw_lm.set_default_lutmap()
                mw_lm.load_phase_pulses_to_AWG_lookuptable()
        # Wrapper function for parity check ramsey detector function.
        def wrapper(Q_target, Q_control,
                    flux_cw,
                    downsample_angle_points,
                    extract_only):
            a = self.measure_parity_check_ramsey(
                Q_target = [Q_target],
                Q_control = [Q_control],
                flux_cw_list = [flux_cw],
                control_cases = None,
                downsample_angle_points = downsample_angle_points,
                prepare_for_timedomain = False,
                pc_repetitions=1,
                solve_for_phase_gate_model = False,
                disable_metadata = True,
                extract_only = extract_only)
            pm = { f'Phase_model_{op}' : a['Phase_model'][Q_target][op]\
                   for op in a['Phase_model'][Q_target].keys()}
            mf = { f'missing_fraction_{q}' : a['Missing_fraction'][q]\
                  for q in [Q_control] }
            return { **pm, **mf} 
        # n = len([Q_control])
        # Operators = ['{:0{}b}'.format(i, n).replace('0','I').replace('1','Z')\
        #              for i in range(2**n)]
        Operators = ['I', 'Z']
        d = det.Function_Detector(
            wrapper,
            msmt_kw={'Q_target' : Q_ramsey,
                     'Q_control' : Q_control,
                     'flux_cw': flux_cw,
                     'downsample_angle_points': downsample_angle_points,
                     'extract_only': extract_only},
            result_keys=[f'Phase_model_{op}' for op in Operators]+\
                        [f'missing_fraction_{Q_control}'],
            value_names=[f'Phase_model_{op}' for op in Operators]+\
                        [f'missing_fraction_{Q_control}'],
            value_units=['deg' for op in Operators]+\
                        ['fraction'])
        nested_MC.set_detector_function(d)
        # Set sweep function
        swf1 = swf.flux_make_pulse_netzero(
            flux_lutman = fl_lm,
            wave_id = f'cz_{directions[1]}')
        nested_MC.set_sweep_function(swf1)
        nested_MC.set_sweep_points(Sample_points)

        MC.live_plot_enabled(False)
        label = f'Pad_samples_calibration_{"_".join([Q_ramsey, Q_control])}'
        nested_MC.run(label, disable_snapshot_metadata=disable_metadata)
        # Run analysis
        a = ma2.Basic1DAnalysis(label=label)
        # get minimum missing fraction
        mf = a.raw_data_dict['measured_values_ord_dict'][f'missing_fraction_{Q_control}'][0]
        sample_points = a.raw_data_dict['xvals'][0]
        min_idx = np.argmin(mf)
        opt_val = sample_points[min_idx]
        if update:
            fl_lm.set(f'vcz_amp_pad_samples_{directions[1]}', opt_val)
        return a

    def measure_parity_check_fidelity(
        self,
        Q_ancilla: List[str],
        Q_control: List[str],
        flux_cw_list: List[str],
        control_cases: List[str] = None,
        prepare_for_timedomain: bool = True,
        initialization_msmt: bool = False,
        disable_metadata: bool = False, 
        nr_shots_per_case: int = 2**12,
        wait_time_before_flux_ns: int = 0,
        wait_time_after_flux_ns: int = 0
        ):
        '''
        Measures parity check fidelity by preparing each.
        Note: When using heralded initialization, Q_control has
        to be given in ascending order
        '''
        assert self.ro_acq_weight_type().lower() == 'optimal'
        assert len(Q_ancilla) == 1
        MC = self.instr_MC.get_instr()
        if control_cases == None:
            control_cases = ['{:0{}b}'.format(i, len(Q_control))\
                              for i in range(2**len(Q_control))]
        else:
            for case in control_cases:
                assert len(case) == len(Q_control)
        qubit_list = Q_ancilla+Q_control
        if prepare_for_timedomain:
            self.prepare_for_timedomain(qubits=qubit_list)
            if not initialization_msmt:
                self.prepare_readout(qubits=Q_ancilla)

        Q_ancilla_idx = [self.find_instrument(q).cfg_qubit_nr() 
                         for q in Q_ancilla]
        Q_control_idx = [self.find_instrument(q).cfg_qubit_nr() 
                         for q in Q_control]
        p = mqo.parity_check_fidelity(
            Q_ancilla_idx = Q_ancilla_idx,
            Q_control_idx = Q_control_idx,
            control_cases = control_cases,
            flux_cw_list = flux_cw_list,
            initialization_msmt = initialization_msmt,
            wait_time_before_flux = wait_time_before_flux_ns,
            wait_time_after_flux = wait_time_after_flux_ns,
            platf_cfg = self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name="Cases",
            unit="a.u.")
        d = self.get_int_logging_detector(qubits=qubit_list)
        total_shots = nr_shots_per_case*(len(control_cases)+2)
        if initialization_msmt:
            n = len(Q_control)
            total_shots = nr_shots_per_case*(len(control_cases)*2+2**(n+1))
        for detector in d.detectors:
            detector.nr_shots = total_shots
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(total_shots))
        MC.set_detector_function(d)
        label = f'Parity_check_fidelity_{"_".join(qubit_list)}'
        MC.run(label,disable_snapshot_metadata=disable_metadata)
        a = ma2.tqg.Parity_check_fidelity_analysis(
            label=label,
            Q_ancilla=Q_ancilla[0],
            Q_control=Q_control,
            control_cases=control_cases,
            post_selection=initialization_msmt)
        return a.qoi

    def measure_sandia_parity_benchmark(self,
                                        ancilla_qubit: str,
                                        data_qubits: list,
                                        flux_cw_list: list,
                                        wait_time_before_flux: int = 0,
                                        wait_time_after_flux: int = 0,
                                        prepare_for_timedomain:bool=True):
        ###################
        # setup qubit idxs
        ###################
        all_qubits = [ancilla_qubit]+data_qubits
        ancilla_idx = self.find_instrument(ancilla_qubit).cfg_qubit_nr()
        data_idxs = [ self.find_instrument(q).cfg_qubit_nr() for q in data_qubits ]
        ###########################################
        # RO preparation (assign res_combinations)
        ###########################################
        RO_lms = np.unique([self.find_instrument(q).instr_LutMan_RO() for q in all_qubits])
        qubit_RO_lm = { self.find_instrument(q).cfg_qubit_nr() : 
                      (self.find_instrument(q).name, 
                       self.find_instrument(q).instr_LutMan_RO()) for q in all_qubits }
        main_qubits = []
        exception_qubits = []
        res_combs = {}
        for lm in RO_lms:
            res_combs[lm] = []
            comb = []
            for idx in data_idxs+[ancilla_idx]:
                  if qubit_RO_lm[idx][1] == lm:
                        comb += [idx]
            res_combs[lm] += [comb]
            if qubit_RO_lm[ancilla_idx][1] == lm:
                  res_combs[lm] += [[ancilla_idx]]
                  main_qubits = [qubit_RO_lm[idx][0] for idx in comb]
            else:
                  exception_qubits += [qubit_RO_lm[idx][0] for idx in comb]
        # Time-domain preparation
        ordered_qubits = main_qubits+exception_qubits
        if prepare_for_timedomain:
            assert self.ro_acq_weight_type() == 'optimal'
            self.prepare_for_timedomain(ordered_qubits)
            for lm in RO_lms:
                ro_lm = self.find_instrument(lm)
                ro_lm.resonator_combinations(res_combs[lm])
                ro_lm.load_DIO_triggered_sequence_onto_UHFQC()
        ########################
        # SETUP MC and detector
        ########################
        uhfqc_max_avg = 2**17
        d = self.get_int_logging_detector(qubits=ordered_qubits, result_logging_mode='raw')
        for detector in d.detectors:
            detector.nr_shots = int(uhfqc_max_avg/5)*5
        p = mqo.Parity_Sandia_benchmark(qA=ancilla_idx,
                                        QDs=data_idxs,
                                        flux_cw_list=flux_cw_list,
                                        wait_time_before_flux=wait_time_before_flux,
                                        wait_time_after_flux=wait_time_after_flux,
                                        platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        MC = self.instr_MC.get_instr()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(int(uhfqc_max_avg/5)*5))
        MC.set_detector_function(d)
        MC.run(f"Sandia_parity_benchmark_{ancilla_qubit}_{data_qubits[0]}_{data_qubits[1]}_{data_qubits[2]}_{data_qubits[3]}")

        ma2.pba.Sandia_parity_benchmark(label='Sandia',
                                        ancilla_qubit=ancilla_qubit, 
                                        data_qubits=data_qubits,
                                        exception_qubits=exception_qubits)

    def measure_weight_n_parity_tomography(
            self,
            ancilla_qubit: str,
            data_qubits: list,
            flux_cw_list: list,
            sim_measurement: bool,
            prepare_for_timedomain: bool=True,
            initialization_msmt: bool = False,
            repetitions: int=3,
            wait_time_before_flux: int = 0,
            wait_time_after_flux: int = 0,
            n_rounds = 1,
            disable_metadata=True,
            readout_duration_ns: int = 480
            ):
        assert self.ro_acq_weight_type().lower() == 'optimal'
        assert self.ro_acq_digitized() == False
        ###################
        # setup qubit idxs
        ###################
        n = len(data_qubits)
        all_qubits = [ancilla_qubit]+data_qubits
        ancilla_idx = self.find_instrument(ancilla_qubit).cfg_qubit_nr()
        data_idxs = [ self.find_instrument(q).cfg_qubit_nr() for q in data_qubits ]
        ###########################################
        # RO preparation (assign res_combinations)
        ###########################################
        RO_lms = np.unique([self.find_instrument(q).instr_LutMan_RO() for q in all_qubits])
        qubit_RO_lm = {self.find_instrument(q).cfg_qubit_nr() : 
            (self.find_instrument(q).name, 
             self.find_instrument(q).instr_LutMan_RO()) for q in all_qubits }
        main_qubits = []
        exception_qubits = []
        res_combs = {}
        for lm in RO_lms:
            res_combs[lm] = []
            comb1= [] # comb used for MUX of all qubits (final meas.)
            comb2= [] # comb used for MUX of just data qubits (final meas.) 
            # ancilla + data qubits resonators
            for idx in [ancilla_idx]+data_idxs:
                if qubit_RO_lm[idx][1] == lm:
                    comb1+= [idx]
                    comb2+= [idx]
            res_combs[lm] += [comb1]
            if qubit_RO_lm[ancilla_idx][1] == lm:
                if not ([ancilla_idx] in res_combs[lm]):
                    res_combs[lm] += [[ancilla_idx]] # comb of just anc. qubit
                comb2.remove(ancilla_idx)
                if comb2 != []:
                    res_combs[lm] += [comb2]
                main_qubits = [qubit_RO_lm[idx][0] for idx in comb1]
            else:
                  exception_qubits += [qubit_RO_lm[idx][0] for idx in comb1]
        # Time-domain preparation
        ordered_qubits = main_qubits+exception_qubits
        if prepare_for_timedomain:
            assert self.ro_acq_weight_type() == 'optimal'
            self.prepare_for_timedomain(ordered_qubits)
            for lm in RO_lms:
                ro_lm = self.find_instrument(lm)
                ro_lm.resonator_combinations(res_combs[lm])
                ro_lm.load_DIO_triggered_sequence_onto_UHFQC()
        # Sequence is compiled with qubit order
        # matching the detector order.
        ord_data_qubit_idxs = [self.find_instrument(q).cfg_qubit_nr()
                               for q in ordered_qubits[1:]]
        exc_qubit_idxs = [self.find_instrument(q).cfg_qubit_nr()
                               for q in exception_qubits]
        p = mqo.Weight_n_parity_tomography(
            Q_anc=ancilla_idx,
            Q_D=ord_data_qubit_idxs,
            flux_cw_list=flux_cw_list,
            Q_exception=exc_qubit_idxs,
            simultaneous_measurement=sim_measurement,
            initialization_msmt = initialization_msmt,
            wait_time_before_flux=wait_time_before_flux,
            wait_time_after_flux=wait_time_after_flux,
            n_rounds = n_rounds,
            readout_duration_ns=readout_duration_ns,
            platf_cfg=self.cfg_openql_platform_fn())

        uhfqc_max_avg = 2**17
        if sim_measurement:
            readouts_per_round = (3**n)*n_rounds+2**(n+1)
            if initialization_msmt:
                readouts_per_round = ((3**n)*n_rounds)*2+2**(n+1)
        else:
            readouts_per_round = (3**n)*(n_rounds+1)+2**(n+1)
            if initialization_msmt:
                readouts_per_round = ((3**n)*(n_rounds+1))*2+2**(n+1)
        
        d = self.get_int_logging_detector(qubits=all_qubits,
                                          result_logging_mode='raw')
        for det in d.detectors:
            det.nr_shots = int(uhfqc_max_avg/readouts_per_round)*readouts_per_round
        s = swf.OpenQL_Sweep(openql_program=p,
                     CCL=self.instr_CC.get_instr())
        MC = self.instr_MC.get_instr()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(int(uhfqc_max_avg/readouts_per_round) 
                                        * readouts_per_round * repetitions))
        MC.set_detector_function(d)
        MC.run(f'Weight_{n}_parity_tomography_{ancilla_qubit}_'+\
               f'{"_".join(data_qubits)}_sim-msmt-{sim_measurement}'+\
               f'_rounds-{n_rounds}',disable_snapshot_metadata=disable_metadata)
        # Warning analysis requires that the detector function is ordered
        # as: [anc_qubit, data_qubit[0],[1],[2],[3]]
        ma2.pba.Weight_n_parity_tomography(
            sim_measurement=sim_measurement,
            n_rounds=n_rounds,
            exception_qubits=exception_qubits,
            post_selection=initialization_msmt)

    ################################################
    # Surface-17 specific functions
    ################################################
    def measure_defect_rate(
            self,
            ancilla_qubit: str,
            data_qubits: list,
            experiments: list,
            lru_qubits: list = None,
            Rounds: list = [1, 2, 4, 6, 10, 15, 25, 50],
            repetitions: int = 20,
            prepare_for_timedomain: bool = True,
            prepare_readout: bool = True,
            heralded_init: bool = True,
            stabilizer_type: str = 'X',
            initial_state_qubits: list = None,
            measurement_time_ns: int = 500,
            analyze: bool = True,
            Pij_matrix: bool = True,
            ):
        # assert self.ro_acq_weight_type() == 'optimal IQ'
        assert self.ro_acq_digitized() == False
        Valid_experiments = ['single_stabilizer', 'single_stabilizer_LRU',
                             'surface_13', 'surface_13_LRU', 'surface_17',
                             'repetition_code']
        for exp in experiments:
            assert exp in Valid_experiments, f'Experiment {exp} not a valid experiment'
        number_of_kernels = len(experiments)
        # Surface-17 qubits
        X_ancillas = ['X1', 'X2', 'X3', 'X4']
        Z_ancillas = ['Z1', 'Z2', 'Z3', 'Z4']
        Data_qubits = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']
        X_anci_idxs = [ self.find_instrument(q).cfg_qubit_nr() for q in X_ancillas ]
        Z_anci_idxs = [ self.find_instrument(q).cfg_qubit_nr() for q in Z_ancillas ]
        Data_idxs = [ self.find_instrument(q).cfg_qubit_nr() for q in Data_qubits ]
        ancilla_qubit_idx = self.find_instrument(ancilla_qubit).cfg_qubit_nr()
        data_qubits_idx = [self.find_instrument(q).cfg_qubit_nr() for q in data_qubits] 
        if lru_qubits:
            lru_qubits_idxs = [self.find_instrument(q).cfg_qubit_nr() for q in lru_qubits]
        else:
            lru_qubits_idxs = []
        ######################################################
        # Prepare for timedomain
        ######################################################
        # prepare mw lutmans
        # for q in [ancilla_qubit]+data_qubits:
        for q in Data_qubits + X_ancillas + Z_ancillas:
            mw_lm = self.find_instrument(f'MW_lutman_{q}')
            mw_lm.set_default_lutmap()
            mw_lm.load_waveforms_onto_AWG_lookuptable()

        if prepare_for_timedomain:
            # Redundancy just to be sure we are uploading every parameter
            # for q_name in data_qubits+[ancilla_qubit]:
            for q_name in Data_qubits+X_ancillas+Z_ancillas:
                q = self.find_instrument(q_name)
                q.prepare_for_timedomain()
            self.prepare_for_timedomain(qubits=Data_qubits+X_ancillas+Z_ancillas,
                                        prepare_for_readout=False)
        if (prepare_for_timedomain or prepare_readout):
            ##################################################
            # Prepare acquisition with custom channel map
            ##################################################
            # Need to create ordered list of experiment qubits
            # and remaining ancilla qubits
            ordered_qubit_dict = {}
            # _qubits = [ancilla_qubit]+data_qubits
            _qubits = [ancilla_qubit]+Data_qubits
            # Add qubits in experiment
            for _q in _qubits:
                acq_instr = self.find_instrument(_q).instr_acquisition()
                if acq_instr not in ordered_qubit_dict.keys():\
                    ordered_qubit_dict[acq_instr] = [_q]
                else:
                    ordered_qubit_dict[acq_instr].append(_q)
            # Add remaining ancilla qubits
            _remaining_ancillas = X_ancillas + Z_ancillas
            _remaining_ancillas.remove(ancilla_qubit)
            _remaining_ancillas.remove('X4')
            for _q in _remaining_ancillas:
                acq_instr = self.find_instrument(_q).instr_acquisition()
                if acq_instr not in ordered_qubit_dict.keys():\
                    ordered_qubit_dict[acq_instr] = [_q]
                else:
                    ordered_qubit_dict[acq_instr].append(_q)
            ordered_qubit_list = [ x for v in ordered_qubit_dict.values() for x in v ]
            # ordered_chan_map = {q:'optimal IQ' if q in _qubits else 'optimal'\
            #                     for q in ordered_qubit_list}
            ordered_chan_map = {q:'optimal IQ' if q in _qubits+_remaining_ancillas else 'optimal'\
                                for q in ordered_qubit_list}
            print(ordered_qubit_list)
            print(ordered_chan_map)
            ## expect IQ mode for D8 & D9 [because we have 6 qubits in this feedline]
            # if 'D8' in ordered_chan_map.keys() and 'D9' in ordered_chan_map.keys():
            #     ordered_chan_map['D8'] = 'optimal'
            #     ordered_chan_map['D9'] = 'optimal'
            self.ro_acq_weight_type('custom')
            self.prepare_readout(qubits=ordered_qubit_list,
                qubit_int_weight_type_dict=ordered_chan_map)
            ##################################################
            # Prepare readout pulses with custom channel map
            ##################################################
            RO_lutman_1 = self.find_instrument('RO_lutman_1')
            RO_lutman_2 = self.find_instrument('RO_lutman_2')
            RO_lutman_3 = self.find_instrument('RO_lutman_3')
            RO_lutman_4 = self.find_instrument('RO_lutman_4')
            if [11] not in RO_lutman_1.resonator_combinations():
                RO_lutman_1.resonator_combinations([[11], 
                    RO_lutman_1.resonator_combinations()[0]])
            RO_lutman_1.load_waveforms_onto_AWG_lookuptable()

            if [3, 7] not in RO_lutman_2.resonator_combinations():
                RO_lutman_2.resonator_combinations([[3, 7],
                    RO_lutman_2.resonator_combinations()[0]])
            RO_lutman_2.load_waveforms_onto_AWG_lookuptable()

            if [8, 12] not in RO_lutman_4.resonator_combinations():
                RO_lutman_4.resonator_combinations([[8, 12],
                    RO_lutman_4.resonator_combinations()[0]])
            RO_lutman_4.load_waveforms_onto_AWG_lookuptable()

            # if [9, 14, 10] not in RO_lutman_3.resonator_combinations():
            #     RO_lutman_3.resonator_combinations([[9, 14, 10], 
            #         RO_lutman_3.resonator_combinations()[0]])
            # RO_lutman_3.load_waveforms_onto_AWG_lookuptable()
            if [14, 10] not in RO_lutman_3.resonator_combinations():
                RO_lutman_3.resonator_combinations([[14, 10], 
                    RO_lutman_3.resonator_combinations()[0]])
            RO_lutman_3.load_waveforms_onto_AWG_lookuptable()
        # Generate compiler sequence
        p = mqo.repeated_stabilizer_data_measurement_sequence(
                target_stab = ancilla_qubit,
                Q_anc = ancilla_qubit_idx,
                Q_D = data_qubits_idx,
                X_anci_idxs = X_anci_idxs,
                Z_anci_idxs = Z_anci_idxs,
                data_idxs = Data_idxs,
                lru_idxs = lru_qubits_idxs,
                platf_cfg = self.cfg_openql_platform_fn(),
                experiments = experiments,
                Rounds = Rounds,
                stabilizer_type=stabilizer_type,
                initial_state_qubits=initial_state_qubits,
                measurement_time_ns=measurement_time_ns)
        # Set up nr_shots on detector
        d = self.int_log_det
        uhfqc_max_avg = 2**19
        for det in d.detectors:
            readouts_per_round = np.sum(np.array(Rounds)+heralded_init)*number_of_kernels\
                                 + 3*(1+heralded_init)
            det.nr_shots = int(uhfqc_max_avg/readouts_per_round)*readouts_per_round

        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        MC = self.instr_MC.get_instr()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(int(uhfqc_max_avg/readouts_per_round) 
                                        * readouts_per_round * repetitions))
        MC.set_detector_function(d)
        if initial_state_qubits:
            _title = f'Surface_13_experiment_{"_".join([str(r) for r in Rounds])}rounds'+\
                     f'_excited_qubits_{"_".join(initial_state_qubits)}'
        else:
            _title = f'Repeated_stab_meas_{"_".join([str(r) for r in Rounds])}rounds'+\
                     f'_{ancilla_qubit}_{data_qubits}_data_qubit_measurement'
        if len(_title) > 96:
            _title = _title[:96]    # this is to avoid failure in creating hdf5 file.
        try:
            MC.run(_title)
            a = None
            if analyze:
                a = ma2.pba.Repeated_stabilizer_measurements(
                    ancilla_qubit=ancilla_qubit,
                    data_qubits = data_qubits,
                    Rounds=Rounds,
                    heralded_init=heralded_init,
                    number_of_kernels=number_of_kernels,
                    experiments=experiments,
                    Pij_matrix=Pij_matrix,
                    label=_title)
            self.ro_acq_weight_type('optimal')
        except:
            print_exception()
            self.ro_acq_weight_type('optimal')
            raise ValueError('Somtehing happened!')
        return a

    def measure_repetition_code_defect_rate(
            self,
            involved_ancilla_ids: List[str],
            involved_data_ids: List[str],
            rounds: list = [1, 2, 4, 6, 10, 15, 25, 50],
            repetitions: int = 20,
            prepare_for_timedomain: bool = True,
            prepare_readout: bool = True,
            heralded_init: bool = True,
            stabilizer_type: str = 'X',
            measurement_time_ns: int = 500,
            analyze: bool = True,
            Pij_matrix: bool = True,
            disable_metadata: bool = False,
            initial_state: list = None,
            ):
        # assert self.ro_acq_weight_type() == 'optimal IQ'
        assert self.ro_acq_digitized() == False
        
        # Surface-17 qubits
        ancilla_x_names = ['X1', 'X2', 'X3', 'X4']
        ancilla_z_names = ['Z1', 'Z2', 'Z3', 'Z4']
        data_names = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']
        ancilla_x_indices: List[int] = [ self.find_instrument(q).cfg_qubit_nr() for q in ancilla_x_names ]
        ancilla_z_indices: List[int] = [ self.find_instrument(q).cfg_qubit_nr() for q in ancilla_z_names ]
        all_ancilla_indices: List[int] = ancilla_x_indices + ancilla_z_indices
        all_data_indices: List[int] = [ self.find_instrument(q).cfg_qubit_nr() for q in data_names ]
        
        involved_ancilla_indices = [self.find_instrument(q).cfg_qubit_nr() for q in involved_ancilla_ids] 
        involved_data_indices = [self.find_instrument(q).cfg_qubit_nr() for q in involved_data_ids] 
        lru_qubits_indices = []
        ######################################################
        # Prepare for timedomain
        ######################################################
        def internal_prepare_for_timedomain():
            """:return: Void."""
            # prepare mw lutmans
            # for q in [ancilla_qubit]+data_qubits:
            for q in data_names + ancilla_x_names + ancilla_z_names:
                mw_lm = self.find_instrument(f'MW_lutman_{q}')
                mw_lm.set_default_lutmap()
                mw_lm.load_waveforms_onto_AWG_lookuptable()

            if prepare_for_timedomain:
                # Redundancy just to be sure we are uploading every parameter
                # for q_name in data_qubits+[ancilla_qubit]:
                for q_name in data_names+ancilla_x_names+ancilla_z_names:
                    q = self.find_instrument(q_name)
                    q.prepare_for_timedomain()
                self.prepare_for_timedomain(qubits=data_names+ancilla_x_names+ancilla_z_names, prepare_for_readout=False)
            if (prepare_for_timedomain or prepare_readout):
                ##################################################
                # Prepare acquisition with custom channel map
                ##################################################
                # Need to create ordered list of experiment qubits
                # and remaining ancilla qubits
                ordered_qubit_dict = {}
                # _qubits = [ancilla_qubit]+data_qubits
                _qubits = involved_ancilla_ids + data_names
                # Add qubits in experiment
                for _q in _qubits:
                    acq_instr = self.find_instrument(_q).instr_acquisition()
                    if acq_instr not in ordered_qubit_dict.keys():\
                        ordered_qubit_dict[acq_instr] = [_q]
                    else:
                        ordered_qubit_dict[acq_instr].append(_q)
                # Add remaining ancilla qubits
                _remaining_ancillas = ancilla_x_names + ancilla_z_names
                for involved_ancilla_id in involved_ancilla_ids:
                    _remaining_ancillas.remove(involved_ancilla_id)
                    
                _remaining_ancillas.remove('X4')
                
                for _q in _remaining_ancillas:
                    acq_instr = self.find_instrument(_q).instr_acquisition()
                    if acq_instr not in ordered_qubit_dict.keys():\
                        ordered_qubit_dict[acq_instr] = [_q]
                    else:
                        ordered_qubit_dict[acq_instr].append(_q)
                ordered_qubit_list = [ x for v in ordered_qubit_dict.values() for x in v ]
                # ordered_chan_map = {q:'optimal IQ' if q in _qubits else 'optimal'\
                #                     for q in ordered_qubit_list}
                ordered_chan_map = {q:'optimal IQ' if q in _qubits+_remaining_ancillas else 'optimal'\
                                    for q in ordered_qubit_list}
                print(ordered_qubit_list)
                print(ordered_chan_map)
                ## expect IQ mode for D8 & D9 [because we have 6 qubits in this feedline]
                # if 'D8' in ordered_chan_map.keys() and 'D9' in ordered_chan_map.keys():
                #     ordered_chan_map['D8'] = 'optimal'
                #     ordered_chan_map['D9'] = 'optimal'
                self.ro_acq_weight_type('custom')
                self.prepare_readout(qubits=ordered_qubit_list,
                    qubit_int_weight_type_dict=ordered_chan_map)
                ##################################################
                # Prepare readout pulses with custom channel map
                ##################################################
                RO_lutman_1 = self.find_instrument('RO_lutman_1')
                RO_lutman_2 = self.find_instrument('RO_lutman_2')
                RO_lutman_3 = self.find_instrument('RO_lutman_3')
                RO_lutman_4 = self.find_instrument('RO_lutman_4')
                if [11] not in RO_lutman_1.resonator_combinations():
                    RO_lutman_1.resonator_combinations([[11], 
                        RO_lutman_1.resonator_combinations()[0]])
                RO_lutman_1.load_waveforms_onto_AWG_lookuptable()

                if [3, 7] not in RO_lutman_2.resonator_combinations():
                    RO_lutman_2.resonator_combinations([[3, 7],
                        RO_lutman_2.resonator_combinations()[0]])
                RO_lutman_2.load_waveforms_onto_AWG_lookuptable()

                if [8, 12] not in RO_lutman_4.resonator_combinations():
                    RO_lutman_4.resonator_combinations([[8, 12],
                        RO_lutman_4.resonator_combinations()[0]])
                RO_lutman_4.load_waveforms_onto_AWG_lookuptable()

                # if [9, 14, 10] not in RO_lutman_3.resonator_combinations():
                #     RO_lutman_3.resonator_combinations([[9, 14, 10], 
                #         RO_lutman_3.resonator_combinations()[0]])
                # RO_lutman_3.load_waveforms_onto_AWG_lookuptable()
                if [14, 10] not in RO_lutman_3.resonator_combinations():
                    RO_lutman_3.resonator_combinations([[14, 10], 
                        RO_lutman_3.resonator_combinations()[0]])
                RO_lutman_3.load_waveforms_onto_AWG_lookuptable()
        # TODO: This should be refactored in a more general approach that handles all the fun things related to (timedomain) readout.
        internal_prepare_for_timedomain()
        
        # Generate compiler sequence
        p = mqo.repetition_code_sequence_old(
            involved_ancilla_indices=involved_ancilla_indices,
            involved_data_indices=involved_data_indices,
            all_ancilla_indices=all_ancilla_indices,
            all_data_indices=all_data_indices,
            array_of_round_number=rounds,
            platf_cfg=self.cfg_openql_platform_fn(),
            stabilizer_type=stabilizer_type,
            measurement_time_ns=measurement_time_ns,
            initial_state=initial_state
        )
        # Set up nr_shots on detector
        d = self.int_log_det
        uhfqc_max_avg = 2**19  # 2**19
        number_of_kernels: int = 1  # Performing only a single experiment
        for det in d.detectors:
            nr_acquisitions = [1 if round == 0 else round for round in rounds]
            readouts_per_round = np.sum(np.array(nr_acquisitions)+heralded_init) * number_of_kernels + 3*(1+heralded_init)
            det.nr_shots = int(uhfqc_max_avg/readouts_per_round)*readouts_per_round

        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())
        MC = self.instr_MC.get_instr()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(int(uhfqc_max_avg/readouts_per_round) * readouts_per_round * repetitions))
        MC.set_detector_function(d)
        
        _title = f'Repeated_stab_meas_{rounds[0]}_to_{rounds[-1]}_rounds'+\
                    f'_{"_".join(involved_ancilla_ids)}_{"_".join(involved_data_ids)}_data_qubit_measurement'
        if len(_title) > 90:
            _title = _title[:90]
        try:
            MC.run(_title, disable_snapshot_metadata=disable_metadata)
            a = None
            if analyze:
                # a = ma2.pba.Repeated_stabilizer_measurements(
                #     ancilla_qubit=involved_ancilla_ids,
                #     data_qubits = involved_data_ids,
                #     Rounds=rounds,
                #     heralded_init=heralded_init,
                #     number_of_kernels=number_of_kernels,
                #     experiments=['repetition_code'],
                #     Pij_matrix=Pij_matrix,
                #     label=_title)
                
                fillvalue = None
                involved_qubit_names: List[str] = [
                    item
                    for pair in itt.zip_longest(involved_data_ids, involved_ancilla_ids, fillvalue=fillvalue) for item in pair if item != fillvalue
                ]
                a = RepeatedStabilizerAnalysis(
                    involved_qubit_names=involved_qubit_names,
                    qec_cycles=rounds,
                    initial_state=InitialStateContainer.from_ordered_list(
                        [InitialStateEnum.ZERO] * len(involved_data_ids)
                    ),  # TODO: Construct initial state from arguments
                    label=_title,
                )
                a.run_analysis()
            self.ro_acq_weight_type('optimal')
        except:
            print_exception()
            self.ro_acq_weight_type('optimal')
            raise ValueError('Somtehing happened!')
        return a
