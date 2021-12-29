"""
File:   HAL_ShimMQ.py : HAL shim Multi Qubit
Note:   extracted from HAL_Device.py (originally device_object_CCL.py)

see the notes in qubit_objects/HAL_ShimSQ.py

Note:   a lot code was moved around within this file in December 2021. As a consequence, the author information provided
        by 'git blame' makes little sense. See GIT tag 'release_v0.3' for the original file.
"""

import logging
import warnings
from collections import OrderedDict
import numpy as np
from deprecated import deprecated

from pycqed.measurement import detector_functions as det

# Imported for type annotations
from pycqed.instrument_drivers.physical_instruments.QuTech_AWG_Module import QuTech_AWG_Module
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.measurement.measurement_control import MeasurementControl
from pycqed.measurement.detector_functions import Detector_Function

from qcodes import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter


log = logging.getLogger(__name__)


def _acq_ch_map_to_IQ_ch_map(acq_ch_map):
    acq_ch_map_IQ = {}
    for acq_instr, ch_map in acq_ch_map.items():
        acq_ch_map_IQ[acq_instr] = {}
        for qubit, ch in ch_map.items():
            acq_ch_map_IQ[acq_instr]["{} I".format(qubit)] = ch
            acq_ch_map_IQ[acq_instr]["{} Q".format(qubit)] = ch + 1
    return acq_ch_map_IQ


class HAL_ShimMQ(Instrument):
    # Constants
    _NUM_INSTR_ACQ = 3    # S17 has 3 acquisition instruments (for 3 feedlines)
    _NUM_INSTR_AWG_MW = 5
    _NUM_INSTR_AWG_FLUX = 3

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self._add_parameters()

    ##########################################################################
    # public functions: prepare
    ##########################################################################

    def prepare_timing(self):
        """
        Responsible for ensuring timing is configured correctly.
        Takes parameters starting with `tim_` and uses them to set the correct
        latencies on the DIO ports of the CC.
        N.B. latencies are set in multiples of 20ns in the DIO.
        Latencies shorter than 20ns are set as channel delays in the AWGs.
        These are set globally. If individual (per channel) setting of latency
        is required in the future, we can add this.
        """

        # 2. Setting the latencies
        cc = self.instr_CC.get_instr()
        if cc.IDN()['model']=='CCL':  # FIXME: CCL is deprecated
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
        # Triple check everything if any changes are to be made

        # Subtract lowest value to ensure minimal latency is used.
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

        # Setting the latencies in the CC
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
                                # FIXME: the line below assumes AWG8_MW_LutMan, incompatible with AWG8_VSM_MW_LutMan (PR #658)
                                #  move delay setting to lutman
                                awg_chs     = MW_lm.channel_I(), MW_lm.channel_Q()
                                log.debug("Setting `sigouts_{}_delay` to {:4g}"
                                          " in {}".format(awg_chs[0], lat_fine, AWG.name))
                                AWG.set("sigouts_{}_delay".format(awg_chs[0]-1), lat_fine+extra_delay)
                                AWG.set("sigouts_{}_delay".format(awg_chs[1]-1), lat_fine+extra_delay)
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

    def prepare_for_timedomain(
            self,
            qubits: list,
            reduced: bool = False,
            bypass_flux: bool = False,
            prepare_for_readout: bool = True
    ):
        """
        Prepare setup for a time domain experiment:

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

    # FIXME: setup dependent
    def prepare_for_inspire(self):
        for lutman in ['mw_lutman_QNW','mw_lutman_QNE','mw_lutman_QC','mw_lutman_QSW','mw_lutman_QSE']:
            self.find_instrument(lutman).set_inspire_lutmap()
        self.prepare_for_timedomain(qubits=self.qubits())
        self.find_instrument(self.instr_MC()).soft_avg(1)
        return True

    ##########################################################################
    # public functions: get_*_detector
    ##########################################################################

    def get_correlation_detector(
            self,
            qubits: list,
            single_int_avg: bool = False,
            seg_per_point: int = 1,
            always_prepare: bool = False
    ) -> Detector_Function:
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


    def get_int_logging_detector(
            self,
            qubits=None,
            result_logging_mode='raw'
    ) -> Detector_Function:
        # FIXME: qubits passed to but not used in function

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
            # # update by Tim [2021-06-01]
            # channel_dict = {}
            # for q in qubits:

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


    def get_input_avg_det(self, **kw) -> Detector_Function:
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


    def get_int_avg_det(self, qubits=None, **kw) -> Detector_Function:
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

    ##########################################################################
    # private functions: add parameters
    ##########################################################################

    def _add_instr_parameters(self):
        self.add_parameter(
            "instr_MC",
            label="MeasurementControl",
            parameter_class=InstrumentRefParameter, )

        self.add_parameter(
            'instr_nested_MC',
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
            docstring="Device responsible for controlling the experiment.",
            parameter_class=InstrumentRefParameter,
        )

        for i in range(self._NUM_INSTR_ACQ):
            self.add_parameter(f"instr_acq_{i}", parameter_class=InstrumentRefParameter)

        for i in range(self._NUM_INSTR_AWG_MW):
            self.add_parameter(f"instr_AWG_mw_{i}", parameter_class=InstrumentRefParameter)

        for i in range(self._NUM_INSTR_AWG_FLUX):
            self.add_parameter(f"instr_AWG_flux_{i}", parameter_class=InstrumentRefParameter)

    def _add_tim_parameters(self):
        # Timing related parameters
        for i in range(self._NUM_INSTR_ACQ):
            self.add_parameter(
                f"tim_ro_latency_{i}",
                unit="s",
                label=f"Readout latency {i}",
                parameter_class=ManualParameter,
                initial_value=0,
                vals=vals.Numbers(),
            )

        for i in range(self._NUM_INSTR_AWG_FLUX):
            self.add_parameter(
                f"tim_flux_latency_{i}",
                unit="s",
                label="Flux latency 0",
                parameter_class=ManualParameter,
                initial_value=0,
                vals=vals.Numbers(),
            )

        for i in range(self._NUM_INSTR_AWG_MW):
            self.add_parameter(
                f"tim_mw_latency_{i}",
                unit="s",
                label=f"Microwave latency {i}",
                parameter_class=ManualParameter,
                initial_value=0,
                vals=vals.Numbers(),
            )

    def _add_ro_parameters(self):
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
                "The time between the instruction that triggers the"
                " readout pulse and the instruction that triggers the "
                "acquisition. The positive number means that the "
                "acquisition is started after the pulse is send."
            ),
        )

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
            "ro_always_all",
            docstring="If true, configures the UHFQC to RO all qubits "
                      "independent of codeword received.",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
        )

    def _add_parameters(self):
        self._add_instr_parameters()
        self._add_tim_parameters()
        self._add_ro_parameters()

        self.add_parameter(
            "cfg_openql_platform_fn",
            label="OpenQL platform configuration filename",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
        )

        self.add_parameter(
            'qubits',
            parameter_class=ManualParameter,
            initial_value=[],
            vals=vals.Lists(elt_validator=vals.Strings())
        )

        self.add_parameter(
            'qubit_edges',
            parameter_class=ManualParameter,
            docstring="Denotes edges that connect qubits. Used to define the device topology.",
            initial_value=[[]],
            vals=vals.Lists(elt_validator=vals.Lists(elt_validator=vals.Strings()))
        )

        self.add_parameter(
            'qubits_by_feedline',
            parameter_class=ManualParameter,
            docstring="Qubits divided by feedline. Used to sort qubits for timedomain preparation.",
            initial_value=[[]],
            vals=vals.Lists(elt_validator=vals.Lists(elt_validator=vals.Strings()))
        )

        self.add_parameter(
            "dio_map",
            docstring="The map between DIO channel number and functionality (ro_x, mw_x, flux_x). "
            "Tip: run `device.dio_map?` to print the docstring of this parameter",
            initial_value=None,
            set_cmd=self._set_dio_map,
            vals=vals.Dict(),
        )

    ##########################################################################
    # private functions: parameter helpers
    ##########################################################################

    def _set_dio_map(self, dio_map_dict):
        allowed_keys = {"ro_", "mw_", "flux_"}
        for key in dio_map_dict:
            assert np.any(
                [a_key in key and len(key) > len(a_key) for a_key in allowed_keys]
            ), "Key `{}` must start with:" " `{}`!".format(key, list(allowed_keys))
        return dio_map_dict

    ##########################################################################
    # private functions: prepare
    ##########################################################################

    # FIXME: unused
    # def _grab_instruments_from_qb(self):
    #     """
    #     initialize instruments that should only exist once from the first
    #     qubit. Maybe must be done in a more elegant way (at least check
    #     uniqueness).
    #     """
    #
    #     qb = self.find_instrument(self.qubits()[0])
    #     self.instr_MC(qb.instr_MC())
    #     self.instr_VSM(qb.instr_VSM())
    #     self.instr_CC(qb.instr_CC())
    #     self.cfg_openql_platform_fn(qb.cfg_openql_platform_fn())

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
            # FIXME: implementation differs from HAL_Transmon::_prep_ro_sources
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
        for acq_instr_name in acq_ch_map.keys():  # FIXME: acq_instr_name not used, but acq_instr is
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
                    acq_instr.set("qas_0_integration_weights_{}_real".format(qb.ro_acq_weight_chI()), opt_WI,)
                    acq_instr.set("qas_0_integration_weights_{}_imag".format(qb.ro_acq_weight_chI()), opt_WQ,)
                    acq_instr.set("qas_0_rotations_{}".format(
                                qb.ro_acq_weight_chI()), 1.0 - 1.0j)
                    if self.ro_acq_weight_type() == 'optimal IQ':
                        print('setting the optimal Q')
                        acq_instr.set('qas_0_integration_weights_{}_real'.format(qb.ro_acq_weight_chQ()), opt_WQ)
                        acq_instr.set('qas_0_integration_weights_{}_imag'.format(qb.ro_acq_weight_chQ()), opt_WI)
                        acq_instr.set('qas_0_rotations_{}'.format(qb.ro_acq_weight_chQ()), 1.0 + 1.0j)

                if self.ro_acq_digitized():
                    # Update the RO theshold
                    if (qb.ro_acq_rotated_SSB_when_optimal() and
                            abs(qb.ro_acq_threshold()) > 32):
                        threshold = 32
                        log.warning("Clipping ro_acq threshold of {} to 32".format(qb.name))
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
            # resonator_combs = [[r] for r in resonators_in_lm[ro_lm.name]] + \
            #     [resonators_in_lm[ro_lm.name]]
            resonator_combs = [resonators_in_lm[ro_lm.name]]
            log.info('Setting resonator combinations for {} to {}'.format(
                ro_lm.name, resonator_combs))

            # FIXME: temporary fix so device object doesnt mess with
            #       the resonator combinations. Better strategy should be implemented
            ro_lm.resonator_combinations(resonator_combs)
            ro_lm.load_DIO_triggered_sequence_onto_UHFQC()

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

    ##########################################################################
    # private functions
    ##########################################################################

