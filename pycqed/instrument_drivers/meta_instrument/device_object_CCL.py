import numpy as np
import logging

from collections import OrderedDict
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from pycqed.analysis import multiplexed_RO_analysis as mra
from pycqed.measurement import detector_functions as det
from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import measurement_analysis as ma2
import networkx as nx
import datetime


try:
    from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
    import pycqed.measurement.openql_experiments.multi_qubit_oql as mqo
except ImportError:
    logging.warning('Could not import OpenQL')
    mqo = None
    sqo = None

from pycqed.analysis import tomography as tomo

from collections import defaultdict


class DeviceCCL(Instrument):

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

        self.add_parameter('ro_acq_integration_length', initial_value=500e-9,
                           vals=vals.Numbers(min_value=0, max_value=20e6),
                           parameter_class=ManualParameter)

        self.add_parameter('ro_pow_LO', label='RO power LO',
                           unit='dBm', initial_value=20,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_acq_averages', initial_value=1024,
                           vals=vals.Numbers(min_value=0, max_value=1e6),
                           parameter_class=ManualParameter)

        self.add_parameter(
            'ro_acq_delay',  unit='s',
            label='Readout acquisition delay',
            vals=vals.Numbers(min_value=0),
            initial_value=0,
            parameter_class=ManualParameter,
            docstring=('The time between the instruction that trigger the'
                       ' readout pulse and the instruction that triggers the '
                       'acquisition. The positive number means that the '
                       'acquisition is started after the pulse is send.'))

        self.add_parameter('instr_MC', label='MeasurementControl',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_VSM', label='Vector Switch Matrix',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_CC', label='Central Controller',
            docstring=('Device responsible for controlling the experiment'
                       ' using eQASM generated using OpenQL, in the near'
                       ' future will be the CC_Light.'),
            parameter_class=InstrumentRefParameter)

        ro_acq_docstr = (
            'Determines what type of integration weights to use: '
            '\n\t SSB: Single sideband demodulation\n\t'
            'DSB: Double sideband demodulation\n\t'
            'optimal: waveforms specified in "RO_acq_weight_func_I" '
            '\n\tand "RO_acq_weight_func_Q"')

        self.add_parameter('ro_acq_weight_type',
                           initial_value='SSB',
                           vals=vals.Enum('SSB', 'DSB', 'optimal'),
                           docstring=ro_acq_docstr,
                           parameter_class=ManualParameter)

        self.add_parameter('ro_acq_digitized', vals=vals.Bool(),
                           initial_value=False,
                           parameter_class=ManualParameter)

        self.add_parameter('cfg_openql_platform_fn',
                           label='OpenQL platform configuration filename',
                           parameter_class=ManualParameter,
                           vals=vals.Strings())

        self.add_parameter('ro_always_all',
                           label='If true, configures the UHFQC to RO all qubits '
                           'independent of codeword received.',
                           parameter_class=ManualParameter,
                           vals=vals.Bool())

        # Timing related parameters
        self.add_parameter('tim_ro_latency_0',
                           unit='s',
                           label='readout latency DIO 1',
                           parameter_class=ManualParameter,
                           initial_value=0,
                           vals=vals.Numbers())
        self.add_parameter('tim_ro_latency_1',
                           unit='s',
                           label='readout latency DIO 2',
                           parameter_class=ManualParameter,
                           initial_value=0,
                           vals=vals.Numbers())
        self.add_parameter('tim_flux_latency',
                           unit='s',
                           label='flux latency DIO 3',
                           parameter_class=ManualParameter,
                           initial_value=0,
                           vals=vals.Numbers())
        self.add_parameter('tim_mw_latency_0',
                           unit='s',
                           label='microwave latency DIO 4',
                           parameter_class=ManualParameter,
                           initial_value=0,
                           vals=vals.Numbers())
        self.add_parameter('tim_mw_latency_1',
                           unit='s',
                           label='microwave latency DIO 5',
                           parameter_class=ManualParameter,
                           initial_value=0,
                           vals=vals.Numbers())

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
        latencies on the DIO ports of the CCL.

        N.B. As latencies here are controlled through the DIO delays it can
        only be controlled in multiples of 20 ns.
        """
        # 2. Setting the latencies
        latencies = OrderedDict([('ro_latency_0', self.tim_ro_latency_0()),
                                 ('ro_latency_1', self.tim_ro_latency_1()),
                                 ('flux_latency_0', self.tim_flux_latency()),
                                 ('mw_latency_0', self.tim_mw_latency_0()),
                                 ('mw_latency_1', self.tim_mw_latency_1())])

        # Substract lowest value to ensure minimal latency is used.
        # note that this also supports negative delays (which is useful for
        # calibrating)
        lowest_value = min(latencies.values())
        for key, val in latencies.items():
            latencies[key] = val - lowest_value

        # Setting the latencies in the CCL
        CCL = self.instr_CC.get_instr()
        for i, val in enumerate(latencies.values()):
            CCL.set('dio{}_out_delay'.format(i+1), val //
                    20e-9)  # Convert to CCL dio value

    def prepare_readout(self):
        self._prep_ro_setup_qubits()
        self._prep_ro_sources()
        # commented out because it conflicts with setting in the qubit object
        # self._prep_ro_pulses()
        self._prep_ro_integration_weights()
        self._prep_ro_instantiate_detectors()

    def prepare_fluxing(self):
        # prepares by loading the awg_hack_program
        q0 = self.qubits()[0]
        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()
        # fl_lutman.load_waveforms_onto_awg_lookuptable()
        fl_lutman.load_waveforms_onto_AWG_lookuptable()
        awg = fl_lutman.AWG.get_instr()
        if awg.__class__.__name__ == 'QuTech_AWG_Module':
            using_QWG = True
        else:
            using_QWG = False
        if not using_QWG:
            # awg.upload_codeword_program(awgs=[0])
            awg_hack_program_cz = """
          while (1) {
            waitDIOTrigger();
            playWave("dev8005_wave_ch1_cw001", "dev8005_wave_ch2_cw001");
          }
          """
            awg.configure_awg_from_string(0, awg_hack_program_cz)
            awg.configure_codeword_protocol()

        awg.start()

    def _prep_ro_setup_qubits(self):
        """
        set the parameters of the individual qubits to be compatible
        with multiplexed readout.
        """

        if self.ro_acq_weight_type() == 'optimal':
            nr_of_acquisition_channels_per_qubit = 1
        else:
            nr_of_acquisition_channels_per_qubit = 2

        used_acq_channels = defaultdict(int)

        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)

            # all qubits use the same acquisition type
            qb.ro_acq_weight_type(self.ro_acq_weight_type())
            qb.ro_acq_integration_length(self.ro_acq_integration_length())
            qb.ro_acq_digitized(self.ro_acq_digitized())
            # hardcoded because UHFLI is not stable for arbitrary values
            qb.ro_acq_input_average_length(4096/1.8e9)

            acq_device = qb.instr_acquisition()

            # allocate different acquisition channels
            # use next available channel as I
            index = used_acq_channels[acq_device]
            used_acq_channels[acq_device] += 1
            qb.ro_acq_weight_chI(index)

            # use next available channel as Q if needed
            if nr_of_acquisition_channels_per_qubit > 1:
                index = used_acq_channels[acq_device]
                used_acq_channels[acq_device] += 1
                qb.ro_acq_weight_chQ(index)

            # set RO modulation to use common LO frequency
            qb.ro_freq_mod(qb.ro_freq() - self.ro_lo_freq())
            qb._prep_ro_pulse(upload=False)
        qb._prep_ro_pulse(upload=True)

    def _prep_ro_integration_weights(self):
        """
        Set the acquisition integration weights on each channel
        """
        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)
            qb._prep_ro_integration_weights()

            if self.ro_acq_digitized():
                # Update the RO theshold
                acq_ch = qb.ro_acq_weight_chI()

                # The threshold that is set in the hardware  needs to be
                # corrected for the offset as this is only applied in
                # software.
                threshold = qb.ro_acq_threshold()
                offs = qb.instr_acquisition.get_instr().get(
                    'quex_trans_offset_weightfunction_{}'.format(acq_ch))
                hw_threshold = threshold + offs
                qb.instr_acquisition.get_instr().set(
                    'quex_thres_{}_level'.format(acq_ch), hw_threshold)

    def get_correlation_detector(self, single_int_avg: bool =False,
                                 seg_per_point: int=1):
        qnames = self.qubits()
        q0 = self.find_instrument(qnames[0])
        q1 = self.find_instrument(qnames[1])

        w0 = q0.ro_acq_weight_chI()
        w1 = q1.ro_acq_weight_chI()

        d = det.UHFQC_correlation_detector(
            UHFQC=q0.instr_acquisition.get_instr(),  # <- hack line
            thresholding=self.ro_acq_digitized(),
            AWG=self.instr_CC.get_instr(),
            channels=[w0, w1], correlations=[(w0, w1)],
            nr_averages=self.ro_acq_averages(),
            integration_length=q0.ro_acq_integration_length(),
            single_int_avg=single_int_avg,
            seg_per_point=seg_per_point)
        d.value_names = [qnames[0], qnames[1],
                         'Corr ({}, {})'.format(qnames[0], qnames[1])]
        return d

    def get_int_logging_detector(self, qubits: list=None,
                                 result_logging_mode='lin_trans'):
        acq_instrs, ro_ch_idx, value_names = \
            self._get_ro_channels_and_labels(qubits)

        UHFQC = self.find_instrument(acq_instrs[0])
        int_log_det = det.UHFQC_integration_logging_det(
            UHFQC=UHFQC, AWG=self.instr_CC.get_instr(),
            channels=ro_ch_idx,
            result_logging_mode=result_logging_mode,
            integration_length=self.ro_acq_integration_length())

        int_log_det.value_names = value_names

        return int_log_det

    def _get_ro_channels_and_labels(self, qubits: list=None):
        """
        Returns
            acq_instruments     : list of acquisition instruments
            ro_ch_idx           : channel indices for acquisition
            value_names         : convenient labels
        """
        if qubits is None:
            qubits = self.qubits()

        channels_list = []  # tuples (instrumentname, channel, description)

        for qb_name in reversed(qubits):
            # ensures that the LSQ (last one) get's assigned the lowest ch_idx
            qb = self.find_instrument(qb_name)
            acq_instr_name = qb.instr_acquisition()

            # one channel per qb
            if self.ro_acq_weight_type() == 'optimal':
                ch_idx = qb.ro_acq_weight_chI()
                channels_list.append((acq_instr_name, ch_idx,
                                      'w{} {}'.format(ch_idx, qb_name)))
            else:
                ch_idx = qb.ro_acq_weight_chI()
                channels_list.append((acq_instr_name, ch_idx,
                                      'w{} {} I'.format(ch_idx, qb_name)))
                ch_idx = qb.ro_acq_weight_chQ()
                channels_list.append((acq_instr_name, ch_idx,
                                      'w{} {} Q'.format(ch_idx, qb_name)))

        # for now, implement only working with one UHFLI
        acq_instruments = list(set([inst for inst, _, _ in channels_list]))
        if len(acq_instruments) != 1:
            raise NotImplementedError("Only one acquisition"
                                      "instrument supported so far")

        ro_ch_idx = [ch for _, ch, _ in channels_list]
        value_names = [n for _, _, n in channels_list]

        return acq_instruments, ro_ch_idx, value_names

    def _prep_ro_instantiate_detectors(self):
        """
        collect which channels are being used for which qubit and make
        detectors.
        """
        acq_instruments, ro_ch_idx, value_names = \
            self._get_ro_channels_and_labels(self.qubits())

        if self.ro_acq_weight_type() == 'optimal':
            # todo: digitized mode
            result_logging_mode = 'lin_trans'
            if self.ro_acq_digitized():
                result_logging_mode = 'digitized'
        else:
            result_logging_mode = 'raw'

        if 'UHFQC' in acq_instruments[0]:
            UHFQC = self.find_instrument(acq_instruments[0])

            self.input_average_detector = det.UHFQC_input_average_detector(
                UHFQC=UHFQC,
                AWG=self.instr_CC.get_instr(),
                nr_averages=self.ro_acq_averages(),
                nr_samples=int(self.ro_acq_integration_length()*1.8e9))

            self.int_avg_det = self.get_int_avg_det()
            self.int_avg_det.value_names = value_names

            self.int_avg_det_single = det.UHFQC_integrated_average_detector(
                UHFQC=UHFQC, AWG=self.instr_CC.get_instr(),
                channels=ro_ch_idx,
                result_logging_mode=result_logging_mode,
                nr_averages=self.ro_acq_averages(),
                real_imag=True, single_int_avg=True,
                integration_length=self.ro_acq_integration_length())

            self.int_avg_det_single.value_names = value_names

    def get_int_avg_det(self, **kw):
        """
        Instantiates an integration average detector using parameters from
        the qubit object. **kw get passed on to the class when instantiating
        the detector function.
        """
        if self.ro_acq_weight_type() == 'optimal':
            if self.ro_acq_digitized():
                result_logging_mode = 'digitized'
            else:
                result_logging_mode = 'lin_trans'
        else:
            result_logging_mode = 'raw'

        acq_instruments, ro_ch_idx, value_names = \
            self._get_ro_channels_and_labels()

        int_avg_det = det.UHFQC_integrated_average_detector(
            channels=ro_ch_idx,
            UHFQC=self.find_instrument(acq_instruments[0]),
            AWG=self.instr_CC.get_instr(),
            result_logging_mode=result_logging_mode,
            nr_averages=self.ro_acq_averages(),
            integration_length=self.ro_acq_integration_length(), **kw)

        return int_avg_det

    def _prep_ro_sources(self):
        """
        turn on and configure the RO LO's of all qubits to be measured.
        """

        ro_qb_list = self.qubits()

        for qb_name in ro_qb_list:
            LO = self.find_instrument(qb_name).instr_LO_ro.get_instr()
            LO.frequency.set(self.ro_lo_freq())
            LO.power(self.ro_pow_LO())
            LO.on()

    def _prep_ro_pulses(self):
        """
        configure lutmans to measure the qubits and
        let the lutman configure the readout AWGs.
        """
        # these are the qubits that should be possible to read out
        ro_qb_list = self.qubits()
        if ro_qb_list == []:
            ro_qb_list = self.qubits()

        lutmans_to_configure = {}

        # calculate the combinations that each ro_lutman should be able to do
        combs = defaultdict(lambda: [[]])

        for qb_name in ro_qb_list:
            qb = self.find_instrument(qb_name)

            ro_lm = qb.instr_LutMan_RO.get_instr()
            lutmans_to_configure[ro_lm.name] = ro_lm
            res_nr = qb.ro_pulse_res_nr()

            # extend the list of combinations to be set for the lutman

            combs[ro_lm.name] = (combs[ro_lm.name] +
                                 [c + [res_nr]
                                  for c in combs[ro_lm.name]])

            # These parameters affect all resonators.
            # Should not be part of individual qubits

            ro_lm.set('pulse_type', 'M_' + qb.ro_pulse_type())
            ro_lm.set('mixer_alpha',
                      qb.ro_pulse_mixer_alpha())
            ro_lm.set('mixer_phi',
                      qb.ro_pulse_mixer_phi())
            ro_lm.set('mixer_offs_I', qb.ro_pulse_mixer_offs_I())
            ro_lm.set('mixer_offs_Q', qb.ro_pulse_mixer_offs_Q())
            ro_lm.acquisition_delay(qb.ro_acq_delay())

            # configure the lutman settings for the pulse on the resonator
            # of this qubit

            ro_lm.set('M_modulation_R{}'.format(res_nr), qb.ro_freq_mod())
            ro_lm.set('M_length_R{}'.format(res_nr),
                      qb.ro_pulse_length())
            ro_lm.set('M_amp_R{}'.format(res_nr),
                      qb.ro_pulse_amp())
            ro_lm.set('M_phi_R{}'.format(res_nr),
                      qb.ro_pulse_phi())
            ro_lm.set('M_down_length0_R{}'.format(res_nr),
                      qb.ro_pulse_down_length0())
            ro_lm.set('M_down_amp0_R{}'.format(res_nr),
                      qb.ro_pulse_down_amp0())
            ro_lm.set('M_down_phi0_R{}'.format(res_nr),
                      qb.ro_pulse_down_phi0())
            ro_lm.set('M_down_length1_R{}'.format(res_nr),
                      qb.ro_pulse_down_length1())
            ro_lm.set('M_down_amp1_R{}'.format(res_nr),
                      qb.ro_pulse_down_amp1())
            ro_lm.set('M_down_phi1_R{}'.format(res_nr),
                      qb.ro_pulse_down_phi1())

        for ro_lm_name, ro_lm in lutmans_to_configure.items():
            if self.ro_always_all():
                all_qubit_idx_list = combs[ro_lm_name][-1]
                no_of_pulses = len(combs[ro_lm_name])
                combs[ro_lm_name] = [all_qubit_idx_list]*no_of_pulses
                ro_lm.hardcode_cases(
                    list(range(1, no_of_pulses)))

            ro_lm.resonator_combinations(combs[ro_lm_name][1:])
            ro_lm.load_DIO_triggered_sequence_onto_UHFQC()
            ro_lm.set_mixer_offsets()

    def _prep_td_configure_VSM(self):
        """
        turn off all VSM channels and then use qubit settings to
        turn on the required channels again.
        """

        # turn all channels on all VSMnS off
        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)
            VSM = qb.instr_VSM.get_instr()
            # VSM.set_all_switches_to('OFF')

        # turn the desired channels on
        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)

            # Configure VSM
            # N.B. This configure VSM block is geared specifically to the
            # Duplexer/BlueBox VSM
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

    def prepare_for_timedomain(self):
        self.prepare_readout()
        self.prepare_fluxing()
        self.prepare_timing()

        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)
            qb._prep_td_sources()
            qb._prep_mw_pulses()

        # self._prep_td_configure_VSM()

    ########################################################
    # Measurement methods
    ########################################################

    def measure_conditional_oscillation(self, q0: str, q1: str,
                                        prepare_for_timedomain=True, MC=None,
                                        wait_time_ns: int=0,
                                        label='',
                                        flux_codeword='fl_cw_01',
                                        verbose=True, disable_metadata=False):
        """
        Measures the "conventional cost function" for the CZ gate that
        is a conditional oscillation.
        """
        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        if MC is None:
            MC = self.instr_MC.get_instr()
        assert q0 in self.qubits()
        assert q1 in self.qubits()
        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()

        # These are hardcoded angles in the mw_lutman for the AWG8
        angles = np.arange(0, 341, 20)
        p = mqo.conditional_oscillation_seq(q0idx, q1idx,
                                            platf_cfg=self.cfg_openql_platform_fn(),
                                            angles=angles, wait_time=wait_time_ns,
                                            flux_codeword=flux_codeword,
                                            CZ_disabled=False)
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='Phase', unit='deg')
        MC.set_sweep_function(s)
        MC.set_sweep_points(p.sweep_points)
        MC.set_detector_function(self.get_correlation_detector())
        MC.run('conditional_oscillation{}{}'.format(self.msmt_suffix, label),
               disable_snapshot_metadata=disable_metadata)

        a = ma2.Conditional_Oscillation_Analysis(
            options_dict={'ch_idx_osc': self.qubits().index(q0),
                          'ch_idx_spec': self.qubits().index(q1)})

        if verbose:
            info_msg = (print(a.plot_dicts['phase_message']['text_string']))
            print(info_msg)

        return a

    def measure_two_qubit_tomo_bell(self, q0: str, q1: str,
                                    bell_state=0,
                                    analyze=True, close_fig=True,
                                    prepare_for_timedomain=True, MC=None):

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        assert q1 in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()

        p = mqo.two_qubit_tomo_bell(bell_state, q0idx, q1idx,
                                    platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        d = self.get_correlation_detector()
        MC.set_sweep_function(s)
        # 36 tomo rotations + 7*4 calibration points
        MC.set_sweep_points(np.arange(36+7*4))
        MC.set_detector_function(d)
        MC.run('TwoQubitBellTomo_{}_{}{}'.format(q0, q1, self.msmt_suffix))
        if analyze:
            a = tomo.Tomo_Multiplexed(
                label='Tomo',
                MLE=True, target_bell=bell_state, single_shots=False,
                q0_label=q0, q1_label=q1)
            return a

    def measure_two_qubit_allxy(self, q0: str, q1: str,
                                sequence_type='sequential',
                                replace_q1_pulses_X180: bool=False,
                                analyze: bool=True, close_fig: bool=True,
                                prepare_for_timedomain: bool=True, MC=None):

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        assert q1 in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()

        p = mqo.two_qubit_AllXY(q0idx, q1idx,
                                platf_cfg=self.cfg_openql_platform_fn(),
                                sequence_type=sequence_type,
                                replace_q1_pulses_X180=replace_q1_pulses_X180,
                                double_points=True)
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())

        d = self.get_correlation_detector()
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(42))
        MC.set_detector_function(d)
        MC.run('TwoQubitAllXY_{}_{}{}'.format(q0, q1, self.msmt_suffix))
        if analyze:
            a = ma.MeasurementAnalysis(close_main_fig=close_fig)
        return a

    def measure_two_qubit_SSRO(self,
                               qubits: list,
                               detector=None,
                               nr_shots: int=4088*4,
                               prepare_for_timedomain: bool =True,
                               result_logging_mode='lin_trans',
                               initialize: bool=False,
                               analyze=True,
                               MC=None):
        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        # count from back because q0 is the least significant qubit
        q0 = qubits[-1]
        q1 = qubits[-2]

        assert q0 in self.qubits()
        assert q1 in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()

        # p = mqo.two_qubit_off_on(q0idx, q1idx,
        #                          platf_cfg=self.cfg_openql_platform_fn())

        p = mqo.multi_qubit_off_on([q1idx, q0idx],
                                   initialize=initialize,
                                   second_excited_state=False,
                                   platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        if detector is None:
            # right is LSQ
            d = self.get_int_logging_detector([q1, q0],
                                              result_logging_mode='lin_trans')
            d.nr_shots = 4088  # To ensure proper data binning
        else:
            d = detector

        old_soft_avg = MC.soft_avg()
        old_live_plot_enabled = MC.live_plot_enabled()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        MC.set_detector_function(d)
        MC.run('SSRO_{}_{}_{}'.format(q1, q0, self.msmt_suffix))

        MC.soft_avg(old_soft_avg)
        MC.live_plot_enabled(old_live_plot_enabled)
        if analyze:
            a = mra.two_qubit_ssro_fidelity('SSRO_{}_{}'.format(q1, q0))
            a = ma2.Multiplexed_Readout_Analysis()
        return a

    def measure_msmt_induced_dephasing_matrix(self, qubits: list,
                                              analyze=True, MC=None,
                                              prepare_for_timedomain=True,
                                              n_amps_rel: int=None,
                                              verbose=True,
                                              get_quantum_eff: bool=False):
        '''
        Measures the msmt induced dephasing for readout the readout of qubits
        i on qubit j. Additionally measures the SNR as a function of amplitude
        for the diagonal elements to obtain the quantum efficiency.
        In order to use this: make sure that
        - all readout_and_depletion pulses are of equal total length
        - the cc light to has the readout time configured equal to the
            measurement and depletion time + 60 ns buffer

        fixme: not sure if the weight function assignment is working correctly.

        the qubit objects will use SSB for the ramsey measurements.
        '''
        lpatt = '_trgt_{TQ}_measured_{RQ}'
        if prepare_for_timedomain:
            #for q in qubits:
            #    q.prepare_for_timedomain()
            self.prepare_for_timedomain()

        old_suffixes = [q.msmt_suffix for q in qubits] #Save old qubit suffixes
        old_suffix = this.msmt_suffix

        # Save the start-time of the experiment for analysis
        start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Loop over all target and measurement qubits
        target_qubits = qubits[:]
        measured_qubits = qubits[:]
        for target_qubit in target_qubits:
            for measured_qubit in measured_qubits:
                # Set measurement label suffix
                s = lpatt.replace('{TQ}', target_qubit.name)
                s = s.replace('{RQ}', measured_qubit.name)
                measured_qubit.msmt_suffix = s
                target_qubit.msmt_suffix = s

                #Print label
                if verbose:
                    print(s)

                # Slight differences if diagonal element
                if target_qubit == measured_qubit:
                    amps_rel = np.linspace(0, 1, n_amps_rel)
                    mqp = None
                else:
                    t_amp_max = max(target_qubit.ro_pulse_down_amp0(),
                                    target_qubit.ro_pulse_down_amp1(),
                                    target_qubit.ro_pulse_amp())
                    amp_max = max(t_amp_max, measured_qubit.ro_pulse_amp())
                    amps_rel = np.linspace(0, 0.99/(amp_max), n_amps_rel)
                    mqp = self.cfg_openql_platform_fn()

                # If a diagonal element, consider doing the full quantum
                # efficiency matrix.
                if target_qubit == measured_qubit and get_quantum_eff:
                    res = measured_qubit.measure_quantum_efficiency(
                                                verbose=verbose,
                                                amps_rel=amps_rel)
                else:
                    res = measured_qubit.measure_msmt_induced_dephasing_sweeping_amps(
                            verbose=verbose,
                            amps_rel=amps_rel,
                            cross_target_qubits=[target_qubit],
                            multi_qubit_platf_cfg=mqp,
                            analyze=True
                        )
                # Print the result of the measurement
                if verbose:
                    print(res)

        # Save the end-time of the experiment
        stop = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        #reset the msmt_suffix'es
        for qi, q in enumerate(qubits):
            q.msmt_suffix = old_suffixes[qi]
        self.msmt_suffix = old_suffix

        # Run the analysis for this experiment
        if analyze:
            options_dict = {
                'verbose': True,
            }
            qarr = [q.name for q in qubits]
            labelpatt = 'ro_amp_sweep_ramsey'+lpatt
            ca = ma2.CrossDephasingAnalysis(t_start=start, t_stop=stop,
                                            label_pattern=labelpatt,
                                            qubit_labels=qarr,
                                            options_dict=options_dict)

    def measure_chevron(self, q0: str, q_spec: str,
                        amps, lengths,
                        prepare_for_timedomain=True, MC=None,
                        waveform_name='square'):
        """
        Measure a chevron by flux pulsing q0.
        q0 is put in an excited at the beginning of the sequence and pulsed
        back at the end.
        The spectator qubit (q_spec) performs a ramsey experiment over
        the flux pulse.
        """

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        assert q_spec in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q_specidx = self.find_instrument(q_spec).cfg_qubit_nr()

        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()

        if waveform_name == 'square':
            length_par = fl_lutman.sq_length
        elif waveform_name == 'cz_z':
            length_par = fl_lutman.cz_length,
        else:
            raise ValueError('Waveform shape not understood')

        awg = fl_lutman.AWG.get_instr()
        using_QWG = (awg.__class__.__name__ == 'QuTech_AWG_Module')

        if using_QWG:
            awg_ch = fl_lutman.cfg_awg_channel()
            amp_par = awg.parameters['ch{}_amp'.format(awg_ch)]
            sw = swf.FLsweep_QWG(fl_lutman, length_par,
                                 realtime_loading=False,
                                 waveform_name=waveform_name)
            flux_cw = 0

        else:
            awg_ch = fl_lutman.cfg_awg_channel()-1  # -1 is to account for starting at 1
            ch_pair = awg_ch % 2
            awg_nr = awg_ch//2

            amp_par = awg.parameters['awgs_{}_outputs_{}_amplitude'.format(
                awg_nr, ch_pair)]
            sw = swf.FLsweep(fl_lutman, length_par,
                             realtime_loading=False,
                             waveform_name=waveform_name)
            flux_cw = 2
        # buffer times are hardcoded for now FIXME!
        p = mqo.Chevron(q0idx, q_specidx, buffer_time=100e-9,
                        buffer_time2=200e-9,
                        flux_cw=flux_cw,
                        platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)
        self.instr_CC.get_instr().start()

        d = self.get_correlation_detector(single_int_avg=True,
                                          seg_per_point=1)

        MC.set_sweep_function(amp_par)
        MC.set_sweep_function_2D(sw)
        MC.set_sweep_points(amps)
        MC.set_sweep_points_2D(lengths)
        MC.set_detector_function(d)

        MC.run('Chevron {} {}'.format(q0, q_spec), mode='2D')
        ma.TwoD_Analysis()

    def measure_cryoscope(self, q0: str, times,
                          MC=None,
                          experiment_name='Cryoscope',
                          waveform_name: str='square',
                          max_delay: float='auto',
                          prepare_for_timedomain: bool=True):
        """
        Performs a cryoscope experiment to measure the shape of a flux pulse.

        Args:
            q0  (str)     :
                name of the target qubit

            times   (array):
                array of measurment times

            experiment_name (str):
                used to label the experiment

            waveform_name (str {"square", "custom_wf"}) :
                defines the name of the waveform used in the
                cryoscope. Valid values are either "square" or "custom_wf"

            max_delay {float, "auto"} :
                determines the delay in the delay in the pusle sequence
                if set to "auto" this is automatically set to the largest
                pulse duration for the cryoscope.

            prepare_for_timedomain (bool):
                calls self.prepare_for_timedomain on start
        """

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        q0idx = self.find_instrument(q0).cfg_qubit_nr()

        if max_delay == 'auto':
            max_delay = np.max(times) + 40e-9
        p = mqo.Cryoscope(q0idx, buffer_time1=20e-9,
                          buffer_time2=max_delay,
                          platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)
        self.instr_CC.get_instr().start()

        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()

        if waveform_name == 'square':
            sw = swf.FLsweep(fl_lutman, fl_lutman.sq_length,
                             realtime_loading=True,
                             waveform_name='square')
        elif waveform_name == 'custom_wf':
            sw = swf.FLsweep(fl_lutman, fl_lutman.custom_wf_length,
                             realtime_loading=True,
                             waveform_name='custom_wf')
        else:
            raise ValueError('waveform_name "{}" should be either '
                             '"square" or "custom_wf"'.format(waveform_name))
        MC.set_sweep_function(sw)
        MC.set_sweep_points(times)
        d = self.get_int_avg_det(values_per_point=2,
                                 values_per_point_suffex=['cos', 'sin'],
                                 single_int_avg=True,
                                 always_prepare=True)
        MC.set_detector_function(d)
        MC.run(experiment_name)

    ########################################################
    # Calibration methods
    ########################################################

    def calibrate_mux_RO(self,
                         calibrate_optimal_weights=True,
                         verify_optimal_weights=False,
                         update: bool=True,
                         update_threshold: bool=True)-> bool:
        """
        Calibrates multiplexed Readout.
        N.B. Currently only works for 2 qubits
        """

        q0 = self.find_instrument(self.qubits()[0])
        q1 = self.find_instrument(self.qubits()[1])

        q0idx = q0.cfg_qubit_nr()
        q1idx = q1.cfg_qubit_nr()

        UHFQC = q0.instr_acquisition.get_instr()
        self.ro_acq_weight_type('optimal')
        self.prepare_for_timedomain()

        if calibrate_optimal_weights:
            # Important that this happens before calibrating the weights
            # 5 is the number of channels in the UHFQC
            for i in range(5):
                UHFQC.set('quex_trans_offset_weightfunction_{}'.format(i), 0)

            UHFQC.upload_transformation_matrix(np.eye(5))
            q0.calibrate_optimal_weights(
                analyze=True, verify=verify_optimal_weights)
            q1.calibrate_optimal_weights(
                analyze=True, verify=verify_optimal_weights)

        self.measure_two_qubit_SSRO([q1.name, q0.name],
                                    result_logging_mode='lin_trans')

        res_dict = mra.two_qubit_ssro_fidelity(
            label='{}_{}'.format(q0.name, q1.name),
            qubit_labels=[q0.name, q1.name])
        V_offset_cor = res_dict['V_offset_cor']

        # weights 0 and 1 are the correct indices because I set the numbering
        # at the start of this calibration script.
        UHFQC.quex_trans_offset_weightfunction_0(V_offset_cor[0])
        UHFQC.quex_trans_offset_weightfunction_1(V_offset_cor[1])

        # Does not work because axes are not normalized
        matrix_normalized = res_dict['mu_matrix_inv']
        matrix_rescaled = matrix_normalized/abs(matrix_normalized).max()
        UHFQC.upload_transformation_matrix(matrix_rescaled)

        # a = self.check_mux_RO(update=update, update_threshold=update_threshold)
        return True

    def calibrate_cz_single_q_phase(self, q0: str, q1: str,
                                    amps,
                                    waveform='cz_z',
                                    update: bool = True,
                                    prepare_for_timedomain: bool=True, MC=None):

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()
        fl_lutman_q0 = self.find_instrument(q0).instr_LutMan_Flux.get_instr()
        fl_lutman_q1 = self.find_instrument(q1).instr_LutMan_Flux.get_instr()

        p = mqo.conditional_oscillation_seq(q0idx, q1idx,
                                            platf_cfg=self.cfg_openql_platform_fn(),
                                            CZ_disabled=False, add_cal_points=False,
                                            angles=[90])

        CC = self.instr_CC.get_instr()
        CC.eqasm_program(p.filename)
        CC.start()

        # other waveform is required for real-time reloading as the other
        # waveform is normally overwritten
        if waveform == 'cz_z':
            other_waveform = fl_lutman_q1._wave_dict_dist['idle_z']
        else:
            other_waveform = fl_lutman_q1._wave_dict_dist['cz_z']
        self.other_waveform = other_waveform

        s = swf.FLsweep(fl_lutman_q0, fl_lutman_q0.cz_phase_corr_amp,
                        waveform, realtime_loading=True,
                        other_waveform=other_waveform)

        d = self.get_correlation_detector(single_int_avg=True, seg_per_point=2)
        d.detector_control = 'hard'
        # the order of self.qubits is used in the correlation detector
        # and is required for the analysis
        ch_idx = self.qubits().index(q0)

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.repeat(amps, 2))
        MC.set_detector_function(d)
        MC.run('{}_CZphase'.format(q0))

        a = ma2.Intersect_Analysis(options_dict={'ch_idx_A': ch_idx,
                                                 'ch_idx_B': ch_idx})

        phase_corr_amp = a.get_intersect()[0]
        if phase_corr_amp > np.max(amps) or phase_corr_amp < np.min(amps):
            print('Calibration failed, intersect outside of initial range')
            return False
        else:
            if update:
                self.find_instrument(q0).fl_cz_phase_corr_amp(phase_corr_amp)
            return True

    def calibrate_flux_timing(self, q0: str, q1: str,
                              times,
                              waveform='cz_z',
                              update: bool = True,
                              prepare_for_timedomain: bool=True, MC=None):

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()
        fl_lutman = self.find_instrument(q0).instr_LutMan_Flux.get_instr()

        p = mqo.conditional_oscillation_seq(q0idx, q1idx,
                                            platf_cfg=self.cfg_openql_platform_fn(),
                                            CZ_disabled=False, add_cal_points=False,
                                            angles=[90])

        CC = self.instr_CC.get_instr()
        CC.eqasm_program(p.filename)
        CC.start()

        s = swf.FLsweep(fl_lutman, fl_lutman.cz_phase_corr_amp,
                        waveform)

        d = self.get_correlation_detector(single_int_avg=True, seg_per_point=2)
        d.detector_control = 'hard'
        # the order of self.qubits is used in the correlation detector
        # and is required for the analysis
        ch_idx = self.qubits().index(q0)

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.repeat(amps, 2))
        MC.set_detector_function(d)
        MC.run('{}_CZphase'.format(q0))

        a = ma2.CZ_1QPhaseCal_Analysis(options_dict={'ch_idx': ch_idx})

        phase_corr_amp = a.get_zero_phase_diff_intersect()
        if phase_corr_amp > np.max(amps) or phase_corr_amp < np.min(amps):
            print('Calibration failed, intersect outside of initial range')
            return False
        else:
            if update:
                self.find_instrument(q0).fl_cz_phase_corr_amp(phase_corr_amp)
            return True

    def create_dep_graph(self):
        dags = []
        for qi in self.qubits():
            q_obj = self.find_instrument(qi)
            if hasattr(q_obj, '_dag'):
                dag = q_obj._dag
            else:
                dag = q_obj.create_dep_graph()
            dags.append(dag)

        dag = nx.compose_all(dags)

        dag.add_node(self.name+' multiplexed readout')
        dag.add_node(self.name+' resonator frequencies coarse')
        dag.add_node('AWG8 MW-staircase')
        dag.add_node('AWG8 Flux-staircase')

        # Timing of channels can be done independent of the qubits
        # it is on a per frequency per feedline basis so not qubit specific
        dag.add_node(self.name + ' mw-ro timing')
        dag.add_edge(self.name + ' mw-ro timing', 'AWG8 MW-staircase')

        dag.add_node(self.name + ' mw-vsm timing')
        dag.add_edge(self.name + ' mw-vsm timing', self.name + ' mw-ro timing')

        for edge_L, edge_R in self.qubit_edges():
            dag.add_node('Chevron {}-{}'.format(edge_L, edge_R))
            dag.add_node('CZ {}-{}'.format(edge_L, edge_R))

            dag.add_edge('CZ {}-{}'.format(edge_L, edge_R),
                         'Chevron {}-{}'.format(edge_L, edge_R))
            dag.add_edge('CZ {}-{}'.format(edge_L, edge_R),
                         '{} cryo dist. corr.'.format(edge_L))
            dag.add_edge('CZ {}-{}'.format(edge_L, edge_R),
                         '{} cryo dist. corr.'.format(edge_R))

            dag.add_edge('Chevron {}-{}'.format(edge_L, edge_R),
                         '{} single qubit gates fine'.format(edge_L))
            dag.add_edge('Chevron {}-{}'.format(edge_L, edge_R),
                         '{} single qubit gates fine'.format(edge_R))
            dag.add_edge('Chevron {}-{}'.format(edge_L, edge_R),
                         'AWG8 Flux-staircase')
            dag.add_edge('Chevron {}-{}'.format(edge_L, edge_R),
                         self.name+' multiplexed readout')

            dag.add_node('{}-{} mw-flux timing'.format(edge_L, edge_R))

            dag.add_edge(edge_L+' cryo dist. corr.',
                         '{}-{} mw-flux timing'.format(edge_L, edge_R))
            dag.add_edge(edge_R+' cryo dist. corr.',
                         '{}-{} mw-flux timing'.format(edge_L, edge_R))

            dag.add_edge('Chevron {}-{}'.format(edge_L, edge_R),
                         '{}-{} mw-flux timing'.format(edge_L, edge_R))
            dag.add_edge('{}-{} mw-flux timing'.format(edge_L, edge_R),
                         'AWG8 Flux-staircase')

            dag.add_edge('{}-{} mw-flux timing'.format(edge_L, edge_R),
                         self.name + ' mw-ro timing')

        for qubit in self.qubits():
            dag.add_edge(qubit + ' ro pulse-acq window timing',
                         'AWG8 MW-staircase')

            dag.add_edge(qubit+' room temp. dist. corr.',
                         'AWG8 Flux-staircase')
            dag.add_edge(self.name+' multiplexed readout',
                         qubit+' optimal weights')

            dag.add_edge(qubit+' resonator frequency',
                         self.name+' resonator frequencies coarse')
            dag.add_edge(qubit+' pulse amplitude coarse', 'AWG8 MW-staircase')

        for qi in self.qubits():
            q_obj = self.find_instrument(qi)
            # ensures all references are to the main dag
            q_obj._dag = dag

        self._dag = dag
        return dag
