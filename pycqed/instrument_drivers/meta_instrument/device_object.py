import numpy as np
import qcodes as qc
import os
import logging
import pycqed.measurement.waveform_control_CC.qasm_compiler as qcx
from copy import deepcopy
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
from pycqed.analysis import multiplexed_RO_analysis as mra
from pycqed.measurement.waveform_control_CC import multi_qubit_module_CC\
    as mqmc
from pycqed.measurement.waveform_control_CC import multi_qubit_qasm_seqs\
    as mqqs
import pycqed.measurement.waveform_control_CC.single_qubit_qasm_seqs as sqqs
from pycqed.measurement import detector_functions as det
from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import measurement_analysis as ma
import pycqed.measurement.gate_set_tomography.gate_set_tomography_CC as gstCC
import pygsti
from pycqed.utilities.general import gen_sweep_pts

# from pycqed_scripts.experiments.Starmon_1702.clean_scripts.functions import \
#     CZ_cost_Z_amp


class DeviceObject(Instrument):
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.msmt_suffix = '_' + name  # used to append to measurement labels
        self.add_parameter(
            'qasm_config',
            docstring='used for generating qumis instructions',
            parameter_class=ManualParameter,
            vals=vals.Anything())
        self.add_parameter(
            'qubits',
            parameter_class=ManualParameter,
            initial_value=[],
            vals=vals.Lists(elt_validator=vals.Strings()))

        self.add_parameter(
            'acquisition_instrument', parameter_class=InstrumentParameter)
        self.add_parameter(
            'RO_acq_averages',
            initial_value=1024,
            vals=vals.Ints(),
            parameter_class=ManualParameter)

        self._sequencer_config = {}
        self.delegate_attr_dicts += ['_sequencer_config']

        # Add buffer parameters for every pulse type
        pulse_types = ['MW', 'Flux', 'RO']
        self.add_parameter(
            'sequencer_config',
            get_cmd=self._get_sequencer_config,
            vals=vals.Anything())
        for pt_a in pulse_types:
            for pt_b in pulse_types:
                self.add_parameter(
                    'Buffer_{}_{}'.format(pt_a, pt_b),
                    unit='s',
                    initial_value=0,
                    parameter_class=ManualParameter)
                self.add_sequencer_config_param(
                    self.parameters['Buffer_{}_{}'.format(pt_a, pt_b)])

        self.add_parameter(
            'RO_fixed_point',
            unit='s',
            initial_value=1e-6,
            docstring=('The Tektronix sequencer shifts all elements in a ' +
                       'sequence such that the first RO encountered in each ' +
                       'element is aligned with the fixpoint.\nIf the ' +
                       'sequence length is longer than the fixpoint, ' +
                       'it will shift such that the first RO aligns with a ' +
                       'multiple of the fixpoint.'),
            parameter_class=ManualParameter,
            vals=vals.Numbers(1e-9, 500e-6))
        self.add_sequencer_config_param(self.RO_fixed_point)
        self.add_parameter(
            'Flux_comp_dead_time',
            unit='s',
            initial_value=3e-6,
            docstring=('Used to determine the wait time between the end of a' +
                       'sequence and the beginning of Flux compensation' +
                       ' pulses.'),
            parameter_class=ManualParameter,
            vals=vals.Numbers(1e-9, 500e-6))
        self.add_sequencer_config_param(self.Flux_comp_dead_time)

    def get_idn(self):
        return self.name

    def _get_sequencer_config(self):
        seq_cfg = {}
        for par_name, config_par in self._sequencer_config.items():
            seq_cfg[par_name] = config_par()
        return seq_cfg

    def add_sequencer_config_param(self, seq_config_par):
        """
        """
        name = seq_config_par.name
        self._sequencer_config[name] = seq_config_par
        return name

    def get_operation_dict(self, operation_dict={}):
        # uses the private qubits list
        for qname in self.qubits():
            q = self.find_instrument(qname)
            q.get_operation_dict(operation_dict)
        operation_dict['sequencer_config'] = self.sequencer_config()
        return operation_dict

    def calibrate_mux_RO(self):
        raise NotImplementedError

    def prepare_for_timedomain(self):
        """
        Calls the prepare for timedomain on each of the constituent qubits.
        """
        RO_LMM = self.RO_LutManMan.get_instr()

        q0 = self.find_instrument(self.qubits()[0])
        q1 = self.find_instrument(self.qubits()[1])

        # this is to ensure that the cross talk correction matrix coefficients
        # can rey on the weight function indices being consistent
        q0.RO_acq_weight_function_I(0)
        q1.RO_acq_weight_function_I(1)

        # Set the modulation frequencies so that the LO's match
        q0.f_RO_mod(q0.f_RO() - self.RO_LO_freq())
        q1.f_RO_mod(q1.f_RO() - self.RO_LO_freq())

        # Update parameters
        q0.RO_pulse_type('IQmod_multiplexed_UHFQC')
        q1.RO_pulse_type('IQmod_multiplexed_UHFQC')

        q1.prepare_for_timedomain()
        q0.prepare_for_timedomain()
        RO_LMM.acquisition_delay(q0.RO_acq_marker_delay())
        #aligining the acquisition delays according to q0, important for optimal
        #weight calibrations
        q0.RO_acq_marker_delay(q1.RO_acq_marker_delay())

        # Generate multiplexed pulse
        multiplexed_wave = [[q0.RO_LutMan(), 'M_simple'],
                            [q1.RO_LutMan(), 'M_simple']]
        RO_LMM.generate_multiplexed_pulse(multiplexed_wave)
        RO_LMM.load_pulse_onto_AWG_lookuptable('Multiplexed_pulse')

    def prepare_for_fluxing(self, reset: bool = True):
        '''
        Calls the prepare for timedomain on each of the constituent qubits.
        The reset option is passed to the first qubit that is called. For the
        rest of the qubit, reset is always disabled, in order not to overwrite
        the settings of the first qubit.
        '''
        for q in self.qubits():
            q_obj = self.find_instrument(q)
            q_obj.prepare_for_fluxing(reset=reset)
            reset = False
