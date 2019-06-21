from copy import deepcopy

import numpy as np
import itertools
import logging

from pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts import \
    generate_mux_ro_pulse_list
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    add_preparation_pulses
from pycqed.measurement.waveform_control import segment

log = logging.getLogger(__name__)


class CalibrationPoints:
    def __init__(self, qb_names, states, **kwargs):
        self.qb_names = qb_names
        self.states = states
        default_map = dict(g=['I '], e=["X180 "], f=['X180 ', "X180_ef "])
        self.pulse_label_map = kwargs.get("pulse_label_map", default_map)

    def create_segments(self, operation_dict, **prep_params):
        segments = []

        for i, seg_states in enumerate(self.states):
            pulse_list = []
            for j, qbn in enumerate(self.qb_names):
                for k, pulse_name in enumerate(self.pulse_label_map[seg_states[j]]):
                    pulse = deepcopy(operation_dict[pulse_name + qbn])
                    if k == 0:
                        pulse['ref_pulse'] = 'segment_start'
                    pulse_list.append(pulse)
                pulse_list = add_preparation_pulses(pulse_list, operation_dict,
                                                    [qbn], **prep_params)

            pulse_list += generate_mux_ro_pulse_list(self.qb_names,
                                                     operation_dict)
            seg = segment.Segment(f'calibration_{i}', pulse_list)
            segments.append(seg)
        return segments
    # def create_segments(self, operations_dict, qb_names=None, **prep_params):
    #     if qb_names is None:
    #         qb_names = self.qb_names
    #
    #     step = np.abs(sweep_points[-1] - sweep_points[-2])
    #
    #     if loc == 'end':
    #         sweep_points = np.concatenate(
    #             [sweep_points, [sweep_points[-1] + i * step for
    #                             i in range(1, self.n_cal_points + 1)]])
    #         start_idx =  \
    #             -np.flip(np.linspace(self.n_per_state,
    #                                  len(self.states) * self.n_per_state,
    #                                  len(self.states)))
    #         cal_pt_indices = \
    #             {s: [idx + i for i in range(self.n_per_state)]
    #                     for s, idx in zip(self.states, start_idx)}
    #         self.segments.update(cal_pt_indices)
    #         if self.verbose:
    #             logging.info("Calibration Points Indices: {}"
    #                          .format(self.segments))
    #     else:
    # #         raise NotImplementedError("Location {} not implemented"
    # #                                   .format(loc))
    #
    #     return sweep_points

    @staticmethod
    def single_qubit(qubit_name, states, n_per_state=2):
        return CalibrationPoints.multi_qubit([qubit_name], states, n_per_state)

    @staticmethod
    def multi_qubit(qb_names, states, n_per_state=2, all_combinations=False):
        n_qubits = len(qb_names)
        if n_qubits == 0:
            return CalibrationPoints(qb_names, [])

        if all_combinations:
            labels_array = np.tile(
                list(itertools.product(states, repeat=n_qubits)), n_per_state)
            labels = [tuple(seg_label)
                      for seg_label in labels_array.reshape((-1, n_qubits))]
        else:
            labels =[tuple(np.repeat(tuple([state]), n_qubits))
                     for state in states for _ in range(n_per_state)]

        return CalibrationPoints(qb_names, labels)

    def __repr__(self):
        return "Calibration:\n    Qubits: {}\n    Labels: {}" \
            .format(self.qb_names, self.states)




