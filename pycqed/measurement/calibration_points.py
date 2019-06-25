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

    def get_states(self, qb_names=None):
        """
        Get calibrations states given a (subset) of qubits of self.qb_names.
        This function is a helper for the analysis which works with information
        per qubit.
        Args:
            qb_names: list of qubit names

        Returns: dict where keys are qubit names and values are the calibration
            states for this particular qubit
        """

        qb_names = self._check_qb_names(qb_names)

        return {qbn: [s[self.qb_names.index(qbn)] for s in self.states]
                for qbn in qb_names}

    def get_indices(self, qb_names=None):
        """
        Get calibration indices
        Args:
            qb_names: qubit name or list of qubit names to retrieve
                the indices of. Defaults to all.

        Returns: dict where keys are qb_names and values dict of {state: ind}

        """
        qb_names = self._check_qb_names(qb_names)
        indices = dict()
        states = self.get_states(qb_names)

        for qbn in qb_names:
            unique, idx, inv = np.unique(states[qbn], return_inverse=True,
                                        return_index=True)
            indices[qbn] = {s: [-len(states[qbn]) + j
                                for j in range(len(inv)) if i == inv[j]]
                            for i, s in enumerate(unique)}

        log.info(f"Calibration Points Indices: {indices}")
        return indices

    def _check_qb_names(self, qb_names):
        if qb_names is None:
            qb_names = self.qb_names
        elif np.ndim(qb_names) == 0:
            qb_names = [qb_names]
        for qbn in qb_names:
            assert qbn in self.qb_names, f"{qbn} not in Calibrated Qubits: " \
                f"{self.qb_names}"

        return qb_names

    def get_rotations(self, last_ge_pulses, qb_names=None):
        """
        Get rotation dictionaries for each qubit in qb_names,
        as used by the analysis for plotting.
        Args:
            qb_names (list or string): qubit names. Defaults to all.
            last_ge_pulses (list or bool): one for each qb in the same order as
                specified in qb_names

        Returns:
             dict where keys are qb_names and values are dict specifying
             rotations.

        """
        qb_names = self._check_qb_names(qb_names)
        rotations = dict()

        if len(qb_names) == 0:
            return rotations
        if len(qb_names) == 1:
            last_ge_pulses = [last_ge_pulses] if \
                isinstance(last_ge_pulses, bool) else last_ge_pulses
        else:
            i, j = len(qb_names), \
                   1 if isinstance(last_ge_pulses, bool) else len(last_ge_pulses)
            assert i == j, f"Size of qb_names and last_ge_pulses don't " \
                f"match: {i} vs {j}"

        states = self.get_states(qb_names)
        for i, qbn in enumerate(qb_names):
            # get unique states in reversed alphabetical order: g, [e, f]
            order = {"g":0, "e": 1, "f": 2}
            unique = list(np.unique(states[qbn]))
            unique.sort(key=lambda s: order[s])
            if len(unique) == 3:
                states = np.delete(unique, 1 if last_ge_pulses[i] else 0)
            rotations[qbn] = {states[i]: i for i in range(len(states))}
        log.info(f"Calibration Points Rotation: {rotations}")
        return rotations

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




