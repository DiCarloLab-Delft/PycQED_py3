# Redefining names defined in the global namespace (only for linter)
station = station

from modules.measurement import detector_functions as det
from modules.measurement.waveform_control import sequence
from modules.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from modules.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars


def capacity_tomo_seq(bases, states, idle_time, pulse_pars, RO_pars,
                      upload=True,
                      verbose=False):
    """
    input pars
        bases list of ints:
            0 - Z-basis
            1 - X-basis
            2 - Y-basis
        states (list of ints):
            0 - |0>
            1 - |1>
        idle_time (s)


    """
    assert(len(bases) == len(states))
    seq_name = 'Capacity_tomo_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # Create a dict with the parameters for all the pulses
    pulse_dict = get_pulse_dict_from_pars(pulse_pars)
    pulse_dict['I']['pulse_delay'] = idle_time
    pulse_dict['RO'] = RO_pars
    pulse_combinations = []
    for basis, state in zip(bases, states):
        if state not in [0, 1]:
            raise ValueError('state {} not recognized'.format(state))
        if basis == 0:  # computational(Z)-basis
            if state == 0:
                pulse_combinations += [['I', 'RO']]
            elif state == 1:
                pulse_combinations += [['X180', 'I', 'RO']]
        elif basis == 1:  # X basis
            if state == 0:
                pulse_combinations += [['Y90', 'I', 'mY90']]
            if state == 1:
                pulse_combinations += [['mY90', 'I', 'mY90']]
        elif basis == 2:  # Y basis
            if state == 0:
                pulse_combinations += [['X90', 'I', 'mX90']]
            if state == 1:
                pulse_combinations += [['mX90', 'I', 'mX90']]
        else:
            raise ValueError('basis {} not recognized'.formate(basis))

    for i, pulse_comb in enumerate(pulse_combinations):
        pulse_list = []
        for pulse_key in pulse_comb:
            pulse_list += [pulse_dict[pulse_key]]
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq, el_list


class Capacity_tomo_detector(det.CBox_digitizing_shots_det):
    def __init__(self, CBox, AWG, threshold, idle_time, pulse_pars, RO_pars,
                 LutMan=None, reload_pulses=False, awg_nrs=None):
        super().__init__(CBox, AWG, LutMan, reload_pulses, awg_nrs)
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars

    def prepare(self, sweep_points):
        self.i = 0
        self.sweep_points = sweep_points

    def get_values(self):
        return self.get()

    def get(self):
        start_idx = self.i*self.chunk_size
        end_idx = start_idx + self.chunk_size
        self.i += 1
        bases = self.sweep_points[start_idx:end_idx, 0]
        states = self.sweep_points[start_idx:end_idx, 1]

        # load sequence
        capacity_tomo_seq(bases, states, idle_time, pulse_pars, RO_pars,

        # if len(np.shape(self.sweep_points))==2:
        #     return self.sweep_points[start_idx:end_idx, :].T
        # else:
        #     return self.sweep_points[start_idx:end_idx]




