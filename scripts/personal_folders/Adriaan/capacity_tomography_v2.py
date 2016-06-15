# Redefining names defined in the global namespace (only for linter)

station = station

import numpy as np
from qcodes.instrument.parameter import ManualParameter
from modules.measurement import detector_functions as det
from modules.measurement.waveform_control import sequence
from modules.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from modules.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars


# Function definitions


def capacity_tomo_seq(prep_bases, states, RO_bases, idle_time,
                      pulse_pars, RO_pars,
                      upload=True,
                      verbose=False):
    """
    input pars
    prep_bases list of ints:
        0 - Z-basis
        1 - X-basis
        2 - Y-basis
    states (list of ints):
        0 - |0>
        1 - |1>
    RO_bases list of ints:
        0 - Z-basis
        1 - X-basis
        2 - Y-basis
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

    prep_pulses = [['I'],
                   ['X180', 'I'],
                   ['Y90', 'I'],
                   ['mY90', 'I'],
                   ['X90', 'I'],
                   ['mX90', 'I']]
    RO_pulses = [['RO'],
                 ['mX90', 'RO'],
                 ['mY90', 'RO']]

    pulse_combinations = []
    for RO_pulse_comb in RO_pulses:
        for prep_pulse_comb in prep_pulses:
            pulse_combinations += [prep_pulse_comb + RO_pulse_comb]

    # Creates elements containing the primitive pulse sequences
    for i, pulse_comb in enumerate(pulse_combinations):
        pulse_list = []
        for pulse_key in pulse_comb:
            pulse_list += [pulse_dict[pulse_key]]
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    seq_idx = []
    # Selects the corresponding pulse combination
    for basis, state in zip(prep_bases, states, RO_bases):
        if state not in [0, 1]:
            raise ValueError('state {} not recognized'.format(state))
        if basis == 0:  # computational(Z)-basis
            if state == 0:
                seq_idx += [0]
            elif state == 1:
                seq_idx += [1]
        elif basis == 1:  # X basis
            if state == 0:
                seq_idx += [2]
            if state == 1:
                seq_idx += [3]
        elif basis == 2:  # Y basis
            if state == 0:
                seq_idx += [4]
            if state == 1:
                seq_idx += [5]
        else:
            raise ValueError('basis {} not recognized'.formate(basis))
    # Creates a sequence by selecting the right primitive element
    for i, idx in enumerate(seq_idx):
        seq.append(name='elt_{}'.format(i),
                   wfname=el_list[idx].name, trigger_wait=True)
    if upload:
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq, el_list


class Capacity_tomo_detector(det.CBox_digitizing_shots_det):
    def __init__(self, CBox, AWG, threshold, chunk_size,
                 idle_time, pulse_pars, RO_pars,
                 LutMan=None, reload_pulses=False, awg_nrs=None):
        super().__init__(CBox=CBox, AWG=AWG, LutMan=LutMan,
                         threshold=threshold,
                         reload_pulses=reload_pulses, awg_nrs=awg_nrs)
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.idle_time = idle_time
        self.chunk_size = chunk_size

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
        capacity_tomo_seq(bases, states,
                          self.idle_time, self.pulse_pars, self.RO_pars,)
        return super().get_values()

###################################
# Script that runs the experiment #
###################################


chunk_size = 8000
number_of_shots = chunk_size*130


# Parameters are only used for labels and units in the datafile
basis = ManualParameter('basis', units='')
state = ManualParameter('state', units='')

# MC is the MeasurementControl that controls the data acquisition loop

CBox.log_length(chunk_size)

# base_combinations = ['ZX', 'XY']
base_combinations = ['ZXY']
idle_times = [0, 5e-6, 10e-6, 15e-6, 20e-6, 25e-6]
for base in base_combinations:
    if base == 'ZX':
        b = [0, 2]
    elif base == 'XY':
        b = [1, 3]
    else:
        b = [0, 3]
    for idle_time in idle_times:
        bases = np.random.randint(b[0], b[1], number_of_shots)
        states = np.random.randint(0, 2, number_of_shots)
        sweep_points = np.array([bases, states]).T
        calibrate_RO_threshold_no_rotation()
        log_length = CBox.log_length()
        d = Capacity_tomo_detector(
            CBox=CBox, AWG=AWG, threshold=CBox.sig0_threshold_line(),
            chunk_size=log_length,
            idle_time=idle_time, pulse_pars=pulse_pars, RO_pars=RO_pars)

        MC.set_sweep_functions([basis, state])
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(d)
        MC.run('Capacity_tomo_idle_time_{:.4g}s_base_{}'.format(idle_time, base))
