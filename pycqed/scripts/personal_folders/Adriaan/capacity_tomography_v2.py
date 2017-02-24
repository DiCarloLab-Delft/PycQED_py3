# Redefining names defined in the global namespace (only for linter)

station = station

import numpy as np
from qcodes.instrument.parameter import ManualParameter
from pycqed.measurement import detector_functions as det
from pycqed.measurement.waveform_control import sequence
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars


# Function definitions


def capacity_tomo_seq(RO_bases, prep_bases, states, idle_time,
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
    assert(len(RO_bases) == len(states))
    assert(len(prep_bases) == len(states))
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
    for RO_base, prep_base, state in zip(RO_bases, prep_bases, states):
        seq_idx += [3*2*RO_base + 2*prep_base + state]
    # Creates a sequence by selecting the right primitive element
    for i, idx in enumerate(seq_idx):
        seq.append(name='elt_{}'.format(i),
                   wfname=el_list[idx].name, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
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
        RO_bases = self.sweep_points[start_idx:end_idx, 0]
        prep_bases = self.sweep_points[start_idx:end_idx, 1]
        states = self.sweep_points[start_idx:end_idx, 2]

        # load sequence
        capacity_tomo_seq(RO_bases=RO_bases, prep_bases=prep_bases,
                          states=states,
                          idle_time=self.idle_time,
                          pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,)
        return super().get_values()

###################################
# Script that runs the experiment #
###################################


chunk_size = 8000
number_of_shots = chunk_size*3


# Parameters are only used for labels and units in the datafile
RO_basis = ManualParameter('RO_basis', units='')
prep_basis = ManualParameter('prep_basis', units='')
state = ManualParameter('state', units='')
sweep_pars = [RO_basis, prep_basis, state]

# MC is the MeasurementControl that controls the data acquisition loop

CBox.log_length(chunk_size)
base_combinations = ['ZXY']
idle_times = [.3e-6, 2e-6]
base = 'ZXY'

for idle_time in idle_times:
    RO_bases = np.random.randint(0, 3, number_of_shots)
    prep_bases = np.random.randint(0, 3, number_of_shots)
    states = np.random.randint(0, 2, number_of_shots)

    sweep_points = np.array([RO_bases, prep_bases, states]).T
    calibrate_RO_threshold_no_rotation()
    log_length = CBox.log_length()
    d = Capacity_tomo_detector(
        CBox=CBox, AWG=AWG, threshold=CBox.sig0_threshold_line(),
        chunk_size=log_length,
        idle_time=idle_time, pulse_pars=pulse_pars, RO_pars=RO_pars)

    MC.set_sweep_functions(sweep_pars)
    MC.set_sweep_points(sweep_points)
    MC.set_detector_function(d)
    MC.run('Capacity_tomo_v2_idle_time_{:.4g}s_base_{}'.format(idle_time, base))


