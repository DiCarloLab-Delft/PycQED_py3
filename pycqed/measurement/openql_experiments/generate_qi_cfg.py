import json
import numpy as np
from numpy import pi, sin, cos, exp
from scipy.linalg import expm
from itertools import chain, repeat


_rot = "rot"
_pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
_pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _dict_merge(*args):
    """merge some small dictionaries to one big"""
    out = {}
    for d in args:
        out.update(d)
    return out


def concat(iterables_list):
    return list(chain(*iterables_list))


def _kraus_jsonify(kr_list):
    out = []
    for mat in kr_list:
        re = np.real(mat).reshape((len(mat)*len(mat[0]),))
        im = np.imag(mat).reshape((len(mat)*len(mat[0]),))
        out.append(list(zip(re, im)))
    return out


def _angle_pi_output(angle):
    return "{:.1f}".format(angle*180)


def _rotation_kraus(phi, theta):
    return [expm(-0.5j*pi*theta*(_pauli_x*cos(phi*pi) + _pauli_y*sin(phi*pi)))]


def _cphase_kraus(theta):
    return [np.diag((1., 1., 1., exp(1j*theta*pi)))]


def _rotation_instruction(phi, theta):
    return '{} {} {} {{}}'.format(_rot, _angle_pi_output(phi),
                                  _angle_pi_output(theta)) \
                         .replace('-', 'm')


_rotation_expansions = {
        # Expanding RX phi -> R 0 phi
        'x': lambda angle_pi: ((0, angle_pi), ),
        # Expanding RY phi -> R 0.5pi phi
        'y': lambda angle_pi: ((0.5, angle_pi), ),
        # Expanding RZ phi -> RY pi + R 0.5(phi - pi) pi
        'z': lambda angle_pi: ((0.5, 1), (0.5*(angle_pi - 1.), 1))
    }


def _rotation_decomposition(ax, angle):
    key = 'R{} %0 {}'.format(ax.upper(), _angle_pi_output(angle)) \
                       .replace('-', 'm')
    value = ['{} {} {} %0'.format(_rot, _angle_pi_output(phi),
                                  _angle_pi_output(theta))
                          .replace('-', 'm')
             for phi, theta in _rotation_expansions[ax](angle)]
    return (key, value)


def generate_config(filename: str,
                    qubits_active: list = None,
                    mw_pulse_duration: int = 20,
                    flux_pulse_duration: int = 40,
                    ro_duration: int = 800,
                    mw_mw_buffer=0,
                    mw_flux_buffer=0,
                    flux_mw_buffer=0,
                    ro_latency: int = 0,
                    mw_latency: int = 0,
                    fl_latency: int = 0,
                    init_duration: int = 200000,
                    simulation_t1=3e7,
                    simulation_t2=1e7,
                    simulation_frac1_0=0.0001,
                    simulation_frac1_1=0.9999):
    """
    Generates a configuration file for OpenQL for use with the CCLight.
    Args:
        filename (str)          : location where to write the config json file
        mw_pulse_duration (int) : duration of the mw_pulses in ns.
            N.B. this should be 20 as the VSM marker is hardcoded to be of that
            length.
        flux_pulse_duration (int) : duration of flux pulses in ns.
        ro_duration       (int) : duration of the readout, including depletion
                                    in ns.
        init_duration     (int) : duration of the initialization/reset
            operation in ns. This corresponds to the wait time before every
            experiment.

    The format for the configuration is a completely flattened file, this means
    that for every operation (including it's target) there is a separate entry
    in the JSON. The details of what can be specified are given in the OpenQL
    documentation under "configuration_specification".
    """

    angles_pi = {
        'x': [0.67, -0.23, -0.08, 0.08, -0.5, 0.5, 1],
        'y': [-0.5, 0.5, 1],
        'z': [-0.38, 0.08, -0.1, -0.5, 0.5, 1]
    }

    # Set of tuples of all needed rotations in form (rot_axis_angle, rot_angle)
    # `rot_axis_angle` is an angle in xy plane, 0 for x and pi/2 for y.
    rotations = [[[_rotation_expansions[ax](angle)] for angle in angles_pi[ax]]
                 for ax in 'xyz']
    # sorry
    rotations = list(sorted(set(
        concat(concat(concat(rotations))))))
    rots_kraus = [_rotation_kraus(*rot) for rot in rotations]
    rots_instr = [_rotation_instruction(*rot) for rot in rotations]

    rots_decomposition = dict(concat(
        [_rotation_decomposition(*args)
         for args in zip(repeat(ax), angles_pi[ax])]
        for ax in 'xyz'
    ))

    decompositions_aliases = {
        "rx90 %0": ["{} 0.0 90.0 %0".format(_rot)],
        "ry90 %0": ["{} 90.0 90.0 %0".format(_rot)],
        "rx180 %0": ["{} 0.0 180.0 %0".format(_rot)],
        "ry180 %0": ["{} 90.0 180.0 %0".format(_rot)],
        "x %0": ["rx180 %0"],
        "y %0": ["ry180 %0"],
        "roty90 %0": ["ry90 %0"],
        "cnot %0,%1": ["ry90 %1", "cz %0,%1", "ry90 %1"],
        "CZ %0,%1": ["cz %0,%1"],

        # To support other forms of writing the same gates
        "x180 %0": ["rx180 %0"],
        "y180 %0": ["ry180 %0"],
        "y90 %0": ["ry90 %0"],
        "x90 %0": ["rx90 %0"],
        "my90 %0": ["rym90 %0"],
        "mx90 %0": ["rxm90 %0"],

        # Clifford decomposition per
        # Eptstein et al. Phys. Rev. A 89, 062321 (2014)
        "cl_0 %0": ['i %0'],
        "cl_1 %0": ['ry90 %0', 'rx90 %0'],
        "cl_2 %0": ['rxm90 %0', 'rym90 %0'],
        "cl_3 %0": ['rx180 %0'],
        "cl_4 %0": ['rym90 %0', 'rxm90 %0'],
        "cl_5 %0": ['rx90 %0', 'rym90 %0'],
        "cl_6 %0": ['ry180 %0'],
        "cl_7 %0": ['rym90 %0', 'rx90 %0'],
        "cl_8 %0": ['rx90 %0', 'ry90 %0'],
        "cl_9 %0": ['rx180 %0', 'ry180 %0'],
        "cl_10 %0": ['ry90 %0', 'rxm90 %0'],
        "cl_11 %0": ['rxm90 %0', 'ry90 %0'],

        "cl_12 %0": ['ry90 %0', 'rx180 %0'],
        "cl_13 %0": ['rxm90 %0'],
        "cl_14 %0": ['rx90 %0', 'rym90 %0', 'rxm90 %0'],
        "cl_15 %0": ['rym90 %0'],
        "cl_16 %0": ['rx90 %0'],
        "cl_17 %0": ['rx90 %0', 'ry90 %0', 'rx90 %0'],
        "cl_18 %0": ['rym90 %0', 'rx180 %0'],
        "cl_19 %0": ['rx90 %0', 'ry180 %0'],
        "cl_20 %0": ['rx90 %0', 'rym90 %0', 'rx90 %0'],
        "cl_21 %0": ['ry90 %0'],
        "cl_22 %0": ['rxm90 %0', 'ry180 %0'],
        "cl_23 %0": ['rx90 %0', 'ry90 %0', 'rxm90 %0']
    }

    decompositions = _dict_merge(
        rots_decomposition,
        decompositions_aliases
    )

    qubits = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']
    qubits_active = qubits_active or ['q0', 'q2']
    flux_tuples = [("q2", "q0"), ("q0", "q2"),
                   ("q0", "q3"), ("q3", "q0"),
                   ("q3", "q1"), ("q1", "q3"),
                   ("q1", "q4"), ("q4", "q1"),
                   ("q2", "q5"), ("q5", "q2"),
                   ("q5", "q3"), ("q3", "q5"),
                   ("q3", "q6"), ("q6", "q3"),
                   ("q6", "q4"), ("q4", "q6")]

    lut_kraus_map = (
        [('i {}', [np.array([[1, 0], [0, 1]], dtype=complex)])] +
        [(rot, kraus) for rot, kraus in zip(rots_instr, rots_kraus)]
    )
    assert len(lut_kraus_map) <= 32

    print("    Declaring following assembler commands:")
    for lut, kraus in lut_kraus_map:
        print(lut)
    print("    Declaring following gate decompositions:")
    for item in decompositions.items():
        print(item)

    cfg = {
        "eqasm_compiler": "cc_light_compiler",
        "hardware_settings": {
            "qubit_number": 7,
            "cycle_time": 20,
            "mw_mw_buffer": mw_mw_buffer,
            "mw_flux_buffer": mw_flux_buffer,
            "mw_readout_buffer": 0,
            "flux_mw_buffer": flux_mw_buffer,
            "flux_flux_buffer": 0,
            "flux_readout_buffer": 0,
            "readout_mw_buffer": 0,
            "readout_flux_buffer": 0,
            "readout_readout_buffer": 0
        },
        "instructions": {},
        "resources": {
            "qubits": {"count": 7},
            "qwgs": {
                "count": 3,
                "connection_map": {
                    "0": [0, 1],
                    "1": [2, 3, 4],
                    "2": [5, 6]
                }
            },
            "meas_units": {
                "count": 7,
                "connection_map": {
                    "0": [0],
                    "1": [1],
                    "2": [2],
                    "3": [3],
                    "4": [4],
                    "5": [5],
                    "6": [6]
                }
            },
            # FIXME OpenQL #103
            # "meas_units": {"count": 2,
            #                "connection_map": {"0": [0, 2, 3, 5, 6],
            #                                   "1": [1, 4]}
            #                },
            "edges": {
                "count": 8,
                "connection_map": {
                    "0": [2],
                    "1": [3],
                    "2": [0],
                    "3": [1],
                    "4": [6],
                    "5": [7],
                    "6": [4],
                    "7": [5]
                }
            }
        },

        "topology": {
            "x_size": 5,
            "y_size": 3,
            "qubits": [
                {"id": 0,  "x": 1, "y": 2},
                {"id": 1,  "x": 3, "y": 2},
                {"id": 2,  "x": 0, "y": 1},
                {"id": 3,  "x": 2, "y": 1},
                {"id": 4,  "x": 4, "y": 1},
                {"id": 5,  "x": 1, "y": 0},
                {"id": 6,  "x": 3, "y": 0}
            ],
            "edges": [
                {"id": 0,  "src": 2, "dst": 0},
                {"id": 1,  "src": 0, "dst": 3},
                {"id": 2,  "src": 3, "dst": 1},
                {"id": 3,  "src": 1, "dst": 4},
                {"id": 4,  "src": 2, "dst": 5},
                {"id": 5,  "src": 5, "dst": 3},
                {"id": 6,  "src": 3, "dst": 6},
                {"id": 7,  "src": 6, "dst": 4}
            ]
        },

        "gate_decomposition": decompositions,
    }

    for q in qubits_active:
        cfg["instructions"]["prepz {}".format(q)] = {
            "duration": init_duration,
            "latency": 0,
            "qubits": [q],
            "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "kraus_repr": None,
            "disable_optimization": True,
            "type": "none",
            "cc_light_instr_type": "single_qubit_gate",
            "cc_light_instr": "prepz",
            "cc_light_codeword": 0,
            "cc_light_opcode": 2
        }

    for q in qubits_active:
        cfg["instructions"]["measure {}".format(q)] = {
            "duration": ro_duration,
            "latency": ro_latency,
            "qubits": [q],
            "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "kraus_repr": None,
            "disable_optimization": False,
            "type": "readout",
            "cc_light_instr_type": "single_qubit_gate",
            "cc_light_instr": "measz",
            "cc_light_codeword": 0,
            "cc_light_opcode": 4
        }

    for cw, (instr, kraus) in enumerate(lut_kraus_map):
        for q in qubits_active:
            cfg["instructions"][instr.format(q)] = {
                "duration": mw_pulse_duration,
                "latency": mw_latency,
                "qubits": [q],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "kraus_repr": _kraus_jsonify(kraus),
                "disable_optimization": False,
                "type": "mw",
                "cc_light_instr_type": "single_qubit_gate",
                "cc_light_instr": "cw_{:02}".format(cw),
                "cc_light_codeword": cw,
                "cc_light_opcode": 8+cw}

            cfg["instructions"]['C1'+instr.format(q)] = {
                "duration": mw_pulse_duration,
                "latency": mw_latency,
                "qubits": [q],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "kraus_repr": None,
                "disable_optimization": False,
                "type": "mw",
                "cc_light_instr_type": "single_qubit_gate",
                "cc_light_instr": "C1_cw_{:02}".format(cw),
                "cc_light_codeword": cw,
                "cc_light_opcode": 32+8+cw,
                "cc_light_cond": 1}  # 1 means : do if last meas. == 1

            cfg["instructions"]['C0'+instr.format(q)] = {
                "duration": mw_pulse_duration,
                "latency": mw_latency,
                "qubits": [q],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "kraus_repr": None,
                "disable_optimization": False,
                "type": "mw",
                "cc_light_instr_type": "single_qubit_gate",
                "cc_light_instr": "C0_cw_{:02}".format(cw),
                "cc_light_codeword": cw,
                "cc_light_opcode": 32+16+cw,
                "cc_light_cond": 2}  # 2 means : do if last meas. == 0

    for cw in range(32):
        for q in qubits_active:
            cfg["instructions"]["cw_{:02} {}".format(cw, q)] = {
                "duration": mw_pulse_duration,
                "latency": mw_latency,
                "qubits": [q],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "kraus_repr": None,
                "disable_optimization": False,
                "type": "mw",
                "cc_light_instr_type": "single_qubit_gate",
                "cc_light_instr": "cw_{:02}".format(cw),
                "cc_light_codeword": cw,
                "cc_light_opcode": 8+cw}

    for q in qubits_active:
        cfg["instructions"]["compensate {}".format(q)] = {
            "duration": mw_pulse_duration,
            "latency": mw_latency,
            "qubits": [q],
            "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "kraus_repr": None,
            "disable_optimization": False,
            "type": "mw",
            "cc_light_instr_type": "single_qubit_gate",
            "cc_light_instr": "cw_00",
            "cc_light_codeword": 0,
            "cc_light_opcode": 8+0}

    # N.B. The codewords for CZ pulses need to be further specified.
    # I do not expect this to be correct for now.
    for ft in flux_tuples:
        if ft[0] in qubits_active and ft[1] in qubits_active:
            # FIXME add space back in
            cfg["instructions"]["cz {},{}".format(ft[0], ft[1])] = {
                "duration": flux_pulse_duration,
                "latency": fl_latency,
                "qubits": [ft[0], ft[1]],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "kraus_repr": _kraus_jsonify(_cphase_kraus(1.)),
                "disable_optimization": True,
                "type": "flux",
                "cc_light_instr_type": "two_qubits_gate",
                "cc_light_instr": "fl_cw_{:02}".format(1),
                "cc_light_right_codeword": 1,
                "cc_light_left_codeword": 1,
                "cc_light_opcode": 128+1
            }

    for cw_flux in range(8):
        for ft in flux_tuples:
            if ft[0] in qubits_active and ft[1] in qubits_active:
                cfg["instructions"]["fl_cw_{:02} {},{}".format(
                        cw_flux, ft[0], ft[1])] = {
                    "duration": flux_pulse_duration,
                    "latency": fl_latency,
                    "qubits": [ft[0], ft[1]],
                    "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                    "kraus_repr": _kraus_jsonify([
                        [[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., -1.]]
                    ]),
                    "disable_optimization": True,
                    "type": "flux",
                    "cc_light_instr_type": "two_qubits_gate",
                    "cc_light_instr": "fl_cw_{:02}".format(cw_flux),
                    "cc_light_right_codeword": cw_flux,
                    "cc_light_left_codeword": cw_flux,
                    "cc_light_opcode": 128+cw_flux
                }

    cfg['simulation_settings'] = {'error_models': {}}
    for qubit in qubits:
        cfg['simulation_settings']['error_models'][qubit] = {
            'error_model': 't1t2',
            't1': simulation_t1,
            't2': simulation_t2,
            'frac1_0': simulation_frac1_0,
            'frac1_1': simulation_frac1_1,
        }

    with open(filename, 'w') as f:
        json.dump(cfg, f, indent=4)
