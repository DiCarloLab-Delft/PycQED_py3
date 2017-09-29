import json


def generate_config(filename: str,
                    mw_pulse_duration: int = 20,
                    ro_duration: int = 800,
                    mw_mw_buffer=20,
                    init_duration: int = 200000):
    """
    Generates a configuration file for OpenQL for use with the CCLight.
    Args:
        filename (str)          : location where to write the config json file
        mw_pulse_duration (int) : duration of the mw_pulses in ns.
            N.B. this should be 20 as the VSM marker is hardcoded to be of that
            length.
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

    qubits = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']
    lut_map = ['i {}', 'rX180 {}', 'rY180 {}', 'rX90 {}', 'rY90 {}',
               'rXm90 {}', 'rYm90 {}', 'spec {}']
    flux_tuples = [("q2", "q0"), ("q0", "q2"),
                   ("q0", "q3"), ("q3", "q0"),
                   ("q3", "q1"), ("q1", "q3"),
                   ("q1", "q4"), ("q4", "q1"),
                   ("q2", "q5"), ("q5", "q2"),
                   ("q5", "q3"), ("q3", "q5"),
                   ("q3", "q6"), ("q6", "q3"),
                   ("q6", "q4"), ("q4", "q6")]

    cfg = {
        "eqasm_compiler": "cc_light_compiler",
        "hardware_settings": {
            "qubit_number": 7,
            "cycle_time": 20,
            "mw_mw_buffer": mw_mw_buffer,
            "mw_flux_buffer": 0,
            "mw_readout_buffer": 0,
            "flux_mw_buffer": 0,
            "flux_flux_buffer": 0,
            "flux_readout_buffer": 0,
            "readout_mw_buffer": 0,
            "readout_flux_buffer": 0,
            "readout_readout_buffer": 0},
        "instructions": {},
        "resources":
            {"qubits": {"count": 7},
             "qwgs": {"count": 3,
                      "connection_map":
                      {
                          "0": [0, 1],
                          "1": [2, 3, 4],
                          "2": [5, 6]
                      }
                      },
             "meas_units": {"count": 2,
                            "connection_map": {"0": [0, 2, 3, 5, 6],
                                               "1": [1, 4]}
                            },
             "edges": {"count": 8,
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
            "topology":
            {
            "x_size": 5,
            "y_size": 3,
            "qubits":
            [
                {"id": 0,  "x": 1, "y": 2},
                {"id": 1,  "x": 3, "y": 2},
                {"id": 2,  "x": 0, "y": 1},
                {"id": 3,  "x": 2, "y": 1},
                {"id": 4,  "x": 4, "y": 1},
                {"id": 5,  "x": 1, "y": 0},
                {"id": 6,  "x": 3, "y": 0}
            ],
            "edges":
            [
                {"id": 0,  "src": 2, "dst": 0},
                {"id": 1,  "src": 0, "dst": 3},
                {"id": 2,  "src": 3, "dst": 1},
                {"id": 3,  "src": 1, "dst": 4},
                {"id": 4,  "src": 2, "dst": 5},
                {"id": 5,  "src": 5, "dst": 3},
                {"id": 6,  "src": 3, "dst": 6},
                {"id": 7,  "src": 6, "dst": 4}
            ]
        }, }

    # cfg["gate_decomposition"]: {
    #     "x q0": ["x q0"],
    #     "ry180 q0": ["ry180 q0"],
    #     "z q0": ["z q0"],
    #     "h q0": ["h q0"],
    #     "t q0": ["t q0"],
    #     "tdag q0": ["tdag q0"],
    #     "s q0": ["s q0"],
    #     "sdag q0": ["sdag q0"],
    #     "cnot q0,q1": ["cnot q0,q1"]
    # }

    for q in qubits:
        cfg["instructions"]["prepz {}".format(q)] = {
            "duration": init_duration,
            "latency": 0,
            "qubits": [q],
            "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "disable_optimization": True,
            "type": "mw",
            "cc_light_instr_type": "single_qubit_gate",
            "cc_light_instr": "prepz",
            "cc_light_codeword": 0,
            "cc_light_opcode": 2
        }

    for q in qubits:
        cfg["instructions"]["measure {}".format(q)] = {
            "duration": ro_duration,
            "latency": 0,
            "qubits": [q],
            "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "disable_optimization": False,
            "type": "readout",
            "cc_light_instr_type": "single_qubit_gate",
            "cc_light_instr": "measz",
            "cc_light_codeword": 0,
            "cc_light_opcode": 4
        }

    for CW in range(len(lut_map)):
        for q in qubits:
            cfg["instructions"][lut_map[CW].format(q)] = {
                "duration": mw_pulse_duration,
                "latency": 0,
                "qubits": [q],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "disable_optimization": False,
                "type": "mw",
                "cc_light_instr_type": "single_qubit_gate",
                "cc_light_instr": "CW_{:02}".format(CW),
                "cc_light_codeword": CW,
                "cc_light_opcode": 8+CW}

    for CW in range(32):
        for q in qubits:
            cfg["instructions"]["CW_{:02} {}".format(CW, q)] = {
                "duration": mw_pulse_duration,
                "latency": 0,
                "qubits": [q],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "disable_optimization": False,
                "type": "mw",
                "cc_light_instr_type": "single_qubit_gate",
                "cc_light_instr": "CW_{:02}".format(CW),
                "cc_light_codeword": CW,
                "cc_light_opcode": 8+CW}

    # N.B. The codewords for CZ pulses need to be further specified.
    # I do not expect this to be correct for now.
    for ft in flux_tuples:
        cfg["instructions"]["CZ {}, {}".format(ft[0], ft[1])] = {
            "duration": 80,
            "latency": 0,
            "qubits": [ft[0], ft[1]],
            "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "disable_optimization": True,
            "type": "flux",
            "cc_light_instr_type": "two_qubits_gate",
            "cc_light_instr": "cz",
            "cc_light_right_codeword": 1,
            "cc_light_left_codeword": 2,
            "cc_light_opcode": 128
        }

    with open(filename, 'w') as f:
        json.dump(cfg, f, indent=4)
