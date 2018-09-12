import json


def generate_config(filename: str,
                    mw_pulse_duration: int = 20,
                    flux_pulse_duration: int=40,
                    ro_duration: int = 800,
                    mw_mw_buffer=0,
                    init_duration: int = 200000):
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

    qubits = ['q0', 'q1']
    lut_map = ['i {}', 'rx180 {}', 'ry180 {}', 'rx90 {}', 'ry90 {}',
               'rxm90 {}', 'rym90 {}']

    cfg = {
        "eqasm_compiler": "qumis_compiler",
        "hardware_settings": {
            "qubit_number": 2,
            "cycle_time": 5,
            "mw_mw_buffer": mw_mw_buffer,
            "mw_flux_buffer": 0,
            "mw_readout_buffer": 0,
            "flux_mw_buffer": 0,
            "flux_flux_buffer": 0,
            "flux_readout_buffer": 0,
            "readout_mw_buffer": 0,
            "readout_flux_buffer": 0,
            "readout_readout_buffer": 0},
        # initializing as an empty dict and then adding to it.
        "instructions": {},
        "resources": {},
        "topology": {},

        "gate_decomposition": {
            "x %0": ["rx180 %0"],
            "y %0": ["ry180 %0"],
            "roty90 %0": ["ry90 %0"],
            "cnot %0,%1": ["ry90 %1", "cz %0,%1", "ry90 %1"],
            # Clifford decomposition per Eptstein et al. Phys. Rev. A 89, 062321
            # (2014)
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
        },
    }

    for qnum, q in enumerate(qubits):
        cfg["instructions"]["prepz {}".format(q)] = {
            "duration": init_duration,
            "latency": 0,
            "qubits": [q],
            "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "disable_optimization": True,
            "type": "mw",
            "qumis_instr": "pulse",
            "qumis_instr_kw": {
               "codeword": 0,
               "awg_nr": 2
            }
        }

    for qnum, q in enumerate(qubits):
        cfg["instructions"]["measure {}".format(q)] = {
            "duration": ro_duration,
            "latency": 0,
            "qubits": [q],
            "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "disable_optimization": True,
            "type" : "readout",
            "qumis_instr": "trigger",
            "qumis_instr_kw": {
                "trigger_bit": 7,
                "trigger_duration": 10
            }
        }

    for CW in range(len(lut_map)):
        for qnum, q in enumerate(qubits):
            cfg["instructions"][lut_map[CW].format(q)] = {
                "duration": mw_pulse_duration,
                "latency": 0,
                "qubits": [q],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "disable_optimization": True,
                "type": "mw",
                "qumis_instr": "pulse",
                "qumis_instr_kw": {
                    "codeword": CW,
                    "awg_nr": qnum
                }
            }

    for CW in range(len(lut_map)):
        for qnum, q in enumerate(qubits):
            cfg["instructions"]["cw_{:02} {}".format(CW, q)] = {
                "duration": mw_pulse_duration,
                "latency": 0,
                "qubits": [q],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "disable_optimization": True,
                "type": "mw",
                "qumis_instr": "pulse",
                "qumis_instr_kw": {
                    "codeword": CW,
                    "awg_nr": qnum
                }
            }


    with open(filename, 'w') as f:
        json.dump(cfg, f, indent=4)
