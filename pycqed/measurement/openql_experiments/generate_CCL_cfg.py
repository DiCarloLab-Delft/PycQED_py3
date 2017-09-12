import json

mw_pulse_duration = 20
RO_duration = 3500
init_duration = 200000

RO_latency = 0
Flux_latency_q0 = -80  # -95 the last 15 is incorporated in the flux lutman
Flux_latency_q1 = -80
MW_latency_q0 = -70
MW_latency_q1 = -70

square_flux_duration_ns = 100
CZ_duration_ns = 280  # 280


def generate_config(filename: str):

    qubits = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']
    lut_map = ['i {}', 'rX180 {}', 'rY180 {}', 'rX90 {}', 'rY90 {}',
               'rXm90 {}', 'rYm90 {}']
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
            "cycle_time": 5,
            "mw_mw_buffer": 0,
            "mw_flux_buffer": 0,
            "mw_readout_buffer": 20,
            "flux_mw_buffer": 0,
            "flux_flux_buffer": 0,
            "flux_readout_buffer": 0,
            "readout_mw_buffer": 0,
            "readout_flux_buffer": 0,
            "readout_readout_buffer": 0},
        "instructions": {},
    }

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
            "duration": RO_duration,
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
                "duration": 20,
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
                "duration": 20,
                "latency": 0,
                "qubits": [q],
                "matrix": [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                "disable_optimization": False,
                "type": "mw",
                "cc_light_instr_type": "single_qubit_gate",
                "cc_light_instr": "CW_{:02}".format(CW),
                "cc_light_codeword": CW,
                "cc_light_opcode": 30+CW}

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
