q0_name = 'ql'
q1_name = 'qr'


mw_pulse_duration = 40
RO_length = 6000

q0_RO_delay = 10
q1_RO_delay = 10
square_flux_duration_ns = 100
CZ_duration_ns = 200


def create_config():
    cfg = {
        "qubit_map": {q0_name: 0, q1_name: 1},
        "operation dictionary": {
            "x180": {
                "parameters": 1,
                "duration": mw_pulse_duration,
                "type": "rf",
                "matrix": []
            },
            "x90": {
                "parameters": 1,
                "duration": mw_pulse_duration,
                "type": "rf",
                "matrix": []
            },
            "y180": {
                "parameters": 1,
                "duration": mw_pulse_duration,
                "type": "rf",
                "matrix": []
            },

            "y90": {
                "parameters": 1,
                "duration": mw_pulse_duration,
                "type": "rf",
                "matrix": []
            },


            "mx90": {
                "parameters": 1,
                "duration": mw_pulse_duration,
                "type": "rf",
                "matrix": []
            },

            "my90": {
                "parameters": 1,
                "duration": mw_pulse_duration,
                "type": "rf",
                "matrix": []
            },

            "cz": {
                "parameters": 2,
                "duration": CZ_duration_ns,
                "type": "flux",
                "matrix": []
            },
            "square": {
                "parameters": 1,
                "duration": square_flux_duration_ns,
                "type": "flux",
                "matrix": []
            },
            "ro": {
                "parameters": 1,
                "duration": RO_length,
                "type": "measurement"
            }

        },

        "hardware specification": {
            "qubit list": [0, 1],
            "init time": 200000,
            "cycle time": 5,
            "qubit_cfgs": [
                {
                    "rf": {
                        "qumis": "pulse",
                        "latency": 0,
                        "awg_nr": 0,
                        "lut": 0
                    },
                    "flux": {
                        "qumis": "trigger",
                        "latency": mw_pulse_duration,
                        "trigger bit": 1,
                        "codeword bit": [1],
                        "format": [5, 10],
                        "lut": 1
                    },
                    "measurement": {
                        "qumis": "trigger",
                        "trigger bit": 7,
                        "format": [15],
                        "latency": q0_RO_delay
                    }
                },
                {
                    "rf": {
                        "qumis": "pulse",
                        "latency": 0,
                        "awg_nr": 2,
                        "lut": 0
                    },
                    "flux": {
                        "qumis": "pulse",
                        "latency": 0,
                        "awg_nr": 0,
                        "lut": 1
                    },
                    "measurement": {
                        "qumis": "trigger",
                        "trigger bit": 7,
                        "format": [15],
                        "latency": q1_RO_delay
                    }
                }

            ]
        },

        "luts": [
            {
                "x180": 1,
                "y180": 2,
                "x90": 3,
                "y90": 4,
                "mx90": 5,
                "my90": 6
            },
            {
                "cz": -2,
                "square": -2,
                "cz90": 1,
                "cz45": 2,
                "dummy_other": 3
            }
        ]
    }
    for i in range(32):
        cfg['luts'][1]['qwg_trigger_{}'.format(i)] = i
        cfg["operation dictionary"]["qwg_trigger_{}".format(i)] = {
            "parameters": 1,
            "duration": 15,
            "type": "flux",
            "matrix": []
        }
    return cfg
