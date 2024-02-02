# -------------------------------------------
# Module describing the OpenQL schedule used for AC flux-crosstalk experiment
# -------------------------------------------
import numpy as np
from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram


def schedule_flux_crosstalk(
        qubit_echo_index: int,
        flux_pulse_cw: str,  # = 'fl_cw_06',
        platf_cfg: str = '',
        half_echo_delay_ns: int = 0,  # Delay in ns with respect to end of qubit gate
    ) -> OqlProgram:
    """
    """

    p = OqlProgram("FluxCrosstalk", platf_cfg)

    # Echo kernel
    k = p.create_kernel("echo")
    k.prepz(qubit_echo_index)
    k.barrier([])
    # Echo part 1
    k.gate('rx90', [qubit_echo_index])
    k.gate('wait', [], half_echo_delay_ns)
    k.barrier([])
    # Echo part 2
    k.gate('ry180', [qubit_echo_index])
    k.gate(flux_pulse_cw, [qubit_echo_index])
    k.barrier([])  # alignment workaround
    k.gate('wait', [], half_echo_delay_ns)
    k.gate('ry90', [qubit_echo_index])
    k.barrier([])
    # Measure
    k.measure(qubit_echo_index)
    p.add_kernel(k)

    # Calibration kernel
    # p.add_single_qubit_cal_points(qubit_idx=qubit_echo_index)

    p.compile()
    return p
