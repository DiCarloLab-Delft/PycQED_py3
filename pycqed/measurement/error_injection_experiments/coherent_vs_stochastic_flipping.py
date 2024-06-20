# -------------------------------------------
# Module containing measurement functionality that attempts to setup coherent vs. stochastic measurement error.
# -------------------------------------------
import numpy as np
import logging
from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
from pycqed.measurement import sweep_functions as swf
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon import CCLight_Transmon as Transmon
from pycqed.instrument_drivers.meta_instrument.LutMans.mw_lutman import AWG8_MW_LutMan
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8_PRNG import ZI_HDAWG8_PRNG


log = logging.getLogger(__name__)



def flipping_default_coherent_and_stochastic(
        qubit_idx: int,
        number_of_flips,
        platf_cfg: str,
        coherence_codeword: str,
        stochastic_codeword: str = 'PRNG',
        equator: bool = True,
        cal_points: bool = True,
        ax: str = 'x',
        angle: str = '180',
    ) -> OqlProgram:
    """
    Generates a flipping sequence that performs multiple pi-pulses
    Basic sequence:
        - (X)^n - RO
        or
        - (Y)^n - RO
        or
        - (X90)^2n - RO
        or
        - (Y90)^2n - RO


    Input pars:
        qubit_idx:      int specifying the target qubit (starting at 0)
        number_of_flips: array of ints specifying the sweep points
        platf_cfg:      filename of the platform config file
        equator:        if True add an extra pi/2 pulse at the end to
                        make the state end at the equator.
        cal_points:     replaces last 4 points by calibration points

    Returns:
        p:              OpenQL Program object
    """
    p = OqlProgram("flipping_default_coherent_stochastic", platf_cfg)
    normal_rotation_cw: str = stochastic_codeword  # coherence_codeword  # 'rx180'

    for i, n in enumerate(number_of_flips):
        k = p.create_kernel('flipping_{}'.format(i))
        k.prepz(qubit_idx)
        if cal_points and (i == (len(number_of_flips)-4) or i == (len(number_of_flips)-3)):
            k.measure(qubit_idx)
        elif cal_points and (i == (len(number_of_flips)-2) or i == (len(number_of_flips)-1)):
            if ax == 'y':
                k.gate('ry180', [qubit_idx])
            else:
                k.gate(normal_rotation_cw, [qubit_idx])
            k.measure(qubit_idx)
        else:
            if equator:
                if ax == 'y':
                    k.gate('ry90', [qubit_idx])
                else:
                    k.gate('rx90', [qubit_idx])

            for j in range(n):
                if ax == 'y' and angle == '90':
                    k.gate('ry90', [qubit_idx])
                    k.gate('ry90', [qubit_idx])
                elif ax == 'y' and angle == '180':
                    k.gate('ry180', [qubit_idx])
                elif angle == '90':
                    k.gate('rx90', [qubit_idx])
                    k.gate('rx90', [qubit_idx])
                else:
                    k.gate(normal_rotation_cw, [qubit_idx])
            k.measure(qubit_idx)
        p.add_kernel(k)

    p.compile()
    return p


def measure_stochastic_flipping(
        transmon: Transmon,
        number_of_flips = np.arange(0, 61, 2),
        equator = True,
        prepare_for_timedomain = True,
        MC = None,
        ax = 'x',
        angle = '180',
        label = '',
        disable_metadata = False
    ):
    """
    Measurement for fine-tuning of the pi and pi/2 pulse amplitudes. Executes sequence
    pi (repeated N-times) - pi/2 - measure
    with variable number N. In this way the error in the amplitude of the MW pi pulse
    accumulate allowing for fine tuning. Alternatively N repetitions of the pi pulse
    can be replaced by 2N repetitions of the pi/2-pulse

    Args:
        number_of_flips (array):
            number of pi pulses to apply. It is recommended to use only even numbers,
            since then the expected signal has a sine shape. Otherwise it has -1^N * sin shape
            which will not be correctly analyzed.

        equator (bool);
            specify whether to apply the final pi/2 pulse. Setting to False makes the sequence
            first-order insensitive to pi-pulse amplitude errors.

        ax (str {'x', 'y'}):
            axis arour which the pi pulses are to be performed. Possible values 'x' or 'y'

        angle (str {'90', '180'}):r
            specifies whether to apply pi or pi/2 pulses. Possible values: '180' or '90'

        update (bool):
            specifies whether to update parameter controlling MW pulse amplitude.
            This parameter is mw_vsm_G_amp in VSM case or mw_channel_amp in no-VSM case.
            Update is performed only if change by more than 0.2% (0.36 deg) is needed.
    """

    if MC is None:
        MC = transmon.instr_MC.get_instr()
    instr_lutman: AWG8_MW_LutMan = transmon.instr_LutMan_MW.get_instr()
    instr_hdawg: ZI_HDAWG8 = instr_lutman.find_instrument(instr_lutman.AWG())

    coherent_codeword: str = "rx175"
    coherent_rotation: int = 175
    lutmap_index: int = 31
    instr_lutman.LutMap()[lutmap_index] = {"name": coherent_codeword, "theta": coherent_rotation, "phi": 0, "type": "ge"}
    instr_lutman.LutMap()[32] = {"name": "rx170", "theta": 180, "phi": 0, "type": "ge"}
    log.warn(f"Overwriting {instr_lutman} lutmap, at index: {lutmap_index} with key: {coherent_codeword}.")
    instr_lutman.load_waveforms_onto_AWG_lookuptable()

    # allow flipping only with pi/2 or pi, and x or y pulses
    assert angle in ['90', '180']
    assert ax.lower() in ['x', 'y']
    assert isinstance(instr_hdawg, ZI_HDAWG8_PRNG), "Requires codeword program that handles stochastic processing"

    # append the calibration points, times are for location in plot
    nf = np.array(number_of_flips)
    dn = nf[1] - nf[0]
    nf = np.concatenate([nf,
                         (nf[-1] + 1 * dn,
                          nf[-1] + 2 * dn,
                          nf[-1] + 3 * dn,
                          nf[-1] + 4 * dn)])
    if prepare_for_timedomain:
        transmon.prepare_for_timedomain()
    p = flipping_default_coherent_and_stochastic(
        number_of_flips=nf,
        equator = equator,
        qubit_idx = transmon.cfg_qubit_nr(),
        platf_cfg = transmon.cfg_openql_platform_fn(),
        coherence_codeword=coherent_codeword,
        stochastic_codeword='PRNG',
        ax = ax.lower(),
        angle = angle,
    )
    s = swf.OpenQL_Sweep(
        openql_program=p,
        unit = '#',
        CCL = transmon.instr_CC.get_instr()
    )
    d = transmon.int_avg_det
    MC.set_sweep_function(s)
    MC.set_sweep_points(nf)
    MC.set_detector_function(d)
    MC.run('flipping_' + ax + angle + label + transmon.msmt_suffix, disable_snapshot_metadata = disable_metadata)

    a = ma2.FlippingAnalysis(options_dict = {'scan_label': 'flipping'})
    return a
