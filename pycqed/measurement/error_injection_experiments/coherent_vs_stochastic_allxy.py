# -------------------------------------------
# Module containing measurement functionality that attempts to setup coherent vs. stochastic measurement error.
# -------------------------------------------
import numpy as np
import logging
from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon import CCLight_Transmon as Transmon
from pycqed.instrument_drivers.meta_instrument.LutMans.mw_lutman import AWG8_MW_LutMan
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8_PRNG import ZI_HDAWG8_PRNG


log = logging.getLogger(__name__)


def allxy_default_coherent_and_stochastic(
        qubit_idx: int,
        platf_cfg: str,
        coherence_codeword: str,
        stochastic_codeword: str = 'PRNG',
        double_points: bool = True,
        prepend_msmt=False,
        wait_time_after_prepend_msmt=0
    ):
    """
    Single qubit AllXY sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
        double_points:  if true repeats every element twice
                        intended for evaluating the noise at larger time scales
    Returns:
        p:              OpenQL Program object


    """
    p = OqlProgram("AllXY_default_coherent_stochastic", platf_cfg)
    normal_rotation_cw: str = stochastic_codeword  # stochastic_codeword  # coherence_codeword  # 'rx180'

    allXY = [['i', 'i'], [normal_rotation_cw, normal_rotation_cw], ['ry180', 'ry180'],
             [normal_rotation_cw, 'ry180'], ['ry180', normal_rotation_cw],
             ['rx90', 'i'], ['ry90', 'i'], ['rx90', 'ry90'],
             ['ry90', 'rx90'], ['rx90', 'ry180'], ['ry90', normal_rotation_cw],
             [normal_rotation_cw, 'ry90'], ['ry180', 'rx90'], ['rx90', normal_rotation_cw],
             [normal_rotation_cw, 'rx90'], ['ry90', 'ry180'], ['ry180', 'ry90'],
             [normal_rotation_cw, 'i'], ['ry180', 'i'], ['rx90', 'rx90'],
             ['ry90', 'ry90']]

    # this should be implicit
    if 0: # FIXME: p.set_sweep_points has been replaced by p.sweep_points, since that was missing here they are probably not necessary for this function
        p.set_sweep_points(np.arange(len(allXY), dtype=float))

    for i, xy in enumerate(allXY):
        if double_points:
            js = 2
        else:
            js = 1
        for j in range(js):
            k = p.create_kernel("AllXY_{}_{}".format(i, j))
            k.prepz(qubit_idx)
            if prepend_msmt:
                k.measure(qubit_idx)
                if wait_time_after_prepend_msmt:
                    k.gate("wait", [qubit_idx], wait_time_after_prepend_msmt)
                k.gate("wait", [])
            k.gate(xy[0], [qubit_idx])
            k.gate(xy[1], [qubit_idx])
            k.measure(qubit_idx)
            p.add_kernel(k)

    p.compile()
    return p


def measure_stochastic_allxy(
        transmon: Transmon,
        MC=None,
        label: str = '',
        analyze=True,
        close_fig=True,
        prepare_for_timedomain=True,
        prepend_msmt: bool = False,
        wait_time_after_prepend_msmt: int = 0,
        disable_metadata=False,
):
    if MC is None:
        MC = transmon.instr_MC.get_instr()
    instr_lutman: AWG8_MW_LutMan = transmon.instr_LutMan_MW.get_instr()
    instr_hdawg: ZI_HDAWG8 = instr_lutman.find_instrument(instr_lutman.AWG())

    coherent_codeword: str = "rx175"
    coherent_rotation: int = 175
    lutmap_index: int = 31
    instr_lutman.LutMap()[lutmap_index] = {"name": coherent_codeword, "theta": coherent_rotation, "phi": 0, "type": "ge"}
    log.warn(f"Overwriting {instr_lutman} lutmap, at index: {lutmap_index} with key: {coherent_codeword}.")
    instr_lutman.load_waveforms_onto_AWG_lookuptable()
    assert isinstance(instr_hdawg, ZI_HDAWG8_PRNG), "Requires codeword program that handles stochastic processing"

    if prepare_for_timedomain:
        transmon.prepare_for_timedomain()
    p = allxy_default_coherent_and_stochastic(
        qubit_idx=transmon.cfg_qubit_nr(),
        platf_cfg=transmon.cfg_openql_platform_fn(),
        coherence_codeword=coherent_codeword,
        stochastic_codeword='PRNG',
        double_points=True,
        prepend_msmt=prepend_msmt,
        wait_time_after_prepend_msmt=wait_time_after_prepend_msmt
    )
    s = swf.OpenQL_Sweep(
        openql_program=p,
        CCL=transmon.instr_CC.get_instr(),
    )
    d = transmon.int_avg_det
    MC.set_sweep_function(s)
    MC.set_sweep_points(np.arange(42) if not prepend_msmt else np.arange( 2 *42))
    MC.set_detector_function(d)
    MC.run('AllXY ' +transmon.msmt_suffix +label, disable_snapshot_metadata=disable_metadata)
    if analyze:
        a = ma.AllXY_Analysis(close_main_fig=close_fig, prepend_msmt=prepend_msmt)
    return a
