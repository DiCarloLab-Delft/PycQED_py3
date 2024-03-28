# -------------------------------------------
# Module containing measurement functionality that attempts to setup coherent vs. stochastic measurement error.
# -------------------------------------------
import numpy as np
import logging
from typing import List
from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon import CCLight_Transmon as Transmon
from pycqed.instrument_drivers.meta_instrument.LutMans.mw_lutman import AWG8_MW_LutMan
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8_PRNG import ZI_HDAWG8_PRNG
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8_PRNG import (
    ZI_HDAWG8_PRNG,
    StochasticTrigger,
)


log = logging.getLogger(__name__)


def ramsey_stochastic(
        qubit_idx: int,
        platf_cfg: str,
        probabilities: List[float],
        stochastic_codeword: str = 'PRNG',
    ):
    """
    """
    p = OqlProgram("Ramsey_stochastic", platf_cfg)

    for i, probs in enumerate(probabilities[:-4]):
        k = p.create_kernel("Ramsey_{}".format(i))
        k.prepz(qubit_idx)
        k.gate('rx90', [qubit_idx])
        # wait_nanoseconds = int(round(time/1e-9))
        k.gate(stochastic_codeword, [qubit_idx])
        # k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx90', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx)

    p.compile()
    return p


class OpenQLSweepUpdateHDAWG(swf.Soft_Sweep):

    def __init__(self, openql_program, CCL, awg: ZI_HDAWG8_PRNG, default_codeword: int, trigger_codeword: int,
                 parameter_name: str ='Points', unit: str='a.u.',
                 upload: bool=True):
        super().__init__()
        self.name = 'OpenQL_Sweep'
        self.openql_program = openql_program
        self.CCL = CCL
        self.upload = upload
        self.parameter_name = parameter_name
        self.sweep_control = 'soft'
        self.unit = unit
        self.awg = awg
        self.default_codeword = default_codeword
        self.trigger_codeword = trigger_codeword

    def prepare(self, **kw):
        print('Hi we prepare here')
        if self.upload:
            self.CCL.eqasm_program(self.openql_program.filename)

    def set_parameter(self, val):
        print('Hi we set-value here', f'probability: {val}')
        awg_sequencer_nr = 2  # X1 -> AWG8_8499
        stochastic_trigger_info = StochasticTrigger(
            awg_core_index=awg_sequencer_nr,
            trigger_probability=min(1.0, val),
            default_codeword=self.default_codeword,
            trigger_codeword=self.trigger_codeword,
        )
        self.awg.set_awg_stochastic_trigger(stochastic_trigger_info)
        self.awg.upload_codeword_program()


def measure_stochastic_ramsey(
        transmon: Transmon,
        label: str = '',
        MC=None,
        prepare_for_timedomain=True,
        disable_metadata=False,
):
    """
    """
    if MC is None:
        MC = transmon.instr_MC.get_instr()

    # append the calibration points, times are for location in plot
    probabilities = np.linspace(0.01, 1.0, 10)
    probabilities = np.concatenate([probabilities, [1.0] * 4])

    awg = transmon.instr_LutMan_MW.get_instr().AWG.get_instr()
    transmon.ro_acq_averages(2**11)

    if prepare_for_timedomain:
        transmon.prepare_for_timedomain()

    p = ramsey_stochastic(
        qubit_idx=transmon.cfg_qubit_nr(),
        platf_cfg=transmon.cfg_openql_platform_fn(),
        probabilities=probabilities
    )

    s = OpenQLSweepUpdateHDAWG(
        openql_program=p,
        CCL=transmon.instr_CC.get_instr(),
        awg=awg,
        default_codeword=0,
        trigger_codeword=1,
        parameter_name='Time',
        unit='s',
    )

    MC.set_sweep_function(s)
    MC.set_sweep_points(probabilities)
    MC.set_detector_function(transmon.int_avg_det)
    print(MC.get_datawriting_start_idx())
    response = MC.run('Ramsey_stochastic' + label + transmon.msmt_suffix, disable_snapshot_metadata=disable_metadata)
    return response
