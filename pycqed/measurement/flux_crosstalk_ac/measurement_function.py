# -------------------------------------------
# Module describing the measurement functionality used for AC flux-crosstalk experiment
# -------------------------------------------
import numpy as np
from pycqed.instrument_drivers.meta_instrument.device_object_CCL import DeviceCCL as Device
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon import CCLight_Transmon as Transmon
from pycqed.measurement.measurement_control import MeasurementControl as MeasurementControl
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC as CentralControl
from pycqed.instrument_drivers.meta_instrument.LutMans.flux_lutman_vcz import HDAWG_Flux_LutMan as FluxLutMan
from pycqed.measurement.sweep_functions import (
    FLsweep as FluxSweepFunctionObject,
    OpenQL_Sweep as OpenQLSweep,
)
from pycqed.measurement.flux_crosstalk_ac.schedule import (
    OqlProgram,
    schedule_flux_crosstalk,
    schedule_ramsey,
)


def measure_ac_flux_crosstalk(device: Device, qubit_echo_id: str, prepare_for_timedomain: bool = False, disable_metadata: bool = False):
    """
    Performs an experiment
    """
    # Data allocation
    qubit_echo: Transmon = device.find_instrument(qubit_echo_id)
    flux_lutman: FluxLutMan = qubit_echo.instr_LutMan_Flux.get_instr()
    times: np.ndarray = np.arange(0, 20e-6, 1e-6)  # Echo time
    # sweep_function: FluxSweepFunctionObject = FluxSweepFunctionObject(
    #     flux_lutman,
    #     flux_lutman.sq_length,  # Square length (qcodes) parameter
    #     waveform_name="square",  # Meaning we are going to sweep the square-pulse parameters only
    # )
    meas_control: MeasurementControl = device.instr_MC.get_instr()
    meas_control_nested: MeasurementControl = device.instr_nested_MC.get_instr()
    central_control: CentralControl = device.instr_CC.get_instr()

    # Prepare for time-domain if requested
    if prepare_for_timedomain:
        device.prepare_for_timedomain(qubits=[qubit_echo])

    # schedule: OqlProgram = schedule_flux_crosstalk(
    #     qubit_echo_index=qubit_echo.cfg_qubit_nr(),
    #     flux_pulse_cw='fl_cw_06',  # Square pulse?
    # )

    schedule: OqlProgram = schedule_ramsey(
        qubit_echo_index=qubit_echo.cfg_qubit_nr(),
        half_echo_delays=times,
        flux_pulse_cw='fl_cw_06',  # Square pulse?
    )

    sweep_function: OpenQLSweep = OpenQLSweep(
        openql_program=schedule,
        CCL=central_control,
        parameter_name='Time',
        unit='s',
    )

    meas_control.set_sweep_function(sweep_function)
    meas_control.set_sweep_points(times)
    meas_control.set_detector_function(qubit_echo.int_avg_det)
    label = f'FluxCrosstalk_{qubit_echo.name}'
    result = meas_control.run(label, disable_snapshot_metadata=disable_metadata)

    return result
