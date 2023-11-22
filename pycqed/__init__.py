# flake8: noqa (we don't need the "<...> imported but unused" error)

from pycqed.version import __version__
import sys
import warnings

if 1:   # FIXME: hack should be removed
    module_name = "qcodes"
    if module_name in sys.modules:
        # This is needed so that the `qcodes_QtPlot_monkey_patching` works for any
        # subsequent `qcodes` import
        raise ImportError("`pycqed` must be imported before `qcodes`! See "
            "__init__.py in `pycqed` folder for more information.\n"
            "NB: Any `qcodes` submodule must also be imported after pycqed.")
    # We need to import this here so that any later imports of `QtPlot` from qcodes
    # KEEP ABOVE any QtPlot import!!!
    warnings.warn(message="Attempts to apply qcodes QtPlot monkey patch.")
    try:
        from qcodes import plots
        from pycqed.measurement import qcodes_QtPlot_monkey_patching
    except ImportError as e:
        warnings.warn(message="Failed to apply qcodes QtPlot monkey patch.")

# from pycqed import measurement
# from pycqed import analysis
# from pycqed import instrument_drivers
# from pycqed import utilities


# # General PycQED modules
# from pycqed.measurement import measurement_control as mc
# from pycqed.measurement import sweep_functions as swf
# from pycqed.measurement import awg_sweep_functions as awg_swf
# from pycqed.measurement import detector_functions as det
# from pycqed.measurement import composite_detector_functions as cdet
# from pycqed.measurement import calibration_toolbox as cal_tools
# from pycqed.measurement import mc_parameter_wrapper as pw
# from pycqed.measurement import CBox_sweep_functions as cb_swf
# from pycqed.measurement.optimization import nelder_mead
# from pycqed.analysis import measurement_analysis as ma
# from pycqed.analysis import analysis_toolbox as a_tools
# from pycqed.utilities import general as gen
