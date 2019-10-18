"""
This file imports all the relevant classes for daily use.
"""

# This snippet ensures all submodules get reloaded properly as we like to
# modify things when using it.
from importlib import reload

# Do not comment this out as other modules rely on this import being present
from pycqed.analysis import analysis_toolbox as a_tools

import pycqed.analysis_v2.base_analysis as ba
reload(ba)
import pycqed.analysis_v2.simple_analysis as sa
reload(sa)
import pycqed.analysis_v2.timedomain_analysis as ta
reload(ta)
import pycqed.analysis_v2.readout_analysis as ra
reload(ra)
import pycqed.analysis_v2.syndrome_analysis as synda
reload(sa)
# only one of these two files should exist in the end
import pycqed.analysis_v2.cryo_scope_analysis as csa
reload(csa)
import pycqed.analysis_v2.cryo_spectrumanalyzer_analysis as csa
reload(csa)
import pycqed.analysis_v2.distortions_analysis as da

import pycqed.analysis_v2.optimization_analysis as oa
reload(da)
import pycqed.analysis_v2.coherence_analysis as cs
reload(cs)
import pycqed.analysis_v2.spectroscopy_analysis as sa
reload(sa)
import pycqed.analysis_v2.dac_scan_analysis as da
reload(da)
import pycqed.analysis_v2.quantum_efficiency_analysis as qea
reload(qea)
import pycqed.analysis_v2.cross_dephasing_analysis as cda
reload(cda)
import pycqed.analysis_v2.randomized_benchmarking_analysis as rba
reload(rba)
# import pycqed.analysis_v2.gate_set_tomography_analysis as gsa
# reload(gsa)

import pycqed.analysis_v2.fluxing_analysis as fla
reload(fla)

import pycqed.analysis_v2.timing_cal_analysis as ta
reload(ta)


# from pycqed.analysis_v2.base_analysis import  # remove me if everything still works*
from pycqed.analysis_v2.simple_analysis import (
    Basic1DAnalysis, Basic1DBinnedAnalysis,
    Basic2DAnalysis, Basic2DInterpolatedAnalysis)
from pycqed.analysis_v2.timedomain_analysis import (
    FlippingAnalysis, Intersect_Analysis, CZ_1QPhaseCal_Analysis,
    Oscillation_Analysis,
    Conditional_Oscillation_Analysis, Idling_Error_Rate_Analyisis,
    Grovers_TwoQubitAllStates_Analysis)
from pycqed.analysis_v2.readout_analysis import Singleshot_Readout_Analysis

from pycqed.analysis_v2.multiplexed_readout_analysis import \
    Multiplexed_Readout_Analysis

from pycqed.analysis_v2.syndrome_analysis import (
    Single_Qubit_RoundsToEvent_Analysis, One_Qubit_Paritycheck_Analysis)


from pycqed.analysis_v2.cryo_scope_analysis import RamZFluxArc, \
    SlidingPulses_Analysis, Cryoscope_Analysis
from pycqed.analysis_v2.cryo_spectrumanalyzer_analysis import Cryospec_Analysis
from pycqed.analysis_v2.distortions_analysis import Scope_Trace_analysis

from pycqed.analysis_v2.optimization_analysis import OptimizationAnalysis
from pycqed.analysis_v2.timing_cal_analysis import Timing_Cal_Flux_Coarse, \
    Timing_Cal_Flux_Fine

from pycqed.analysis_v2.coherence_analysis import CoherenceAnalysis, \
    CoherenceTimesAnalysisSingle, AliasedCoherenceTimesAnalysisSingle, \
    CoherenceTimesAnalysis_old, CoherenceAnalysisDataExtractor
from pycqed.analysis_v2.spectroscopy_analysis import Spectroscopy, \
    ResonatorSpectroscopy, VNA_analysis, complex_spectroscopy, VNA_DAC_Analysis
from pycqed.analysis_v2.dac_scan_analysis import FluxFrequency, \
    Susceptibility_to_Flux_Bias, DACarcPolyFit
from pycqed.analysis_v2.quantum_efficiency_analysis import \
    QuantumEfficiencyAnalysis, DephasingAnalysisSingleScans, \
    DephasingAnalysisSweep, SSROAnalysisSingleScans, SSROAnalysisSweep, \
    QuantumEfficiencyAnalysisTWPA
from pycqed.analysis_v2.cross_dephasing_analysis import CrossDephasingAnalysis
from pycqed.analysis_v2.randomized_benchmarking_analysis import (
    RandomizedBenchmarking_SingleQubit_Analysis,
    RandomizedBenchmarking_TwoQubit_Analysis,
    UnitarityBenchmarking_TwoQubit_Analysis,
    InterleavedRandomizedBenchmarkingAnalysis,
    CharacterBenchmarking_TwoQubit_Analysis)
# from pycqed.analysis_v2.gate_set_tomography_analysis import \
#     GST_SingleQubit_DataExtraction, GST_TwoQubit_DataExtraction


from pycqed.analysis_v2.fluxing_analysis import Chevron_Analysis, Conditional_Oscillation_Heatmap_Analysis

