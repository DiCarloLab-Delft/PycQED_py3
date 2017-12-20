"""
This is the file one imports for daily use.
This file should only contain import statements to import functions
from other files in the analysis_v2 module.
"""

# This snippet ensures all submodules get reloaded properly
from importlib import reload
import pycqed.analysis_v2.base_analysis as ba
reload(ba)
import pycqed.analysis_v2.simple_analysis as sa
reload(sa)
import pycqed.analysis_v2.timedomain_analysis as ta
reload(ta)
import pycqed.analysis_v2.readout_analysis as ra
reload(ra)
import pycqed.analysis_v2.syndrome_analysis as sa
reload(sa)
# only one of these two files should exist in the end
import pycqed.analysis_v2.cryo_scope_analysis as csa
reload(csa)
import pycqed.analysis_v2.cryo_scope_analysis_v2 as csa
reload(csa)
import pycqed.analysis_v2.distortions_analysis as da
reload(da)


from pycqed.analysis_v2.base_analysis import *
from pycqed.analysis_v2.simple_analysis import (
    Basic1DAnalysis, Basic2DAnalysis)
from pycqed.analysis_v2.timedomain_analysis import (
    FlippingAnalysis, Intersect_Analysis, CZ_1QPhaseCal_Analysis)
from pycqed.analysis_v2.readout_analysis import *
from pycqed.analysis_v2.syndrome_analysis import *
from pycqed.analysis_v2.cryo_scope_analysis import *
from pycqed.analysis_v2.cryo_scope_analysis_v2 import RamZFluxArc
from pycqed.analysis_v2.distortions_analysis import Scope_Trace_analysis
