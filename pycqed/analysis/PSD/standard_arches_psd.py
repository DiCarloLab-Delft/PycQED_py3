import numpy as np
import lmfit
from matplotlib import pyplot as plt
import os
##############################################################################################
# These functions are VERY deprecated! Use the 'coherence_analysis from analysis_v2 instead! #
##############################################################################################


def PSD_Analysis(table, freq_resonator=None, Qc=None, chi_shift=None, path=None, plot: bool = True, verbose: bool = True):
    raise DeprecationWarning("This function is VERY deprecated! Use the 'coherence_analysis from analysis_v2 instead!")



def prepare_input_table(dac, frequency, T1, T2_star, T2_echo,
                        T1_mask=None, T2_star_mask=None, T2_echo_mask=None):
    raise DeprecationWarning("This function is VERY deprecated! Use the 'coherence_analysis from analysis_v2 instead!")