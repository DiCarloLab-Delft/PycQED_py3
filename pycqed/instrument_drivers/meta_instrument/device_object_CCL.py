"""
This file provides compatibility for existing code. The functionality of this file had been moved to HAL_Device.py
"""

# these imports just rename the new names to the legacy names
from .HAL_Device import HAL_Device as DeviceCCL
from .HAL_Device import _acq_ch_map_to_IQ_ch_map
