from .defHeaders import *


# #################### Master controller headers ########################### #
UpdateModeHeader = None  # This command is replaced with UpdateMCWorkingState
UpdateMCWorkingState = b"\x5A"
UpdateSequencerParametersHeader = None       # sequencer is removed in CBox_v3


# Memory  Update transactions
LoadInstructionsHeader = b"\x42"

# Memory read transactions
ReadSequencerCounters = None                 # sequencer is removed in CBox_v3


# Modes index in list corresponds to the integer that will be sent
core_states = ['0: idle',
               '1: active']

acquisition_modes = ['0: idle',
                     '1: integration logging mode',
                     '2: integration averaging mode',
                     '3: input averaging mode',
                     '4: integration streaming mode']

trigger_sources = ['0: internal MC trigger',
                   '1: external MC trigger',
                   '2: mixed MC trigger']

demodulation_modes = ['0: double side band demodulation',
					  '1: single side band demodulation']