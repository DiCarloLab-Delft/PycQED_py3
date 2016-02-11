'''
defHeaders.py
used by QuTech_ControlBox.py
===============================================================================
   Ver. |     Date   |  Author   | Remarks
--------+------------+-----------+---------------------------------------------
   0.1  |  unknown   |           | First version
--------+------------+-----------+---------------------------------------------
   0.2  |  15-1-2015 | FU Xiang  | * Removed CompareSign related commands.
--------+------------+-----------+---------------------------------------------
   0.3  |  16-1-2015 | FU Xiang  | * Added command updateLinTransCoeffHeader.
--------+------------+-----------+---------------------------------------------
   0.4  | 26-02-2015 |JdeSterke  | redefined header codes
--------+------------+-----------+---------------------------------------------
        |            |           |
===============================================================================
This file define the headers used to communicate with Master Controller.

Description :  Specifies the different headers

    +---+---+---+---+---+---+---+---+
    | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
    +---+---+---+---+---+---+---+---+
      |   |   |   |   |   |   |   |
      |   |   |   |   +---+---+---+--- Request/Command
      |   |   |   |                     Update commands start from 0 upwards
      |   |   |   |                     Read commands start from F downwards
      |   |   |   |                     This to ease differentiating between
      |   |   |   |                     read and write.
      |   |   |   +------------------- Transaction type
      |   |   |                         0 = Memory transaction
      |   |   |                         1 = Register transaction
      |   +---+----------------------- Destination
      |                                 00 = Not used
      |                                 01 = AWG
      |                                 10 = MC
      |                                 11 = Test and EndOfMessage
      +------------------------------- Always "0" indicates header byte
                                        ("1" indicates data)
'''


# bit 4 of the header indicates whether it is a register or memory transaction
MemoryTransaction = b"\x00"
RegisterTransaction = b"\x10"

# ######################### Awg headers #################################### #
AwgBaseAddress = b"\x20"

# Register update transactions
AwgOffsetHeader = b"\x31"
AwgDisableHeader = b"\x32"
AwgEnableHeader = b"\x33"
AwgModeHeader = b"\x34"
<<<<<<< HEAD

=======
>>>>>>> origin/master
AwgNoCodewordTrigHeader = b"\x34"
# FIXME: should be removed once integrated with AWGMode function
AwgRestartTapeHeader = b"\x35"

# Memory  Update transactions
AwgLUTHeader = b"\x21"
AwgTapeHeader = b"\x22"  # this header is obselete in timing tape AWG.
AwgCondionalTape = b"\x22"
AwgSegmentedTape = b"\x23"

# #################### Master controller headers ########################### #
MC_BaseAddress = b"\x40"

UpdAncillaTruthTableHeader = b"\x50"
UpdIntegrationDelayHeader = b"\x51"
UpdIntegrationLengthHeader = b"\x52"
UpdThresholdZeroHeader = b"\x53"
UpdThresholdOneHeader = b"\x54"
UpdVoffsetCalcDelaytimeHeader = b"\x55"
UpdVoffsetHeader = b"\x56"
UpdLinTransCoeffHeader = b"\x57"
UpdateAverageSettings = b"\x58"
UpdateLoggerMaxCounterHeader = b"\x59"

UpdateModeHeader = b"\x5A"
UpdateRunModeHeader = b"\x5B"
UpdateSequencerParametersHeader = b"\x5C"

# Register Read transactions
ReadCalculatedVoffset = b"\x5F"

# Memory  Update transactions
UpdWeightsZeroHeader = b"\x40"
UpdWeightsOneHeader = b"\x41"


# Memory read transactions
GetQubitStateLogResults = b"\x49"
GetQubitStateLogCounterResults = b"\x4A"
ReadSequencerCounters = b"\x4B"
ReadIntStreamingResults = b"\x4C"
ReadIntAverageResults = b"\x4D"
ReadInputAverageResults = b"\x4E"
ReadLoggedResults = b"\x4F"
ReadVersion = b"\x60"


# ########################## Communication headers ######################## #
EndOfDataTrailer = b"\x77"
EndOfStreamingHeader = b"\x78"
IllegalDataHeader = b"\x7C"
DataOverflowHeader = b"\x7D"
IllegalCommandHeader = b"\x7E"

# CommunicationErrorHeader = b"\x7E"
EndOfMessageHeader = b"\x7F"

# Test header to test fpga to PC communication
MC_TestHeader = b"\x40"

# Modes index in list corresponds to the integer that will be sent



acquisition_modes = ['0: idle',
                     '1: integration logging mode',

                     '2: feedback mode,',  # This mode does not do anything
                     '3: input averaging mode',
                     '4: integration averaging mode',




                     '5: integration streaming mode',
                     "6: touch 'n go"]

run_modes = ['0: idle',
             '1: Run mode']

awg_modes = ['0: Codeword-trigger mode',
             '1: No-codeword mode',
             '2: Tape mode']
