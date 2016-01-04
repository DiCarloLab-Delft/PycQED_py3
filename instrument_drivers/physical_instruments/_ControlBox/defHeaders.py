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
MemoryTransaction = "\x00"
RegisterTransaction = "\x10"

# ######################### Awg headers #################################### #
AwgBaseAddress = "\x20"

# Register update transactions
AwgOffsetHeader = "\x31"
AwgDisableHeader = "\x32"
AwgEnableHeader = "\x33"
AwgModeHeader = "\x34"
AwgNoCodewordTrigHeader = "\x34" # FIXME: should be removed once integrated with AWGMode function
AwgRestartTapeHeader = "\x35"

# Memory  Update transactions
AwgLUTHeader = "\x21"
AwgTapeHeader = "\x22"

# #################### Master controller headers ########################### #
MC_BaseAddress = "\x40"

UpdAncillaTruthTableHeader = "\x50"
UpdIntegrationDelayHeader = "\x51"
UpdIntegrationLengthHeader = "\x52"
UpdThresholdZeroHeader = "\x53"
UpdThresholdOneHeader = "\x54"
UpdVoffsetCalcDelaytimeHeader = "\x55"
UpdVoffsetHeader = "\x56"
UpdLinTransCoeffHeader = "\x57"
UpdateAverageSettings = "\x58"
UpdateLoggerMaxCounterHeader = "\x59"
UpdateModeHeader = "\x5A"
# EndOfStreaming = "\x5B"
UpdateRunModeHeader = "\x5B"
UpdateSequencerParametersHeader = "\x5C"
EndOfStreaming = "\x5B"   # FIXME: this header is not defined in the FPGA headers - JdS

# Register Read transactions
ReadCalculatedVoffset = "\x5F"

# Memory  Update transactions
UpdWeightsZeroHeader = "\x40"
UpdWeightsOneHeader = "\x41"

# Memory read transactions
ReadSequencerCounters = "\x4B"
ReadIntStreamingResults = "\x4C"
ReadIntAverageResults = "\x4D"
ReadInputAverageResults = "\x4E"
ReadLoggedResults = "\x4F"
ReadVersion = "\x60"
# ########################## Communication headers ######################## #
EndOfDataTrailer = "\x77"
EndOfStreamingHeader = "\x78"
IllegalDataHeader = "\x7C"
DataOverflowHeader = "\x7D"
IllegalCommandHeader = "\x7E"
#CommunicationErrorHeader = "\x7E"
EndOfMessageHeader = "\x7F"

# Test header to test fpga to PC communication
MC_TestHeader = "\x40"


