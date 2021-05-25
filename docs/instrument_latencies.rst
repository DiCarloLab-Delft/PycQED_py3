Feedback latencies in QuSurf architecture
-----------------------------------------

.. list-table:: Latencies
    :widths: 20 15 25 40
    :header-rows: 1

    *   -   Identifier
        -   Latency [ns]
        -   Condition
        -   Description
    *   -   **HDAWG**
        -
        -
        -
    *   -   tHdawgSyncDio
        -   ~10 (0-20)
        -   TBC
        -   synchronize incoming signal on DIO interface to 50 MHz grid. Depends on arrival time and DIO timing calibration
    *   -   tHdawgTriggerDio
        -   180
        -   HDAWG8 v2, filter disabled, no output delay
        -   delay from DIO trigger to first analog output. Depends on arrival time and DIO timing calibration
    *   -   tHdawgFilter
        -   30
        -
        -   extra delay if the filter is enabled at all (not bypassed)
    *   -   tHdawgFilterHighPass
        -   40
        -
        -   extra delay if high pass filter is enabled
    *   -   tHdawgFilterExpComp
        -   36.67
        -
        -   extra delay per enabled exponential compensation filter stage (8 stages available)
    *   -   tHdawgFilterBounceComp
        -   13.33
        -
        -   extra delay if bounce compensation filter is enabled
    *   -   tHdawgFilterFir
        -   56.67
        -
        -   extra delay if FIR filter is enabled
    *   -   tHdawgOutputDelay
        -   0-TBD
        -
        -   output delay configurable by user (/DEV..../SIGOUTS/n/DELAY)
    *   -
        -
        -
        -
    *   -   **QWG**
        -
        -
        -
    *   -   tQwgSyncDio
        -   ~10 (0-20)
        -
        -   synchronize incoming signal on DIO interface to 50 MHz grid. Depends on arrival time and DIO timing calibration
    *   -   tQwgTriggerDio
        -   80
        -   using LVDS input
        -   delay from DIO trigger to first analog output. Includes sideband modulation and mixer correction
    *   -
        -
        -
        -
    *   -   **UHFQA:AWG**
        -
        -
        -
    *   -   tUhfqaSyncDio
        -   ~10 (0-20)
        -   TBC
        -   synchronize incoming signal on DIO interface to 50 MHz grid. Depends on arrival time and DIO timing calibration
    *   -   tUhfqaTriggerDio
        -   314
        -   codewords: 5+default, firmware: 67225
        -   delay from DIO trigger to first analog output. Depends on number of codeword possibilities in sequencing program, and DIO arrival time and calibration
    *   -   tUhfqaWaveformPlay
        -   0-TBD
        -
        -   duration of the output waveform set by user
    *   -   tUhfqaOutputDelay
        -   0-TBD
        -
        -   output delay configurable by user
    *   -
        -
        -
        -
    *   -   **UHFQA:input**
        -
        -
        -
    *   -   tUhfqaIntegrationTime
        -   0-TBD
        -
        -   integration time set by user
    *   -   tUhfqaReadoutProcessing
        -   135.5
        -   Deskew, Rotation, and Crosstalk units bypassed
        -   delay between the end of a readout pulse at the Signal Inputs and the QA Result Trigger on any Trigger output
    *   -   tUhfqaDeskew
        -   ~8.8
        -
        -   delay introduced by enabling Deskew unit
    *   -   tUhfqaRotation
        -   ~57.7
        -
        -   delay introduced by enabling Rotation unit
    *   -   tUhfqaCrosstalk
        -   ~91.6
        -
        -   delay introduced by enabling Crosstalk unit
    *   -   tUhfqaReadoutProcessing
        -   293.3
        -   Deskew, Rotation, and Crosstalk units enabled
        -   delay between the end of a readout pulse at the Signal Inputs and the QA Result Trigger on any Trigger output
    *   -
        -
        -
        -
    *   -   tUhfqaHoldoff
        -
        -
        -   TBW
    *   -
        -
        -
        -
    *   -   **VSM**
        -
        -
        -
    *   -   tVsmDelay
        -   12
        -   VSM v3
        -   delay from digital input to signal starts turning on/off
    *   -   tVsmTransition
        -
        -
        -   transition time of VSM switch from on to off or vice versa
    *   -
        -
        -
        -
    *   -   **Central Controller**
        -
        -
        -
    *   -   tCcInputDio
        -   ~23
        -
        -   delay of DIO input interface and serializer
    *   -   tCcSyncDio
        -   ~10 (0-20)
        -
        -   synchronize incoming signal on DIO interface to 50 MHz grid. Depends on arrival time and DIO timing calibration
    *   -   tCcDistDsm
        -   20
        -
        -   read DIO interface and dispatch DSM data distribution
    *   -   tCcWaitDsm
        -   80
        -   S-17 (3 parallel 8 bit transfers)
        -   wait for DSM transfers to be completed
    *   -   tCcSyncDsm
        -
        -
        -
    *   -   tCcCondgate
        -   20
        -
        -   output a gate conditional on DSM data
    *   -   tCcOutputDio
        -   ~10
        -
        -   delay of serializer and DIO output interface
    *   -   **tCcDioToDio**
        -   **~163**
        -   S-17
        -   total latency from DIO data arriving to DIO output, depends on DIO timing calibration
    *   -
        -
        -
        -
    *   -   tCcCondBreak
        -   150
        -
        -   perform a break conditional on DSM data
    *   -
        -
        -
        -
    *   -   **System**
        -
        -
        -
    *   -   tSysReadoutRoundtrip
        -   ~40 ?
        -
        -   round trip delay from UHFQA signal output to UHFQA signal input: cables, mixers, filters, amplifiers

Information sources:

-   tHdawgTriggerDio: table 5.5 of https://docs.zhinst.com/pdf/ziHDAWG_UserManual.pdf (revision 21.02.0)
-   tHdawgFilter*: section 4.6.2 of same document
-   tQwg*: 20171511_pitch_qwg_final.pptx
-   tUhfqaReadoutProcessing: mail Niels H. 20210317, replaces ziUHFQA_UserManual.pdf (revision 21.02.01)
-   tUhfqaTriggerDio: measurement Miguel 20210519
-   tCc*: CC-SiteVisitVirtual-20200506.pptx


Notes:

-   20210319: measurement with CC from TRACE_DEV_OUT to TRACE_DEV_IN takes 1370 ns (274 ticks of 5 ns), with a measurement
    output signal duration of 780 ns (39 * 20 ns)), so total overhead is 1370 - 780 = 590 ns
    (tCcOutputDio + tCable + tUhfqaTriggerDio + tUhfqaReadoutProcessing + tCable + tCcInputDio)

