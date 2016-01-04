from time import sleep, time
import select
import socket
import qt
import numpy as np
import ctypes as ct
import types
import ctypes.util as ctu
import sys
import os
from instrument import Instrument
import logging
# import matplotlib.pyplot as plt

waittime = 0.01


def has_newline(ans):
    if len(ans) > 0 and ans.find('\n') != -1:
        return True
    return False


# initialize with
# qt.instruments.create(name='Signal_Hound',instype='SignalHound_USB_SA124B',dummy_instrument=True)
# dummy instrument because it does not use VISA
class SignalHound_USB_SA124B(Instrument):
    saStatus = {
        "saUnknownErr"                  : -666,
        "saFrequencyRangeErr"           : -99,
        "saInvalidDetectorErr"          : -95,
        "saInvalidScaleErr"             : -94,
        "saBandwidthErr"                : -91,
        "saExternalReferenceNotFound"   : -89,
        "saOvenColdErr"                 : -20,
        "saInternetErr"                 : -12,
        "saUSBCommErr"                  : -11,
        "saTrackingGeneratorNotFound"   : -10,
        "saDeviceNotIdleErr"            : -9,
        "saDeviceNotFoundErr"           : -8,
        "saInvalidModeErr"              : -7,
        "saNotConfiguredErr"            : -6,
        "saDeviceNotConfiguredErr"      : -6, # Added because key error raised
        "saTooManyDevicesErr"           : -5,
        "saInvalidParameterErr"         : -4,
        "saDeviceNotOpenErr"            : -3,
        "saInvalidDeviceErr"            : -2,
        "saNullPtrErr"                  : -1,
        "saNoError"                     : 0,
        "saNoCorrections"               : 1,
        "saCompressionWarning"          : 2,
        "saParameterClamped"            : 3
    }


    def __init__(self,name):
        self.log = logging.getLogger("Main.DeviceInt")
        logging.info(__name__ + ' : Initializing instrument SignalHound USB 124A')
        Instrument.__init__(self, name, tags=['physical'])
        self.dll = ct.CDLL("C:\Windows\System32\sa_api.dll")
        self.hf = constants()
        #parameter settings
        self.add_parameter('frequency', flags=Instrument.FLAG_GETSET, units='GHz',
                           type=float)
        self.add_parameter('span', flags=Instrument.FLAG_GETSET, units='GHz',
                           type=float)
        self.add_parameter('power', flags=Instrument.FLAG_GETSET, units='dBm',
                           maxval=20, type=float)
        self.add_parameter('ref_lvl', flags=Instrument.FLAG_GETSET, units='dBm',
                           maxval=20, type=float)
        self.add_parameter('external_reference', flags=Instrument.FLAG_GETSET,type=bool)
        self.add_parameter('device_type',type=bytes, flags=Instrument.FLAG_GETSET)
        self.add_parameter('device_mode',type=bytes, flags=Instrument.FLAG_GETSET)
        self.add_parameter('acquisition_mode',type=bytes, flags=Instrument.FLAG_GETSET)
        self.add_parameter('scale',type=bytes, flags=Instrument.FLAG_GETSET)
        self.add_parameter('running', flags=Instrument.FLAG_GETSET,type=bool)
        self.add_parameter('decimation', flags=Instrument.FLAG_GETSET,maxval=8,minval=1,type=int)

        self.add_parameter('bandwidth', flags=Instrument.FLAG_GETSET, units='Hz',
                           type=float)
        self.add_parameter('rbw', flags=Instrument.FLAG_GETSET, units='Hz',
                           type=float)
        self.add_parameter('vbw', flags=Instrument.FLAG_GETSET, units='Hz',
                           type=float)
        self.set_frequency(5)
        self.set_span(.25e-3)
        self.set_power(0)
        self.set_ref_lvl(0)
        self.set_external_reference(False)
        self.set_device_type('Not loaded')
        self.set_device_mode('sweeping')
        self.set_acquisition_mode('average')
        self.set_scale('log-scale')
        self.set_running(False)
        self.set_decimation(1)
        self.set_bandwidth(0)
        self.set_rbw(1e3)
        self.set_vbw(1e3)
        self.openDevice()

    def openDevice(self):
        self.log.info("Opening Device")
        self.deviceHandle = ct.c_int(0)
        deviceHandlePnt = ct.pointer(self.deviceHandle)
        ret = self.dll.saOpenDevice(deviceHandlePnt)
        if ret != self.hf.saNoError:
            if ret == self.hf.saNullPtrErr:
                raise ValueError("Could not open device due to null-pointer error!")
            elif ret == self.hf.saDeviceNotOpenErr:
                raise ValueError("Could not open device!")
            else:
                raise ValueError("Could not open device due to unknown reason! Error = %d" % ret)

        self.devOpen = True
        self._devType = self.get_device_type()
        self.log.info("Opened Device with handle num: %s", self.deviceHandle.value)


    def closeDevice(self):
        self.log.info("Closing Device with handle num: %s", self.deviceHandle.value)

        try:
            self.dll.saAbort(self.deviceHandle)
            self.log.info("Running acquistion aborted.")
        except Exception as e:
            self.log.info("Could not abort acquisition: %s", e)

        ret = self.dll.saCloseDevice(self.deviceHandle)
        if ret != self.hf.saNoError:
            raise ValueError("Error closing device!")
        self.log.info("Closed Device with handle num: %s", self.deviceHandle.value)
        self.devOpen = False
        self.running = False

    def abort(self):
        self.log.info("Stopping acquisition")

        err = self.dll.saAbort(self.deviceHandle)
        if err == self.saStatus["saNoError"]:
            self.log.info("Call to abort succeeded.")
            self.running = False
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        elif err == self.saStatus["saDeviceNotConfiguredErr"]:
            raise IOError("Device was already idle! Did you call abort without ever calling initiate()?")
        else:
            raise IOError("Unknown error setting abort! Error = %s" % err)

    def preset(self):
        self.log.warning("Performing hardware-reset of device!")
        self.log.warning("Please ensure you close the device handle within two seconds of this call!")

        err = self.dll.saPreset(self.deviceHandle)
        if err == self.saStatus["saNoError"]:
            self.log.info("Call to preset succeeded.")
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        else:
            raise IOError("Unknown error calling preset! Error = %s" % err)



    def _do_get_frequency(self):
        return self.frequency

    def _do_set_frequency(self,freq):
        self.frequency = freq

    def _do_get_span(self):
        return self.span
    def _do_set_span(self,span):
        self.span = span

    def _do_get_power(self):
        return self.power
    def _do_set_power(self,power):
        self.power = power

    def _do_get_ref_lvl(self):
        return self.ref_lvl

    def _do_set_ref_lvl(self,ref_lvl):
        self.ref_lvl = ref_lvl

    def _do_get_external_reference(self):
        return self.external_reference

    def _do_set_external_reference(self,external_reference):
        self.external_reference = external_reference

    def _do_get_running(self):
        return self.running
    def _do_set_running(self,running):
        self.running = running

    def _do_get_device_type(self):
        self.log.info("Querying device for model information")

        devType = ct.c_uint(0)
        devTypePnt = ct.pointer(devType)

        err = self.dll.saGetDeviceType(self.deviceHandle, devTypePnt)
        if err == self.saStatus["saNoError"]:
            pass
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        elif err == self.saStatus["saNullPtrErr"]:
            raise IOError("Null pointer error!")
        else:
            raise IOError("Unknown error setting getDeviceType! Error = %s" % err)

        if devType.value == self.hf.saDeviceTypeNone:
            dev = "No device"
        elif devType.value == self.hf.saDeviceTypeSA44:
            dev = "sa44"
        elif devType.value == self.hf.saDeviceTypeSA44B:
            dev = "sa44B"
        elif devType.value == self.hf.saDeviceTypeSA124A:
            dev = "sa124A"
        elif devType.value == self.hf.saDeviceTypeSA124B:
            dev = 'sa124B'
        else:
            raise ValueError("Unknown device type!")
        return dev

    def _do_set_device_type(self, device_type):
        self.device_type = device_type

    def _do_get_device_mode(self):
        return self.device_mode
    def _do_set_device_mode(self,device_mode):
        self.device_mode = device_mode
        return

    def _do_get_acquisition_mode(self):
        return self.acquisition_mode
    def _do_set_acquisition_mode(self,acquisition_mode):
        self.acquisition_mode = acquisition_mode
        return

    def _do_get_scale(self):
        return self.scale
    def _do_set_scale(self,scale):
        self.scale = scale

    def _do_get_decimation(self):
        return self.decimation
    def _do_set_decimation(self,decimation):
        self.decimation = decimation

    def _do_get_bandwidth(self):
        return self.bandwidth
    def _do_set_bandwidth(self,bandwidth):
        self.bandwidth = bandwidth

    def _do_get_rbw(self):
        return self.rbw
    def _do_set_rbw(self,rbw):
        self.rbw = rbw

    def _do_get_vbw(self):
        return self.vbw
    def _do_set_vbw(self,vbw):
        self.vbw = vbw


############################################################################



    def initialisation(self,flag=0):
        mode=self.device_mode
        modeOpts = {
            "sweeping"       : self.hf.sa_SWEEPING,
            "real_time"      : self.hf.sa_REAL_TIME,
            "IQ"             : self.hf.sa_IQ,
            "idle"           : self.hf.sa_IDLE
        }
        if mode in modeOpts:
            mode = modeOpts[mode]
        else:
            raise ValueError("Mode must be one of %s. Passed value was %s." % (modeOpts, mode))
        err = self.dll.saInitiate(self.deviceHandle, mode, flag)
        if err == self.saStatus["saNoError"]:
            self.running = True
            self.log.info("Call to initiate succeeded.")
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        elif err == self.saStatus["saInvalidParameterErr"]:
            self.log.error("saInvalidParameterErr!")
            self.log.error('''In real-time mode, this value may be returned if the span limits defined in the API header are broken. Also in real-time mode, this error will be
                returned if the resolution bandwidth is outside the limits defined in the API header.''')
            self.log.error('''In time-gate analysis mode this error will be returned if span limits defined in the API header are broken. Also in time gate analysis, this
                error is returned if the bandwidth provided require more samples for processing than is allowed in the gate length. To fix this, increase rbw/vbw.''')
            raise IOError("The value for mode did not match any known value.")
        elif err == self.saStatus["saAllocationLimitError"]:
            self.log.error('''This value is returned in extreme circumstances. The API currently limits the amount of RAM usage to 1GB. When exceptional parameters are
                provided, such as very low bandwidths, or long sweep times, this error may be returned. At this point you have reached the boundaries of the
                device. The processing algorithms are optimized for speed at the expense of space, which is the reason this can occur.''')
            raise IOError("Could not allocate sufficent RAM!")
        elif err == self.saStatus["saBandwidthErr"]:
            raise IOError("RBW is larger than your span. (Sweep Mode)!")
        else:
            raise IOError("Unknown error setting initiate! Error = %s" % err)

        return

    def QuerySweep(self):
        sweep_len = ct.c_int(0)
        start_freq=ct.c_double(0)
        stepsize=ct.c_double(0)
        err = self.dll.saQuerySweepInfo(self.deviceHandle, ct.pointer(sweep_len), ct.pointer(start_freq), ct.pointer(stepsize))
        if err == self.saStatus["saNoError"]:
            pass
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        elif err == self.saStatus["saDeviceNotConfiguredErr"]:
            raise IOError("The device specified is not currently streaming!")
        elif err == self.saStatus["saNullPtrErr"]:
            raise IOError("Null pointer error!")
        else:
            raise IOError("Unknown error!")

        info = np.array([sweep_len.value,start_freq.value,stepsize.value])
        return info

    def configure(self,rejection=True):
        decimation = self.decimation
        # bandwidth=self.bandwidth
        ref_lvl = self.ref_lvl
        external_reference = self.external_reference
        rbw = self.rbw
        vbw = self.vbw

    # CenterSpan Configuration


        frequency = self.frequency * 1e9
        span = self.span * 1e9
        center = ct.c_double(frequency)
        span   = ct.c_double(span)
        self.log.info("Setting device CenterSpan configuration.")
        err = self.dll.saConfigCenterSpan(self.deviceHandle, center, span)
        if err == self.saStatus["saNoError"]:
            self.log.info("Call to configureCenterSpan succeeded.")
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        elif err == self.saStatus["saFrequencyRangeErr"]:
            raise IOError("The calculated start or stop frequencies fall outside of the operational frequency range of the specified device.")
        else:
            raise IOError("Unknown error setting configureCenterSpan! Error = %s" % err)

    # Acquisition configuration
        detectorVals = {
            "min-max" : ct.c_uint(self.hf.sa_MIN_MAX),
            "average" : ct.c_uint(self.hf.sa_AVERAGE)
        }
        scaleVals = {
            "log-scale"      : ct.c_uint(self.hf.sa_LOG_SCALE),
            "lin-scale"      : ct.c_uint(self.hf.sa_LIN_SCALE),
            "log-full-scale" : ct.c_uint(self.hf.sa_LOG_FULL_SCALE),
            "lin-full-scale" : ct.c_uint(self.hf.sa_LIN_FULL_SCALE)
        }
        if self.acquisition_mode in detectorVals:
            detector = detectorVals[self.acquisition_mode]
        else:
            raise ValueError("Invalid Detector mode! Detector  must be one of %s. Specified detector = %s" % (list(detectorVals.keys()), detector))
        if self.scale in scaleVals:
            scale = scaleVals[self.scale]
        else:
            raise ValueError("Invalid Scaling mode! Scaling mode must be one of %s. Specified scale = %s" % (list(scaleVals.keys()), scale))
        err = self.dll.saConfigAcquisition(self.deviceHandle, detector, scale)
        if err == self.saStatus["saNoError"]:
            self.log.info("Call to configureAcquisition succeeded.")
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        elif err == self.saStatus["saInvalidDetectorErr"]:
            raise IOError("Invalid Detector mode!")
        elif err == self.saStatus["saInvalidScaleErr"]:
            raise IOError("Invalid scale setting error!")
        else:
            raise IOError("Unknown error setting configureAcquisition! Error = %s" % err)

    # Reference Level configuration

        ref = ct.c_double(ref_lvl)
        # atten = ct.c_double(atten)
        self.log.info("Setting device reference level configuration.")
        err = self.dll.saConfigLevel(self.deviceHandle, ct.c_double(ref_lvl))
        if err == self.saStatus["saNoError"]:
            self.log.info("Call to configureLevel succeeded.")
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        elif err == self.saStatus["saParameterClamped"]:
            raise IOError("The attenuation value provided exceeds 20 db.")
        else:
            raise IOError("Unknown error setting configureLevel! Error = %s" % err)
    # External Reference configuration
        if self.external_reference:
            self.log.info("Setting reference frequency from external source.")
            err = self.dll.saEnableExternalReference(self.deviceHandle)
            if err == self.saStatus["saNoError"]:
                pass
            elif err == self.saStatus["saInvalidDeviceErr"]:
                raise IOError("Can not set an external reference on this model!")
            elif err == self.saStatus["saExternalReferenceNotFound"]:
                raise IOError("No external reference found!")
            else:
                raise IOError("Unknown error!")

        if self.device_mode =='sweeping':
            # Sweeping Configuration
            rbw        = self.rbw
            vbw        = self.vbw
            reject_var = ct.c_bool(rejection)
            self.log.info("Setting device Sweeping configuration.")
            err = self.dll.saConfigSweepCoupling(self.deviceHandle, ct.c_double(rbw), ct.c_double(vbw), reject_var)
            if err == self.saStatus["saNoError"]:
                self.log.info("configureSweepCoupling Succeeded.")
            elif err == self.saStatus["saDeviceNotOpenErr"]:
                raise IOError("Device not open!")
            elif err == self.saStatus["saBandwidthErr"]:
                raise IOError("'rbw' falls outside device limits or 'vbw' is greater than resolution bandwidth.")
            elif err == self.saStatus["saInvalidBandwidthTypeErr"]:
                raise IOError("'rbwType' is not one of the accepted values.")
            elif err == self.saStatus["saInvalidParameterErr"]:
                raise IOError("'rejection' value is not one of the accepted values.")
            else:
                raise IOError("Unknown error setting configureSweepCoupling! Error = %s" % err)
        elif self.device_mode == 'IQ':
            err = self.dll.saConfigIQ(self.deviceHandle, ct.c_int(decimation), ct.c_double(bandwidth))
            if err == self.saStatus["saNoError"]:
                raise IOError("configureSweepCoupling Succeeded.")
            elif err == self.saStatus["saDeviceNotOpenErr"]:
                raise IOError("Device not open!")
            elif err == self.saStatus["saInvalidParameterErr"]:
                raise IOError("Invalid Parameter passed!")
            else:
                raise IOError("Unknown error setting configureSweepCoupling! Error = %s" % err)
        return

    def sweep(self):
        # this needs an initialized device. Originally used by read_power()
        sweep_len = ct.c_int(0)
        start_freq = ct.c_double(0)
        stepsize = ct.c_double(0)
        err = self.dll.saQuerySweepInfo(self.deviceHandle,
                                        ct.pointer(sweep_len),
                                        ct.pointer(start_freq),
                                        ct.pointer(stepsize))
        if not err == self.saStatus['saNoError']:
            # if an error occurs tries preparing the device and then asks again
            print('Error raised in Sweep Info Query, preparing for measurement')
            qt.msleep(.5)
            self.prepare_for_measurement()
            qt.msleep(.5)
            err = self.dll.saQuerySweepInfo(self.deviceHandle,
                                            ct.pointer(sweep_len),
                                            ct.pointer(start_freq),
                                            ct.pointer(stepsize))
        if err == self.saStatus["saNoError"]:
            pass
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        elif err == self.saStatus["saDeviceNotConfiguredErr"]:
            raise IOError("The device specified is not currently streaming!")
        elif err == self.saStatus["saNullPtrErr"]:
            raise IOError("Null pointer error!")
        else:
            raise IOError("Unknown error!")
        end_freq = start_freq.value + stepsize.value*sweep_len.value
        freq_points = np.arange(start_freq.value*1e-9, end_freq*1e-9,
                                stepsize.value*1e-9)

        minarr = (ct.c_float * sweep_len.value)()
        maxarr = (ct.c_float * sweep_len.value)()
        err = self.dll.saGetSweep_32f(self.deviceHandle, minarr, maxarr)

        if not err == self.saStatus['saNoError']:
            # if an error occurs tries preparing the device and then asks again
            print('Error raised in Sweep Info Query, preparing for measurement')
            qt.msleep(.5)
            self.prepare_for_measurement()
            qt.msleep(.5)
            minarr = (ct.c_float * sweep_len.value)()
            maxarr = (ct.c_float * sweep_len.value)()
            err = self.dll.saGetSweep_32f(self.deviceHandle, minarr, maxarr)

        if err == self.saStatus["saNoError"]:
            pass
        elif err == self.saStatus["saDeviceNotOpenErr"]:
            raise IOError("Device not open!")
        elif err == self.saStatus["saDeviceNotConfiguredErr"]:
            raise IOError("The device specified is not currently streaming!")
        elif err == self.saStatus["saNullPtrErr"]:
            raise IOError("Null pointer error!")
        elif err == self.saStatus["saInvalidModeErr"]:
            raise IOError("Invalid mode error!")
        elif err == self.saStatus["saCompressionWarning"]:
            raise IOError("Input voltage overload!")
        elif err == self.saStatus["sCUSBCommErr"]:
            raise IOError("Error ocurred in the USB connection!")
        else:
            raise IOError("Unknown error!")

        datamin = np.array([minarr[elem] for elem in range(sweep_len.value)])
        datamax = np.array([minarr[elem] for elem in range(sweep_len.value)])
        info = np.array([sweep_len.value, start_freq.value,
                        stepsize.value])

        return np.array([freq_points[0:sweep_len.value-1],
                        datamin, datamax, info])

    def get_power_at_freq(self, Navg=1):
        '''
        Returns the maximum power in a window of 250kHz
        around the specified  frequency.
        The integration window is specified by the VideoBandWidth (set by vbw)
        '''
        poweratfreq = 0
        for i in range(Navg):
            qt.msleep(.1)  # is this delay long enough to work with the vbw
            data = self.sweep()
            max_power = np.max(data[1][:])
            poweratfreq += max_power
        self.power = poweratfreq / Navg
        return self.power

    def get_spectrum(self, Navg=1):
        sweep_params = self.QuerySweep()
        data_spec = np.zeros(sweep_params[0])
        for i in range(Navg):
            data = self.sweep()
            data_spec[:] += data[1][:]
        data_spec[:] = data_spec[:] / Navg
        sweep_points = data[0][:]
        return np.array([data_spec, sweep_points])

    def prepare_for_measurement(self):
        self.set_device_mode('sweeping')
        self.configure()
        self.initialisation()
        return

    def safe_reload(self):
        self.closeDevice()
        self.reload()


class constants():
    def __init__(self):
        self.SA_MAX_DEVICES = 8

        self.saDeviceTypeNone = 0
        self.saDeviceTypeSA44 = 1
        self.saDeviceTypeSA44B = 2
        self.saDeviceTypeSA124A= 3
        self.saDeviceTypeSA124B = 4

        self.sa44_MIN_FREQ = 1.0
        self.sa124_MIN_FREQ = 100.0e3
        self.sa44_MAX_FREQ = 4.4e9
        self.sa124_MAX_FREQ = 13.0e9
        self.sa_MIN_SPAN = 1.0
        self.sa_MAX_REF = 20
        self.sa_MAX_ATTEN = 3
        self.sa_MAX_GAIN = 2
        self.sa_MIN_RBW = 0.1
        self.sa_MAX_RBW = 6.0e6
        self.sa_MIN_RT_RBW = 100.0
        self.sa_MAX_RT_RBW = 10000.0
        self.sa_MIN_IQ_BANDWIDTH = 100.0
        self.sa_MAX_IQ_DECIMATION = 128

        self.sa_IQ_SAMPLE_RATE = 486111.111

        self.sa_IDLE      = -1
        self.sa_SWEEPING  = 0x0
        self.sa_REAL_TIME = 0x1
        self.sa_IQ        = 0x2
        self.sa_AUDIO     = 0x3
        self.sa_TG_SWEEP  = 0x4

        self.sa_MIN_MAX = 0x0
        self.sa_AVERAGE = 0x1

        self.sa_LOG_SCALE      = 0x0
        self.sa_LIN_SCALE      = 0x1
        self.sa_LOG_FULL_SCALE = 0x2
        self.sa_LIN_FULL_SCALE = 0x3

        self.sa_AUTO_ATTEN = -1
        self.sa_AUTO_GAIN  = -1

        self.sa_LOG_UNITS   = 0x0
        self.sa_VOLT_UNITS  = 0x1
        self.sa_POWER_UNITS = 0x2
        self.sa_BYPASS      = 0x3

        self.sa_AUDIO_AM  = 0x0
        self.sa_AUDIO_FM  = 0x1
        self.sa_AUDIO_USB = 0x2
        self.sa_AUDIO_LSB = 0x3
        self.sa_AUDIO_CW  = 0x4

        self.TG_THRU_0DB  = 0x1
        self.TG_THRU_20DB  = 0x2

        self.saUnknownErr = -666

        self.saFrequencyRangeErr = -99
        self.saInvalidDetectorErr = -95
        self.saInvalidScaleErr = -94
        self.saBandwidthErr = -91
        self.saExternalReferenceNotFound = -89

        self.saOvenColdErr = -20

        self.saInternetErr = -12
        self.saUSBCommErr = -11

        self.saTrackingGeneratorNotFound = -10
        self.saDeviceNotIdleErr = -9
        self.saDeviceNotFoundErr = -8
        self.saInvalidModeErr = -7
        self.saNotConfiguredErr = -6
        self.saTooManyDevicesErr = -5
        self.saInvalidParameterErr = -4
        self.saDeviceNotOpenErr = -3
        self.saInvalidDeviceErr = -2
        self.saNullPtrErr = -1

        self.saNoError = 0

        self.saNoCorrections = 1
        self.saCompressionWarning = 2
        self.saParameterClamped = 3
        self.saBandwidthClamped = 4



'''
######################### OLD FUNCTIONS ###################################################
    def setup_LO(self,centerFrequency,devNum=0):
        self.S.write('SetupLO('+str(centerFrequency)+','+str(self.mixerband(centerFrequency))+','+str(devNum)+')')
        return self.S.read()

    def write(self,string):
        print string
        self.S.write(string)

    def read(self):
        print self.S.read()


    def start_streaming(self,devNum=0):
        self.S.write('StartStreamingData('+str(devNum)+')')
        return self.S.read()

    def stop_streaming(self,devNum=0):
        self.S.write('StopStreamingData('+str(devNum)+')')
        return self.S.read()


    def get_data_packet(self,devNum=0):
        ''
        It's required to empty the buffer of the Signal Hound before taking
        a data point, else you will measure some old data. The buffer
        contains 18 data points and seems no to refresh all the time
        when data is being streamed.
        ''
        for i in range(18):
            self.S.write('GetStreamingPacket('+str(devNum)+')')
            self.S.read()
        self.S.write('GetStreamingPacket('+str(devNum)+')')
        return float(self.S.read())


    def mixerband(self,CenterFrequency):
        ''
        This is to choose the correct mixer band. Note that this is poorly
        documented information fluthi obtained by personal correspondence
        with Justin Crooks from SignalHound, Dec. 2014
        ''
        if (CenterFrequency <= 150000000):
            return 0
        elif (CenterFrequency <= 4000000000):
            return 1
        elif (CenterFrequency <= 6500000000):
            return 2
        elif (CenterFrequency <= 8300000000):
            return 3
        elif (CenterFrequency <= 9750000000):
            return 4
        else:
            return 5

    def get_data_point(self,centerFrequency,devNum=0):
        self.configure(centerFrequency)
        self.setup_LO(centerFrequency)
        self.start_streaming()
        value = self.get_data_packet()
        self.stop_streaming()
        return float(value)

    def get_sweep(self,centerFrequency,span,FFT=16,ImageHandling=0,devNum=0):
        self.configure(centerFrequency,10,1,1,0,0,devNum)
        self.S.write('GetSweep('+str(centerFrequency)+','+str(span)+','+str(FFT)+','+str(ImageHandling)+','+str(devNum)+')')
        return [float(i) for i in self.S.read().split('\t')]

    def get_sweep_plot(self,centerFrequency,span,FFT=16,ImageHandling=0,devNum=0):
        sweepPoints = self.get_sweep(centerFrequency,span,FFT,ImageHandling,devNum)
        frequencyPoints = arange(centerFrequency - span/2, centerFrequency + span/2, float(span)/len(sweepPoints))
        plt.plot(frequencyPoints,sweepPoints)


    def calibrate_mixer(self, set_offset_1, set_offset_2,
                        centerFrequency, devNum):
        ''
        This is not really working...
        set_offset_1 needs to be the I channel dc-offset function,
        like AWG.set_ch1_offset(1)
        set_offset_2 needs to be the Q channel dc-offset function,
         like AWG.set_ch2_offset(1)
        ''
        ivoltage = 0
        qvoltage = 0
        voltageGrid = [-0.06,-0.021,-0.009,-0.003]
        stepList = [0.02,0.007,0.003,0.001]
        dBmTable = array([[0 for x in range(7)] for x in range(7)])
        minimaTable = [[0 for x in range(2)] for x in range(5)]
        minIndex = (0,0)
        self.configure(centerFrequency,devNum) # This should be changed to a fastConfig at some point
        self.setupLO(centerFrequency,devNum) # Just to get the Hound streaming
        self.startStreaming(devNum)
        for q in range(4):
            ivoltage = minimaTable[q][0] + voltageGrid[q] #Get starting point for new grid
            qvoltage = minimaTable[q][1] + voltageGrid[q]
            for i in range(7): # Step the i voltage in this loop
                # Set the I voltage of the AWG
                self.set_offset_1(ivoltage)
                for j in range(7): # Step the q voltage in this loop
                    self.set_offset_2(qvoltage) # Set the q voltage of the AWG
                    dBmTable[i][j] = self.getDataPacket(devNum)
                    qvoltage += stepList[q]
                qvoltage = minimaTable[q][0] + voltageGrid[q]
                ivoltage += stepList[q]
            minIndex = unravel_index(dBmTable.argmin(),bla.shape) # (ivoltIndex,qvoltIndex)
            minimaTable[q+1][0] = minimaTable[q][0] + voltageGrid[q] + minIndex[0]*stepList[q]
            minimaTable[q+1][1] = minimaTable[q][1] + voltageGrid[q] + minIndex[1]*stepList[q]
        # Result for i and q voltage is stored in minimaTable[5]
        # Set the outputs of the AWG to the values we found
        self.set_offset_1(minimaTable[5][0])
        self.set_offset_2(minimaTable[5][1])
        streamingValue = self.getDataPacket(devNum)
        # Stop the Data streaming
        self.stopStreaming(devNum)
        print "%s dBm at %d Hz for I-Voltage %d and Q-Voltage %d." % (streamingValue, centerFrequency, minimaTable[5][0],minimaTable[5][1])
        self.getSweepPlot(centerFrequency,30000000,256,0,devNum)







#execdir = qt.config['execdir']
#signalhound_folder = '\\instrument_plugins\\user_instruments\\_SignalHound'
#self.signalhoundfolderpath = os.path.join(execdir,signalhound_folder)

# class Socket:
#     # Should be moved to qtlab source as it is currently being used in multiple user instruments
#     def __init__(self, host, port):
#         self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self._socket.connect((host, port))
#         self._socket.settimeout(20)

#     def clear(self):
#         rlist, wlist, xlist = select.select([self._socket], [], [])
#         if len(rlist) == 0:
#             return
#         ret = self.read()
#         print 'Unexpected data before ask(): %r' % (ret, )

#     def properwrite(self, data):
#         self.clear()
#         qt.msleep(waittime)
#         if len(data) > 0 and data[-1] != '\r\n':
#             data += '\n'
#         self._socket.send(data+'\r\n')

#     def write(self,data):
#         qt.msleep(waittime)
#         self._socket.send(data+'\r\n')

#     def read(self,timeouttime=20):
#         start = time()
#         try:
#             ans = ''
#             while len(ans) == 0 and (time() - start) < timeouttime or not has_newline(ans):
#                 ans2 = self._socket.recv(8192)
#                 ans += ans2
#                 if len(ans2) == 0:
#                     qt.msleep(waittime)
#             AWGlastdataread=ans
#         except socket.timeout, e:
#             print 'Timed out'
#             return ''

#         if len(ans) > 0:
#             ans = ans.rstrip('\r\n')
#         return ans

#     def ask(self, data):
#         self.clear()
#         qt.msleep(waittime)
#         self.write(data)
#         return self.read()

'''