#talking to oscilloscope
#Input the relevant parameters, returns the trace of signal(multiple acquisitions)
import qcodes as qc
from qcodes.instrument import visa
instr = visa.VisaInstrument('SCOPE', address='TCPIP0::192.168.0.3')
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("D:/Repository/PyCQED_py3")
import PycQED_py3.pycqed.measurement.waveform_control_CC.waveform as wvf
from time import sleep
def setup_oscilloscope(start,stop, acq_rate = 10000000000)
	"""
	Initializes oscilloscope 
	acquisition rate is set to 10000000000
	takes as an input starting/stopping point in seconds
	returns nothing
	"""
	opc = instr.visa_handle.ask('STOP;*OPC?') 
	instr.visa_handle.write('ACQuire:SRATe 10000000000')
	instr.visa_handle.write('EXPort:WAVeform:FASTexport ON')
    instr.visa_handle.write('CHANnel1:WAVeform1:STATe 1')
    instr.visa_handle.write('EXPort:WAVeform:STARt start')
    instr.visa_handle.write('EXPort:WAVeform:STOP stop')
    sampling_rate = float(instr.visa_handle.ask('ACQuire:POINts:ARATe?'))*1e-9
    print('sampling_rate')
    print('oscilloscope initialized')
    return 

def trace_oscilloscope(acq_count,x_range,y_range,start,stop):
    '''
    function to get data out from oscilloscope as a numpy array
    takes as an input 
    '''
    setup_oscilloscope(start,stop)
    # defines the acquisition no. for run single mode
    instr.visa_handle.write('ACQuire:COUNt acq_count')
    # we use manual measurements of how much time it takes to
    # acquire the one waveform and interpolate to wait accordingly 
    #before sending extracting data
    sleep(0.21*acquire_count + 1.16)
    instr.visa_handle.write('EXPort:WAVeform:SOURce C1W1') 
    instr.visa_handle.write('CHANnel1:ARIThmetics AVERage')
    ret_str = instr.visa_handle.ask('CHANNEL1:WAVEFORM1:DATA?')
    array = ret_str.split(',')
    array = np.double(array)
    x_values = array[::2]
    y_values = array[1::2]
    inline matplotlib
    plt.plot(x_values, y_values)
    return x_values, y_values
    
    
