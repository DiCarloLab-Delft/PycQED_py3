import numpy as np
# import qt
from matplotlib import pyplot as plt
from pycqed.measurement import hdf5_data
import time
import h5py


###################
# Pulse envelopes #
###################

#This is a general container of pulse functions that will be used for
#qubit pulses and RO pulses that are generated from all look-up-table based AWG's (so not
#tektronix  5014 and 520). This module replaces the CBox pulse generator.

def gauss_pulse(amp, sigma_length, axis='x', nr_sigma=4, sampling_rate=2e8,
                motzoi=0, delay=0):
    '''
    All inputs are in s and Hz.
    '''
    # why the Gaussian pulse has different values at [0] and [-1]?
    nr_sigma_samples = int(sigma_length * sampling_rate)
    nr_pulse_samples = int(nr_sigma*nr_sigma_samples)
    mu = ((nr_pulse_samples-1)/2.)
    pulse_samples = np.linspace(0, nr_pulse_samples, nr_pulse_samples,
                                endpoint=False)
    delay_samples = int(delay*sampling_rate)
    # generate pulses
    if axis == 'x':
        pulse_I = amp*np.exp(-0.5*(np.square((pulse_samples-mu) /
                             nr_sigma_samples)))
        pulse_Q = -amp*motzoi*(((pulse_samples-mu)/(nr_sigma_samples**2)) *
                               np.exp(-0.5*(np.square((pulse_samples-mu) /
                                      nr_sigma_samples))))
        # remove offset from I
        offset_I = (pulse_I[0]+pulse_I[-1])/2
        pulse_I = pulse_I-offset_I
    elif axis == 'y':
        pulse_I = amp*motzoi*(((pulse_samples-mu)/(nr_sigma_samples**2)) *
                               np.exp(-0.5*(np.square((pulse_samples-mu) /
                                      nr_sigma_samples))))
        pulse_Q = amp*np.exp(-0.5*(np.square((pulse_samples-mu) /
                             nr_sigma_samples)))
        # remove offset from Q
        offset_Q = (pulse_Q[0]+pulse_Q[-1])/2
        pulse_Q = pulse_Q-offset_Q
    Zeros = np.zeros(delay_samples)
    pulse_I = list(Zeros)+list(pulse_I)
    pulse_Q = list(Zeros)+list(pulse_Q)
    return pulse_I, pulse_Q


def block_pulse(amp, length, sampling_rate=2e8, delay=0, phase=0):
    '''
    Generates the envelope of a block pulse.
        length in s
        amp in V
        sampling_rate in Hz
        empty delay in s
        phase in degrees
    '''
    nr_samples = (length+delay)*sampling_rate
    delay_samples = int(delay*sampling_rate)
    pulse_samples = int(nr_samples - delay_samples)
    amp_I = amp*np.cos(phase*2*np.pi/360)
    amp_Q = amp*np.sin(phase*2*np.pi/360)
    block_I = amp_I * np.ones(pulse_samples)
    block_Q = amp_Q * np.ones(pulse_samples)
    Zeros = np.zeros(delay_samples)
    pulse_I = list(Zeros)+list(block_I)
    pulse_Q = list(Zeros)+list(block_Q)
    return pulse_I, pulse_Q

####################
# Pulse modulation #
####################


def mod_pulse(pulse_I, pulse_Q, f_modulation,
              Q_phase_delay=0, sampling_rate=2e8):
    '''
    inputs are in s and Hz.
    Q_phase_delay is in degree

    transformation:
    [I_mod] = [cos(wt)            sin(wt)] [I_env]
    [Q_mod]   [-sin(wt+phi)   cos(wt+phi)] [Q_env]

    phase delay is applied to Q_mod as a whole because it is to correct a
    mixer phase offset.
    To add phase to the pulse itself edit the envelope function.
    '''
    Q_phase_delay_rad = 2*np.pi * Q_phase_delay/360.
    nr_pulse_samples = len(pulse_I)
    f_mod_samples = f_modulation/sampling_rate
    pulse_samples = np.linspace(0, nr_pulse_samples, nr_pulse_samples,
                                endpoint=False)

    pulse_I_mod = pulse_I*np.cos(2*np.pi*f_mod_samples*pulse_samples) + \
        pulse_Q*np.sin(2*np.pi*f_mod_samples*pulse_samples)
    pulse_Q_mod = pulse_I*-np.sin(2*np.pi*f_mod_samples*pulse_samples +
                                  Q_phase_delay_rad) + \
        pulse_Q*np.cos(2*np.pi*f_mod_samples*pulse_samples + Q_phase_delay_rad)

    return pulse_I_mod, pulse_Q_mod


def simple_mod_pulse(pulse_I, pulse_Q, f_modulation,
                     Q_phase_delay=0, sampling_rate=2e8):
    '''
    inputs are in s and Hz.
    Q_phase_delay is in degree

    transformation:
    [I_mod] = [cos(wt)            0] [I_env]
    [Q_mod]   [0        sin(wt+phi)] [Q_env]

    phase delay is applied to Q_mod as a whole because it is to correct a
    mixer phase offset.
    To add phase to the pulse itself edit the envelope function.
    '''
    Q_phase_delay_rad = 2*np.pi * Q_phase_delay/360.
    nr_pulse_samples = len(pulse_I)
    f_mod_samples = f_modulation/sampling_rate
    pulse_samples = np.linspace(0, nr_pulse_samples, nr_pulse_samples,
                                endpoint=False)

    pulse_I_mod = pulse_I*np.cos(2*np.pi*f_mod_samples*pulse_samples)
    pulse_Q_mod = pulse_Q*np.sin(2*np.pi*f_mod_samples*pulse_samples +
                                 Q_phase_delay_rad)
    return pulse_I_mod, pulse_Q_mod


def mod_gauss(amp, sigma_length, f_modulation, axis='x',
              motzoi=0, sampling_rate=2e8,
              Q_phase_delay=0, delay=0, nr_sigma=4):
    '''
    Simple gauss pulse maker for CBOX. All inputs are in s and Hz.
    '''
    pulse_I, pulse_Q = gauss_pulse(amp, sigma_length, nr_sigma=nr_sigma,
                                   sampling_rate=sampling_rate, axis=axis,
                                   motzoi=motzoi, delay=delay)
    pulse_I_mod, pulse_Q_mod = mod_pulse(pulse_I, pulse_Q, f_modulation,
                                         sampling_rate=sampling_rate,
                                         Q_phase_delay=Q_phase_delay)
    return pulse_I_mod, pulse_Q_mod

#####################################################
# Sequences (should maybe not be in the pulse gen)  #
#####################################################


def Rabi_CBox(n_waves):
    CBox = qt.instruments['CBox']
    # samples, pulse_I_mod, pulse_Q_mod = Rabi_pulses_CBox(n_waves,amplitude=8000,sigma_length=25, f_modulation=-0.02, sampling_rate=0.2,nr_sigma=4)
    # loop has to be added
    list_amplitudes = np.linspace(-8000, 8000, n_waves)
    CBox.set_AWG_triggermode(0, 'False')

    #list_amplitudes=8000*np.sin(list_amplitudes*2*np.pi/n_waves)

    print('Starting test measurement')
    # initialization of variables
    InputAvgRes0 = np.zeros(n_waves)
    InputAvgRes1 = np.zeros(n_waves)
    CBox.set_acquisition_mode(0)
    for ii, amp in enumerate(list_amplitudes):
        # generate the sidebanded Gaussian pulse and load it on the FPGA (LUT#0)
        Wave_I, Wave_Q = mod_gauss(amp, 25, -0.02)
        CBox.set_awg_lookuptable(0, 0, 1, np.round(Wave_I))
        CBox.set_awg_lookuptable(0, 0, 0, np.round(Wave_Q))
        # print "n_samples", len(list_amplitudes)
        # print 'list_amps', list_amplitudes
        #qt.msleep(0.5)
        No_samples = 1  # max 2000
        NoAvg = 12  # max 17
        CBox.set_averaging_parameters(No_samples, NoAvg)
        CBox.set_acquisition_mode(4)
        #qt.msleep(1)
        [InputAvgRes0[ii], InputAvgRes1[ii]] = CBox.get_integrated_avg_results(
            timeout=60)
        CBox.set_acquisition_mode(0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('pulse amplitude (DAC value)', fontsize=14)
    ax.set_ylabel('output amplitude (Integrated ADC value)', fontsize=14)
    ax.plot(list_amplitudes, InputAvgRes0, label='I')
    ax.plot(list_amplitudes, InputAvgRes1, label='Q')
    ax.set_title('Rabi with CBox generating pulses and acquiring data')
    plt.legend()
    plt.show()

    return InputAvgRes0, InputAvgRes1


def Check_trace(No_points, NoAvg=8):
    CBox = qt.instruments['CBox']
    CBox.set_acquisition_mode(0)
    CBox.set_averaging_parameters(No_points, NoAvg)
    CBox.set_acquisition_mode(3)
    InputAvgRes0, InputAvgRes1 = CBox.get_input_avg_results(timeout=30)
    plt.plot(InputAvgRes0, label='1')

    return InputAvgRes0, InputAvgRes1


def T1_CBox(n_waves=70, time_step=2000, amp180=4000):
    #input("stop AWG and hit enter....")
    CBox = qt.instruments['CBox']
    AWG = qt.instruments['AWG']
    CBox.set_acquisition_mode(0)
    # sets the codeword to False
    CBox.set_AWG_triggermode(0, 'False')
    No_points = 70
    NoAvg = 6 #14 # This is a lot and should not be hardcoded
    # Generates wavetables
    Wave_I, Wave_Q = mod_gauss(amp180, 25, -0.02)
    CBox.set_awg_lookuptable(0, 0, 1, np.round(Wave_I))
    CBox.set_awg_lookuptable(0, 0, 0, np.round(Wave_Q))
    CBox.set_averaging_parameters(No_points, NoAvg)
    CBox.set_acquisition_mode(4)
    # This trigger starts the measurement
    AWG.start()
    [InputAvgRes0, InputAvgRes1] = CBox.get_integrated_avg_results(timeout=360)

    # Creates the corresponding sweeppoints
    time_samples = np.linspace(0, (n_waves-1)*time_step, n_waves)

    # Below is data saving
    AWG.stop()
    filename = hdf5_data.Data(name="T1")

    if "Experimental Data" in filename:
        grp = filename["Experimental Data"]
    else:
        grp = filename.create_group('Experimental Data')

    dataset_samples = grp.create_dataset('time', time_samples.shape,
                                         dtype=np.float)
    dataset_InputAvgRes0 = grp.create_dataset('I_raw', InputAvgRes0.shape,
                                              dtype=np.float)
    dataset_InputAvgRes1 = grp.create_dataset('Q_raw', InputAvgRes1.shape,
                                              dtype=np.float)
    dataset_samples = time_samples
    dataset_InputAvgRes0[:] = InputAvgRes0[:]
    dataset_InputAvgRes1[:] = InputAvgRes1[:]
    filename.close()

    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.set_xlabel('time (us)', fontsize=14)
    # ax.set_ylabel('output amplitude (Integrated ADC value)', fontsize=14)
    # ax.plot(time_samples,InputAvgRes0, label = 'I')
    # ax.plot(time_samples,InputAvgRes1, label = 'Q')
    # ax.set_title('T1 with CBox generating pulses and acquiring data')
    # plt.show()
    return InputAvgRes0, InputAvgRes1

def T2_Echo_CBox(n_waves=70,time_step=600,amp180=4000, amp90=2000):
    #input("stop AWG and hit enter....")
    CBox = qt.instruments['CBox']
    CBox.set_acquisition_mode(0)
    CBox.set_AWG_triggermode(0, 'True')
    No_points = 70
    NoAvg = 12
    CBox.set
    Wave_I, Wave_Q = mod_gauss(amp180, 25, -0.02)
    CBox.set_awg_lookuptable(0, 7, 1, np.round(Wave_I))
    CBox.set_awg_lookuptable(0, 7, 0, np.round(Wave_Q))
    CBox.set_averaging_parameters(No_points, NoAvg)

    Wave_I, Wave_Q = mod_gauss(amp90, 25, -0.02)
    CBox.set_awg_lookuptable(0, 0, 1, np.round(Wave_I))
    CBox.set_awg_lookuptable(0, 0, 0, np.round(Wave_Q))
    CBox.set_averaging_parameters(No_points, NoAvg)

    CBox.set_averaging_parameters(No_points, NoAvg)
    CBox.set_acquisition_mode(4)
    AWG.start()
    [InputAvgRes0, InputAvgRes1] = CBox.get_integrated_avg_results(timeout=60)
    time_samples = np.linspace(0, (n_waves-1)*time_step, n_waves)
    AWG.stop()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('time (us)', fontsize=14)
    ax.set_ylabel('output amplitude (Integrated ADC value)', fontsize=14)
    ax.plot(time_samples, InputAvgRes0, label='I')
    ax.plot(time_samples, InputAvgRes1, label='Q')
    ax.set_title('T_Echo with CBox generating pulses and acquiring data')
    plt.legend()
    plt.show()
    return InputAvgRes0, InputAvgRes1


def T1_night(n_rep):
    for i in range(n_rep):
        print("mearurement", i)
        T1_CBox(n_waves=70, time_step=2000, amp180=4000)
