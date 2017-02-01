import numpy as np
import qcodes as qc

f_span = 30e6
# These names have to be replaced with whatever you have named them
q0 = qc.station['q0_name']
device = qc.station['device_name']
qubits = [q0]

for q in qubits:
    # CW experiments
    # MC.soft_avg(1)
    # q.RO_acq_averages(2**8)
    # q.find_resonator_frequency(freqs=None,
    #                            use_min=True, update=False)
    # # Updates the RO but not the resonator frequency
    # q.RO_acq_averages(2**13)

    # q.find_frequency(freqs=np.arange(q.f_max()-f_span/2,
    #                                  q.f_max()+f_span/2, 0.2e6), pulsed=True)

    q.RO_acq_averages(512)
    MC.soft_avg(4)
    q.measure_rabi()
    q.find_pulse_amplitude(np.linspace(-.5, .5, 31))
    # q.find_frequency(method='ramsey')
    q.measure_motzoi_XY(np.linspace(-.3, .3, 61))
    q.find_pulse_amplitude(q.amp180())
    times = np.arange(0.5e-6, 80e-6, 1e-6)
    # q.measure_T1(times)
    q.measure_echo(times, artificial_detuning=4/times[-1])
    q.measure_allxy()

    # leakage and skewness calibration
    # q.measure_ssro()


########################################
# Bus characterization T1

sq_flm.measure_SWAPN(device, q0.name,
                     swap_amps = np.arange(1.04, 1.06, 0.01))
sq_flm.measure_BusT1(device, q0.name,
                     times=np.arange(0, 40e-6, 1e-6))

