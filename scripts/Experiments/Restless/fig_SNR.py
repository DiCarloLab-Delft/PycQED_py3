import numpy as np
import time
from modules.measurement import detector_functions as det

station = station
PyCQEDpath = PyCQEDpath
VIP_mon_2_dux = station.VIP_mon_2_dux
MC = station.MC
Dux = station.Dux
CBox = station.CBox
AWG = station.AWG


exec(open(PyCQEDpath+'\scripts\Experiments\Restless\prepare_for_restless.py').read())

nr_cliffords = [2, 4, 6, 8, 12, 16, 20, 24, 30, 40,
                60, 80, 100, 140, 180, 200, 240, 280, 300, 380,
                500, 600, 800, 1200, 1500]
nr_seeds = 200
nr_iterations = 300
center = 0.315338304148
attenuations = np.linspace(center-.05, center+.05, 21)
# attenuations = np.arange(.27 + 0.000338, .32001, .001)


############################################################
# Short sequences for testing and building analysis
nr_cliffords = [2, 8,  20, 60, 100,  300, 600, 1200]
nr_seeds = 5
nr_iterations = 3
# Comment out between the brackets for the night run
############################################################

DUX_1_default = 0.3
DUX_2_default = 0.700729836874
print('Setting dux attenuations')
Dux.in1_out1_attenuation(DUX_1_default)
Dux.in2_out1_attenuation(DUX_2_default)
Dux.in1_out1_switch('ON')
Dux.in2_out1_switch('ON')
pulse_pars, RO_pars = VIP_mon_2_dux.get_pulse_pars()


detector_restless = det.CBox_single_qubit_event_s_fraction(CBox)
detector_traditional = det.CBox_single_qubit_frac1_counter(CBox)

t0 = time.time()
VIP_mon_2_dux.measure_ssro(close_fig=True, set_integration_weights=True)


for i in range(nr_iterations):
    try:
        # Restless heatmap
        set_trigger_fast()
        calibrate_RO_threshold_no_rotation()
        for i, ncl in enumerate(nr_cliffords):
            set_trigger_fast()
            sq.Randomized_Benchmarking_seq(
                pulse_pars, RO_pars, [ncl], nr_seeds=nr_seeds,
                net_clifford=3, post_msmt_delay=3e-6,
                cal_points=False, resetless=True)
            AWG.start()
            MC.set_sweep_function(Dux.in1_out1_attenuation)
            MC.set_sweep_points(attenuations)
            MC.set_detector_function(detector_restless)
            MC.run('RB_restless_{}cl_{}sds'.format(ncl, nr_seeds))
            ma.MeasurementAnalysis()

            # Conventional
            set_trigger_slow()
            sq.Randomized_Benchmarking_seq(
                pulse_pars, RO_pars, [ncl], nr_seeds=nr_seeds,
                net_clifford=0, post_msmt_delay=3e-6,
                cal_points=False, resetless=True)
            MC.set_detector_function(detector_traditional)
            AWG.start()
            MC.run('RB_conventional_{}cl_{}sds'.format(ncl, nr_seeds))
            ma.MeasurementAnalysis()
    except Exception:
        print('excepting error')
