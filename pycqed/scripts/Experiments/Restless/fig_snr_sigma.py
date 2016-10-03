import numpy as np
import time
from pycqed.measurement import detector_functions as det

station = station
PyCQEDpath = PyCQEDpath
VIP_mon_2_dux = station.VIP_mon_2_dux
MC = station.MC
Dux = station.Dux
CBox = station.CBox
AWG = station.AWG


exec(open(PyCQEDpath+'\scripts\Experiments\Restless\prepare_for_restless.py').read())

nr_cliffords = [2, 4, 5, 6, 8, 10, 12, 13, 16, 20, 24, 30, 40,
                60, 80, 100, 140, 180, 200, 240, 280, 300, 380, 450,
                500, 550, 600, 700, 800, 1000, 1200, 1500]
nr_seeds = 200
nr_iterations = 50
pts_per_iteration = 50
# Attenuations should correspond to
#             .999, .998, .995, .994, .99, .985 .98
attenuations = [.4, .41, .417, .422, .431, .435, .446, .457, .467]
#                  4.1 and      4.22 are extra safety points

############################################################
# Short sequences for testing and building analysis
# nr_cliffords = [2, 8,  20, 60, 100]
# nr_iterations = 2
# pts_per_iteration = 100
# attenuations = [.4, .417, 4.7]
# Comment out between the brackets for the night run
############################################################

# print('Setting dux attenuations')
Dux.in1_out1_switch('ON')
Dux.in2_out1_switch('ON')
detector_restless = det.CBox_single_qubit_event_s_fraction(CBox)
detector_traditional = det.CBox_single_qubit_frac1_counter(CBox)

t0 = time.time()

par = pw.wrap_par_remainder(Dux.in1_out1_attenuation, remainder=1)
log_length = (8000)

for i in range(nr_iterations):
    try:
        set_CBox_cos_sine_weigths(VIP_mon_2_dux.f_RO_mod())
        pulse_pars, RO_pars = calibrate_pulse_pars_conventional()
        VIP_mon_2_dux.measure_ssro(close_fig=True, set_integration_weights=True)
        set_trigger_fast()
        for i, ncl in enumerate(nr_cliffords):
            sq.Randomized_Benchmarking_seq(
                pulse_pars, RO_pars, [ncl], nr_seeds=nr_seeds,
                net_clifford=3, post_msmt_delay=3e-6,
                cal_points=False, resetless=True)
            calibrate_RO_threshold_no_rotation()
            for att in attenuations:
                sweep_pts = np.ones(pts_per_iteration)*att
                AWG.start()
                MC.set_sweep_function(par)
                MC.set_sweep_points(sweep_pts)
                MC.set_detector_function(detector_restless)
                MC.run('RB_restless_noise_{}att_{}cl_{}sds'.format(att, ncl, nr_seeds))
                ma.MeasurementAnalysis()
    except Exception:
        print('excepting error')
