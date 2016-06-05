import numpy as np
import time
from modules.measurement import detector_functions as det
from modules.analysis import measurement_analysis as ma

station = station
PyCQEDpath = PyCQEDpath
set_trigger_fast = set_trigger_fast
set_trigger_slow = set_trigger_slow
calibrate_RO_threshold_no_rotation = calibrate_RO_threshold_no_rotation

VIP_mon_2_dux = station.VIP_mon_2_dux
MC = station.MC
Dux = station.Dux
CBox = station.CBox
AWG = station.AWG


exec(open(PyCQEDpath+'\scripts\Experiments\Restless\prepare_for_restless.py').read())

nr_cliffords = [20, 40, 100, 300]
nr_seeds = 200
nr_iterations = 3
attenuations = np.linspace(0, 1, 21)

############################################################
# Short sequences for testing and building analysis
nr_cliffords = [20, 40]
nr_seeds = 10
nr_iterations = 3
attenuations = np.linspace(0, 1, 11)

# Comment out between the brackets for the night run
############################################################

Dux.in1_out1_switch('ON')
Dux.in2_out1_switch('ON')
pulse_pars, RO_pars = VIP_mon_2_dux.get_pulse_pars()

detector_restless = det.CBox_single_qubit_event_s_fraction(CBox)
detector_traditional = det.CBox_single_qubit_frac1_counter(CBox)

t0 = time.time()
set_trigger_slow()
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
            MC.set_sweep_functions([Dux.in1_out1_attenuation,
                                    Dux.in2_out1_attenuation])
            MC.set_sweep_points(attenuations)
            MC.set_sweep_points_2D(attenuations)
            MC.set_detector_function(detector_restless)
            MC.run('RB_restless_heatmap_{}cl_{}sds'.format(ncl, nr_seeds),
                   mode='2D')
            ma.TwoD_Analysis()

            # Conventional
            set_trigger_slow()
            sq.Randomized_Benchmarking_seq(
                pulse_pars, RO_pars, [ncl], nr_seeds=nr_seeds,
                net_clifford=0, post_msmt_delay=3e-6,
                cal_points=False, resetless=True)
            MC.set_detector_function(detector_traditional)
            AWG.start()
            MC.run('RB_conventional_heatmap_{}cl_{}sds'.format(ncl, nr_seeds),
                   mode='2D')
            ma.TwoD_Analysis()
    except Exception:
        print('excepting error')
