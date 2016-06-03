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

# nr_cliffords = [2, 4, 6, 8, 12, 16, 20, 24, 30, 40,
#                 60, 80, 100, 140, 180, 200, 240, 280, 300, 380,
#                 500, 600, 800, 1200, 1500]
nr_seeds = 200
nr_cliffords = [2, 4, 8, 16, 30, 60, 100, 200, 300, 400, 600, 800, 1200, 1500]
# nr_seeds = 5
center = 0.315338304148
attenuations = np.arange(.27 + 0.000338, .32001, .001)


nr_averages = 2048*4  # Approx 8000
DUX_1_default = 0.3
DUX_2_default = 0.700729836874
print('Setting dux attenuations')
Dux.in1_out1_attenuation(DUX_1_default)
Dux.in2_out1_attenuation(DUX_2_default)
Dux.in1_out1_switch('ON')
Dux.in2_out1_switch('ON')
pulse_pars, RO_pars = VIP_mon_2_dux.get_pulse_pars()


detector_restless = det.CBox_single_qubit_event_s_fraction(CBox)
detector_traditional = det.CBox_integrated_average_detector(CBox, AWG)
CBox.nr_averages(nr_averages)

t0 = time.time()
nr_cliff_cal_elts = np.concatenate([nr_cliffords,
                                   [nr_cliffords[-1]+.2, nr_cliffords[-1]+.3,
                                    nr_cliffords[-1]+.7, nr_cliffords[-1]+.8]])


# Long sequence stored here to save time on regenerating
# Need to do this slightly smarter
VIP_mon_2_dux.measure_ssro(close_fig=True, set_integration_weights=True)


for i in range(100):
    try:
        # Restless heatmap
        set_trigger_fast()
        calibrate_RO_threshold_no_rotation()
        MC.live_plot_enabled = True
        for i, ncl in enumerate(nr_cliffords):
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

        # # Conventional 2D heatmap
        set_trigger_slow()
        print('starting generation')
        conv_seq, conv_el_list = sq.Randomized_Benchmarking_seq(
            pulse_pars, RO_pars, nr_seeds=nr_seeds,
            nr_cliffords=nr_cliff_cal_elts, upload=True)
        t1 = time.time()
        print('generating conv RB seq took {:.1f}s'.format(t1-t0))
        # AWG.stop()
        # station.pulsar.program_awg(conv_seq, *conv_el_list, verbose=False)
        MC.live_plot_enabled = False
        MC.set_sweep_function(awg_swf.Randomized_Benchmarking(
            pulse_pars, RO_pars, nr_seeds=nr_seeds,
            nr_cliffords=nr_cliffords, upload=False))
        # Watch out ! due to the nature of the 2D sweep the sequence that get's
        # uploaded goes out of
        MC.set_detector_function(detector_traditional)
        MC.set_sweep_function_2D(Dux.in1_out1_attenuation)
        MC.set_sweep_points_2D(attenuations)
        MC.run('RB_conventional_2D_{}sds'.format(nr_seeds), mode='2D')
        ma.TwoD_Analysis()
    except Exception:
        print('excepting error')